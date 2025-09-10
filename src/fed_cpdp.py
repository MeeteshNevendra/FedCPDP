# ============================================================
# Federated, Privacy-Preserving CPDP (Improved + Ensembling)
# - Heterogeneous features -> hashing
# - 2-layer MLP + Dropout, Adam (fixed), optional cosine LR
# - FedAvg / FedProx with Attention / Graph / Hybrid aggregation
# - Client-level DP (clip + Gaussian noise)
# - Hidden mean+variance alignment (CORAL-style)
# - Focal loss or weighted BCE
# - Threshold tuning: max-F1 / max-recall@pmin / max-AUCEC
# - Ensembling across multiple seeds (probability averaging)
# - Safe AUC and AUCEC/Popt (np.trapezoid), LOPO eval
# ============================================================

import os, math, glob, pickle, random
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

# -----------------------------
# Reproducibility & Config
# -----------------------------
GLOBAL_BASE_SEED = 42

DATA_MODE   = "csv_folder"                # "csv_folder" or "synthetic"
CSV_FOLDER  = "/content/fed_cpdp_csvs"    # <- your 30-CSV folder
OUTPUT_DIR  = "./fed_cpdp_outputs_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_RESULTS_PATH = os.path.join(OUTPUT_DIR, "fed_cpdp_results.csv")
SAVE_MODELS_DIR   = os.path.join(OUTPUT_DIR, "models")
os.makedirs(SAVE_MODELS_DIR, exist_ok=True)

# Dataset columns
LABEL_CANDIDATES  = ["bug","defect","label","is_buggy","defective","target"]
EFFORT_CANDIDATES = ["loc","LOC","size","lines_of_code"]

# Feature hashing (heterogeneous -> common space)
HASH_DIM = 1024
STANDARDIZE_PER_CLIENT = True  # per-client zscore

# Model (2-layer MLP)
H1_DIM = 128
H2_DIM = 64
DROPOUT_P = 0.2                 # hidden dropout prob
USE_COSINE_LR = True            # cosine schedule within local epochs

# Training budget
ROUNDS        = 80
LOCAL_EPOCHS  = 12
BATCH_SIZE    = 128
LEARNING_RATE = 0.003
L2_WEIGHT_DECAY = 1e-4          # keep small (Adam uses L2-style reg here)

# Optimizer = Adam (fixed)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8

# Federated settings
USE_FEDPROX = True
FEDPROX_MU  = 0.005

# Aggregator: "avg", "attn", "graph", "hybrid"
AGGREGATOR = "hybrid"
# attention weights (delta-to-momentum similarity; loss-aware)
ATTN_ALPHA = 2.0      # weight on cosine(delta, momentum)
ATTN_BETA  = 0.5      # weight on inverse loss
MOMENTUM_EMA = 0.9    # server momentum on aggregated delta

# graph weights (representation similarity)
GRAPH_GAMMA_MEAN = 1.0  # weight on cos(hmean_i, global_mean)
GRAPH_GAMMA_VAR  = 0.5  # weight on -||hvar_i - global_var||

# Privacy (client-delta DP)
USE_DP = False
DP_CLIP_NORM = 1.0
DP_NOISE_MULTIPLIER = 0.1  # try 0.05–0.2

# Domain alignment (hidden mean+var alignment on last hidden)
USE_ALIGN = True
ALIGN_MEAN_WEIGHT = 0.003
ALIGN_VAR_WEIGHT  = 0.001

# Loss
USE_FOCAL   = True
FOCAL_GAMMA = 2.0

# Threshold tuning
THR_MODE = "max_f1"             # "max_f1", "max_recall_at_pmin", "max_aucec"
P_MIN    = 0.60                 # used only if mode == max_recall_at_pmin

# Ensembling
ENSEMBLE_SEEDS = [42, 43, 44]   # set [42] to disable ensembling

# -----------------------------
# Utility: AUC & Effort-aware
# -----------------------------
def safe_roc_auc(y_true, y_scores):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        return float('nan')

def compute_aucec_popt(y_true, y_scores, effort):
    y_true   = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    effort   = np.asarray(effort).astype(float)
    if len(y_true) == 0 or np.all(~np.isfinite(effort)):
        return float('nan'), float('nan')
    effort = np.maximum(effort, 1e-12)

    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    eff_sorted = effort[order]

    cum_eff = np.cumsum(eff_sorted)
    tot_eff = cum_eff[-1]
    if tot_eff <= 0: return float('nan'), float('nan')
    eff_frac = cum_eff / tot_eff

    cum_bug = np.cumsum(y_sorted)
    tot_bug = cum_bug[-1]
    if tot_bug <= 0: return float('nan'), float('nan')
    bug_frac = cum_bug / tot_bug

    aucec = np.trapezoid(bug_frac, eff_frac)

    density = y_true / effort
    opt_order = np.argsort(-density)
    opt_bug_frac   = np.cumsum(y_true[opt_order]) / tot_bug
    opt_eff_frac   = np.cumsum(effort[opt_order]) / tot_eff
    aucec_opt      = np.trapezoid(opt_bug_frac, opt_eff_frac)

    worst_order    = np.argsort(density)
    worst_bug_frac = np.cumsum(y_true[worst_order]) / tot_bug
    worst_eff_frac = np.cumsum(effort[worst_order]) / tot_eff
    aucec_worst    = np.trapezoid(worst_bug_frac, worst_eff_frac)

    denom = (aucec_opt - aucec_worst)
    if abs(denom) < 1e-12:
        return float(aucec), float('nan')
    popt = (aucec - aucec_worst) / denom
    return float(aucec), float(popt)

# -----------------------------
# Data loading + hashing
# -----------------------------
def _find_label_column(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # heuristic fallback: binary-like col
    for c in df.columns:
        u = df[c].dropna().unique()
        if len(u) <= 3 and set(u).issubset({0,1,"0","1",True,False}):
            return c
    return None

def _find_effort_column(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def _numeric_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def _hash_index(name: str, dim: int) -> int:
    return abs(hash(name)) % dim

def hash_numeric_features(df: pd.DataFrame, label_col: str, hash_dim: int, standardize: bool=True):
    effort_col = _find_effort_column(df, EFFORT_CANDIDATES)
    exclude = [label_col]
    if effort_col is not None:
        exclude.append(effort_col)
    num_cols = _numeric_cols(df, exclude)
    if len(num_cols) == 0:
        raise ValueError("No numeric features found besides label/effort.")

    X = np.zeros((len(df), hash_dim), dtype=np.float32)
    y = df[label_col].astype(int).values
    effort = df[effort_col].values if effort_col is not None else None

    stats = {}
    if standardize:
        for c in num_cols:
            s = df[c].astype(float).values
            m, v = np.nanmean(s), np.nanstd(s)
            v = 1.0 if (v == 0 or np.isnan(v)) else v
            stats[c] = (m, v)

    for c in num_cols:
        idx = _hash_index(c, hash_dim)
        col = df[c].astype(float).values
        if standardize:
            m, v = stats[c]
            col = (col - m) / v
        col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        X[:, idx] += col
    return X, y, effort

def load_csv_clients(csv_folder: str) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    clients = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue
        label_col = _find_label_column(df, LABEL_CANDIDATES)
        if label_col is None:
            print(f"[WARN] No label column in {path}; skipping.")
            continue
        X, y, effort = hash_numeric_features(df, label_col, HASH_DIM, STANDARDIZE_PER_CLIENT)
        clients.append({"name": os.path.basename(path), "X": X, "y": y, "effort": effort})
    return clients

def make_synthetic_clients(n_clients=6, n_features_each=(10, 30), n_samples_each=(400, 800), hetero=True):
    clients = []
    for i in range(n_clients):
        f = np.random.randint(n_features_each[0], n_features_each[1]+1)
        n = np.random.randint(n_samples_each[0], n_samples_each[1]+1)
        feats = [f"feat_{i}_{k}" if hetero else f"feat_{k}" for k in range(f)]
        data = {c: (np.random.randn(n)*np.random.uniform(0.5, 2.0) + np.random.uniform(-1.0, 1.0)) for c in feats}
        true_w = np.random.randn(f)
        logits = sum(data[c]*true_w[idx] for idx, c in enumerate(feats)) + np.random.randn(n)*0.5
        probs = 1/(1+np.exp(-logits))
        y = (probs > 0.5).astype(int)
        loc = np.random.lognormal(mean=5.5 + 0.4*y, sigma=0.6, size=n).astype(float)
        df = pd.DataFrame(data); df["bug"]=y; df["loc"]=loc
        X, y_vec, effort = hash_numeric_features(df, "bug", HASH_DIM, STANDARDIZE_PER_CLIENT)
        clients.append({"name": f"client_{i+1}.csv", "X": X, "y": y_vec, "effort": effort})
    return clients

# -----------------------------
# Model: 2-layer MLP + Dropout
# -----------------------------
class MLP2:
    def __init__(self, in_dim, h1, h2, out_dim, l2=0.0, seed=GLOBAL_BASE_SEED):
        rs = np.random.RandomState(seed)
        self.W1 = rs.randn(in_dim, h1) * (1.0/ math.sqrt(in_dim))
        self.b1 = np.zeros(h1)
        self.W2 = rs.randn(h1, h2) * (1.0/ math.sqrt(h1))
        self.b2 = np.zeros(h2)
        self.W3 = rs.randn(h2, out_dim) * (1.0/ math.sqrt(h2))
        self.b3 = np.zeros(out_dim)
        self.l2 = l2

    def forward(self, X, train=False, dropout_p=0.0, rs: Optional[np.random.RandomState]=None):
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        # dropout on h1 if train
        if train and dropout_p > 0.0:
            if rs is None: rs = np.random.RandomState(GLOBAL_BASE_SEED)
            mask1 = (rs.rand(*h1.shape) > dropout_p).astype(float) / (1.0 - dropout_p)
            h1d = h1 * mask1
        else:
            mask1 = None; h1d = h1

        z2 = h1d @ self.W2 + self.b2
        h2 = np.maximum(0, z2)
        if train and dropout_p > 0.0:
            if rs is None: rs = np.random.RandomState(GLOBAL_BASE_SEED+1)
            mask2 = (rs.rand(*h2.shape) > dropout_p).astype(float) / (1.0 - dropout_p)
            h2d = h2 * mask2
        else:
            mask2 = None; h2d = h2

        z3 = h2d @ self.W3 + self.b3
        yhat = 1.0 / (1.0 + np.exp(-z3))
        return (z1,h1,mask1,z2,h2,mask2,z3,yhat)

    def predict_proba(self, X):
        return self.forward(X, train=False)[-1].ravel()

    def _pos_weight(self, y):
        n = y.shape[0]
        pos = max(1.0, float(y.sum()))
        neg = max(1.0, float(n - y.sum()))
        return neg / pos

    def loss_and_grads(self, X, y,
                       global_params=None,
                       fedprox_mu=0.0,
                       align_mean_w=0.0, align_var_w=0.0,
                       global_mean=None, global_var=None,
                       use_focal=False, focal_gamma=2.0,
                       dropout_p=0.0, rs=None):
        n = X.shape[0]
        z1,h1,m1,z2,h2,m2,z3,yhat = self.forward(X, train=True, dropout_p=dropout_p, rs=rs)
        y = y.reshape(-1,1).astype(float)
        eps = 1e-12

        # Loss
        if not use_focal:
            pos_w = self._pos_weight(y)
            bce = -(pos_w*y*np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps)).mean()
            dz3 = (yhat - y) * (1 + (pos_w-1)*y) / n
        else:
            pos_w = self._pos_weight(y)
            p = yhat
            fl_pos = - pos_w * ((1-p)**focal_gamma) * y * np.log(p+eps)
            fl_neg = - (p**focal_gamma) * (1-y) * np.log(1-p+eps)
            bce = (fl_pos + fl_neg).mean()
            dz3 = (p - y) * (1 + (pos_w-1)*y) * ((1-p)**focal_gamma + (p**focal_gamma)) / n

        l2 = 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))

        # FedProx
        prox = 0.0
        if fedprox_mu > 0.0 and global_params is not None:
            gW1,gb1,gW2,gb2,gW3,gb3 = global_params
            prox = 0.5*fedprox_mu*(np.sum((self.W1-gW1)**2)+np.sum((self.b1-gb1)**2)
                                   +np.sum((self.W2-gW2)**2)+np.sum((self.b2-gb2)**2)
                                   +np.sum((self.W3-gW3)**2)+np.sum((self.b3-gb3)**2))

        # Alignment (on h2)
        align = 0.0
        h_mean = h2.mean(axis=0)
        h_var  = h2.var(axis=0)
        if align_mean_w > 0.0 and global_mean is not None:
            align += 0.5*align_mean_w*np.sum((h_mean - global_mean)**2)
        if align_var_w > 0.0 and global_var is not None:
            align += 0.5*align_var_w*np.sum((h_var - global_var)**2)

        loss = bce + l2 + prox + align

        # Backprop
        dW3 = (h2 if m2 is None else (h2*m2)).T @ dz3 + self.l2*self.W3
        db3 = dz3.sum(axis=0)

        dh2 = dz3 @ self.W3.T
        if m2 is not None: dh2 = dh2 * m2
        dz2 = dh2 * (z2 > 0)

        dW2 = (h1 if m1 is None else (h1*m1)).T @ dz2 + self.l2*self.W2
        db2 = dz2.sum(axis=0)

        dh1 = dz2 @ self.W2.T
        if m1 is not None: dh1 = dh1 * m1
        dz1 = dh1 * (z1 > 0)

        dW1 = X.T @ dz1 + self.l2*self.W1
        db1 = dz1.sum(axis=0)

        # FedProx grads
        if fedprox_mu > 0.0 and global_params is not None:
            dW1 += fedprox_mu*(self.W1 - gW1); db1 += fedprox_mu*(self.b1 - gb1)
            dW2 += fedprox_mu*(self.W2 - gW2); db2 += fedprox_mu*(self.b2 - gb2)
            dW3 += fedprox_mu*(self.W3 - gW3); db3 += fedprox_mu*(self.b3 - gb3)

        # Alignment grads (mean)
        if align_mean_w > 0.0 and global_mean is not None:
            grad_h_mean = (h_mean - global_mean) * align_mean_w / n
            dz2_m = np.where(z2>0, grad_h_mean, 0.0)
            dW2 += (h1 if m1 is None else h1*m1).T @ dz2_m
            db2 += dz2_m.sum(axis=0)

        # Alignment grads (var approx)
        if align_var_w > 0.0 and global_var is not None:
            grad_var = (h_var - global_var) * align_var_w
            dh_v = (2.0/n) * (h2 - h_mean) * grad_var
            dz2_v = np.where(z2>0, dh_v, 0.0)
            dW2 += (h1 if m1 is None else h1*m1).T @ dz2_v
            db2 += dz2_v.sum(axis=0)

        grads = (dW1,db1,dW2,db2,dW3,db3)
        return loss, grads, h_mean, h_var

    def get_params(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), self.W3.copy(), self.b3.copy())

    def set_params(self, params):
        self.W1,self.b1,self.W2,self.b2,self.W3,self.b3 = [p.copy() for p in params]

# -----------------------------
# Optimizer: Adam (fixed)
# -----------------------------
class AdamState:
    def __init__(self, params):
        W1,b1,W2,b2,W3,b3 = params
        self.mW1 = np.zeros_like(W1); self.vW1 = np.zeros_like(W1)
        self.mb1 = np.zeros_like(b1); self.vb1 = np.zeros_like(b1)
        self.mW2 = np.zeros_like(W2); self.vW2 = np.zeros_like(W2)
        self.mb2 = np.zeros_like(b2); self.vb2 = np.zeros_like(b2)
        self.mW3 = np.zeros_like(W3); self.vW3 = np.zeros_like(W3)
        self.mb3 = np.zeros_like(b3); self.vb3 = np.zeros_like(b3)
        self.t = 0

def adam_step(params, grads, state: AdamState, lr, beta1, beta2, eps):
    W1,b1,W2,b2,W3,b3 = params
    dW1,db1,dW2,db2,dW3,db3 = grads
    state.t += 1

    def upd(p, g, m, v):
        m[:] = beta1*m + (1-beta1)*g
        v[:] = beta2*v + (1-beta2)*(g*g)
        mhat = m / (1-beta1**state.t)
        vhat = v / (1-beta2**state.t)
        return p - lr * mhat / (np.sqrt(vhat)+eps)

    W1 = upd(W1, dW1, state.mW1, state.vW1)
    b1 = upd(b1, db1, state.mb1, state.vb1)
    W2 = upd(W2, dW2, state.mW2, state.vW2)
    b2 = upd(b2, db2, state.mb2, state.vb2)
    W3 = upd(W3, dW3, state.mW3, state.vW3)
    b3 = upd(b3, db3, state.mb3, state.vb3)
    return (W1,b1,W2,b2,W3,b3), state

def sgd_step(params, grads, lr):
    return tuple(p - lr*g for p,g in zip(params, grads))

# -----------------------------
# Federated helpers
# -----------------------------
def clip_and_add_noise(delta_params, clip_norm, noise_multiplier, rng):
    flat = np.concatenate([p.ravel() for p in delta_params])
    norm = np.linalg.norm(flat) + 1e-12
    scale = min(1.0, clip_norm / norm)
    flat = flat * scale
    sigma = noise_multiplier * clip_norm
    flat += rng.normal(0.0, sigma, size=flat.shape)
    outs, idx = [], 0
    for p in delta_params:
        sz = int(np.prod(p.shape))
        outs.append(flat[idx:idx+sz].reshape(p.shape)); idx += sz
    return tuple(outs)

def cosine(a, b, eps=1e-12):
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na*nb))

def local_train(client, global_params, global_mean, global_var, base_lr, local_epochs, seed):
    X, y = client["X"], client["y"]
    n = len(y); idx = np.arange(n)
    rs = np.random.RandomState(seed)
    model = MLP2(HASH_DIM, H1_DIM, H2_DIM, 1, l2=L2_WEIGHT_DECAY, seed=seed)
    model.set_params(global_params)
    opt_state = AdamState(model.get_params())

    fedprox_mu = FEDPROX_MU if USE_FEDPROX else 0.0
    losses, hmeans, hvars = [], [], []

    total_steps = int(math.ceil(n / BATCH_SIZE)) * local_epochs
    step_counter = 0

    for ep in range(local_epochs):
        rs.shuffle(idx)
        for start in range(0, n, BATCH_SIZE):
            end = min(start+BATCH_SIZE, n)
            b = idx[start:end]
            Xb, yb = X[b], y[b]

            # cosine LR schedule
            if USE_COSINE_LR:
                t = step_counter / max(1, total_steps)
                lr = 0.5*base_lr*(1 + math.cos(math.pi*t))
            else:
                lr = base_lr

            loss, grads, h_mean, h_var = model.loss_and_grads(
                Xb, yb,
                global_params=global_params,
                fedprox_mu=fedprox_mu,
                align_mean_w=ALIGN_MEAN_WEIGHT if USE_ALIGN else 0.0,
                align_var_w=ALIGN_VAR_WEIGHT  if USE_ALIGN else 0.0,
                global_mean=global_mean,
                global_var=global_var,
                use_focal=USE_FOCAL,
                focal_gamma=FOCAL_GAMMA,
                dropout_p=DROPOUT_P,
                rs=rs
            )
            new_params, opt_state = adam_step(model.get_params(), grads, opt_state,
                                              lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS)
            model.set_params(new_params)
            losses.append(loss); hmeans.append(h_mean); hvars.append(h_var)
            step_counter += 1

    new_params = model.get_params()
    delta = tuple(new - old for new, old in zip(new_params, global_params))
    return delta, float(np.mean(losses)), np.mean(hmeans, axis=0), np.mean(hvars, axis=0)

def aggregate(deltas, losses, hmeans, hvars, weights, server_state):
    """
    Aggregation modes:
      - avg:    size-weighted average
      - attn:   attention on delta-to-momentum & inverse loss
      - graph:  similarity to global (hmean,hvar)
      - hybrid: combine attn + graph
    """
    global_mean = server_state["global_mean"]
    global_var  = server_state["global_var"]
    momentum    = server_state["momentum"]

    n_clients = len(deltas)
    size_w = np.array(weights, dtype=float)
    size_w = size_w / max(1e-12, size_w.sum())

    # base weights start as size weights
    w = size_w.copy()

    if AGGREGATOR in ("attn", "hybrid"):
        # attention score: softmax of alpha*cos(delta, momentum) + beta*(1/loss)
        mom_flat = momentum
        if mom_flat is None or mom_flat.shape[0] == 0:
            mom_flat = np.zeros_like(np.concatenate([p.ravel() for p in deltas[0]]))
        attn_scores = []
        for d, L in zip(deltas, losses):
            d_flat = np.concatenate([p.ravel() for p in d])
            s = ATTN_ALPHA * cosine(d_flat, mom_flat) + ATTN_BETA * (1.0 / (L + 1e-6))
            attn_scores.append(s)
        attn_scores = np.array(attn_scores)
        # softmax normalize
        attn_w = np.exp(attn_scores - attn_scores.max())
        attn_w = attn_w / max(1e-12, attn_w.sum())
        w = w * attn_w
        w = w / max(1e-12, w.sum())

    if AGGREGATOR in ("graph", "hybrid"):
        # graph sim score: gamma_mean*cos(hmean_i, global_mean) + gamma_var*(-||hvar_i - global_var||)
        gm = global_mean if global_mean is not None else np.zeros_like(hmeans[0])
        gv = global_var  if global_var  is not None else np.ones_like(hvars[0])
        g_scores = []
        for hm, hv in zip(hmeans, hvars):
            s_cos = cosine(hm, gm)
            s_var = - np.linalg.norm(hv - gv)
            s = GRAPH_GAMMA_MEAN*s_cos + GRAPH_GAMMA_VAR*s_var
            g_scores.append(s)
        g_scores = np.array(g_scores)
        g_w = np.exp(g_scores - g_scores.max())
        g_w = g_w / max(1e-12, g_w.sum())
        w = w * g_w
        w = w / max(1e-12, w.sum())

    # final aggregate by w
    agg = []
    for layer in range(len(deltas[0])):
        agg.append(sum(deltas[i][layer]*w[i] for i in range(n_clients)))
    agg = tuple(agg)

    # update momentum
    agg_flat = np.concatenate([p.ravel() for p in agg])
    if server_state["momentum"].shape[0] == 0:
        server_state["momentum"] = agg_flat.copy()
    else:
        server_state["momentum"] = MOMENTUM_EMA*server_state["momentum"] + (1-MOMENTUM_EMA)*agg_flat

    return agg

def federated_round(train_clients, global_params, server_state, rng, base_lr, local_epochs, seed):
    deltas, losses, hmeans, hvars, weights = [], [], [], [], []
    for c in train_clients:
        delta, loss, hmean, hvar = local_train(c, global_params,
                                               server_state["global_mean"], server_state["global_var"],
                                               base_lr, local_epochs, seed)
        if USE_DP:
            delta = clip_and_add_noise(delta, DP_CLIP_NORM, DP_NOISE_MULTIPLIER, rng)
        deltas.append(delta); losses.append(loss); hmeans.append(hmean); hvars.append(hvar)
        weights.append(len(c["y"]))

    agg = aggregate(deltas, losses, hmeans, hvars, weights, server_state)
    updated_params = tuple(old + d for old, d in zip(global_params, agg))

    # update global stats (weighted by client sizes)
    total = sum(weights)
    gm = np.average(np.vstack(hmeans), axis=0, weights=weights)
    gv = np.average(np.vstack(hvars),  axis=0, weights=weights)
    server_state["global_mean"] = gm
    server_state["global_var"]  = gv

    return updated_params, float(np.mean(losses))

# -----------------------------
# Evaluation + threshold tuning
# -----------------------------
def model_predict_scores(params, X):
    model = MLP2(HASH_DIM, H1_DIM, H2_DIM, 1, l2=L2_WEIGHT_DECAY, seed=GLOBAL_BASE_SEED)
    model.set_params(params)
    return model.predict_proba(X)

def evaluate_scores(y_true, y_scores, effort=None, thr=0.5):
    y_pred = (y_scores >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = safe_roc_auc(y_true, y_scores)
    aucec = popt = None
    if effort is not None:
        try:
            aucec, popt = compute_aucec_popt(y_true, y_scores, effort)
        except Exception:
            aucec = popt = None
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "auc":auc, "aucec":aucec, "popt":popt}

def find_best_threshold_from_scores(train_scores: List[np.ndarray], train_labels: List[np.ndarray],
                                    train_efforts: List[Optional[np.ndarray]],
                                    mode="max_f1", pmin=0.6):
    y_all = np.concatenate(train_labels)
    p_all = np.concatenate(train_scores)
    grid = np.linspace(0.05, 0.95, 37)

    if mode == "max_f1":
        best, thr = -1.0, 0.5
        for t in grid:
            y_pred = (p_all >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_all, y_pred, average="binary", zero_division=0)
            if f1 > best: best, thr = f1, float(t)
        return thr

    elif mode == "max_recall_at_pmin":
        best, thr = -1.0, 0.5
        for t in grid:
            y_pred = (p_all >= t).astype(int)
            prec, rec, _, _ = precision_recall_fscore_support(y_all, y_pred, average="binary", zero_division=0)
            if prec >= pmin and rec > best:
                best, thr = rec, float(t)
        return thr

    elif mode == "max_aucec":
        # needs effort; only use clients that have effort
        scores_e, labels_e, effort_e = [], [], []
        for s, y, e in zip(train_scores, train_labels, train_efforts):
            if e is not None:
                scores_e.append(s); labels_e.append(y); effort_e.append(e)
        if len(scores_e) == 0:
            # fallback to max_f1 if no effort available
            return find_best_threshold_from_scores(train_scores, train_labels, train_efforts, mode="max_f1")
        p_all = np.concatenate(scores_e)
        y_all = np.concatenate(labels_e)
        e_all = np.concatenate(effort_e)
        best, thr = -1.0, 0.5
        for t in grid:
            # AUCEC is threshold-free usually; here we use scores only.
            # To incorporate threshold, we use scores directly; threshold doesn't change AUCEC.
            # So we instead maximize Recall@P>=pmin on effort-weighted subset as proxy:
            y_pred = (p_all >= t).astype(int)
            prec, rec, _, _ = precision_recall_fscore_support(y_all, y_pred, average="binary", zero_division=0)
            # Proxy objective: prioritize recall but ensure precision not terrible
            score = rec if prec >= 0.5 else 0.0
            if score > best:
                best, thr = score, float(t)
        return thr

    else:
        return 0.5

# -----------------------------
# LOPO + Ensembling
# -----------------------------
def train_one_seed(train_clients, seed):
    # set per-seed RNGs
    rng = np.random.RandomState(seed+777)
    # init global model + server state
    model = MLP2(HASH_DIM, H1_DIM, H2_DIM, 1, l2=L2_WEIGHT_DECAY, seed=seed)
    global_params = model.get_params()
    server_state = {
        "global_mean": np.zeros(H2_DIM, dtype=np.float32),
        "global_var":  np.ones (H2_DIM, dtype=np.float32),
        "momentum": np.zeros(0, dtype=np.float32)  # flat momentum buffer
    }

    for r in range(ROUNDS):
        global_params, loss = federated_round(
            train_clients, global_params, server_state, rng,
            base_lr=LEARNING_RATE, local_epochs=LOCAL_EPOCHS, seed=seed
        )
    return global_params, server_state

def run_lopo_ensemble(clients: List[Dict]):
    results = []
    n = len(clients)
    for holdout in range(n):
        test_client = clients[holdout]
        train_clients = [c for i,c in enumerate(clients) if i != holdout]

        # --- Train K models (one per seed)
        seed_models = []
        train_scores_per_seed = []  # list of list per seed: [ per-client scores ]
        for sd in ENSEMBLE_SEEDS:
            random.seed(sd); np.random.seed(sd)
            params, server_state = train_one_seed(train_clients, sd)
            seed_models.append((params, server_state))

            # get training scores for threshold tuning
            per_client_scores = []
            for c in train_clients:
                s = model_predict_scores(params, c["X"])
                per_client_scores.append(s)
            train_scores_per_seed.append(per_client_scores)

        # --- Ensemble training scores (average per-sample across seeds)
        train_scores_ens = []
        train_labels = []
        train_efforts = []
        for ci, c in enumerate(train_clients):
            # collect this client's scores from each seed
            arr = np.stack([train_scores_per_seed[k][ci] for k in range(len(ENSEMBLE_SEEDS))], axis=0)
            s_mean = arr.mean(axis=0)
            train_scores_ens.append(s_mean)
            train_labels.append(c["y"])
            train_efforts.append(c.get("effort"))

        # pick threshold on aggregated training predictions
        thr = find_best_threshold_from_scores(train_scores_ens, train_labels, train_efforts,
                                              mode=THR_MODE, pmin=P_MIN)

        # --- Ensemble test scores
        test_scores = []
        for (params, _) in seed_models:
            s = model_predict_scores(params, test_client["X"])
            test_scores.append(s)
        test_scores = np.stack(test_scores, axis=0).mean(axis=0)

        metrics = evaluate_scores(test_client["y"], test_scores, effort=test_client.get("effort"), thr=thr)
        metrics.update({
            "test_client": test_client["name"],
            "n_train_clients": len(train_clients),
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "thr": thr,
            "aggregator": AGGREGATOR,
            "seeds": len(ENSEMBLE_SEEDS),
            "dp": USE_DP,
            "fedprox": USE_FEDPROX
        })
        results.append(metrics)

        # (optional) save one of the models
        with open(os.path.join(SAVE_MODELS_DIR, f"model_after_{test_client['name']}_seed{ENSEMBLE_SEEDS[0]}.pkl"), "wb") as f:
            pickle.dump(seed_models[0][0], f)

    df = pd.DataFrame(results)
    df.to_csv(SAVE_RESULTS_PATH, index=False)
    return df

# -----------------------------
# Main
# -----------------------------
def main():
    random.seed(GLOBAL_BASE_SEED); np.random.seed(GLOBAL_BASE_SEED)
    if DATA_MODE == "synthetic":
        clients = make_synthetic_clients(n_clients=6, hetero=True)
    else:
        clients = load_csv_clients(CSV_FOLDER)
        if len(clients)==0:
            raise RuntimeError(f"No usable CSVs in {CSV_FOLDER}. Check label names & formats.")
    # sanity
    for c in clients:
        assert c["X"].shape[1] == HASH_DIM, "Hashed feature dim mismatch."
        assert set(np.unique(c["y"])) <= {0,1}, "Labels must be 0/1."

    if True:
        df = run_lopo_ensemble(clients)
        print("LOPO (ensemble) results:")
        print(df.head())
        print(f"Saved to: {SAVE_RESULTS_PATH}")
    else:
        # (not typical CPDP) train on all, eval per client
        pass

if __name__ == "__main__":
    main()
