# Federated and Privacy-Preserving Cross-Project Defect Prediction (FedCPDP)

## 🔑 Overview
Organizations are often reluctant to share defect-labeled source code due to privacy and intellectual property concerns.  
**FedCPDP** enables **collaborative cross-project defect prediction (CPDP)** without exchanging raw project data and without requiring a common feature schema:contentReference[oaicite:1]{index=1}.

**Key Features:**
- **Heterogeneous features** handled via **feature hashing**
- **Federated optimization** with FedAvg / FedProx
- **Privacy-preserving** training with client-level Differential Privacy (DP)
- **Representation alignment** (CORAL-style mean & variance)
- **Adaptive server aggregation** (attention-based, graph-based, or hybrid)
- **Focal loss / Weighted BCE** for class imbalance
- **Threshold tuning** (max-F1, recall@precision, AUCEC-based)
- **Effort-aware evaluation** with AUCEC and Popt
- **Seed ensembling** for robust results

---

## 📂 Dataset
- Supports **30 benchmark datasets** across NASA, SOFTLAB, Relink, AEEEM, and PROMISE.


## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/<your-username>/FedCPDP.git
cd FedCPDP


Install dependencies:
pip install -r requirements.txt


🚀 Usage
1. Run with synthetic data
python fed_cpdp.py --mode synthetic

2. Run with your CSV dataset folder
python fed_cpdp.py --mode csv_folder --csv_folder ./fed_cpdp_csvs/


Run
python src/fed_cpdp.py --csv_folder ./data/csvs --out_dir ./outputs
