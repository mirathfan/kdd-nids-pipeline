# KDD Cup 1999 — Network Intrusion Detection Pipeline

A full end-to-end machine learning pipeline for multi-class network intrusion detection, built in **R** (primary) and **Python** (secondary). Trains and evaluates four classifiers on the KDD Cup 1999 dataset, with SHAP-based explainability for security-critical model transparency.

---

## Results

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Decision Tree | 99.88% | 0.821 | Interpretable baseline |
| **Random Forest** | **99.94%** | **0.832** | **Best overall** |
| XGBoost | 99.91% | 0.820 | + SHAP explainability |
| LightGBM | 99.92% | 0.827 | Fastest training |

> U2R class (privilege escalation attacks) is hardest across all models — only 10 test samples, reflecting extreme real-world rarity. All models achieve near-perfect detection on DoS, Normal, and Probe classes.

---

## Dataset

**KDD Cup 1999 — Computer Network Intrusion Detection**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/130/kdd+cup+1999+data)
- 494,021 connection records (10% subset), 41 features
- 5 classes: Normal, DoS, Probe, R2L, U2R
- Severe class imbalance: U2R = 0.01% of data

---

## Pipeline Overview

```
Phase 1 (R)  →  Phase 2 (R)  →  Phase 3 (R)  →  Phase 4 (Python)
  Ingestion       Preprocessing    Baselines        Advanced Models
  & EDA           & SMOTE          DT + RF          XGBoost + LightGBM
                                                    + SHAP
```

### Phase 1 — Data Ingestion & EDA (`R/phase1_eda.R`)
- Downloads and parses KDD Cup 10% subset (494K rows, 41 features)
- Maps 23 granular attack subtypes → 5-class taxonomy
- Generates 5 EDA plots: class distribution, protocol breakdown, feature distributions, correlation heatmap, attack subtype chart
- Exports `data/kdd_labelled.csv`

### Phase 2 — Preprocessing & SMOTE (`R/phase2_preprocessing.R`)
- One-hot encodes 3 categorical features (42 → 116 columns)
- Removes 93 near-zero variance features via `caret::nearZeroVar`
- Stratified 80/20 train/test split (no data leakage)
- Min-max scaling fit on train only, applied to both sets
- Random Forest feature importance → selects top 20 features
- SMOTE oversampling on minority classes: Probe → 8K, R2L → 5K, U2R → 2K
- Exports `data/train_balanced.csv` and `data/test.csv`

### Phase 3 — Baseline Models (`R/phase3_baseline_models.R`)
- **Decision Tree** (rpart): pruned via cross-validation, 105 leaves
- **Random Forest**: 300 trees, mtry = √20 ≈ 4, trained on stratified 50% sample
- Evaluation: per-class precision/recall/F1, confusion matrix heatmaps, macro F1
- Exports `data/baseline_results.csv` for Python compatibility and `data/baseline_results.rds` for R reuse

### Phase 4 — Advanced Models (`python/phase4_xgboost.py`)
- **XGBoost**: 400 trees, depth 6, L1+L2 regularisation, inverse-frequency class weights
- **LightGBM**: 400 trees, leaf-wise growth, 63 leaves
- **SHAP explainability**: TreeExplainer on 2,000 test samples — feature importance bar chart + DoS beeswarm plot
- Loads Phase 3 baseline metrics from `data/baseline_results.csv`
- Exports all metrics to `data/phase4_results.json`

---

## Key Findings

- **`src_bytes`** is the single most predictive feature across all models (RF importance + SHAP)
- **DoS attacks** are trivially separable — near-100% F1 across all classifiers
- **U2R** (user-to-root privilege escalation) is the hardest class — only 52 raw records and 10 held-out test samples, so evaluation remains high-variance even after SMOTE-balanced training
- **R2L** benefits most from minority-class handling, reaching about 0.94 F1 with LightGBM on the held-out test set
- Random Forest outperforms gradient boosting here, suggesting the feature space is well-suited to bagging after careful feature selection

---

## Repo Structure

```
kdd-nids-pipeline/
├── R/
│   ├── phase1_eda.R                # Data ingestion & EDA
│   ├── phase2_preprocessing.R      # Encoding, scaling, SMOTE, feature selection
│   └── phase3_baseline_models.R    # Decision Tree & Random Forest
├── python/
│   └── phase4_xgboost.py           # XGBoost, LightGBM, SHAP
├── outputs/                        # All plots (generated on run)
│   ├── p1_class_distribution.png
│   ├── p2_protocol_by_class.png
│   ├── p3_feature_distributions.png
│   ├── p4_correlation_heatmap.png
│   ├── p5_attack_subtypes.png
│   ├── p6_feature_importance.png
│   ├── p7_smote_class_dist.png
│   ├── p8_dt_confusion.png
│   ├── p8b_dt_tree.png
│   ├── p9_rf_confusion.png
│   ├── p10_model_comparison.png
│   ├── p11_xgb_confusion.png
│   ├── p12_lgbm_confusion.png
│   ├── p13_shap_summary.png
│   ├── p14_shap_beeswarm.png
│   └── p15_all_models_comparison.png
├── data/                           # Gitignored — regenerated on run
└── README.md
```

---

## How to Reproduce

### Requirements

**R packages** (auto-installed via `pacman`):
```
tidyverse, caret, randomForest, smotefamily, rpart, rpart.plot,
ggcorrplot, scales, gridExtra, knitr
```

**Python packages**:
```bash
pip install xgboost lightgbm shap scikit-learn pandas numpy matplotlib seaborn
```

> Mac users: `brew install libomp` is required for XGBoost.

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/mirathfan/kdd-nids-pipeline.git
cd kdd-nids-pipeline

# 2. Create data directory
mkdir -p data outputs

# 3. Run R pipeline (in RStudio or terminal)
Rscript R/phase1_eda.R        # ~60 sec — downloads & parses data
Rscript R/phase2_preprocessing.R  # ~3 min — SMOTE + feature selection
Rscript R/phase3_baseline_models.R # ~3 min — DT + RF training

# 4. Run Python pipeline
python3 python/phase4_xgboost.py   # ~2 min — XGBoost + LightGBM + SHAP
```

> **Note:** The `data/` directory is gitignored. Re-running Phase 1 will automatically download the KDD Cup dataset (~2MB). All subsequent phases read from files generated by the previous phase.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data wrangling | R (`tidyverse`, `dplyr`), Python (`pandas`) |
| Preprocessing | R (`caret`, `smotefamily`) |
| Baseline models | R (`rpart`, `randomForest`) |
| Advanced models | Python (`xgboost`, `lightgbm`) |
| Explainability | Python (`shap`) |
| Visualisation | R (`ggplot2`), Python (`matplotlib`, `seaborn`) |

---

## Author

**Mir Athfan Ali**
MS Computer Science — Illinois Institute of Technology
[LinkedIn](https://linkedin.com/in/mirathfan) · [GitHub](https://github.com/mirathfan)
