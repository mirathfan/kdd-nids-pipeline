# =============================================================================
# KDD Cup 1999 - Network Intrusion Detection
# Phase 4: Advanced Models — XGBoost + LightGBM + SHAP
# Secondary language: Python
# Input : data/train_balanced.csv   (from Phase 2)
#         data/test.csv             (from Phase 2)
# Outputs: outputs/p11_xgb_confusion.png
#          outputs/p12_lgbm_confusion.png
#          outputs/p13_shap_summary.png
#          outputs/p14_shap_beeswarm.png
#          data/phase4_results.json
# =============================================================================

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb
import shap

# ── directories ──────────────────────────────────────────────────────────────
Path("outputs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

CLASSES   = ["Normal", "DoS", "Probe", "R2L", "U2R"]
PALETTE   = {
    "Normal": "#378ADD", "DoS": "#E24B4A",
    "Probe":  "#EF9F27", "R2L": "#1D9E75", "U2R": "#7F77DD"
}

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titleweight":  "bold",
    "axes.titlesize":    14,
    "figure.dpi":        150,
})

# =============================================================================
# 1. Load data
# =============================================================================
print("Loading data...")
train = pd.read_csv("data/train_balanced.csv")
test  = pd.read_csv("data/test.csv")

le = LabelEncoder()
le.fit(CLASSES)

X_train = train.drop("label", axis=1)
y_train = le.transform(train["label"])

X_test  = test.drop("label", axis=1)
y_test  = le.transform(test["label"])

print(f"Train : {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
print(f"Test  : {X_test.shape[0]:,}  rows x {X_test.shape[1]} features")
print(f"Classes: {le.classes_}")

# =============================================================================
# 2. Helper functions
# =============================================================================
def evaluate(y_true, y_pred, model_name):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report   = classification_report(
        y_true, y_pred,
        target_names=CLASSES, output_dict=True, zero_division=0
    )
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)}")
    return {"accuracy": acc, "macro_f1": macro_f1, "report": report}


def plot_confusion(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt=",", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"shrink": 0.75}
    )
    ax.set_title(f"{title}\nAccuracy: {acc:.2%}", pad=12)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"outputs/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/{fname}")


# =============================================================================
# 3. XGBoost
# =============================================================================
print("\n===== MODEL 3: XGBoost =====")

# Class weights — inverse frequency to handle residual imbalance
class_counts  = np.bincount(y_train)
class_weights = len(y_train) / (len(CLASSES) * class_counts)
sample_weight = np.array([class_weights[y] for y in y_train])

xgb_model = xgb.XGBClassifier(
    objective        = "multi:softmax",
    num_class        = len(CLASSES),
    n_estimators     = 400,
    max_depth        = 6,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    gamma            = 0.1,
    reg_alpha        = 0.1,   # L1 regularisation
    reg_lambda       = 1.0,   # L2 regularisation
    use_label_encoder= False,
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 1
)

print("Training XGBoost (400 trees)...")
xgb_model.fit(
    X_train, y_train,
    sample_weight = sample_weight,
    eval_set      = [(X_test, y_test)],
    verbose       = 100
)

xgb_preds   = xgb_model.predict(X_test)
xgb_metrics = evaluate(y_test, xgb_preds, "XGBoost")
plot_confusion(y_test, xgb_preds, "XGBoost — confusion matrix", "p11_xgb_confusion.png")

# =============================================================================
# 4. LightGBM
# =============================================================================
print("\n===== MODEL 4: LightGBM =====")

# Class weights dict for LightGBM
lgb_class_weight = {i: class_weights[i] for i in range(len(CLASSES))}

lgb_model = lgb.LGBMClassifier(
    objective        = "multiclass",
    num_class        = len(CLASSES),
    n_estimators     = 400,
    max_depth        = 8,
    learning_rate    = 0.05,
    num_leaves       = 63,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_samples= 20,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    class_weight     = lgb_class_weight,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = -1    # suppress LightGBM output
)

print("Training LightGBM (400 trees)...")
lgb_model.fit(
    X_train, y_train,
    eval_set        = [(X_test, y_test)],
    callbacks       = [lgb.log_evaluation(period=100)]
)

lgb_preds   = lgb_model.predict(X_test)
lgb_metrics = evaluate(y_test, lgb_preds, "LightGBM")
plot_confusion(y_test, lgb_preds, "LightGBM — confusion matrix", "p12_lgbm_confusion.png")

# =============================================================================
# 5. SHAP explainability (on XGBoost — best interpretable model)
# =============================================================================
print("\n===== SHAP EXPLAINABILITY (XGBoost) =====")
print("Computing SHAP values on 2,000 test samples...")

# Use a sample for speed — SHAP is O(n * trees)
shap_sample_idx = np.random.RandomState(42).choice(len(X_test), size=2000, replace=False)
X_shap = X_test.iloc[shap_sample_idx]

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap)   # shape: (n_samples, n_features, n_classes)

# ── Plot 1: SHAP summary — mean absolute impact per feature (all classes) ──
mean_shap = np.abs(shap_values).mean(axis=(0, 2))   # avg over samples and classes
shap_importance = pd.DataFrame({
    "feature":   X_test.columns,
    "mean_shap": mean_shap
}).sort_values("mean_shap", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(
    shap_importance["feature"][::-1],
    shap_importance["mean_shap"][::-1],
    color="#0C447C", height=0.7
)
ax.set_xlabel("Mean |SHAP value| (impact on model output)")
ax.set_title("SHAP feature importance — XGBoost\nMean absolute impact across all classes")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
plt.savefig("outputs/p13_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/p13_shap_summary.png")

# ── Plot 2: SHAP beeswarm for DoS class (class index 1) ──────────────────────
print("Generating SHAP beeswarm plot for DoS class...")
shap_dos = shap_values[:, :, 1]   # DoS is index 1

fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(
    shap_dos,
    X_shap,
    plot_type  = "dot",
    max_display= 15,
    show       = False,
    color_bar  = True
)
plt.title("SHAP beeswarm — DoS class\nFeature impact direction and magnitude",
          fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/p14_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/p14_shap_beeswarm.png")

# =============================================================================
# 6. Final comparison — all 4 models
# =============================================================================
print("\n===== FINAL MODEL COMPARISON =====")

# Load Phase 3 baseline metrics (hardcoded from R output)
baseline = {
    "Decision Tree": {"accuracy": 0.9988, "macro_f1": 0.8210},
    "Random Forest": {"accuracy": 0.9994, "macro_f1": 0.8310},
}

results = {
    "Decision Tree": baseline["Decision Tree"],
    "Random Forest": baseline["Random Forest"],
    "XGBoost":       {"accuracy": xgb_metrics["accuracy"], "macro_f1": xgb_metrics["macro_f1"]},
    "LightGBM":      {"accuracy": lgb_metrics["accuracy"], "macro_f1": lgb_metrics["macro_f1"]},
}

print(f"\n{'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 42)
for name, m in results.items():
    print(f"{name:<20} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")

# ── Comparison bar chart ──────────────────────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"]  for m in model_names]
macro_f1s   = [results[m]["macro_f1"]  for m in model_names]

x    = np.arange(len(model_names))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5.5))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy",
               color="#B5D4F4", edgecolor="white")
bars2 = ax.bar(x + width/2, macro_f1s,  width, label="Macro F1",
               color="#0C447C", edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0.75, 1.02)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.set_ylabel("Score")
ax.set_title("All models — Accuracy vs Macro F1\nEvaluated on held-out test set (98,802 rows)",
             pad=12)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/p15_all_models_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/p15_all_models_comparison.png")

# =============================================================================
# 7. Save results to JSON for README
# =============================================================================
# Per-class F1 for XGBoost and LightGBM
xgb_report = xgb_metrics["report"]
lgb_report = lgb_metrics["report"]

output = {
    "models": {
        "Decision Tree": {"accuracy": 0.9988, "macro_f1": 0.8210},
        "Random Forest": {"accuracy": 0.9994, "macro_f1": 0.8310},
        "XGBoost":  {
            "accuracy":  round(xgb_metrics["accuracy"],  4),
            "macro_f1":  round(xgb_metrics["macro_f1"],  4),
            "per_class": {c: round(xgb_report[c]["f1-score"], 4) for c in CLASSES}
        },
        "LightGBM": {
            "accuracy":  round(lgb_metrics["accuracy"],  4),
            "macro_f1":  round(lgb_metrics["macro_f1"],  4),
            "per_class": {c: round(lgb_report[c]["f1-score"], 4) for c in CLASSES}
        },
    },
    "top_shap_features": shap_importance["feature"].tolist()
}

with open("data/phase4_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved: data/phase4_results.json")

print("\n===== PHASE 4 COMPLETE =====")
print("Next step -> python/phase5_evaluation.py  OR  write README.md")
