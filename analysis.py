"""
Classification Metrics Analysis Script
================================================
Evaluates the LogisticRegressionScratch model trained on
Kaggle Telco Customer Churn data.

Artifact contract (from train_model.py)
---------------------------------------------
    artifact = {
        "model":         LogisticRegressionScratch instance,
        "scaler":        fitted StandardScaler,
        "feature_names": list[str],
        "threshold":     float (0.5),
        "train_metrics": dict,
        "test_metrics":  dict,
        "encoding_info": { binary_cols, multi_cat_cols, numeric_cols, one_hot_cols }
    }

Usage
-----
    python analysis.py

Prerequisites
-------------
    ./models/churn_model.pkl
    ./data/Telco-Customer-Churn.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix as sk_confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from src.classification_metrics import (
    ClassificationMetrics, confusion_matrix_scratch,
    precision_scratch, recall_scratch, f1_score_scratch,
)

matplotlib.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                             "axes.spines.right": False})

SEED       = 42
MODEL_PATH = "./models/churn_model.pkl"
DATA_PATH  = "./data/Telco-Customer-Churn.csv"

np.random.seed(SEED)

# ─────────────────────────────────────────────
# Replicate preprocessing exactly
# ─────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Mirror of load_and_clean — must stay identical."""
    df = pd.read_csv(path)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def encode_features(df: pd.DataFrame) -> tuple:
    """Mirror of encode_features — must stay identical."""
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    df_encoded = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True, dtype=int)
    feature_cols = [c for c in df_encoded.columns if c != "Churn"]
    new_cols     = [c for c in df_encoded.columns if c not in df.columns]
    return df_encoded, feature_cols, new_cols


# ─────────────────────────────────────────────
# 1. Load data & artifact
# ─────────────────────────────────────────────

print("─" * 60)
print("  Classification Metrics Deep Dive")
print("  Dataset: Telco Customer Churn (Kaggle)")
print("─" * 60)

if not os.path.exists(DATA_PATH):
    sys.exit(
        f"\n✗  Telco CSV not found at: {DATA_PATH}\n"
        "   Copy Telco-Customer-Churn.csv to that path and rerun."
    )

if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"\n✗  artifact not found at: {MODEL_PATH}\n"
        "   Run train_model.py first to generate it."
    )

# --- Data ---
df_raw = load_and_clean(DATA_PATH)
df_encoded, _, _ = encode_features(df_raw.copy())

# --- Artifact ---
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model         = artifact["model"]
scaler        = artifact["scaler"]
feature_names = artifact["feature_names"] 
threshold     = artifact["threshold"]
d3_test_m     = artifact["test_metrics"]

y = df_encoded["Churn"].values

print(f"\n  Customers: {len(df_raw):,} | Churn rate: {y.mean():.1%} ({y.sum()} churners)")
print(f"  Features (encoded): {len(feature_names)}")
print(f"  Logistic regression reported test  →  Acc={d3_test_m['accuracy']:.4f}  "
      f"P={d3_test_m['precision']:.4f}  R={d3_test_m['recall']:.4f}  "
      f"F1={d3_test_m['f1']:.4f}")

# Reconstruct exact train/test split with same column order
X_ordered = df_encoded[feature_names].values
X_train, X_test, y_train, y_test = train_test_split(
    X_ordered, y, test_size=0.2, random_state=SEED, stratify=y
)
# Use saved scaler — do NOT refit
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {len(y_train):,} | Test: {len(y_test):,}\n")

# Get probability scores from scratch model
try:
    y_prob_test  = model.predict_proba(X_test_s)
    y_prob_train = model.predict_proba(X_train_s)
except AttributeError:
    # Fallback: sigmoid(Xw + b) manually if predict_proba not defined
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    y_prob_test  = _sigmoid(X_test_s  @ model.weights + model.bias)
    y_prob_train = _sigmoid(X_train_s @ model.weights + model.bias)

# ─────────────────────────────────────────────
# 2. Evaluate with scratch metrics
# ─────────────────────────────────────────────

metrics = ClassificationMetrics(threshold=threshold)
metrics.evaluate(y_test, y_prob_test)
metrics.print_summary()

print("\n  sklearn classification_report (validation):")
print(classification_report(
    y_test, (y_prob_test >= threshold).astype(int),
    target_names=["Stay", "Churn"]
))

sk_auc = roc_auc_score(y_test, y_prob_test)
print(f"  AUC-ROC  scratch={metrics.auc:.4f}  sklearn={sk_auc:.4f}  "
      f"{'✓ match' if abs(metrics.auc - sk_auc) < 0.005 else '✗ mismatch'}")

# ─────────────────────────────────────────────
# 3. Threshold sweep
# ─────────────────────────────────────────────

sweep_df = pd.DataFrame(metrics.threshold_sweep(np.linspace(0.1, 0.9, 17)))
print("\n  Threshold sweep:")
print(sweep_df[["threshold", "accuracy", "precision", "recall", "f1"]].to_string(index=False))

# ─────────────────────────────────────────────
# 4. Optimal thresholds
# ─────────────────────────────────────────────

opt_f1     = metrics.find_optimal_threshold(method="f1")
opt_youden = metrics.find_optimal_threshold(method="youden")
opt_cost   = metrics.find_optimal_threshold(method="cost", cost_fp=1.0, cost_fn=3.0)

print(f"\n  Optimal thresholds:")
print(f"    F1-optimal:      t={opt_f1['threshold']:.3f}  →  "
      f"P={opt_f1['precision']:.3f}  R={opt_f1['recall']:.3f}  F1={opt_f1['f1']:.3f}")
print(f"    Youden's J:      t={opt_youden['threshold']:.3f}  →  "
      f"P={opt_youden['precision']:.3f}  R={opt_youden['recall']:.3f}  F1={opt_youden['f1']:.3f}")
print(f"    Cost-sensitive:  t={opt_cost['threshold']:.3f}  →  "
      f"P={opt_cost['precision']:.3f}  R={opt_cost['recall']:.3f}  F1={opt_cost['f1']:.3f}")

# ─────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────

# Fig 1 — Three confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrix — Threshold Comparison (Telco Churn)",
             fontsize=13, fontweight="bold")
for ax, (t, label) in zip(axes, [
    (threshold,              f"Logistic Regression default ({threshold:.2f})"),
    (opt_f1["threshold"],    f"F1-optimal ({opt_f1['threshold']:.2f})"),
    (opt_cost["threshold"],  f"Cost 1:3 ({opt_cost['threshold']:.2f})"),
]):
    y_pred_t = (y_prob_test >= t).astype(int)
    tp_, tn_, fp_, fn_ = confusion_matrix_scratch(y_test, y_pred_t)
    p_ = precision_scratch(tp_, fp_)
    r_ = recall_scratch(tp_, fn_)
    f_ = f1_score_scratch(p_, r_)
    sns.heatmap(sk_confusion_matrix(y_test, y_pred_t), annot=True, fmt="d",
                cmap="Blues", ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{label}\nP={p_:.3f}  R={r_:.3f}  F1={f_:.3f}")
plt.tight_layout()
plt.savefig("visualisations/fig1_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("\n  Saved: visualisations/fig1_confusion_matrices.png")

# Fig 2 — ROC
tp0, tn0, fp0, fn0 = confusion_matrix_scratch(y_test, (y_prob_test >= threshold).astype(int))
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(metrics.fpr_curve, metrics.tpr_curve, lw=2, color="#2563EB",
        label=f"Scratch  (AUC={metrics.auc:.3f})")
sk_fpr, sk_tpr, _ = roc_curve(y_test, y_prob_test)
ax.plot(sk_fpr, sk_tpr, lw=1.5, linestyle="--", color="#16A34A",
        label=f"sklearn  (AUC={sk_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (0.50)")
ax.scatter([fp0/(fp0+tn0)], [tp0/(tp0+fn0)], color="red", zorder=5, s=100,
           label=f"Day 3 default (t={threshold})")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve — Telco Churn", fontweight="bold")
ax.legend(loc="lower right"); ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
plt.tight_layout()
plt.savefig("visualisations/fig2_roc_curve.png", bbox_inches="tight")
plt.close()
print("  Saved: visualisations/fig2_roc_curve.png")

# Fig 3 — PR curve
sk_ap = average_precision_score(y_test, y_prob_test)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(metrics.pr_recalls, metrics.pr_precisions, lw=2, color="#7C3AED",
        label=f"Scratch  (AP={metrics.average_precision:.3f})")
sk_pr_p, sk_pr_r, _ = precision_recall_curve(y_test, y_prob_test)
ax.plot(sk_pr_r, sk_pr_p, lw=1.5, linestyle="--", color="#DC2626",
        label=f"sklearn  (AP={sk_ap:.3f})")
ax.axhline(y_test.mean(), color="gray", lw=1, linestyle=":",
           label=f"No-skill ({y_test.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Telco Churn", fontweight="bold")
ax.legend(); ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
plt.tight_layout()
plt.savefig("visualisations/fig3_pr_curve.png", bbox_inches="tight")
plt.close()
print("  Saved: visualisations/fig3_pr_curve.png")

# Fig 4 — Threshold sweep (fine-grained)
sweep_fine_df = pd.DataFrame(metrics.threshold_sweep(np.linspace(0.05, 0.95, 300)))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sweep_fine_df["threshold"], sweep_fine_df["accuracy"],  label="Accuracy",  color="#1D4ED8", lw=2)
ax.plot(sweep_fine_df["threshold"], sweep_fine_df["precision"], label="Precision", color="#16A34A", lw=2)
ax.plot(sweep_fine_df["threshold"], sweep_fine_df["recall"],    label="Recall",    color="#DC2626", lw=2)
ax.plot(sweep_fine_df["threshold"], sweep_fine_df["f1"],        label="F1",        color="#7C3AED", lw=2)
ax.axvline(threshold, color="gray", lw=1.2, linestyle="--",
           label=f"Logistic Regression default ({threshold:.2f})")
ax.axvline(opt_f1["threshold"], color="#7C3AED", lw=1.2, linestyle=":",
           label=f"F1-optimal ({opt_f1['threshold']:.2f})")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Metric Value")
ax.set_title("How Metrics Change with Threshold — Telco Churn", fontweight="bold")
ax.legend(loc="center right"); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig("visualisations/fig4_threshold_sweep.png", bbox_inches="tight")
plt.close()
print("  Saved: visualisations/fig4_threshold_sweep.png")

# Fig 5 — Dashboard
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(metrics.fpr_curve, metrics.tpr_curve, lw=2, color="#2563EB",
         label=f"AUC={metrics.auc:.3f}")
ax1.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
ax1.scatter([fp0/(fp0+tn0)],[tp0/(tp0+fn0)],color="red",zorder=5,s=80)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC Curve"); ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(metrics.pr_recalls, metrics.pr_precisions, lw=2, color="#7C3AED",
         label=f"AP={metrics.average_precision:.3f}")
ax2.axhline(y_test.mean(), color="gray", lw=1, linestyle=":")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("PR Curve"); ax2.legend()

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(sweep_fine_df["threshold"], sweep_fine_df["precision"], label="Precision", color="#16A34A", lw=2)
ax3.plot(sweep_fine_df["threshold"], sweep_fine_df["recall"],    label="Recall",    color="#DC2626", lw=2)
ax3.plot(sweep_fine_df["threshold"], sweep_fine_df["f1"],        label="F1",        color="#7C3AED", lw=2)
ax3.axvline(opt_f1["threshold"], color="#7C3AED", lw=1.2, linestyle=":",
            label=f"opt={opt_f1['threshold']:.2f}")
ax3.set_xlabel("Threshold"); ax3.set_ylabel("Score")
ax3.set_title("Threshold Trade-off"); ax3.legend(fontsize=9)

ax4 = fig.add_subplot(gs[1, 1])
y_pred_opt = (y_prob_test >= opt_f1["threshold"]).astype(int)
sns.heatmap(sk_confusion_matrix(y_test, y_pred_opt), annot=True, fmt="d",
            cmap="Blues", ax=ax4,
            xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
ax4.set_title(f"Confusion Matrix (F1-opt t={opt_f1['threshold']:.2f})")

fig.suptitle("Day 04 — Classification Metrics Dashboard (Telco Churn)",
             fontsize=14, fontweight="bold")
plt.savefig("visualisations/fig5_dashboard.png", bbox_inches="tight")
plt.close()
print("  Saved: visualisations/fig5_dashboard.png")

# ─────────────────────────────────────────────
# 6. Final summary
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("  Summary")
print("─" * 60)
print(f"  {'Config':>20}  {'t':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'FN (missed)':>12}")
print(f"  {'-'*60}")
for label, t in [("Day 3 default", threshold),
                 ("F1-optimal",    opt_f1["threshold"]),
                 ("Cost 1:3",      opt_cost["threshold"])]:
    y_pred_t = (y_prob_test >= t).astype(int)
    tp_, tn_, fp_, fn_ = confusion_matrix_scratch(y_test, y_pred_t)
    p_ = precision_scratch(tp_, fp_)
    r_ = recall_scratch(tp_, fn_)
    f_ = f1_score_scratch(p_, r_)
    print(f"  {label:>20}  {t:.3f}  {p_:.4f}  {r_:.4f}  {f_:.4f}  {fn_:>12}")

print(f"\n  AUC-ROC  scratch={metrics.auc:.4f}  sklearn={sk_auc:.4f}")
print(f"\n  Key finding: Recall at default t={threshold} was {metrics.recall:.2f}.")
print(f"  F1-optimal t={opt_f1['threshold']:.2f} raises recall to {opt_f1['recall']:.2f},")
print(f"  catching {opt_f1['tp'] - metrics.tp} more churners at the cost of "
      f"{opt_f1['fp'] - metrics.fp} extra false alarms.")
print("─" * 60)
