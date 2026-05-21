# Classification Metrics Deep Dive

>*Why 80% accuracy can hide a failing model*

---

## Overview

My Logistic Regression model (within my repo list) produced a churn classifier with **~80% accuracy**. That number sounds reasonable - until you notice that **Recall was only 0.53**, meaning the model silently missed **47% of employees who actually left**.

This project builds every binary classification metric from scratch and develops the intuition for when each one matters. The central argument: accuracy is almost never the right metric for real-world classification problems.

---

## Performance Summary (baseline vs optimised)

| Configuration | Threshold | Precision | Recall | F1 |
|--------------|-----------|-----------|--------|----|
| Default | 0.50 | ~0.65 | ~0.53 | ~0.58 |
| F1-optimal | ~0.40 | — | — | — |
| Cost-sensitive (1:3) | ~0.30 | — | — | — |

*Exact values depend on your Logisitc Regression model. Run `python analysis.py` to see yours.*

---

## Metrics Implemented from Scratch

### Confusion Matrix
```
               PREDICTED
            Stay    Churn
ACTUAL Stay [  TN  |  FP  ]   ← True Negatives, False Positives
      Churn [  FN  |  TP  ]   ← False Negatives, True Positives
```

### Point Metrics (at a fixed threshold)

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness. Misleading if classes are imbalanced |
| **Precision** | TP/(TP+FP) | "Of flagged churners, how many actually left?" |
| **Recall** | TP/(TP+FN) | "Of actual churners, how many did we catch?" |
| **F1-Score** | 2·P·R/(P+R) | Harmonic mean; punishes extreme P/R imbalance |
| **Specificity** | TN/(TN+FP) | "Of stayers, how many did we correctly identify?" |

### Curve Metrics (threshold-independent)

| Metric | What it shows | When to use |
|--------|--------------|-------------|
| **ROC curve** | TPR vs FPR across all thresholds | General model quality comparison |
| **AUC-ROC** | P(score(+) > score(-)) | Single-number model ranking |
| **PR curve** | Precision vs Recall across thresholds | Imbalanced datasets |
| **Average Precision (AP)** | Area under PR curve | Summary of PR performance |

### Threshold Optimisation
- **F1-optimal**: maximises harmonic mean of P and R
- **Youden's J**: maximises (TPR - FPR), best ROC-based cut-point
- **Cost-sensitive**: minimises `cost_FP × FP + cost_FN × FN` — most useful in practice when you know the business cost ratio

---

## Key Finding: When to use each metric

| Problem | Optimise for | Reason |
|---------|-------------|--------|
| Churn prediction | **Recall** | Missing a churner is expensive; false alarms are cheap |
| Medical diagnosis | **Recall** | Missing a sick patient is worse than an unnecessary test |
| Spam filter | **Precision** | A real email in spam is more disruptive than spam in inbox |
| Fraud detection | **PR / AP** | Positives are rare (<1%); ROC overstates performance |
| General comparison | **AUC-ROC** | Threshold-independent; comparable across models |

---

## Project Structure

```
classification-metrics/
├── notebooks/
│   └── metrics_eval.ipynb      # 17 cells — full exploration
├── src/
│   ├── __init__.py
│   └── classification_metrics.py # Production class: ClassificationMetrics
├── visualizations/               # Generated plots (git-ignored)
├── analysis.py                   # Production analysis script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Navigate to project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis script (loads Day 3 model if available, otherwise retrains)
python analysis.py

# 4. Or open notebook
jupyter notebook notebooks/metrics_eval.ipynb
```

**Expected output:**
- Console report with all metrics at threshold=0.50
- sklearn validation (AUC match to 3 decimal places)
- 6 saved visualisation files in `visualizations/`
- Optimal threshold recommendation


---

## Dependency on Logistic Regression

The analysis script attempts to load:
- `../logistic-regression/models/churn_model.pkl`
- `../logistic-regression/data/churn_data.csv`

If these don't exist (e.g. you didn't serialise the model), it regenerates identical synthetic data and trains a fresh sklearn logistic regression. Results will be consistent.

### Saving your logistic regression model (recommended)

Add these lines to your logistic regression `train_model.py` if not already there:

```python
import pickle, os
os.makedirs('models', exist_ok=True)
with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
df.to_csv('data/churn_data.csv', index=False)
```

---

## Design Decisions

**Why harmonic mean (F1) instead of arithmetic mean?**
Arithmetic mean of P=1.0, R=0.01 = 0.505 — sounds decent. F1 = 0.02 — correctly indicates the model is nearly useless at catching positives.

**Why manually sweep thresholds instead of using scipy.optimize?**
Transparency. The sweep makes the precision-recall trade-off visible as a curve rather than a black-box optimisation result. A practitioner needs to see the full trade-off to make an informed business decision.

**Why both ROC and PR curves?**
ROC inflates when negatives dominate (large TN makes FPR look low even with many FP). PR ignores TN entirely. Running both gives a complete picture.

---

## Limitations

- Metrics computed on a single train/test split. Cross-validated estimates would be more robust.
- Threshold optimisation uses the test set — in production this should be a held-out validation set.
- Cost ratio (1:3 FP:FN) is illustrative; the actual ratio requires domain input from HR.
- AUC-ROC assumes the positive class is meaningful — both classes should have business significance.

---

## Next Steps

**FastAPI Deployment**
Deploy the Logistic Regression churn model as a REST API with:
- `POST /predict` — single employee prediction
- `POST /predict_batch` — batch scoring
- Swagger auto-documentation
- Threshold configurable at runtime

---
