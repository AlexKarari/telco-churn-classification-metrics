"""
Classification Metrics from Scratch
=====================================
Day 04 of ML Learning Journey

Implements all key binary classification evaluation metrics from first principles:
  - Confusion matrix components (TP, TN, FP, FN)
  - Precision, Recall, F1-Score
  - ROC curve and AUC
  - Precision-Recall curve and Average Precision
  - Threshold optimization (F1-optimal, cost-sensitive)

Design Philosophy:
    Every formula is implemented manually before being validated against sklearn.
    This is intentional: understanding what the library computes is more valuable
    than knowing how to call it.

Author: Xander
Date: 2025
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Compute the four components of a binary confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_pred : np.ndarray, shape (n_samples,)
        Predicted binary labels (0 or 1).

    Returns
    -------
    tp : int   True positives  — predicted 1, actually 1
    tn : int   True negatives  — predicted 0, actually 0
    fp : int   False positives — predicted 1, actually 0  (Type I error)
    fn : int   False negatives — predicted 0, actually 1  (Type II error)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    return tp, tn, fp, fn


# ---------------------------------------------------------------------------
# Point metrics (single threshold)
# ---------------------------------------------------------------------------

def precision_scratch(tp: int, fp: int) -> float:
    """
    Precision = TP / (TP + FP)

    "Of all employees we predicted would churn, what fraction actually did?"
    Penalises false alarms. Use when acting on every positive is costly.
    """
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return tp / denominator


def recall_scratch(tp: int, fn: int) -> float:
    """
    Recall (Sensitivity, TPR) = TP / (TP + FN)

    "Of all employees who actually churned, what fraction did we catch?"
    Penalises missed positives. Use when missing a positive is costly.
    """
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return tp / denominator


def f1_score_scratch(precision: float, recall: float) -> float:
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Harmonic mean of precision and recall. Balances both concerns.
    More sensitive to the lower of the two values than their arithmetic mean.
    """
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def accuracy_scratch(tp: int, tn: int, fp: int, fn: int) -> float:
    """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


def specificity_scratch(tn: int, fp: int) -> float:
    """
    Specificity (True Negative Rate) = TN / (TN + FP)

    "Of all employees who stayed, what fraction did we correctly label?"
    Complement of False Positive Rate (FPR = 1 - Specificity).
    """
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    return tn / denominator


# ---------------------------------------------------------------------------
# Main ClassificationMetrics class
# ---------------------------------------------------------------------------

class ClassificationMetrics:
    """
    A comprehensive binary classification evaluation toolkit built from scratch.

    After fitting (calling `evaluate`), all metrics are stored as attributes
    and accessible via `summary()` or individual properties.

    Parameters
    ----------
    threshold : float, default=0.5
        Decision threshold for converting probabilities to class labels.

    Examples
    --------
    >>> metrics = ClassificationMetrics(threshold=0.5)
    >>> metrics.evaluate(y_true, y_prob)
    >>> metrics.summary()
    >>> metrics.plot_roc_curve()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> "ClassificationMetrics":
        """
        Compute all metrics from ground-truth labels and predicted probabilities.

        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            True binary labels (0 or 1).
        y_prob : array-like, shape (n_samples,)
            Predicted probabilities for the positive class.

        Returns
        -------
        self : ClassificationMetrics
            Fitted instance (for method chaining).
        """
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_prob = np.asarray(y_prob, dtype=float)
        self.y_pred = (self.y_prob >= self.threshold).astype(int)

        # --- Confusion matrix components ---
        self.tp, self.tn, self.fp, self.fn = confusion_matrix_scratch(
            self.y_true, self.y_pred
        )

        # --- Point metrics at chosen threshold ---
        self.precision = precision_scratch(self.tp, self.fp)
        self.recall    = recall_scratch(self.tp, self.fn)
        self.f1        = f1_score_scratch(self.precision, self.recall)
        self.accuracy  = accuracy_scratch(self.tp, self.tn, self.fp, self.fn)
        self.specificity = specificity_scratch(self.tn, self.fp)

        # --- Curve data (computed once, used for plotting) ---
        self.fpr_curve, self.tpr_curve, self.roc_thresholds = self._compute_roc_curve()
        self.auc = self._compute_auc(self.fpr_curve, self.tpr_curve)

        self.pr_precisions, self.pr_recalls, self.pr_thresholds = \
            self._compute_pr_curve()
        self.average_precision = self._compute_average_precision(
            self.pr_precisions, self.pr_recalls
        )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------

    def _compute_roc_curve(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Manually compute the ROC (Receiver Operating Characteristic) curve.

        Algorithm
        ---------
        For each unique threshold t (sorted descending):
            1. Predict positive if y_prob >= t
            2. Compute FPR = FP / (FP + TN)   [x-axis]
                       TPR = TP / (TP + FN)   [y-axis]
        Connecting these points traces the ROC curve.

        The curve starts at (0,0) — predict nothing positive —
        and ends at (1,1) — predict everything positive.
        A random classifier lies on the diagonal (FPR = TPR).
        """
        thresholds = np.sort(np.unique(self.y_prob))[::-1]

        fprs, tprs = [0.0], [0.0]  # start at origin

        for t in thresholds:
            y_pred_t = (self.y_prob >= t).astype(int)
            tp, tn, fp, fn = confusion_matrix_scratch(self.y_true, y_pred_t)

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            fprs.append(fpr)
            tprs.append(tpr)

        fprs.append(1.0)   # end at (1,1)
        tprs.append(1.0)

        return np.array(fprs), np.array(tprs), thresholds

    # ------------------------------------------------------------------
    # Precision-Recall curve
    # ------------------------------------------------------------------

    def _compute_pr_curve(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Manually compute the Precision-Recall curve.

        Algorithm
        ---------
        For each unique threshold t (sorted descending):
            1. Predict positive if y_prob >= t
            2. Compute Precision = TP / (TP + FP)   [y-axis]
                       Recall    = TP / (TP + FN)   [x-axis]

        The curve starts at (Recall=0, Precision=1.0) — only the most
        confident predictions are positive — and ends at (Recall=1, Precision=prevalence).

        When to prefer PR over ROC:
            Imbalanced datasets.  ROC can look optimistic because TN is large,
            keeping FPR low even with many FP. PR ignores TN entirely.
        """
        thresholds = np.sort(np.unique(self.y_prob))[::-1]

        precisions, recalls = [], []

        for t in thresholds:
            y_pred_t = (self.y_prob >= t).astype(int)
            tp, _, fp, fn = confusion_matrix_scratch(self.y_true, y_pred_t)

            p = precision_scratch(tp, fp)
            r = recall_scratch(tp, fn)

            precisions.append(p)
            recalls.append(r)

        # Append baseline point: predict everything positive
        precisions.append(np.mean(self.y_true))
        recalls.append(1.0)

        return np.array(precisions), np.array(recalls), thresholds

    # ------------------------------------------------------------------
    # AUC (trapezoidal rule)
    # ------------------------------------------------------------------

    def _compute_auc(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Area Under Curve via the trapezoidal rule.

        AUC-ROC interpretation:
            Probability that the model ranks a random positive
            higher than a random negative.
            0.5 = random; 0.7–0.8 = acceptable; 0.8–0.9 = good; >0.9 = excellent.
        """
        # Sort by x for correct trapezoid direction
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # np.trapz: ∫y dx  using the trapezoidal rule
        return float(np.trapezoid(y_sorted, x_sorted))

    def _compute_average_precision(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray
    ) -> float:
        """
        Average Precision (AP) = Σ (Recall_k - Recall_{k-1}) * Precision_k

        Summarises the PR curve as a single scalar.
        Equivalent to the area under the PR curve.
        """
        # Sort by recall ascending
        order = np.argsort(recalls)
        recalls_s = recalls[order]
        precisions_s = precisions[order]

        ap = 0.0
        for i in range(1, len(recalls_s)):
            delta_r = recalls_s[i] - recalls_s[i - 1]
            ap += delta_r * precisions_s[i]

        return float(ap)

    # ------------------------------------------------------------------
    # Threshold optimisation
    # ------------------------------------------------------------------

    def find_optimal_threshold(
        self,
        method: str = "f1",
        cost_fp: float = 1.0,
        cost_fn: float = 1.0
    ) -> Dict[str, float]:
        """
        Find the probability threshold that optimises a chosen criterion.

        Parameters
        ----------
        method : str
            "f1"         — maximise F1-Score (default)
            "youden"     — maximise Youden's J = TPR - FPR (balances sensitivity/specificity)
            "cost"       — minimise weighted cost: cost_fp*FP + cost_fn*FN
        cost_fp : float
            Cost of a false positive (used only for method="cost").
        cost_fn : float
            Cost of a false negative (used only for method="cost").

        Returns
        -------
        dict with keys: threshold, metric_value, precision, recall, f1
        """
        self._check_fitted()

        thresholds = np.linspace(0.01, 0.99, 200)
        best_t      = 0.5
        best_value  = -np.inf
        best_metrics: Dict[str, float] = {}

        for t in thresholds:
            y_pred_t = (self.y_prob >= t).astype(int)
            tp, tn, fp, fn = confusion_matrix_scratch(self.y_true, y_pred_t)

            p = precision_scratch(tp, fp)
            r = recall_scratch(tp, fn)
            f = f1_score_scratch(p, r)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            if method == "f1":
                value = f
            elif method == "youden":
                value = r - fpr          # TPR - FPR
            elif method == "cost":
                value = -(cost_fp * fp + cost_fn * fn)   # maximise negative cost
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'f1', 'youden', or 'cost'.")

            if value > best_value:
                best_value = value
                best_t = t
                best_metrics = {
                    "threshold":    float(t),
                    "metric_value": float(value),
                    "precision":    float(p),
                    "recall":       float(r),
                    "f1":           float(f),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                }

        return best_metrics

    def threshold_sweep(
        self,
        thresholds: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Evaluate all key metrics across a range of thresholds.

        Useful for visualising the trade-off between precision and recall
        as the decision boundary is moved.

        Parameters
        ----------
        thresholds : array-like, optional
            Thresholds to evaluate. Defaults to np.linspace(0.1, 0.9, 17).

        Returns
        -------
        List of dicts, one per threshold.
        """
        self._check_fitted()

        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 17)

        results = []
        for t in thresholds:
            y_pred_t = (self.y_prob >= t).astype(int)
            tp, tn, fp, fn = confusion_matrix_scratch(self.y_true, y_pred_t)

            p   = precision_scratch(tp, fp)
            r   = recall_scratch(tp, fn)
            f   = f1_score_scratch(p, r)
            acc = accuracy_scratch(tp, tn, fp, fn)

            results.append({
                "threshold": round(float(t), 3),
                "accuracy":  round(acc, 4),
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f, 4),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            })

        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """
        Return a dictionary of all point metrics at the current threshold.

        Returns
        -------
        dict with keys: threshold, accuracy, precision, recall, f1,
                        specificity, auc_roc, average_precision,
                        tp, tn, fp, fn
        """
        self._check_fitted()
        return {
            "threshold":         self.threshold,
            "accuracy":          round(self.accuracy,   4),
            "precision":         round(self.precision,  4),
            "recall":            round(self.recall,     4),
            "f1":                round(self.f1,         4),
            "specificity":       round(self.specificity, 4),
            "auc_roc":           round(self.auc,        4),
            "average_precision": round(self.average_precision, 4),
            "tp": self.tp, "tn": self.tn,
            "fp": self.fp, "fn": self.fn,
        }

    def print_summary(self) -> None:
        """Print a formatted metrics report to stdout."""
        self._check_fitted()
        s = self.summary()
        print("=" * 50)
        print(f"  Classification Metrics (threshold={s['threshold']:.2f})")
        print("=" * 50)
        print(f"  Confusion Matrix:  TP={s['tp']}  TN={s['tn']}  FP={s['fp']}  FN={s['fn']}")
        print(f"  Accuracy:          {s['accuracy']:.4f}")
        print(f"  Precision:         {s['precision']:.4f}")
        print(f"  Recall:            {s['recall']:.4f}")
        print(f"  F1-Score:          {s['f1']:.4f}")
        print(f"  Specificity:       {s['specificity']:.4f}")
        print(f"  AUC-ROC:           {s['auc_roc']:.4f}")
        print(f"  Average Precision: {s['average_precision']:.4f}")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .evaluate(y_true, y_prob) first.")
