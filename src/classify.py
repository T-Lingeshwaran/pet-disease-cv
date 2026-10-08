"""
classify.py

Turns a GaitReport (see gait_metrics.py) into a plain-language flag.

v1 = rule-based thresholds. These are starting points, not clinically
validated cutoffs — you do NOT have a labeled dataset of confirmed-healthy
vs. confirmed-limping pets yet, so there's nothing to statistically fit
thresholds against. Once you've run this pipeline on enough of your own
clips with a known ground truth (ask your vet to confirm which clips show
a genuine limp), replace `rule_based_flag()` with `train_classifier()` +
`predict()` below — the feature vector shape is identical, so nothing
upstream changes.

Flag levels:
    "normal"      — no leg stands out as asymmetric or unloaded early
    "mild"        — one metric crosses its threshold
    "significant" — two or more metrics cross their thresholds, or one
                    crosses by a wide margin
"""

from dataclasses import dataclass
import numpy as np

from gait_metrics import GaitReport


# Symmetry score is 1.0 = perfectly symmetric, 0.0 = maximally asymmetric.
SYMMETRY_MILD_THRESHOLD = 0.85   # below this, flag as mildly asymmetric
SYMMETRY_SIGNIFICANT_THRESHOLD = 0.65

STANCE_RATIO_IMBALANCE_MILD = 0.12   # max-min stance ratio across legs
STANCE_RATIO_IMBALANCE_SIGNIFICANT = 0.25

# Head-bob amplitude as a fraction of nose-to-tail body length.
# Starting points only (uncalibrated) — tighten these once you have
# labeled clips to check against.
HEAD_BOB_MILD = 0.15
HEAD_BOB_SIGNIFICANT = 0.30


@dataclass
class ClassificationResult:
    flag: str                 # "normal" | "mild" | "significant" | "inconclusive"
    confidence: float         # 0..1, heuristic — how many signals agree
    suspected_leg: str
    reasons: list


def _suspected_leg(report: GaitReport) -> str:
    if not report.stance_ratio:
        return "unknown"
    return min(report.stance_ratio, key=report.stance_ratio.get)


def rule_based_flag(report: GaitReport) -> ClassificationResult:
    reasons = []
    severity_hits = 0
    mild_hits = 0

    # 1. Stride symmetry
    if report.stride_symmetry:
        worst_pair = min(report.stride_symmetry, key=report.stride_symmetry.get)
        worst_val = report.stride_symmetry[worst_pair]
        if worst_val < SYMMETRY_SIGNIFICANT_THRESHOLD:
            severity_hits += 1
            reasons.append(
                f"Stride symmetry for {worst_pair} is low ({worst_val:.2f})."
            )
        elif worst_val < SYMMETRY_MILD_THRESHOLD:
            mild_hits += 1
            reasons.append(
                f"Stride symmetry for {worst_pair} is slightly reduced ({worst_val:.2f})."
            )

    # 2. Stance ratio imbalance across legs
    if len(report.stance_ratio) >= 2:
        values = list(report.stance_ratio.values())
        imbalance = max(values) - min(values)
        if imbalance > STANCE_RATIO_IMBALANCE_SIGNIFICANT:
            severity_hits += 1
            reasons.append(f"Stance-time imbalance across legs is large ({imbalance:.2f}).")
        elif imbalance > STANCE_RATIO_IMBALANCE_MILD:
            mild_hits += 1
            reasons.append(f"Stance-time imbalance across legs is noticeable ({imbalance:.2f}).")

    # 3. Head bob
    if report.head_bob_normalized > HEAD_BOB_SIGNIFICANT:
        severity_hits += 1
        reasons.append(f"Head-bob amplitude is pronounced ({report.head_bob_normalized:.2f}).")
    elif report.head_bob_normalized > HEAD_BOB_MILD:
        mild_hits += 1
        reasons.append(f"Head-bob amplitude is elevated ({report.head_bob_normalized:.2f}).")

    if not reasons and report.warnings:
        return ClassificationResult(
            flag="inconclusive",
            confidence=0.0,
            suspected_leg="unknown",
            reasons=report.warnings,
        )

    if severity_hits >= 1:
        flag = "significant"
    elif mild_hits >= 1:
        flag = "mild"
    else:
        flag = "normal"
        reasons.append("All measured gait signals are within a symmetric range.")

    total_signals = max(severity_hits + mild_hits, 1)
    confidence = min(0.4 + 0.2 * total_signals, 0.9)  # heuristic, not statistical

    return ClassificationResult(
        flag=flag,
        confidence=round(confidence, 2),
        suspected_leg=_suspected_leg(report),
        reasons=reasons,
    )


def feature_vector(report: GaitReport) -> np.ndarray:
    """
    Flattens a GaitReport into a fixed-order numeric vector, for later use
    with a trained classifier (v2) instead of rule_based_flag (v1).
    """
    legs = ["front_left", "front_right", "back_left", "back_right"]
    pairs = [
        "front_left_vs_right",
        "back_left_vs_right",
        "left_front_vs_back",
        "right_front_vs_back",
    ]
    feats = []
    feats += [report.stance_ratio.get(l, np.nan) for l in legs]
    feats += [report.stride_length.get(l, np.nan) for l in legs]
    feats += [report.stride_symmetry.get(p, np.nan) for p in pairs]
    feats.append(report.head_bob_normalized)
    return np.array(feats, dtype=float)


def train_classifier(reports: list, labels: list):
    """
    v2 path — once you have labeled clips (0 = healthy, 1 = limping,
    confirmed by you/a vet), fit a simple model instead of hand-set
    thresholds. Kept minimal (logistic regression) since a student
    dataset will likely be small (tens, not thousands, of clips).

    Args:
        reports: list[GaitReport]
        labels: list[int], same length, 0/1
    Returns:
        a fitted sklearn-compatible classifier
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    X = np.vstack([feature_vector(r) for r in reports])
    y = np.array(labels)

    clf = make_pipeline(SimpleImputer(strategy="mean"), LogisticRegression(max_iter=1000))
    clf.fit(X, y)
    return clf


def predict(clf, report: GaitReport) -> ClassificationResult:
    x = feature_vector(report).reshape(1, -1)
    proba = clf.predict_proba(x)[0]
    pred = clf.predict(x)[0]
    flag = "significant" if pred == 1 else "normal"
    return ClassificationResult(
        flag=flag,
        confidence=round(float(max(proba)), 2),
        suspected_leg=_suspected_leg(report),
        reasons=[f"Trained-classifier prediction (v2), probability={proba.tolist()}"],
    )
