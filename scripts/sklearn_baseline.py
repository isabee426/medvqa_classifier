"""Sklearn baseline: Logistic Regression + SVM on frozen BLIP hidden states.

Runs in seconds on CPU. Tests several C values and feature subsets.

Usage:
    python scripts/sklearn_baseline.py \
        --features_dir outputs/features/vqarad_stage1_hidden_only \
        --output_dir outputs/eval/sklearn_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Inline feature loading (no medvqa_probe import needed for portability)
# ---------------------------------------------------------------------------

def load_features(directory: str, splits: list[str], label_type: str = "corruption"):
    import json as _json
    d = Path(directory)
    data = np.load(d / "features.npz", allow_pickle=False)
    X, y, ids = [], [], []
    with open(d / "index.jsonl") as f:
        for line in f:
            m = _json.loads(line)
            if m["split"] not in splits:
                continue
            if m["label_type"] != label_type:
                continue
            X.append(data[m["id"]])
            y.append(m["label"])
            ids.append(m["id"])
    return np.stack(X), np.array(y, dtype=int), ids


def run(features_dir: str, output_dir: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
    from sklearn.pipeline import Pipeline

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {features_dir} ...")
    X_train, y_train, _ = load_features(features_dir, splits=["train"])
    X_test,  y_test,  _ = load_features(features_dir, splits=["test"])
    print(f"  train: {X_train.shape}, test: {X_test.shape}")
    print(f"  feature_dim: {X_train.shape[1]}")

    # Feature subsets to try.
    # Full features: all dims.
    # Answer-only:   features are ordered (layer, segment) = (4,vis),(4,q),(4,ans),(8,vis),...
    # 3 layers x 3 segments x 768 = 6144 total.
    # Answer segment is index 2 of each layer block: dims [1536:2304], [3840:4608], [5376:6144]
    feat_dim = X_train.shape[1]
    seg_size = feat_dim // 9  # 768 per segment per layer (approximate)
    answer_idx = np.concatenate([
        np.arange(seg_size * 2, seg_size * 3),   # layer 0, answer
        np.arange(seg_size * 5, seg_size * 6),   # layer 1, answer
        np.arange(seg_size * 8, seg_size * 9),   # layer 2, answer
    ])

    feature_sets = {
        "full":        (X_train, X_test),
        "answer_only": (X_train[:, answer_idx], X_test[:, answer_idx]),
    }

    # Classifiers to try.
    classifiers = {
        "LR_C0.001":  LogisticRegression(C=0.001, max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=42),
        "LR_C0.01":   LogisticRegression(C=0.01,  max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=42),
        "LR_C0.1":    LogisticRegression(C=0.1,   max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=42),
        "LR_C1.0":    LogisticRegression(C=1.0,   max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=42),
    }

    all_results = []
    best_auc = 0.0
    best_name = ""

    for feat_name, (Xtr, Xte) in feature_sets.items():
        print(f"\n=== Feature set: {feat_name} (dim={Xtr.shape[1]}) ===")
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        for clf_name, clf in classifiers.items():
            clf.fit(Xtr_s, y_train)
            probs = clf.predict_proba(Xte_s)[:, 1]
            preds = clf.predict(Xte_s)

            auc   = roc_auc_score(y_test, probs)
            prauc = average_precision_score(y_test, probs)
            f1    = f1_score(y_test, preds)
            acc   = accuracy_score(y_test, preds)

            tag = f"{feat_name}/{clf_name}"
            print(f"  {tag:35s}  AUC={auc:.4f}  PR-AUC={prauc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

            row = {"model": tag, "feat_set": feat_name, "classifier": clf_name,
                   "roc_auc": round(auc, 4), "pr_auc": round(prauc, 4),
                   "f1": round(f1, 4), "accuracy": round(acc, 4),
                   "n_train": len(y_train), "n_test": len(y_test), "feat_dim": Xtr.shape[1]}
            all_results.append(row)

            if auc > best_auc:
                best_auc = auc
                best_name = tag

    print(f"\nBest: {best_name}  →  AUC {best_auc:.4f}")

    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({"best": best_name, "best_auc": best_auc, "results": all_results}, f, indent=2)
    print(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="outputs/features/vqarad_stage1_hidden_only")
    parser.add_argument("--output_dir",   default="outputs/eval/sklearn_baseline")
    args = parser.parse_args()
    run(args.features_dir, args.output_dir)


if __name__ == "__main__":
    main()
