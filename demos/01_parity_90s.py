#!/usr/bin/env python3
"""WSRF Parity Showdown — hidden parity structure that defeats standard ML.

Generates a synthetic dataset whose label is the XOR of K hidden features among
D total features (the other D-K features are pure noise). Standard ML methods
(logistic regression, random forest, gradient boosting, MLP) cannot solve this
above chance because parity is the textbook example of a function with zero
local gradient information — every individual feature is independent of y.

WSRF's parity detector projects the data through P2 boundaries (powers-of-2
thresholds) and runs Gauss-Jordan elimination over GF(2) on the resulting
binary frame. If a low-dimensional XOR structure exists, it's exposed in the
null space, and WSRF builds a zone-stratified forest on top of it.

Usage:
    pip install wsrf-lib
    python demos/01_parity_90s.py
"""

from __future__ import annotations

import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def make_parity_dataset(n_samples: int, n_features: int, parity_arity: int,
                        noise_level: float, seed: int):
    """y = XOR over a hidden subset of `parity_arity` features.

    The features are real-valued in [0, 1]; the binary projection used by the
    parity rule is `feature >= 0.5`. The remaining `n_features - parity_arity`
    columns are pure noise. Optionally flip `noise_level` fraction of labels.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    parity_idx = sorted(rng.choice(n_features, size=parity_arity, replace=False).tolist())
    bits = (X[:, parity_idx] >= 0.5).astype(int)
    y = bits.sum(axis=1) % 2
    if noise_level > 0:
        flips = rng.random(n_samples) < noise_level
        y = np.where(flips, 1 - y, y)
    return X, y, parity_idx


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def fmt_time(s: float) -> str:
    if s < 1: return f"{s*1000:5.0f}ms"
    return f"{s:5.2f}s"


def bench_one(name: str, model, X_train, y_train, X_test, y_test, extra: str = "") -> dict:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    pred_t = time.perf_counter() - t0
    acc = (pred == y_test).mean()
    return {"name": name, "acc": acc, "fit": fit_t, "pred": pred_t, "extra": extra}


def run_baselines(X_train, y_train, X_test, y_test):
    results = []
    results.append(bench_one(
        "LogisticRegression",
        LogisticRegression(max_iter=2000, n_jobs=-1),
        X_train, y_train, X_test, y_test))
    results.append(bench_one(
        "RandomForest(500)",
        RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1, random_state=0),
        X_train, y_train, X_test, y_test))
    results.append(bench_one(
        "GradientBoosting(300)",
        GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=0),
        X_train, y_train, X_test, y_test))
    try:
        from xgboost import XGBClassifier
        results.append(bench_one(
            "XGBoost(500)",
            XGBClassifier(n_estimators=500, max_depth=6, n_jobs=-1,
                          eval_metric='logloss', use_label_encoder=False,
                          verbosity=0, random_state=0),
            X_train, y_train, X_test, y_test))
    except Exception:
        pass
    results.append(bench_one(
        "MLP(256-128-64)",
        MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=400,
                      learning_rate_init=3e-4, random_state=0),
        X_train, y_train, X_test, y_test))
    return results


def run_wsrf(X_train, y_train, X_test, y_test, parity_idx):
    from wsrf import WSRFClassifier
    t0 = time.perf_counter()
    model = WSRFClassifier(random_state=0, n_trees_per_zone=80,
                           max_depth=15, n_jobs=1)
    # auto_discover finds hidden zones (regimes), including parity if present
    model.auto_discover(X_train, y_train)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    pred_t = time.perf_counter() - t0
    acc = (pred == y_test).mean()
    discovery = getattr(model, 'discovery_result_', None)
    extra = ""
    if discovery is not None:
        summary = getattr(discovery, 'summary', None)
        if callable(summary):
            try:
                txt = summary()
                first = txt.splitlines()[0] if txt else ""
                extra = first[:60]
            except Exception:
                pass
    return {"name": "WSRF (auto_discover)", "acc": acc,
            "fit": fit_t, "pred": pred_t, "extra": extra,
            "discovery": discovery}


def banner(s: str):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def print_row(r: dict):
    print(f"  {r['name']:<24s}  acc={fmt_pct(r['acc'])}  "
          f"fit={fmt_time(r['fit']):>7s}  pred={fmt_time(r['pred']):>7s}  "
          f"{r['extra']}")


def main():
    seed = 13
    cases = [
        ("4-way XOR over 12 features", dict(n_features=12, parity_arity=4)),
        ("5-way XOR over 16 features", dict(n_features=16, parity_arity=5)),
        ("6-way XOR over 20 features", dict(n_features=20, parity_arity=6)),
    ]

    for title, params in cases:
        banner(title)
        X, y, idx = make_parity_dataset(
            n_samples=8000, noise_level=0.0, seed=seed, **params)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed)
        print(f"  Hidden parity indices : {idx}")
        print(f"  Noise features        : "
              f"{[i for i in range(params['n_features']) if i not in idx]}")
        print(f"  Train / test split    : {len(X_train)} / {len(X_test)}")
        print(f"  Class balance (train) : {y_train.mean():.3f}")
        print()
        print("  --- Standard ML baselines ---")
        baselines = run_baselines(X_train, y_train, X_test, y_test)
        for r in baselines: print_row(r)
        print()
        print("  --- WSRF (Williams Structured Random Forest) ---")
        wsrf = run_wsrf(X_train, y_train, X_test, y_test, idx)
        print_row(wsrf)
        print()
        best_baseline = max(baselines, key=lambda r: r["acc"])
        gap = wsrf["acc"] - best_baseline["acc"]
        verdict = "WSRF WINS" if gap > 0.01 else ("tie" if abs(gap) <= 0.01 else "baseline wins")
        print(f"  >> best baseline = {best_baseline['name']} @ {fmt_pct(best_baseline['acc'])}")
        print(f"  >> WSRF gap      = {gap*100:+.2f} pp  [{verdict}]")

    banner("TAKEAWAY")
    print("""
  Parity is the canonical 'invisible to local methods' structure: every
  individual feature is independent of the label, so any gradient-based,
  splitting, or kernel-based learner fails to find signal without an
  exponential number of samples.

  WSRF's parity detector projects features through P2 boundaries and runs
  GF(2) Gauss-Jordan on the resulting binary frame. The hidden XOR drops out
  of the null space; the zone-stratified forest then learns on top of the
  recovered structure. No exponential blow-up.
""")


if __name__ == "__main__":
    main()
