"""Controlled frontier-density and pool-size sensitivity.
At fixed frontier density, dominated padding preserves the population oracle,
but changes both the competitor set and confidence allocation; this does not
isolate a pure multiplicity effect. Independent Bernoulli score columns are
simulated using different seeds across design conditions. Only CP and DKW
within a condition and repetition share the same calibration observations.
DKW must use M=K, one fixed binary score per independent policy column.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from policy_certificate import bounds, macro_f1, recommendations

R = Path(__file__).resolve().parent
OUT = R / "results/extended"
OUT.mkdir(parents=True, exist_ok=True)

PS = np.linspace(0.02, 0.50, 25)

# The first four points are the sparse frontier.  Four additional points add
# attainable, non-dominated operating points between them.  All points are
# fixed before calibration, satisfying the CP construction's selection rule.
SPARSE = np.array([
    [0.45, 0.02], [0.60, 0.06], [0.75, 0.15], [0.88, 0.32],
])
DENSE = np.array([
    [0.45, 0.02], [0.52, 0.035], [0.60, 0.06], [0.68, 0.10],
    [0.75, 0.15], [0.82, 0.22], [0.88, 0.32], [0.94, 0.45],
])

# Dominated padding: lower TPR and higher FPR than the worst frontier point.
PADDING = np.array([
    [0.05, 0.60], [0.10, 0.65], [0.15, 0.70], [0.20, 0.75],
    [0.25, 0.80], [0.30, 0.82], [0.35, 0.85], [0.40, 0.88],
    [0.42, 0.90], [0.43, 0.92], [0.44, 0.94], [0.445, 0.96],
    [0.30, 0.90], [0.25, 0.92], [0.20, 0.94], [0.15, 0.96],
    [0.10, 0.97], [0.08, 0.98], [0.06, 0.985], [0.04, 0.99],
    [0.03, 0.995], [0.02, 0.998], [0.01, 0.999], [0.005, 1.0],
    [0.40, 0.91], [0.38, 0.93], [0.36, 0.95], [0.34, 0.97],
])


def pool(frontier_density: int, total_k: int) -> np.ndarray:
    if frontier_density == 4:
        front = SPARSE
    elif frontier_density == 8:
        front = DENSE
    else:
        raise ValueError(frontier_density)
    if total_k < len(front):
        raise ValueError("total_k must contain the requested frontier")
    return np.vstack([front, PADDING[: total_k - len(front)]])


def simulate_cell(frontier_density: int, total_k: int, n1: int,
                  reps: int, seed: int) -> list[dict]:
    points = pool(frontier_density, total_k)
    a, b = points[:, 0], points[:, 1]
    truth = macro_f1(a[:, None], b[:, None], PS[None, :])
    oracle = truth.max(axis=0)
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    n0 = 4 * n1
    for rep in range(reps):
        # Independent columns and observations within each class.
        u1 = rng.uniform(size=(n1, total_k))
        u0 = rng.uniform(size=(n0, total_k))
        ah = (u1 < a[None, :]).mean(axis=0)
        bh = (u0 < b[None, :]).mean(axis=0)
        for kind in ("cp", "dkw"):
            band = bounds(ah, bh, n1, n0, alpha=.05, kind=kind,
                          n_models=total_k)
            rec = recommendations(ah, bh, band, PS)
            idx = np.arange(len(PS))
            selected = rec["selected"]
            actual_regret = oracle - truth[selected, idx]
            certified = rec["certified"]
            false_strict = certified & (actual_regret > 1e-10)
            rows.append({
                "frontier_density": frontier_density,
                "total_k": total_k,
                "n1": n1,
                "n0": n0,
                "rep": rep,
                "band": kind,
                "rate_coverage": float(np.all((a >= band[0] - 1e-12) &
                                                (a <= band[1] + 1e-12) &
                                                (b >= band[2] - 1e-12) &
                                                (b <= band[3] + 1e-12))),
                "metric_coverage": float(np.all((truth >= rec["lower"] - 1e-12) &
                                                  (truth <= rec["upper"] + 1e-12))),
                "regret_coverage": float(np.all(actual_regret <= rec["regret_bound"] + 1e-12)),
                "strict_cert_fraction": float(certified.mean()),
                "strict_cert_any": float(certified.any()),
                "false_strict_fraction": float(false_strict.mean()),
                "false_strict_any": float(false_strict.any()),
                "mean_actual_regret": float(actual_regret.mean()),
                "max_actual_regret": float(actual_regret.max()),
                "mean_regret_bound": float(rec["regret_bound"].mean()),
                "max_regret_bound": float(rec["regret_bound"].max()),
            })
    return rows


def main() -> None:
    tic = time.perf_counter()
    all_rows: list[dict] = []
    # 2 density x 2 total-K x 4 sample sizes x 2 bands x 1,000 reps.
    for di, density in enumerate((4, 8)):
        for ki, total_k in enumerate((8, 32)):
            for ni, n1 in enumerate((25, 100, 400, 800)):
                all_rows.extend(simulate_cell(
                    density, total_k, n1, 1000,
                    2026100830 + di * 100000 + ki * 10000 + ni * 100,
                ))
                print("orthogonal", density, total_k, n1, flush=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT / "orthogonal_pool_ablation_repetitions.csv", index=False)
    keys = ["frontier_density", "total_k", "n1", "band"]
    grouped = raw.groupby(keys, dropna=False)
    summary = grouped.agg(
        repetitions=("rep", "size"),
        rate_coverage=("rate_coverage", "mean"),
        metric_coverage=("metric_coverage", "mean"),
        regret_coverage=("regret_coverage", "mean"),
        strict_cert_fraction=("strict_cert_fraction", "mean"),
        strict_cert_any=("strict_cert_any", "mean"),
        false_strict_fraction=("false_strict_fraction", "mean"),
        false_strict_any=("false_strict_any", "mean"),
        mean_actual_regret=("mean_actual_regret", "mean"),
        max_actual_regret=("max_actual_regret", "mean"),
        mean_regret_bound=("mean_regret_bound", "mean"),
        max_regret_bound=("max_regret_bound", "mean"),
    ).reset_index()
    metric_cols = [c for c in summary.columns if c in {
        "rate_coverage", "metric_coverage", "regret_coverage",
        "strict_cert_fraction", "strict_cert_any", "false_strict_fraction",
        "false_strict_any", "mean_actual_regret", "max_actual_regret",
        "mean_regret_bound", "max_regret_bound",
    }]
    for col in metric_cols:
        summary[col + "_mcse"] = grouped[col].std(ddof=1).fillna(0).to_numpy() / np.sqrt(summary.repetitions)
    summary.to_csv(OUT / "orthogonal_pool_ablation_summary.csv", index=False)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": "2x2 orthogonal rate-level candidate-pool ablation",
        "frontier_density": [4, 8],
        "total_k": [8, 32],
        "n1": [25, 100, 400, 800],
        "n0_ratio": 4,
        "repetitions_per_cell": 1000,
        "calibration_draws": 16000,
        "method_records": 32000,
        "dkw_models": "K independent binary score columns",
        "bands": ["cp", "dkw"],
        "prevalence_grid_points": 25,
        "seed_base": 2026100830,
        "interpretation": "Fixed-density pool expansion changes competitors and confidence allocation; the oracle remains unchanged. Conditions have independent seeds; CP/DKW share each draw. DKW M=K.",
        "scope": "Controlled rate-level sensitivity analysis; not a score-model or deployment experiment.",
    }
    (OUT / "orthogonal_pool_ablation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("DONE", len(raw), "records", time.perf_counter() - tic, "seconds", flush=True)


if __name__ == "__main__":
    main()
