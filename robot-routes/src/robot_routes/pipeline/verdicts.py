"""Hypothesis verdicts with Holm–Bonferroni (§11.7.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_routes.diversity.route_metrics import paired_bootstrap_ci


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p_values[idx] * (m - rank))
    return adjusted.tolist()


def ci_to_p_value(ci: tuple[float, float]) -> float:
    lo, hi = ci
    if lo > 0:
        return 0.01
    if hi < 0:
        return 0.99
    return 0.5


def compute_verdicts(
    eval_json: Path,
    mde: float = 10.0,
    compare_paths: dict[str, Path] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng(0)
    data = json.loads(eval_json.read_text()) if eval_json.exists() else {}
    compare_paths = compare_paths or {}
    verdicts: dict[str, Any] = {}
    hypotheses: list[tuple[str, tuple[float, float]]] = []
    hyp_compare = {"H1": "bc_dagger", "H2": "rac_noreroute"}
    for hyp, _ in [("H1", "h1_diff"), ("H2", "h2_diff")]:
        ci = tuple(data.get(f"{hyp.lower()}_diff", data.get(f"{hyp}_ci", [0.0, 0.0])))
        ref = compare_paths.get(hyp)
        if ref and ref.exists():
            other = json.loads(ref.read_text())
            a = np.array(data.get("scene_success", []))
            b = np.array(other.get("scene_success", []))
            if len(a) and len(a) == len(b):
                ci = paired_bootstrap_ci(a, b, rng)
                verdicts[f"{hyp}_compare"] = hyp_compare.get(hyp, ref.stem)
        hypotheses.append((hyp, ci))
        lo, hi = ci
        mde_frac = mde / 100.0
        if lo > 0:
            verdicts[hyp] = "CONFIRMED"
        elif hi < mde_frac:
            verdicts[hyp] = "REFUTED"
        else:
            verdicts[hyp] = "UNDERPOWERED"
        verdicts[f"{hyp}_ci"] = [float(lo), float(hi)]
    p_vals = [ci_to_p_value(ci) for _, ci in hypotheses]
    adj = holm_bonferroni(p_vals)
    for i, (hyp, _) in enumerate(hypotheses):
        verdicts[f"{hyp}_holm_p"] = adj[i]
        if adj[i] <= 0.05 and verdicts[hyp] == "CONFIRMED":
            verdicts[f"{hyp}_adjusted"] = "CONFIRMED"
        elif adj[i] <= 0.05 and verdicts[hyp] == "REFUTED":
            verdicts[f"{hyp}_adjusted"] = "REFUTED"
        else:
            verdicts[f"{hyp}_adjusted"] = verdicts[hyp]
    h3 = data.get("h3_spearman")
    verdicts["H3"] = "OBSERVATIONAL"
    if h3 is not None and not (isinstance(h3, float) and np.isnan(h3)):
        verdicts["H3_spearman"] = h3
    return verdicts


def write_verdicts(run_dir: Path, eval_json: Path, **kwargs: Any) -> Path:
    v = compute_verdicts(eval_json, **kwargs)
    out = run_dir / "hypotheses_verdicts.json"
    out.write_text(json.dumps(v, indent=2))
    return out
