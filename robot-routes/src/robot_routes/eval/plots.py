"""Report figures from eval JSONs (§10.3)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _metric_value(data: dict[str, Any]) -> float:
    for key in ("success_rate", "mean_n_routes"):
        v = data.get(key)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return 0.0


def plot_success_curves(run_dirs: list[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    plotted = False
    for rd in run_dirs:
        state = (
            json.loads((rd / "pipeline_state.json").read_text())
            if (rd / "pipeline_state.json").exists()
            else {}
        )
        rounds = state.get("dagger_rounds", [])
        if rounds:
            ax.plot(range(len(rounds)), rounds, label=rd.name)
            plotted = True
    ax.set_xlabel("Round")
    ax.set_ylabel("Success rate")
    if plotted:
        ax.legend()
    fig.savefig(out / "success_curves.png")
    plt.close(fig)


def plot_route_bars(metrics: dict[str, float], planner_ceiling: float, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    labels = list(metrics.keys())
    vals = [metrics[k] for k in labels]
    ax.bar(labels, vals)
    if not math.isnan(planner_ceiling):
        ax.axhline(planner_ceiling, color="r", linestyle="--", label="planner ceiling")
    ax.set_ylabel("Metric")
    ax.legend()
    fig.savefig(out / "route_bars.png")
    plt.close(fig)


def build_report(run_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / "eval"
    metrics: dict[str, float] = {}
    ceiling = float("nan")
    for name in ("val_eval.json", "test_eval.json"):
        p = eval_dir / name
        if p.exists():
            data = json.loads(p.read_text())
            metrics[name] = _metric_value(data)
            c = data.get("planner_ceiling_mean", ceiling)
            if isinstance(c, (int, float)) and not math.isnan(c):
                ceiling = float(c)
    if metrics:
        plot_route_bars(metrics, ceiling, out)
    plot_success_curves([run_dir], out)
    summary = _json_safe(
        {
            "run": run_dir.name,
            "metrics": metrics,
            "verdicts": (
                json.loads((run_dir / "hypotheses_verdicts.json").read_text())
                if (run_dir / "hypotheses_verdicts.json").exists()
                else {}
            ),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
