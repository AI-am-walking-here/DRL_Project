"""Overall pipeline progress bar + JSON status (§11.7)."""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot_routes.utils.progress import _bar


def ordered_stages(stages_allowed: set[str]) -> list[str]:
    """Stages this run will execute, in DAG order."""
    always = {"setup", "calibrate_delta"}
    out: list[str] = []
    for name in (
        "setup",
        "scene_sets",
        "calibrate_delta",
        "collect_bc",
        "train_bc",
        "dagger_rac",
        "curriculum",
        "evaluate_val",
        "ppo",
        "evaluate_test",
        "verdicts",
        "report_assets",
    ):
        if name in always:
            out.append(name)
        elif name == "scene_sets":
            if "collect_bc" in stages_allowed or "evaluate_val" in stages_allowed:
                out.append(name)
        elif name in stages_allowed:
            out.append(name)
    return out


@dataclass
class PipelineProgress:
    run_dir: Path
    condition: str
    seed: int
    profile: str
    stages: list[str]
    status_path: Path | None = None
    _stage_status: dict[str, str] = field(default_factory=dict)
    _current: str | None = None
    _detail: str = ""
    _sub: dict[str, Any] = field(default_factory=dict)
    _start: float = field(default_factory=time.monotonic)
    _tqdm: Any = None
    _poll_stop: threading.Event | None = None
    _poll_thread: threading.Thread | None = None
    _last_log_line: float = 0.0

    def __post_init__(self) -> None:
        if self.status_path is None:
            self.status_path = self.run_dir / ".pipeline_progress.json"
        self._sync_from_disk()
        label = f"{self.condition} seed{self.seed} ({self.profile})"
        try:
            from tqdm import tqdm

            done = self._completed_count()
            self._tqdm = tqdm(
                total=len(self.stages),
                initial=done,
                desc=label,
                unit="stage",
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        except ImportError:
            self._tqdm = None
            sys.stderr.write(
                f"{label}: {self._completed_count()}/{len(self.stages)} stages\n"
            )
            sys.stderr.flush()
        self._write()

    def _merge_snapshot(self, snap: dict[str, Any]) -> None:
        for key in (
            "dagger_round",
            "dagger_rounds_total",
            "dagger_round_frac",
            "dagger_phase",
            "dagger_transitions",
            "last_success_rate",
            "curriculum_step",
            "curriculum_steps_total",
            "collect_shards_done",
            "collect_shards_total",
            "liveness",
            "warning",
            "live_desc",
            "live_phase",
        ):
            if key in snap:
                self._sub[key] = snap[key]
        if snap.get("detail"):
            self._detail = str(snap["detail"])

    def refresh_from_disk(self) -> None:
        """Merge live subprocess progress + recompute ETA (for watch / heartbeat loops)."""
        from robot_routes.pipeline.stage_progress import snapshot_run_progress

        snap = snapshot_run_progress(self.run_dir)
        self._sync_from_disk()
        for s in self.stages:
            for entry in snap.get("stages", []):
                if entry["name"] == s:
                    self._stage_status[s] = entry["status"]
        self._current = snap.get("current_stage")
        self._merge_snapshot(snap)
        self._write(
            eta_override=float(snap["eta_s"]) if snap.get("eta_s") is not None else None,
            overall_override=float(snap["overall_pct"]) if snap.get("overall_pct") is not None else None,
        )
        self._refresh_display()
        self._maybe_log_status_line()

    def _maybe_log_status_line(self) -> None:
        """Emit a plain-text status line periodically (visible above tqdm in tmux)."""
        now = time.monotonic()
        if now - self._last_log_line < 60.0:
            return
        self._last_log_line = now
        stage = self._current or "idle"
        overall = self._overall_fraction()
        parts = [f"[pipeline] {overall:.1f}% stage={stage}"]
        if self._sub.get("dagger_transitions"):
            parts.append(f"trans={self._sub['dagger_transitions']}")
        if self._sub.get("dagger_phase"):
            parts.append(str(self._sub["dagger_phase"]))
        if self._sub.get("live_desc"):
            parts.append(str(self._sub["live_desc"]))
        if self._sub.get("liveness") not in (None, "alive", "idle"):
            parts.append(f"!{self._sub['liveness']}")
        sys.stderr.write("\n" + " ".join(parts) + "\n")
        sys.stderr.flush()

    def _completed_count(self) -> int:
        return sum(1 for s in self.stages if self._stage_status.get(s) == "COMPLETED")

    def _sync_from_disk(self) -> None:
        state_path = self.run_dir / "pipeline_state.json"
        disk: dict[str, Any] = {}
        if state_path.exists():
            disk = json.loads(state_path.read_text())
        for stage in self.stages:
            stamp = self.run_dir / f"{stage}.stamp"
            entry = disk.get("stages", {}).get(stage, {})
            st = entry.get("status", "PENDING")
            if stamp.exists() and st != "COMPLETED":
                st = "COMPLETED"
            if st in ("COMPLETED", "SKIPPED", "FAILED", "RUNNING", "WAITING_DEP", "PENDING"):
                self._stage_status[stage] = st
            else:
                self._stage_status.setdefault(stage, "PENDING")

    def stage_skipped(self, stage: str) -> None:
        if self._stage_status.get(stage) == "COMPLETED":
            return
        self._stage_status[stage] = "COMPLETED"
        self._detail = "skipped (cached)"
        self._write()
        if self._tqdm is not None:
            self._tqdm.update(1)
            self._tqdm.set_postfix(stage=stage, status="skip", refresh=False)
        else:
            self._print_fallback()

    def stage_running(self, stage: str) -> None:
        self._current = stage
        self._detail = ""
        self._sub = {}
        self._stage_status[stage] = "RUNNING"
        self._start_poll()
        self._write()
        if self._tqdm is not None:
            self._tqdm.set_postfix(stage=stage, status="RUNNING", refresh=True)

    def stage_waiting(self, stage: str, detail: str) -> None:
        self._current = stage
        self._detail = detail
        self._stage_status[stage] = "WAITING_DEP"
        self._write()
        if self._tqdm is not None:
            self._tqdm.set_postfix(stage=stage, status="WAITING", refresh=True)
        else:
            self._print_fallback()

    def stage_done(self, stage: str) -> None:
        self._stop_poll()
        self._stage_status[stage] = "COMPLETED"
        self._current = stage
        self._detail = ""
        self._sub = {}
        self._write()
        if self._tqdm is not None:
            self._tqdm.update(1)
            self._tqdm.set_postfix(stage=stage, status="done", refresh=True)
        else:
            self._print_fallback()

    def stage_failed(self, stage: str, error: str) -> None:
        self._stop_poll()
        self._stage_status[stage] = "FAILED"
        self._current = stage
        self._detail = error[:200]
        self._write()
        if self._tqdm is not None:
            self._tqdm.set_postfix(stage=stage, status="FAILED", refresh=True)

    def close(self, ok: bool = True) -> None:
        self._stop_poll()
        if self._tqdm is not None:
            self._tqdm.set_postfix(status="complete" if ok else "failed", refresh=True)
            self._tqdm.close()
        self._write(done=True, ok=ok)

    def _start_poll(self) -> None:
        self._stop_poll()
        self._poll_stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _stop_poll(self) -> None:
        if self._poll_stop is not None:
            self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
        self._poll_stop = None
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while self._poll_stop is not None and not self._poll_stop.is_set():
            self._enrich_sub_status()
            self._write()
            self._refresh_display()
            if self._poll_stop.wait(5.0):
                break

    def _enrich_sub_status(self) -> None:
        from robot_routes.pipeline.stage_progress import snapshot_run_progress

        snap = snapshot_run_progress(self.run_dir)
        self._merge_snapshot(snap)
        self._maybe_log_status_line()

    def _stage_fraction(self) -> float:
        """In-stage progress in [0, 1) for long-running stages."""
        stage = self._current
        if stage == "dagger_rac":
            n = float(self._sub.get("dagger_round", 0)) + float(self._sub.get("dagger_round_frac", 0))
            total = int(self._sub.get("dagger_rounds_total", 1))
            return min(n / max(total, 1), 0.999)
        if stage == "curriculum":
            n = int(self._sub.get("curriculum_step", 0))
            total = int(self._sub.get("curriculum_steps_total", 1))
            return min(n / max(total, 1), 0.999)
        if stage == "collect_bc":
            shards = int(self._sub.get("collect_shards_done", 0))
            total = int(self._sub.get("collect_shards_total", 8))
            if total > 0:
                return min(shards / total, 0.999)
        if self._stage_status.get(stage) == "RUNNING":
            return 0.0
        return 0.0

    def _overall_fraction(self) -> float:
        total = len(self.stages)
        if total == 0:
            return 100.0
        done = self._completed_count()
        if self._current and self._stage_status.get(self._current) == "RUNNING":
            done += self._stage_fraction()
        return min(100.0 * done / total, 100.0)

    def _refresh_display(self) -> None:
        overall = self._overall_fraction()
        if self._tqdm is not None:
            desc = f"{self.condition} seed{self.seed}"
            if self._sub.get("dagger_transitions"):
                desc += f" | {self._sub['dagger_transitions']}"
            elif self._sub.get("collect_shards_done") is not None:
                desc += (
                    f" | shards {self._sub['collect_shards_done']}/"
                    f"{self._sub.get('collect_shards_total', '?')}"
                )
            self._tqdm.set_description(desc, refresh=False)
            postfix: dict[str, Any] = {
                "stage": self._current or "-",
                "overall": f"{overall:.1f}%",
            }
            if "dagger_round" in self._sub:
                postfix["dagger"] = (
                    f"{self._sub['dagger_round']}/{self._sub.get('dagger_rounds_total', '?')}"
                )
            if self._sub.get("dagger_transitions"):
                postfix["trans"] = self._sub["dagger_transitions"]
            if self._sub.get("dagger_phase"):
                postfix["phase"] = self._sub["dagger_phase"]
            if self._sub.get("liveness") not in (None, "alive", "idle"):
                postfix["!"] = self._sub["liveness"]
            if "curriculum_step" in self._sub:
                postfix["cur"] = (
                    f"{self._sub['curriculum_step']}/{self._sub.get('curriculum_steps_total', '?')}"
                )
            self._tqdm.set_postfix(**{k: str(v) for k, v in postfix.items()}, refresh=True)
        else:
            self._print_fallback()

    def _print_fallback(self) -> None:
        overall = self._overall_fraction()
        done = self._completed_count()
        total = len(self.stages)
        elapsed = time.monotonic() - self._start
        rate = max(overall, 1e-6) / max(elapsed, 1e-6)
        eta = (100.0 - overall) / max(rate, 1e-6)
        cur = self._current or "-"
        extra = " ".join(f"{k}={v}" for k, v in self._sub.items())
        line = (
            f"\r{_bar(overall)} {overall:5.1f}% {done}/{total} stage={cur} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s {extra} {self._detail}"
        )
        sys.stderr.write(line[:160].ljust(160))
        sys.stderr.flush()

    def _write(
        self,
        done: bool = False,
        ok: bool = True,
        eta_override: float | None = None,
        overall_override: float | None = None,
    ) -> None:
        if self.status_path is None:
            return
        done_n = self._completed_count()
        total = len(self.stages)
        pct = 100.0 * done_n / max(total, 1)
        overall_pct = overall_override if overall_override is not None else self._overall_fraction()
        elapsed = time.monotonic() - self._start
        rate = overall_pct / max(elapsed, 1e-6)
        eta_s = eta_override if eta_override is not None else (
            (100.0 - overall_pct) / max(rate, 1e-6) if not done else 0.0
        )
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "condition": self.condition,
            "seed": self.seed,
            "profile": self.profile,
            "current_stage": self._current,
            "current_status": self._stage_status.get(self._current or "", "PENDING"),
            "detail": self._detail,
            "stages_completed": done_n,
            "stages_total": total,
            "pct": round(pct, 2),
            "overall_pct": round(overall_pct, 2),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta_s, 1),
            "done": done,
            "ok": ok if done else None,
            "stages": [
                {"name": s, "status": self._stage_status.get(s, "PENDING")} for s in self.stages
            ],
            "snapshot": False,
            **self._sub,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.status_path)
            human = self.run_dir / "pipeline_status.txt"
            sub = ""
            if "dagger_round" in self._sub:
                sub = (
                    f" dagger {self._sub['dagger_round']}/"
                    f"{self._sub.get('dagger_rounds_total', '?')}"
                )
                if self._sub.get("dagger_transitions"):
                    sub += f" trans {self._sub['dagger_transitions']}"
                if self._sub.get("dagger_phase"):
                    sub += f" [{self._sub['dagger_phase']}]"
            elif "curriculum_step" in self._sub:
                sub = (
                    f" curriculum {self._sub['curriculum_step']}/"
                    f"{self._sub.get('curriculum_steps_total', '?')}"
                )
            warn = f" | {self._detail}" if self._detail else ""
            live = f" | {self._sub.get('liveness', '')}" if self._sub.get("liveness") not in (
                None,
                "alive",
                "idle",
            ) else ""
            human.write_text(
                f"{overall_pct:5.1f}% | {self._current or 'idle'} | "
                f"{done_n}/{total} stages{sub} | eta {eta_s/3600:.1f}h{warn}{live}\n"
            )
        except OSError:
            pass
