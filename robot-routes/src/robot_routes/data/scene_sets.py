"""Frozen scene-set generation and SHA verification (§10.2, §11.7.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_routes.contracts import Q_HOME, Obstacle, SceneSpec
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.envs.scene_gen import sample_scene
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.utils.config import EnvConfig, ExpertConfig, SceneConfig, load_config
from robot_routes.utils.progress import ProgressReporter

SET_SPECS: list[tuple[str, int, list[int], bool, str]] = [
    ("val_L0", 0, [2, 3], False, "val_L0"),
    ("val_L1", 1, [3, 4], False, "val_L1"),
    ("val_L2", 2, [4, 6], False, "val_L2"),
    ("val_L3", 3, [6, 8], False, "val_L3"),
    ("val_unseen", 3, [6, 8], True, "val_unseen"),
    ("test_L0", 0, [2, 3], False, "test_L0"),
    ("test_L1", 1, [3, 4], False, "test_L1"),
    ("test_L2", 2, [4, 6], False, "test_L2"),
    ("test_L3", 3, [6, 8], False, "test_L3"),
    ("test_unseen", 3, [6, 8], True, "test_unseen"),
]

PROFILE_COUNTS = {
    "smoke": {"val": 5, "test": 5},
    "medium": {"val": 20, "test": 40},
    "day": {"val": 50, "test": 100},
    "full": {"val": 100, "test": 200},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def scene_set_path(root: Path, name: str) -> Path:
    return root / "data" / "scenes" / f"{name}.json"


def manifest_path(root: Path) -> Path:
    return root / "data" / "scenes" / "manifest.json"


def load_manifest(root: Path) -> dict[str, Any]:
    p = manifest_path(root)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def verify_scene_sets(root: Path, profile: str = "smoke") -> None:
    manifest = load_manifest(root)
    if not manifest:
        raise RuntimeError("missing data/scenes/manifest.json — run scripts/generate_scene_sets.py")
    profiles = manifest.get("profiles", {})
    if profile not in profiles:
        raise RuntimeError(
            f"scene profile {profile!r} missing from manifest — run: "
            f"make scene-sets PROFILE={profile}"
        )
    prof = profiles[profile]
    files = prof.get("files", {})
    if not files:
        raise RuntimeError(f"scene profile {profile!r} has no files in manifest")
        path = root / "data" / "scenes" / name
        if not path.exists():
            raise RuntimeError(f"missing scene set: {path}")
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            raise RuntimeError(f"scene set hash mismatch: {name}")


def write_scene_set(path: Path, scenes: list[SceneSpec], meta: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "scenes": [json.loads(s.to_json()) for s in scenes],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return sha256_file(path)


def _seed_for_range(seed_range: list[int], idx: int) -> int:
    lo, hi = seed_range
    span = hi - lo + 1
    return lo + (idx % span)


def generate_verified_scene(
    env: PandaReachEnv,
    expert: ExpertOracle,
    rng: np.random.Generator,
    level: int,
    bounds: list[int],
    unseen: bool,
) -> SceneSpec | None:
    start_ee = env._default_start_ee()

    def penetrates(scene: SceneSpec) -> bool:
        return env._penetrates_start(scene)

    for _ in range(300):
        spec = sample_scene(level, rng, env.scene_cfg, bounds, start_ee, penetrates, unseen=unseen)
        if spec is None:
            continue
        env.reset(options={"scene": spec})
        q = env.data.qpos[:7].copy()
        path = expert.plan(q, spec, spec.seed, time_budget_s=expert.cfg.t_validate_s)
        if path is not None:
            return spec
    return None


def generate_unseen_set(
    env: PandaReachEnv,
    expert: ExpertOracle,
    rng: np.random.Generator,
    count: int,
    seed_range: list[int],
    reject_max: float = 0.5,
    progress: ProgressReporter | None = None,
) -> tuple[list[SceneSpec], dict[str, Any]]:
    shift_steps = 0
    scenes: list[SceneSpec] = []
    rejections = 0
    attempts = 0
    while len(scenes) < count and shift_steps <= 10:
        attempts += 1
        idx = len(scenes)
        s = _seed_for_range(seed_range, idx + shift_steps * count)
        sub = np.random.default_rng(s)
        spec = generate_verified_scene(env, expert, sub, 3, [6, 8], unseen=True)
        if spec is None:
            rejections += 1
            if attempts > count * 20 and rejections / max(attempts, 1) > reject_max:
                shift_steps += 1
                rejections = 0
                attempts = 0
                env.scene_cfg = SceneConfig(
                    unseen_half_min=max(0.08 - 0.01 * shift_steps, 0.03),
                    unseen_half_max=max(0.14 - 0.01 * shift_steps, 0.06),
                )
            if progress is not None:
                progress.set(
                    len(scenes),
                    reject_rate=rejections / max(attempts, 1),
                    shift_steps=shift_steps,
                    attempts=attempts,
                )
            continue
        scenes.append(spec)
        if progress is not None:
            progress.set(
                len(scenes),
                reject_rate=rejections / max(attempts, 1),
                shift_steps=shift_steps,
            )
    meta = {"unseen_shift_steps": shift_steps, "reject_rate": rejections / max(attempts, 1)}
    return scenes, meta


def generate_set(
    root: Path,
    name: str,
    level: int,
    bounds: list[int],
    unseen: bool,
    count: int,
    seed_range: list[int],
    profile: str,
    progress: ProgressReporter | None = None,
) -> str:
    env_cfg = load_config(root / "configs/env/panda_reach.yaml", EnvConfig)
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))
    env = PandaReachEnv(env_cfg)
    rng = np.random.default_rng(42)
    scenes: list[SceneSpec] = []
    if unseen:
        sr = env_cfg.seed_ranges or {}
        scenes, extra = generate_unseen_set(
            env,
            expert,
            rng,
            count,
            seed_range or sr.get(name.replace("test_", "test_"), [1000400, 1000499]),
            progress=progress,
        )
        meta = {"name": name, "profile": profile, "count": count, **extra}
    else:
        sr = (env_cfg.seed_ranges or {}).get(name, [0, count - 1])
        tries = 0
        while len(scenes) < count and tries < count * 50:
            tries += 1
            sub = np.random.default_rng(_seed_for_range(sr, len(scenes) + tries))
            spec = generate_verified_scene(env, expert, sub, level, bounds, unseen=False)
            if spec is not None:
                scenes.append(spec)
                if progress is not None:
                    progress.set(
                        len(scenes),
                        set=name,
                        tries=tries,
                        reject_rate=1.0 - len(scenes) / tries,
                    )
            elif progress is not None and tries % 10 == 0:
                progress.set(
                    len(scenes),
                    set=name,
                    tries=tries,
                    reject_rate=1.0 - len(scenes) / tries,
                )
        meta = {"name": name, "profile": profile, "count": len(scenes), "requested": count}
    path = scene_set_path(root, name)
    return write_scene_set(path, scenes, meta)


def generate_all(root: Path, profile: str = "smoke", status_path: Path | None = None) -> dict[str, Any]:
    counts = PROFILE_COUNTS[profile]
    env_cfg = load_config(root / "configs/env/panda_reach.yaml", EnvConfig)
    sr = env_cfg.seed_ranges or {}
    files: dict[str, Any] = {}
    if status_path is None:
        status_path = root / "data" / "scenes" / ".generation_status.json"
    sets_prog = ProgressReporter(
        job="scene_sets",
        phase="sets",
        total=len(SET_SPECS),
        unit="set",
        status_path=status_path,
        desc=f"scene_sets profile={profile}",
    )
    sets_prog.update(0, profile=profile)
    for i, (name, level, bounds, unseen, key) in enumerate(SET_SPECS):
        n = counts["test" if name.startswith("test_") else "val"]
        seed_range = sr.get(key, [0, n - 1])
        scene_prog = ProgressReporter(
            job="scene_sets",
            phase=name,
            total=n,
            unit="scene",
            status_path=status_path,
            desc=f"{name} ({n} scenes)",
        )
        scene_prog.update(0, set_index=i + 1, set_total=len(SET_SPECS), profile=profile)
        digest = generate_set(
            root, name, level, bounds, unseen, n, seed_range, profile, progress=scene_prog
        )
        scene_prog.close(final=False, wrote=n, sha256=digest[:12])
        files[f"{name}.json"] = {"sha256": digest, "count": n}
        sets_prog.update(set=name, scenes=n, set_index=i + 1)
    sets_prog.close(sets=len(SET_SPECS))
    mpath = manifest_path(root)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if mpath.exists():
        existing = json.loads(mpath.read_text())
    profiles = existing.get("profiles", {})
    profiles[profile] = {"files": files}
    manifest = {
        "profile_default": existing.get("profile_default", profile),
        "profiles": profiles,
    }
    mpath.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_scenes(root: Path, name: str) -> list[SceneSpec]:
    path = scene_set_path(root, name)
    data = json.loads(path.read_text())
    return [SceneSpec.from_json(json.dumps(s)) for s in data["scenes"]]


def generate_pool_rl(
    root: Path,
    count: int = 16,
    profile: str = "smoke",
    status_path: Path | None = None,
) -> str:
    env = PandaReachEnv()
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))
    rng = np.random.default_rng(999)
    scenes: list[SceneSpec] = []
    if status_path is None:
        status_path = root / "data" / "scenes" / ".generation_status.json"
    prog = ProgressReporter(
        job="scene_sets",
        phase="pool_rl",
        total=count,
        unit="scene",
        status_path=status_path,
        desc=f"pool_rl ({count} scenes)",
    )
    prog.update(0, profile=profile)
    for attempt in range(count):
        spec = generate_verified_scene(env, expert, rng, 3, [6, 8], unseen=False)
        if spec:
            scenes.append(spec)
        prog.set(attempt + 1, pool_scenes=len(scenes), requested=count)
    prog.close(pool_scenes=len(scenes))
    path = root / "data" / "pool_rl.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": {"profile": profile, "count": len(scenes)},
                "scenes": [json.loads(s.to_json()) for s in scenes],
            },
            indent=2,
        )
    )
    return sha256_file(path)
