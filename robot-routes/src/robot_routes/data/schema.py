"""HDF5 dataset schema (§5.1, §15.6)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from robot_routes.contracts import SEGMENT_CODE, SEGMENT_NAME, Transition


def write_shard(
    path: Path,
    transitions: list[Transition],
    episode_scenes: list[str],
    git_hash: str = "",
    worker_id: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(transitions)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = 1
        f.attrs["git_hash"] = git_hash
        f.attrs["worker_id"] = worker_id
        f.create_dataset("obs", shape=(n, 79), maxshape=(None, 79), dtype="f4", chunks=(4096, 79))
        f.create_dataset("action", shape=(n, 7), maxshape=(None, 7), dtype="f4", chunks=(4096, 7))
        f.create_dataset("q", shape=(n, 7), maxshape=(None, 7), dtype="f4", chunks=(4096, 7))
        f.create_dataset("ee_pos", shape=(n, 3), maxshape=(None, 3), dtype="f4", chunks=(4096, 3))
        f.create_dataset("done", shape=(n,), maxshape=(None,), dtype="bool", chunks=(4096,))
        f.create_dataset("segment", shape=(n,), maxshape=(None,), dtype="u1", chunks=(4096,))
        f.create_dataset("episode_id", shape=(n,), maxshape=(None,), dtype="i4", chunks=(4096,))
        f.create_dataset("level", shape=(n,), maxshape=(None,), dtype="i4", chunks=(4096,))
        if n:
            f["obs"][:] = np.stack([t.obs for t in transitions])
            f["action"][:] = np.stack([t.action for t in transitions])
            f["q"][:] = np.stack([t.q for t in transitions])
            f["ee_pos"][:] = np.stack([t.ee_pos for t in transitions])
            f["done"][:] = np.array([t.done for t in transitions])
            f["segment"][:] = np.array(
                [SEGMENT_CODE[t.segment] for t in transitions], dtype=np.uint8
            )
            f["episode_id"][:] = np.array([t.episode_id for t in transitions], dtype=np.int32)
            f["level"][:] = np.array([t.level for t in transitions], dtype=np.int32)
        dt = h5py.special_dtype(vlen=str)
        ep_grp = f.create_group("episodes")
        ep_grp.create_dataset("scene_json", data=np.array(episode_scenes, dtype=object), dtype=dt)


def read_shard(path: Path) -> tuple[list[Transition], list[str]]:
    """Load transitions + per-episode scene JSON from a shard."""
    with h5py.File(path, "r") as f:
        n = int(f["obs"].shape[0])
        ep_scenes = list(f["episodes/scene_json"].asstr()[:])
        rows: list[Transition] = []
        for i in range(n):
            rows.append(
                Transition(
                    obs=f["obs"][i],
                    action=f["action"][i],
                    q=f["q"][i],
                    ee_pos=f["ee_pos"][i],
                    done=bool(f["done"][i]),
                    segment=SEGMENT_NAME[int(f["segment"][i])],
                    episode_id=int(f["episode_id"][i]),
                    level=int(f["level"][i]),
                )
            )
    return rows, ep_scenes


def merge_shards(shard_paths: list[Path], out_path: Path) -> int:
    all_trans: list[Transition] = []
    scenes: list[str] = []
    ep_offset = 0
    for sp in shard_paths:
        with h5py.File(sp, "r") as f:
            n = f["obs"].shape[0]
            ep_scenes = list(f["episodes/scene_json"].asstr()[:])
            scenes.extend(ep_scenes)
            for i in range(n):
                all_trans.append(
                    Transition(
                        obs=f["obs"][i],
                        action=f["action"][i],
                        q=f["q"][i],
                        ee_pos=f["ee_pos"][i],
                        done=bool(f["done"][i]),
                        segment=SEGMENT_NAME[int(f["segment"][i])],
                        episode_id=int(f["episode_id"][i]) + ep_offset,
                        level=int(f["level"][i]),
                    )
                )
            ep_offset += len(ep_scenes)
    write_shard(out_path, all_trans, scenes)
    return len(all_trans)
