"""MuJoCo camera rollout videos (non-fatal; §11.7.4)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from robot_routes.contracts import PathTracker, SceneSpec
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.expert.oracle import label as expert_label


def showcase_camera(scene: SceneSpec) -> mujoco.MjvCamera:
    """Third-person view centered on workspace clutter + goal."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    gx, gy, gz = scene.goal
    cam.lookat[:] = [gx * 0.55 + 0.2, gy * 0.4, gz * 0.55 + 0.12]
    cam.distance = 1.85
    cam.azimuth = 132.0
    cam.elevation = -18.0
    return cam


def write_gif(frames: list[np.ndarray], path: Path, *, fps: int = 20) -> bool:
    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    duration_ms = max(1, int(1000 / fps))
    pil[0].save(
        path,
        save_all=True,
        append_images=pil[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return path.exists()


def write_mp4(frames: list[np.ndarray], path: Path, *, fps: int = 20) -> bool:
    try:
        import imageio.v3 as iio
    except ImportError:
        return False
    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", pixelformat="yuv420p")
        return path.exists()
    except Exception:
        return False


def write_frames(frames: list[np.ndarray], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(np.ascontiguousarray(frame)).save(out_dir / f"{stem}_{i:04d}.png")
    return out_dir


def rollout_expert_frames(
    scene: SceneSpec,
    expert: ExpertOracle,
    *,
    max_steps: int = 200,
    settle_steps: int = 5,
) -> tuple[list[np.ndarray], bool]:
    env = PandaReachEnv(render_mode="rgb_array")
    cam = showcase_camera(scene)
    frames: list[np.ndarray] = []
    obs, info = env.reset(options={"scene": scene})
    frame = env.render(camera=cam)
    if frame is not None:
        frames.append(frame.copy())

    path = expert.plan(info["q"], scene, scene.seed, time_budget_s=expert.cfg.t_validate_s)
    if path is None:
        env.close()
        return frames, False

    tracker = PathTracker()
    success = False
    for _ in range(max_steps):
        a = expert_label(info["q"], path, tracker, expert.cfg.lookahead)
        obs, _, term, trunc, info = env.step(a)
        frame = env.render(camera=cam)
        if frame is not None:
            frames.append(frame.copy())
        if info.get("success"):
            success = True
            for _ in range(settle_steps):
                obs, _, term, trunc, info = env.step(np.zeros(7, dtype=np.float64))
                frame = env.render(camera=cam)
                if frame is not None:
                    frames.append(frame.copy())
            break
        if term or trunc:
            break
    env.close()
    return frames, success


def rollout_policy_frames(
    scene: SceneSpec,
    policy: object,
    *,
    max_steps: int = 200,
) -> tuple[list[np.ndarray], bool]:
    env = PandaReachEnv(render_mode="rgb_array")
    cam = showcase_camera(scene)
    frames: list[np.ndarray] = []
    obs, info = env.reset(options={"scene": scene})
    frame = env.render(camera=cam)
    if frame is not None:
        frames.append(frame.copy())

    a_prev = None
    success = False
    for _ in range(max_steps):
        a = policy.act(obs, stochastic=False, a_prev=a_prev)
        a_prev = a
        obs, _, term, trunc, info = env.step(a)
        frame = env.render(camera=cam)
        if frame is not None:
            frames.append(frame.copy())
        if info.get("success"):
            success = True
            break
        if term or trunc:
            break
    env.close()
    return frames, success


def export_rollout(
    frames: list[np.ndarray],
    out_base: Path,
    *,
    fps: int = 20,
    also_png: bool = True,
) -> dict[str, str]:
    out: dict[str, str] = {}
    gif = out_base.with_suffix(".gif")
    mp4 = out_base.with_suffix(".mp4")
    if write_gif(frames, gif, fps=fps):
        out["gif"] = str(gif)
    if write_mp4(frames, mp4, fps=fps):
        out["mp4"] = str(mp4)
    if also_png and frames:
        png_dir = out_base.parent / f"{out_base.stem}_frames"
        write_frames(frames, png_dir, out_base.stem)
        out["frames"] = str(png_dir)
    return out
