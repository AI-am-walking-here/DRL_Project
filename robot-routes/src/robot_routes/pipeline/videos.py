"""Non-fatal showcase rollout videos (§11.7.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_routes.contracts import SceneSpec
from robot_routes.envs.panda_reach_env import PandaReachEnv


def render_showcase_video(
    policy: object,
    scene: SceneSpec,
    out_path: Path,
    *,
    seed: int = 0,
    fps: int = 20,
) -> bool:
    """Write mp4 if matplotlib animation succeeds; otherwise log-only failure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation
    except ImportError:
        return False
    env = PandaReachEnv()
    obs, info = env.reset(seed=seed, options={"scene": scene})
    ee_traj = [info["ee_pos"][:2].copy()]
    a_prev = None
    for _ in range(min(env.cfg.horizon, 150)):
        a = policy.act(obs, stochastic=False, a_prev=a_prev)
        a_prev = a
        obs, _, term, trunc, info = env.step(a)
        ee_traj.append(info["ee_pos"][:2].copy())
        if term or trunc:
            break
    traj = np.array(ee_traj)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.2, 0.9)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    (line,) = ax.plot([], [], "b-", lw=2)
    ax.scatter([scene.goal[0]], [scene.goal[1]], c="g", s=40)

    def update(i: int) -> tuple:
        line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
        return (line,)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim = FuncAnimation(fig, update, frames=len(traj), interval=1000 // fps)
        anim.save(str(out_path), writer=FFMpegWriter(fps=fps))
        plt.close(fig)
        return True
    except Exception:
        plt.close(fig)
        return False


def render_round_videos(
    policy: object,
    scenes: list[SceneSpec],
    out_dir: Path,
    *,
    limit: int = 4,
) -> list[str]:
    written: list[str] = []
    for i, scene in enumerate(scenes[:limit]):
        path = out_dir / f"showcase_{i}.mp4"
        if render_showcase_video(policy, scene, path, seed=scene.seed):
            written.append(str(path))
    return written
