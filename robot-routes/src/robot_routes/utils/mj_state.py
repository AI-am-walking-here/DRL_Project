"""MuJoCo state save/restore via mjSTATE_INTEGRATION (§3.3, §15.3)."""

from __future__ import annotations

import mujoco
import numpy as np

from robot_routes.contracts import MjSimState


def state_size(model: mujoco.MjModel) -> int:
    return int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))


def get_state(model: mujoco.MjModel, data: mujoco.MjData, step_idx: int = 0) -> MjSimState:
    n = state_size(model)
    buf = np.empty(n, dtype=np.float64)
    mujoco.mj_getState(model, data, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
    return MjSimState(state=buf.copy(), step_idx=step_idx)


def set_state(model: mujoco.MjModel, data: mujoco.MjData, sim_state: MjSimState) -> None:
    mujoco.mj_setState(model, data, sim_state.state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
