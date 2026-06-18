"""Gymnasium wrapper around MuJoCo Panda reach task (§3, §15.7.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot_routes.contracts import Q_HOME, MjSimState, SceneSpec
from robot_routes.envs.scene_gen import encode_obs, make_slots, sample_scene
from robot_routes.utils.config import EnvConfig, SceneConfig
from robot_routes.utils.mj_state import get_state, set_state

ASSETS_DIR = Path(__file__).parent / "assets"
SCENE_XML = ASSETS_DIR / "reach_scene.xml"
MAX_OBS = 8


class PandaReachEnv(gym.Env):
    """obs: Box(79,), act: Box(7,) in [-0.05, 0.05]."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        scene_cfg: SceneConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.scene_cfg = scene_cfg or SceneConfig()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self._ee_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self._goal_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self._table_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._obs_gids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            for i in range(MAX_OBS)
        ]
        self._robot_gids = self._collect_robot_geoms()
        link0_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self._link0_gids = {g for g in self._robot_gids if self.model.geom_bodyid[g] == link0_bid}
        self._excluded = self._build_exclusions()
        self._jnt_lo = self.model.jnt_range[:7, 0].copy()
        self._jnt_hi = self.model.jnt_range[:7, 1].copy()
        self.action_space = spaces.Box(
            low=-self.cfg.action_clip, high=self.cfg.action_clip, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.obs_dim,), dtype=np.float32
        )
        self._scene = self._empty_scene()
        self._slots = np.zeros((MAX_OBS, 7), dtype=np.float64)
        self._goal = np.zeros(3, dtype=np.float64)
        self._t = 0
        self._hold = 0
        self._rng = np.random.default_rng(0)
        self._renderer: mujoco.Renderer | None = None

    def _empty_scene(self) -> SceneSpec:
        return SceneSpec((), (0.5, 0.0, 0.3), tuple(float(x) for x in Q_HOME), 0, 0)

    def _collect_robot_geoms(self) -> set[int]:
        geoms: set[int] = set()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("obs_") or name == "table":
                continue
            body = self.model.geom_bodyid[i]
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if bname.startswith("link") or bname in ("link0", "attachment"):
                geoms.add(i)
        return geoms

    def _build_exclusions(self) -> set[tuple[int, int]]:
        excluded: set[tuple[int, int]] = set()
        exclusions = self.cfg.contact_exclusions or [("link0", "table")]
        for a, b in exclusions:
            try:
                if a == "table":
                    g1 = self._table_gid
                else:
                    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, a)
                    g1 = next(
                        i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == bid
                    )
                if b == "table":
                    g2 = self._table_gid
                else:
                    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                    g2 = next(
                        i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == bid
                    )
                excluded.add((g1, g2))
                excluded.add((g2, g1))
            except StopIteration:
                pass
        return excluded

    def _apply_scene(self, scene: SceneSpec) -> None:
        self._scene = scene
        self._goal = np.asarray(scene.goal, dtype=np.float64)
        self.model.site_pos[self._goal_sid] = self._goal
        for i, gid in enumerate(self._obs_gids):
            if i < len(scene.obstacles):
                o = scene.obstacles[i]
                self.model.geom_pos[gid] = o.center
                if o.kind == "box":
                    self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_BOX
                    self.model.geom_size[gid] = o.size
                else:
                    self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_SPHERE
                    self.model.geom_size[gid] = [o.size[0], 0, 0]
                self.model.geom_contype[gid] = 1
                self.model.geom_conaffinity[gid] = 1
            else:
                self.model.geom_pos[gid] = [5, 5, 5]
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0
        self.data.qpos[:7] = scene.q_start
        self.data.qvel[:7] = 0
        self.data.ctrl[:7] = scene.q_start
        mujoco.mj_forward(self.model, self.data)
        ee = self.data.site_xpos[self._ee_sid].copy()
        self._slots = make_slots(scene, ee)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if "scene" in options:
            scene = options["scene"]
            assert isinstance(scene, SceneSpec)
        else:
            level = int(options.get("level", 0))
            bounds = options.get("level_bounds", [2, 3])
            scene = sample_scene(
                level,
                self._rng,
                self.scene_cfg,
                bounds,
                self._default_start_ee(),
                self._penetrates_start,
                unseen=bool(options.get("unseen", False)),
            )
            if scene is None:
                scene = SceneSpec(
                    (),
                    (0.5, 0.0, 0.4),
                    tuple(float(x) for x in Q_HOME),
                    level,
                    int(self._rng.integers(2**31)),
                )
        self._apply_scene(scene)
        self._t = 0
        self._hold = 0
        obs = self._encode()
        info = self._info()
        return obs, info

    def _default_start_ee(self) -> np.ndarray:
        self.data.qpos[:7] = Q_HOME
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self._ee_sid].copy()

    def _penetrates_start(self, scene: SceneSpec) -> bool:
        saved_q = self.data.qpos[:7].copy()
        self._apply_scene(scene)
        if self._contact_violation():
            self.data.qpos[:7] = saved_q
            mujoco.mj_forward(self.model, self.data)
            return True
        self.data.qpos[:7] = saved_q
        mujoco.mj_forward(self.model, self.data)
        return False

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.clip(
            np.asarray(action, dtype=np.float64), -self.cfg.action_clip, self.cfg.action_clip
        )
        q = self.data.qpos[:7].copy()
        q_target = np.clip(
            q + a,
            self._jnt_lo + self.cfg.jlimit_margin_rad,
            self._jnt_hi - self.cfg.jlimit_margin_rad,
        )
        self.data.ctrl[:7] = q_target
        for _ in range(self.cfg.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._t += 1
        collision = self._contact_violation()
        clearance = self.min_clearance()
        ee = self.data.site_xpos[self._ee_sid].copy()
        at_goal = float(np.linalg.norm(ee - self._goal)) < self.cfg.success_tol_m
        self._hold = self._hold + 1 if at_goal else 0
        success = self._hold >= self.cfg.success_hold_steps
        terminated = bool(collision or success)
        truncated = bool(self._t >= self.cfg.horizon and not terminated)
        reward = float(
            -0.1 * np.linalg.norm(ee - self._goal)
            + 10.0 * success
            - 25.0 * collision
            - 1e-3 * float(a @ a)
        )
        info = {
            "collision": collision,
            "min_clearance": clearance,
            "ee_pos": ee,
            "q": self.data.qpos[:7].copy(),
            "success": success,
        }
        return self._encode(), reward, terminated, truncated, info

    def _encode(self) -> np.ndarray:
        q = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        ee = self.data.site_xpos[self._ee_sid].copy()
        return encode_obs(q, qvel, ee, self._goal, self._slots)

    def _info(self) -> dict:
        ee = self.data.site_xpos[self._ee_sid].copy()
        return {
            "collision": False,
            "min_clearance": self.min_clearance(),
            "ee_pos": ee,
            "q": self.data.qpos[:7].copy(),
            "success": False,
        }

    def _contact_violation(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (int(c.geom1), int(c.geom2))
            if pair in self._excluded or pair[::-1] in self._excluded:
                continue
            if pair[0] in self._robot_gids or pair[1] in self._robot_gids:
                g1, g2 = pair[0], pair[1]
                if g1 in self._link0_gids and g2 == self._table_gid:
                    continue
                if g2 in self._link0_gids and g1 == self._table_gid:
                    continue
                if g1 in self._robot_gids and (g2 in self._obs_gids or g2 == self._table_gid):
                    return True
                if g2 in self._robot_gids and (g1 in self._obs_gids or g1 == self._table_gid):
                    return True
        return False

    def min_clearance(self) -> float:
        distmax = 0.25
        best = float("inf")
        for rg in self._robot_gids:
            for og in self._obs_gids + [self._table_gid]:
                if self.model.geom_contype[og] == 0 and og in self._obs_gids:
                    continue
                if og == self._table_gid and rg in self._link0_gids:
                    continue
                d = float(mujoco.mj_geomDistance(self.model, self.data, rg, og, distmax, None))
                if d < best:
                    best = d
        return best if best < float("inf") else distmax

    def get_state(self) -> MjSimState:
        return get_state(self.model, self.data, self._t)

    def set_state(self, s: MjSimState) -> np.ndarray:
        set_state(self.model, self.data, s)
        self._t = s.step_idx
        return self._encode()

    @property
    def scene(self) -> SceneSpec:
        return self._scene

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
