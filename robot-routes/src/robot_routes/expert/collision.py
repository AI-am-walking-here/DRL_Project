"""MuJoCo collision checker for planning (§4.1)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from robot_routes.contracts import SceneSpec

SCENE_XML = Path(__file__).resolve().parents[1] / "envs" / "assets" / "reach_scene.xml"
MAX_OBS = 8


class CollisionChecker:
    def __init__(self, margin_plan: float = 0.03) -> None:
        self.margin_plan = margin_plan
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self._table_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._obs_gids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            for i in range(MAX_OBS)
        ]
        self._robot_gids = self._robot_geom_ids()
        link0_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self._link0_gids = {g for g in self._robot_gids if self.model.geom_bodyid[g] == link0_bid}

    def _robot_geom_ids(self) -> set[int]:
        geoms: set[int] = set()
        for i in range(self.model.ngeom):
            body = self.model.geom_bodyid[i]
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if bname.startswith("link") or bname == "attachment":
                geoms.add(i)
        return geoms

    def set_scene(self, scene: SceneSpec) -> None:
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
            else:
                self.model.geom_pos[gid] = [5, 5, 5]
                self.model.geom_contype[gid] = 0

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)

    def clearance(self) -> float:
        distmax = 0.25
        best = float("inf")
        for rg in self._robot_gids:
            for og in self._obs_gids + [self._table_gid]:
                if self.model.geom_contype[og] == 0 and og in self._obs_gids:
                    continue
                if og == self._table_gid and rg in self._link0_gids:
                    continue  # §3.2: base mounted on table
                d = float(mujoco.mj_geomDistance(self.model, self.data, rg, og, distmax, None))
                best = min(best, d)
        return best if best < float("inf") else distmax

    def free(self, q: np.ndarray) -> bool:
        self.set_q(q)
        return self.clearance() > self.margin_plan

    def in_contact(self) -> bool:
        """True when robot geom contacts obstacle/table (matches PandaReachEnv)."""
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (int(c.geom1), int(c.geom2))
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
