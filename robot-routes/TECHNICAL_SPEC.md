# Technical Specification — *Teaching a Robot Arm Many Ways to Reach a Goal*

**Project:** Deep Reinforcement Learning — Final Project
**Authors:** Dalton Jens · Nils Ludvig
**Document purpose:** Implementation-ready technical specification, written to be executed with Cursor (AI-assisted IDE). Every module below is specified with interfaces, data formats, and acceptance criteria so that implementation can proceed file-by-file with minimal ambiguity.
**Version:** 3.7 — experimental design FROZEN as of v3.4; Revs 17–19 add additive agent-harness and code-level reference layers (§14, §15, Appendix B) without altering any experimental parameter

---

## 1. Background and Purpose

### 1.1 The problem

A robot arm operating in a cluttered workspace must move its end-effector to a target location while avoiding the objects in its way. Classical motion planners (e.g. RRT-Connect) solve this reliably when given a perfect world model, but they are slow at run time, require explicit scene geometry, and produce a single plan per query. Learned policies, by contrast, are fast and reactive — but the standard way of training them, behavior cloning (BC), suffers from a well-known weakness: the policy only ever sees *success stories*. The moment it drifts slightly off the demonstrated path, it lands in a state it has never seen, has no idea how to recover, and small errors compound into collisions or dead-ends. This compounding-error problem grows worse as paths get longer and scenes get more cluttered, and naively fixing it by collecting more demonstrations requires a prohibitively large amount of data.

### 1.2 The idea

Our central hypothesis is that the missing ingredient is *failure data* — specifically, structured examples of how to recover from mistakes and how to reach the same goal by different routes. Three lines of prior work converge on this:

- **DAgger** (Ross, Gordon, Bagnell, 2011) lets the learner act, then asks an expert to label every state the learner actually visits, so the training distribution matches the deployment distribution. It reduces imitation learning to no-regret online learning and breaks the quadratic compounding-error bound of plain BC.
- **Diffusion Meets DAgger** (Zhang et al., 2024) shows that the out-of-distribution states DAgger needs can be *generated artificially* rather than visited physically, getting the same benefit at much lower cost.
- **RaC** (Hu et al., 2025) shows that when a rollout approaches failure, rewinding the robot to a safe, in-distribution state and then demonstrating a corrective route ("recover, then correct") teaches the policy to self-correct, and that keeping a balanced mixture of *recovery* and *correction* segments dramatically improves data efficiency. RaC additionally prescribes terminating the episode once an intervention concludes, so the data budget concentrates on the sub-tasks that actually need it.

All three tackle the same root cause: the robot only learns from successes. The fix is to deliberately add failures, recoveries, and alternative routes to the training data.

### 1.3 The research question

> **Can DAgger and RaC teach a robot arm to find many *different* routes to the same goal — so that it can navigate clutter it has never seen before?**

We measure success not only by goal-reaching but by *route diversity*: a policy that knows several distinct ways around an obstacle field should generalize better to unseen obstacle layouts than one that memorized a single route. Concretely we expect (a) DAgger + RaC to produce a policy that knows more distinct routes to the goal than BC alone, and (b) more recovery attempts per episode to correlate with a higher success rate in unseen scenes.

### 1.4 Pre-registered hypotheses and decision rules

Stated before any training run; the report will evaluate these as written, whatever the outcome.

- **H1 (recovery & correction help):** *BC + DAgger + RaC* exceeds *BC + DAgger* on `test_unseen` success rate, scene-paired bootstrap 95% CI on the difference excluding zero (§10.3). Refuted if the CI includes zero at the full data budget.
- **H2 (route diversity is the active ingredient, the novel claim):** *BC + DAgger + RaC* exceeds *RaC-NoReroute* (identical pipeline, distinct-homotopy rejection disabled — §10.1) on `test_unseen` success, same CI rule; and its `#routes`/route-entropy (§9.2) is higher on training-distribution scenes. If H1 holds but H2 fails, the honest conclusion is "recovery scaling transfers to this domain, route-forcing adds nothing" — that is a publishable course-project result, not a failure.
- **H3 (recovery behavior correlates with unseen-scene success):** within the main policy, per-scene Spearman r > 0 (§9.3), reported as observational evidence only.

### 1.5 Scope of this implementation

Everything runs in simulation. The "expert" is a classical sampling-based planner (RRT-Connect) used **for labeling only** — it is never deployed at test time. Because the expert is automated, the human-in-the-loop intervention of the original RaC paper is replaced by an *automated* near-failure detector + state-rewind mechanism (this is the key engineering adaptation of this project, detailed in §6). The pipeline has four progressive stages:

1. **Behavior Cloning** — copy the planner directly; the arm learns the basics.
2. **DAgger + RaC** — correct the arm's mistakes during rollouts (DAgger); on near-failure, rewind and reroute (RaC). The policy learns to recover.
3. **Synthetic Clutter Scaling** — once the policy is stronger, artificially add more obstacles to force generalization to denser scenes.
4. **Deep RL with a Path-Diversity Reward** — fine-tune with RL using a reward that explicitly bonuses trajectories that differ from previously taken routes (stretch goal; see §8).

---

## 2. System Overview

### 2.1 High-level dataflow

```
                ┌────────────────────────────────────────────────────┐
                │                  Scene Generator                   │
                │   (randomized box & sphere obstacles, curriculum)  │
                └───────────────┬────────────────────────────────────┘
                                │ MJ model + initial state
                                ▼
   ┌─────────────┐     ┌──────────────────┐      ┌───────────────────┐
   │ RRT-Connect │────▶│  MuJoCo Env      │◀────▶│  Policy π_θ (MLP) │
   │ expert      │     │  (Franka Panda)  │      │  joint-space ctrl │
   │ (label only)│     └────────┬─────────┘      └───────────────────┘
   └─────────────┘              │ transitions
                                ▼
              ┌─────────────────────────────────────┐
              │            Data Engine              │
              │  Stage 1: full expert demos (BC)    │
              │  Stage 2: DAgger relabels +         │
              │           RaC recovery/correction   │
              │  Stage 3: curriculum clutter demos  │
              └────────────────┬────────────────────┘
                               │ aggregated dataset D_0:k
                               ▼
              ┌─────────────────────────────────────┐
              │     Trainer (BC objective; later    │
              │     PPO fine-tune w/ diversity)     │
              └────────────────┬────────────────────┘
                               │ checkpoints
                               ▼
              ┌─────────────────────────────────────┐
              │  Evaluator: success rate, route     │
              │  diversity, recoveries per episode  │
              └─────────────────────────────────────┘
```

### 2.2 Repository layout

```
robot-routes/
├── README.md
├── pyproject.toml                  # uv/pip installable; pinned deps
├── configs/
│   ├── env/panda_reach.yaml        # scene, obstacle, success params
│   ├── expert/rrt_connect.yaml
│   ├── train/bc.yaml
│   ├── train/dagger_rac.yaml
│   ├── train/curriculum.yaml
│   ├── train/rl_diversity.yaml     # stage 4 (stretch)
│   └── eval/default.yaml
├── src/robot_routes/
│   ├── envs/
│   │   ├── panda_reach_env.py      # gymnasium.Env wrapper around MuJoCo
│   │   ├── scene_gen.py            # randomized obstacle scenes + curriculum
│   │   └── assets/                 # MJCF: panda + table + obstacle templates
│   ├── expert/
│   │   ├── collision.py            # MuJoCo-based collision/clearance checker
│   │   ├── rrt_connect.py          # joint-space RRT-Connect + shortcutting
│   │   └── oracle.py               # Expert interface: plan(state, scene) -> traj
│   ├── data/
│   │   ├── schema.py               # transition/segment dataclasses, HDF5 I/O
│   │   ├── collect_demos.py        # stage 1 collection script
│   │   └── buffer.py               # aggregated dataset D_0:k
│   ├── agents/
│   │   ├── policy.py               # MLP gaussian policy, action chunking
│   │   ├── bc_trainer.py
│   │   ├── dagger_rac.py           # stage 2 outer loop (core of project)
│   │   └── ppo_diversity.py        # stage 4 (stretch)
│   ├── diversity/
│   │   └── route_metrics.py        # trajectory distance, clustering, #routes
│   ├── eval/
│   │   ├── evaluate.py             # batch evaluation across seeds/scenes
│   │   └── plots.py
│   └── utils/
│       ├── seeding.py
│       ├── logging.py              # wandb/tensorboard wrapper
│       ├── gpu_alloc.py            # NVML-based GPU discovery, scoring, leasing (§10.5)
│       └── mj_state.py             # save/restore full MuJoCo physics state
├── scripts/
│   ├── 01_collect_bc_demos.py
│   ├── 02_train_bc.py
│   ├── 03_run_dagger_rac.py
│   ├── 04_run_curriculum.py
│   ├── 05_train_rl_diversity.py
│   ├── 06_evaluate.py
│   └── 07_launch_grid.py           # multi-GPU grid scheduler (§10.5)
└── tests/
    ├── test_env.py
    ├── test_expert.py
    ├── test_rewind.py
    ├── test_diversity_metrics.py
    └── test_gpu_alloc.py
```

### 2.3 Dependencies (pinned in `pyproject.toml`)

| Package | Purpose | Notes |
|---|---|---|
| `mujoco >= 3.2` | Physics simulation | CPU is sufficient for this scale; MJX/GPU-parallel rollouts optional later |
| `gymnasium >= 0.29` | Env API | |
| `torch >= 2.3` | Policy training | CUDA if available, otherwise CPU |
| `numpy`, `scipy` | Math, KD-trees for RRT | |
| `h5py` | Dataset storage | |
| `pyyaml` | Configs (plain YAML + dataclass parsing; no Hydra to keep it simple) | |
| `wandb` (optional) / `tensorboard` | Logging | |
| `matplotlib` | Plots | |
| `mujoco_menagerie` (vendored asset) | `franka_emika_panda` MJCF | Copy `panda.xml` + meshes into `src/robot_routes/envs/assets/` |
| `pynvml >= 11` | GPU discovery: utilization + memory queries for the allocator (§10.5) | degrade gracefully to CPU if NVML/CUDA absent |
| `pytest` | Tests | |

**Implementation language:** Python 3.11. **Style:** ruff + black defaults, type hints everywhere, dataclasses for configs.

---

## 3. Environment Specification (`envs/`)

### 3.1 Robot and scene

- **Robot:** 7-DoF Franka Emika Panda (MJCF from MuJoCo Menagerie), mounted at the origin on a table plane (`z = 0`). The gripper is fixed (closed); this is a *reaching* task, not grasping. End-effector site: `attachment_site` on link 7.
- **Obstacles:** Axis-aligned **boxes** and **spheres**, static within an episode. Obstacle count `n_obs` is curriculum-controlled (§7), spawned in the reachable workspace annulus `r ∈ [0.25, 0.75] m`, `z ∈ [0.05, 0.7] m` in front of the robot (`x > 0.1`).
  - Box half-extents sampled `U(0.03, 0.10) m` per axis; sphere radii `U(0.04, 0.10) m`.
  - Obstacles may float (no support constraint) — a deliberate abstraction: the project studies route diversity around volumetric occlusions, not physically plausible furniture, and floating spheres create the over/under homotopy choices that table-bound clutter cannot.
  - Rejection-sample obstacle poses so that: (a) no obstacle penetrates the robot's start configuration, (b) no obstacle contains the goal point, (c) obstacles do not overlap each other by more than 20% volume heuristic (simple sphere-bound check is acceptable).
- **Goal:** a 3-D point sampled in the same workspace region, at least `0.35 m` (straight-line) from the start end-effector position, and verified reachable: the scene is accepted **only if RRT-Connect finds a collision-free plan within its time budget, under the full §4.1 planning margin** — so "solvable" already means solvable-with-the-clearance-the-expert-demands, and a goal 1 cm from an obstacle face correctly rejects rather than producing an unplannable frozen scene (record planner failures as scene-generation rejections, not task failures).
- **Start configuration:** fixed neutral pose `q_home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]` with small uniform noise `U(-0.05, 0.05)` rad per joint (collision-checked, resampled on penetration).

### 3.2 MDP definition

| Item | Definition |
|---|---|
| **Observation** (`float32`) | `[q (7), q̇ (7), ee_pos (3), goal_pos (3), goal − ee (3), obstacle_encoding (8 × 7 = 56)]` → **79-D**. Obstacle encoding: fixed-size slot array for up to `MAX_OBS = 8` obstacles; each slot = `[type ∈ {0 box, 1 sphere}, center (3), size (3)]` (sphere: size = `[r, 0, 0]`); unused slots zero-filled. **Slot order is computed once at `reset`** (sorted by distance to the start→goal line segment) **and frozen for the episode** — re-sorting per step would swap slot contents mid-episode and inject step discontinuities into the observation. Obstacles are static within an episode, so the frozen order is also always current. |
| **Action** (`float32`) | Δ joint position command, 7-D, clipped to `[-0.05, +0.05]` rad per control step. Applied as `q_target = clip(q + a, q_limits ± 0.02 rad margin)`, tracked by MuJoCo position actuators (kp from Menagerie defaults) — without the clamp, repeated same-sign actions push targets beyond joint range and the actuator grinds against the limit in poorly-defined states. |
| **Control rate** | 20 Hz control (`n_substeps = 25` at `dt = 0.002` physics). |
| **Horizon** | `H = 300` control steps (15 s). |
| **Success** | `‖ee_pos − goal_pos‖ < 0.05 m` sustained for 5 consecutive control steps (anti-flythrough), reached before termination. Since any collision terminates the episode immediately (below), success implies a collision-free trajectory by construction. |
| **Failure / termination** | (a) any contact between robot bodies and obstacle/table geoms (`terminated = True`, `info["collision"] = True`), **subject to an explicit contact-exclusion list:** base-link (`link0`) ↔ table (the base is mounted on it — without this exclusion every episode terminates at t = 0), and the adjacent-link pairs already excluded by the Menagerie model's `contype/conaffinity` settings. The exclusion list is defined once in `panda_reach.yaml` and consumed by both the env and `expert/collision.py` so planner and env can never disagree about what counts as a collision; (b) timeout at `H` (`truncated = True`). All non-excluded self-collisions count as collisions. |
| **Reward** (used in stages 1–3 only for logging; BC/DAgger do not optimize reward) | `r_t = −‖ee−goal‖ · 0.1 + 10·1[success] − 25·1[collision] − 0.001·‖a‖²`. Stage 4 extends this (§8). |

### 3.3 Env API

```python
class PandaReachEnv(gymnasium.Env):
    """obs: Box(79,), act: Box(7,) in [-0.05, 0.05]."""
    def __init__(self, cfg: EnvConfig, render_mode: str | None = None): ...
    def reset(self, *, seed=None, options=None) -> tuple[obs, info]:
        # options={"scene": SceneSpec} forces a specific scene (needed for
        # paired evaluation and for re-creating scenes during DAgger/RaC).
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]:
        # info: {"collision": bool, "min_clearance": float, "ee_pos": (3,),
        #        "q": (7,), "success": bool}
    # --- extensions beyond gym API ---
    def get_state(self) -> MjSimState        # EXACTLY: qpos, qvel, act, ctrl, time,
                                             # qacc_warmstart, mocap_{pos,quat} — omitting
                                             # qacc_warmstart makes post-restore stepping
                                             # diverge subtly (warm-started solver), which
                                             # the round-trip determinism test would chase
                                             # for days as a "heisenbug"
    def set_state(self, s: MjSimState) -> obs # exact restore — cornerstone of RaC rewind
    def min_clearance(self) -> float          # min signed distance robot↔obstacles
    @property
    def scene(self) -> SceneSpec              # serializable scene description
```

`SceneSpec` is a frozen dataclass: `{obstacles: list[Obstacle], goal: vec3, q_start: vec7, seed: int}` — JSON-serializable so scenes are perfectly reproducible across processes.

**Clearance computation:** use `mujoco.mj_geomDistance` (MuJoCo ≥ 3.2) over robot-geom × {obstacle **and table**} geom pairs (minus the §3.2 contact-exclusion list) — the table is a terminal collision surface, so excluding it from clearance would deny RaC any chance to intervene before table strikes; `min_clearance = min(...)`. This is the per-step hot loop, so bound it: (a) cache geom-id lists at construction; (b) keep the robot at ≤ 10 collision geoms (the Menagerie Panda already uses capsule/primitive collision geoms); (c) broad-phase prune pairs whose bounding-sphere distance exceeds 0.25 m and pass `distmax = 0.25` so narrow-phase exits early; (d) budget ≤ 0.5 ms per control step on desktop CPU, asserted as a soft perf test in `test_env.py`. This powers both the RaC near-failure trigger and evaluation stats.

**Acceptance criteria (tests/test_env.py):**
- `reset` with the same `SceneSpec` twice produces bit-identical observations.
- `set_state(get_state())` round-trips: subsequent `step` sequences are deterministic and identical.
- A scripted straight-line motion into a known obstacle sets `info["collision"]` within the expected step window.
- 1,000 random resets produce zero scenes violating the spawn constraints of §3.1.

---

## 4. Expert Oracle — RRT-Connect (`expert/`)

The expert is used **for labeling only** (stage 1 demos, DAgger relabels, RaC correction segments). It is never run at test time.

### 4.1 Planner

- **Algorithm:** bidirectional RRT-Connect in 7-D joint space, custom Python implementation (no OMPL dependency — keeps install trivial and the code inspectable).
- **Collision checker:** `expert/collision.py` wraps a dedicated `MjModel`/`MjData` pair (separate from the env's) — set `qpos`, call `mj_forward` (not `mj_step`), read contacts. Edge checking: discretize each tree edge so that **max single-joint |Δq| ≤ 0.03 rad between consecutive checks** (the conservative norm; L2-based spacing under-checks distal joints). Use an inflated collision margin of **`margin_plan = 0.03 m`** during planning (via `mj_geomDistance < margin` rather than contact booleans). **Coupling constraint (load-bearing): `margin_plan > ε_danger (0.02 m)`.** If the expert may legally skim obstacles below the near-failure threshold, the policy faithfully imitates those tight passages and (a) fires spurious RaC interventions throughout training, polluting recovery data and wasting budget, and (b) inflates the deployment "recovery attempt" statistic (§9.3) that H3 depends on — normal tight passages would be miscounted as recoveries. With the constraint, dipping under `ε_danger` genuinely means *deviation from expert-like behavior*. This and the other parameter couplings are asserted by the orchestrator's config validation (§11.7.3).
- **Parameters (configs/expert/rrt_connect.yaml):**
  - `step_size: 0.15` rad (extend step, scaled per-joint by range)
  - `goal_bias: 0.10`
  - `max_iters: 20_000`; **two time budgets:** `time_budget_validate_s: 10.0` (one-off scene-solvability checks during scene generation — generous, failure just means resample) and `time_budget_label_s: 3.0` (all in-loop DAgger/RaC calls — tight, because this multiplies across ~40k transitions/round; plan caching in §6.2 keeps the call count to ~1–3 per episode)
  - Goal set: any configuration with `‖fk(q) − goal‖ < 0.04 m`; obtain goal configurations by damped-least-squares IK (MuJoCo Jacobian + `scipy`), **joint-limit aware**: clamp iterates to limits minus a 0.05 rad margin and reject converged solutions on the margin boundary. Restarts are **stratified, not uniform**: 16 initial seeds = 8 with elbow-up joint-4 sign × 8 elbow-down, each ± uniform noise, so both elbow families are represented; deduplicate solutions closer than 0.2 rad (L2) and keep all collision-free survivors as goal-tree roots. **Multi-goal seeding matters: distinct IK families are the seed of route diversity — if all roots collapse to one elbow family, RaC rerouting (§6.3) loses its main source of alternative homotopies.** Log the number of distinct goal roots per plan.
- **Post-processing:** (1) shortcut smoothing — 100 iterations of random two-point shortcuts with edge collision checks; (2) time-parameterize: resample the path to waypoints with ≤ 0.04 rad joint displacement so the tracking controller below stays within the env's per-step action bound.
- **Stochasticity:** the planner takes an explicit `rng` seed. Calling `plan` with different seeds on the same scene yields *different homotopy* routes — this is exploited by RaC rerouting (§6) and by demo diversification (§5).

### 4.2 Expert action interface

The policy acts in Δ-joint space, so expert *labels* must be actions, not paths:

```python
class ExpertOracle:
    def plan(self, q_start: vec7, scene: SceneSpec, rng: int) -> JointPath | None
    def label(self, q_current: vec7, path: JointPath, tracker: PathTracker) -> vec7:
        """Pure-pursuit on the joint path with a MONOTONIC progress index:
        tracker.idx only moves forward, and the nearest-point search (joint-space
        L2) is restricted to the window [idx, idx + 10] so the match can never
        jump backwards when the path passes near itself. Look ahead L=3 waypoints
        from the matched index; return a = clip(q_lookahead − q_current, ±0.05).
        One PathTracker per (episode, plan); reset on every replan.
        TERMINAL HOLD: once tracker.idx reaches the final waypoint, return the
        zero action (hold) — lookahead past path end is otherwise undefined."""
```

During demo collection the expert *executes* via the same `label` loop closed around the env, so demo actions are exactly the labels the policy will regress onto (no train/label mismatch).

**Acceptance criteria (tests/test_expert.py):**
- On 50 random 4-obstacle scenes, plan success rate ≥ 95% within `time_budget_validate_s`, and ≥ 90% within `time_budget_label_s`; every returned path is collision-free under dense re-checking at half the planning discretization.
- Executing `label` closed-loop in the env reaches the goal with zero collisions on ≥ 95% of planned scenes.
- **Constructed-scene diversity test:** on a fixed family of 20 scenes each containing a single large box centered on the start→goal chord (guaranteeing left/right homotopy classes exist), 10 planner seeds per scene produce ≥ 2 routes with pairwise §9.1 distance > `δ_distinct` in ≥ 80% of scenes. (Random scenes are too uneven to anchor a pass/fail bar; constructed ones make the property testable.)

---

## 5. Stage 1 — Behavior Cloning

### 5.1 Demo collection (`scripts/01_collect_bc_demos.py`)

- `N_demo = 2,000` successful expert episodes at curriculum level 0 (`n_obs ∈ {2, 3}`), each with a fresh random scene and a fresh planner seed. Per-scene, sample the planner seed randomly so the dataset already contains some route multimodality.
- **Stopping is a behavior and must be in the data:** every successful demo (and every successful RaC correction in §6.3) records **5 additional zero-action hold steps at the goal** before episode end — §3.2's success criterion requires the tolerance to be *sustained* 5 steps, and a dataset that ends at arrival teaches arrival, not stopping. Without this, the policy orbits the goal and fails its own success test.
- Store per-step records (HDF5, schema in `data/schema.py`): `obs (79,), action (7,), q (7,), ee_pos (3,), done, segment_type = "full_demo", episode_id, scene_json`. **Concurrency rule: one HDF5 shard per worker process** (`demos_w{i}.h5` — HDF5 has no safe concurrent-write mode), merged by a single-process `data/merge_shards.py` that re-indexes `episode_id` globally and verifies row counts. The same shard-then-merge rule applies to every collection stage (§6, §7).
- Expected size: ~2,000 × ~120 steps ≈ 240k transitions (~90 MB).

### 5.2 Policy architecture (`agents/policy.py`)

- Trunk MLP: `obs(79) → 512 → 512 → 256`, LayerNorm + GELU.
- **Head: Mixture Density Network (MDN), `K_mix = 5` Gaussian components** — per component: mixture logit, 7-D mean, 7-D diagonal log-std (clamped to [−4, 0]). **Likelihood semantics (pinned):** the MDN is trained as an *unbounded* density in raw action space — exact mixture NLL against the expert label, no tanh in the likelihood. Expert labels are clipped to ±0.05 by construction, so they live inside the support; clipping is applied only at *execution* time on sampled/мean actions. (A tanh-squashed likelihood plus post-hoc clipping is not a coherent density; keep the two concerns separate.) **Rationale (this is load-bearing):** the project's central claim is multi-route behavior, and a *unimodal* Gaussian trained with NLL mode-averages — given left-route and right-route data around an obstacle it regresses toward their mean, i.e. into the obstacle. Multimodal routes require a multimodal head. (IMLE Policy, Rana et al. 2025, is a noted drop-in alternative if MDN training proves finicky; diffusion heads are out of scope for compute reasons.)
- Loss: exact mixture NLL (logsumexp over components). Standard MDN stabilizers: log-std clamp as above, mixture-entropy bonus `1e-3` for the first 20 epochs to delay component collapse, component-wise gradient clipping.
- **Sampling semantics:** *stochastic* mode (data collection §6, route-diversity eval §9.2) = sample component from logits, then sample the Gaussian. *Deterministic* mode (success-rate eval) = mean of the highest-weight component. **Per-episode route commitment — done honestly:** a fixed per-episode Gumbel vector does *not* work here, because MDN components are permutation-symmetric across states (component k at state A has no learned correspondence to component k at state B), so "keep picking component k" is meaningless. Instead use **action-continuity selection**: at each step, sample a component from the logits *but* multiply each component's selection probability by a continuity kernel `exp(−‖μ_i − a_prev‖² / 2σ_c²)` with `σ_c = 0.02` (first step: plain logits). This biases the rollout to stay on the mode it is already executing while still allowing genuine branch points (where modes converge, the kernel is uninformative). **Scope rule — the kernel is an inference-time wrapper, never part of the trained density:** it is active for stage 1–3 data collection and for all evaluation rollouts (identically across every condition, for fairness), but it is **disabled during PPO data collection** — see §8, where sampling must match the density used in the importance ratio. **Dithering diagnostic:** per episode, count switches of the nearest-mean component index; log its distribution every retrain — route commitment is thereby *measured*, not assumed. If median switches/episode stays > 10 after stage 2, gate G-DITHER (§11.7.2) auto-branches: retrain the IMLE-head fallback on the identical aggregated dataset, evaluate both heads on validation, and carry the winner forward — no human escalation step.
- **Ablation head (config `head: gaussian`):** the original unimodal Gaussian, retained deliberately — comparing it against the MDN on `#routes` (§9.2) turns the mode-averaging pathology into a *reported result* instead of a silent failure.
- Optional **action chunking** flag (predict next `k=4` actions, execute all before re-querying) — behind a config flag, default **off** for stages 1–3 (single-step actions keep DAgger labeling simple), available for ablation.

### 5.3 Training (`agents/bc_trainer.py`, `configs/train/bc.yaml`)

- Loss: negative log-likelihood of expert action (equivalently MSE + std regularization).
- AdamW, lr `3e-4`, cosine decay, batch 1024, 200 epochs, grad-clip 1.0. Normalize observations with dataset running mean/std (persist stats in the checkpoint).
- Validation: hold out 5% of episodes; early-stop on val NLL.
- **Exit criterion for stage 1 (soft gate):** target BC success ≥ 40% on 200 held-out level-0 scenes; **proceed to WP5 regardless once BC ≥ 25%** — downstream stages exist precisely to repair BC's weaknesses, and the plan must not deadlock on an arbitrary absolute. If BC < 25%, the orchestrator auto-runs the diagnostic ladder as code (gate **G-BC**, §11.7.2): (i) expert closed-loop replay still succeeds ≥ 95% (isolates data bugs); (ii) the policy can overfit 10 episodes to near-zero NLL (isolates training bugs); (iii) an automated 2×2 sweep over lr ∈ {1e-4, 3e-4} × chunking on/off, retraining with the best cell. Whatever BC achieves is the baseline row of the results table, not a quality bar.

---

## 6. Stage 2 — DAgger + RaC (core contribution)

This stage merges DAgger's "label the states the learner visits" with RaC's "recover, then correct, then terminate," with the human teleoperator replaced by the automated planner + simulator state rewind.

### 6.1 Outer loop (`agents/dagger_rac.py`)

```
π_0 ← BC checkpoint;  D ← D_BC
for k in 1..K (K = 6 rounds):
    Δd_k ← collect_round(π_{k-1}, budget B = 40,000 transitions)
    D ← D ∪ Δd_k
    π_k ← retrain from scratch on D (same recipe as §5.3, 100 epochs)
    evaluate π_k on the VALIDATION sets only (§10.2 tier 2); log success, recoveries/ep, route count
```

Retraining from scratch each round (true DAgger aggregation) avoids loss-of-plasticity questions; with ~500k transitions max this is cheap. **Normalization semantics (pinned to avoid train/eval skew):** observation mean/std are computed *unweighted* over the raw aggregated dataset at the start of each retrain (the sampling weights of §6.2 shape the loss, not the normalizer), and the stats are serialized *inside* the checkpoint; evaluators must never recompute stats. **Budget accounting (automated-RaC restatement of RaC Alg. 1):** every simulated control step of a collection episode — policy steps, executed recovery steps, and expert-executed correction steps alike — counts against the round budget `B`, while *storage* keeps all of them too (unlike human-RaC there is no labor asymmetry to economize; the budget exists so rounds are comparable in environment interaction, the quantity our scaling plots use on the x-axis).

### 6.2 `collect_round` — per-episode logic

```
env.reset(random scene at current curriculum level)
ring_buffer ← deque(maxlen = R_max = 40) of (MjSimState, step_idx)   # ~2 s of history
while episode active:
    snapshot env.get_state() into ring_buffer every step
    a ← π(obs)                                  # policy drives (HG-DAgger-style gating)
    obs', info ← env.step(a)
    # --- DAgger relabel (always, every visited state) ---
    a* ← expert.label(q, current_plan, tracker) # drift > 0.25 rad → WARM-START replan
    record (obs, a*, segment_type="dagger_label")   # note: expert action, learner state
    # --- RaC trigger: NEAR-failure, before actual failure ---
    if info.min_clearance < ε_danger (=0.02 m) or collision or
       (no EE progress > 0.01 m over last 60 steps):       # stuck detector
        run_rac_intervention(); break           # Rule 2: terminate episode after intervention
budget b += episode length                      # charge full episode (RaC Alg. 1)
```

Notes:
- The learner's *own* action `a` is executed (on-policy states), but the recorded label is the expert's `a*` — that is exactly DAgger. **Storage dedup:** every visited step is recorded exactly once; if the episode finishes with no intervention its already-recorded steps are simply re-tagged `segment_type="clean_rollout"` (no second copy is appended — the original draft double-counted clean episodes).
- **Per-segment-type sampling weights** (training-time weighted sampler in `data/buffer.py`, configurable in `dagger_rac.yaml`): `full_demo 1.0`, `dagger_label 1.0`, `clean_rollout 0.5`, `recovery 1.5`, `correction 1.5`. Rationale: recovery/correction frames are the scarce, high-value skill data RaC scales (they are a minority of total frames); clean rollouts are near-duplicates of demo behavior. Log the *effective* per-type epoch composition every retrain.
- Plan caching: the expert plans once at episode start and replans only on drift or intervention; keeps wall-clock manageable.
- **Homotopy-preserving drift replans:** a drift-triggered replan must not flip the label route mid-episode (whipsawing DAgger labels between homotopies at adjacent states). Warm-start it: seed the start tree with the remaining waypoints of the current plan and reuse the same RNG seed; only RaC corrections (§6.3.3) deliberately use fresh seeds. Log Fréchet distance between old-suffix and replan as a flip detector.
- **Planner-failure fallbacks (`plan → None` within `time_budget_label_s`):** at episode start — discard the scene and resample (counts as scene rejection, not budget). On a drift replan — keep tracking the stale plan and retry replanning every 20 steps. On a RaC correction — retry once with doubled budget; if still `None`, store the recovery segment alone and terminate the episode (an intervention with recovery-only data is valid, just logged as `correction_failed`).

### 6.3 `run_rac_intervention` — automated recover-then-correct

Implements RaC Rule 1 (every intervention = one recovery segment + one correction segment) and Rule 2 (terminate afterward):

1. **Rewind target selection:** scan the ring buffer backwards; pick the most recent snapshot whose `min_clearance ≥ ε_safe (= 0.10 m)` and which lies within `0.4 rad` (joint L2) of the demo-data manifold (cheap proxy: clearance test only, manifold test optional flag). If none qualifies, use the oldest snapshot.
2. **Recovery segment (the data, not a teleport):** the *teleport* `env.set_state(rewind_state)` would give us no recovery data — instead, compute the reversed action sequence `a_rec,t = clip(q_{t-1} − q_t, ±0.05)` from the learner's recorded joint waypoints (trigger step back to rewind step, dropping the last 3 pre-trigger steps). **Reversal start state:** for clearance/stuck triggers, the trigger-time state; for *actual-collision* triggers, first `set_state` to the most recent **pre-contact** snapshot and run 5 zero-action settle steps (starting a reversal in contact with forward momentum is ill-posed). Then **execute the reversed actions live in the simulator**, recording the genuine observations (including velocities) that result. Never construct recovery observations by replaying forward snapshots in reverse order — reversed snapshots carry forward-motion velocity signs, which would train the policy on observation–action pairs that cannot co-occur at deployment. The executed reversal traverses a just-cleared corridor so collisions should be rare; if one occurs, discard the recovery segment and keep only the correction. Record as `segment_type="recovery"`; the *endpoint of the executed reversal* (not the stored snapshot) becomes the rewind state for step 3. This mirrors RaC's "rewind to a prior familiar state."
3. **Correction segment (try a different route):** from the rewind configuration (endpoint of the executed reversal), call the expert with a **fresh random seed**, and — to actively force a *different homotopy* — reject plans too similar to the route that just failed. **Well-posed comparison:** compare like with like — the EE path of the *remaining suffix of the previous plan* (from its tracker index at the rewind configuration, to goal) against the EE path of the *candidate new plan* (rewind configuration to goal); both share endpoints, so §9.1 Fréchet is a meaningful homotopy proxy. Reject candidates with distance below **`δ_reroute = δ_distinct`** (the calibrated value of §9.1 — the original draft's 0.5 m was a units bug ~3× above the distinctness scale, which would have shunted everything through the fallback). Up to 5 reseeded attempts; accept the most distant if all fail and log the fallback rate. Execute the accepted plan closed-loop with `expert.label`; record as `segment_type="correction"` until success/collision/timeout.
4. **Terminate the episode** (RaC Rule 2). One intervention max per episode.

The aggregated round data therefore contains a controlled mixture of `dagger_label`, `recovery`, and `correction` segments. Log the recovery:correction frame ratio per round — RaC reports a healthy band of roughly 1:1 to 1:2, and we should watch ours stays balanced rather than collapsing to corrections.

### 6.4 Hyperparameters (`configs/train/dagger_rac.yaml`)

| Param | Value | Rationale |
|---|---|---|
| Rounds `K` | 6 | proposal timeline fits 6 retrains |
| Budget/round `B` | 40,000 transitions | ≈ 300–400 episodes/round |
| `ε_danger` | 0.02 m | near-failure clearance trigger |
| `ε_safe` | 0.10 m | rewind state must be comfortably clear |
| Ring buffer `R_max` | 40 steps | 2 s of rewind history |
| `δ_reroute` | = calibrated `δ_distinct` (§9.1) | force distinct-homotopy corrections; suffix-vs-candidate comparison (§6.3.3) |
| Stuck window | 60 steps / 0.01 m | secondary trigger |
| Replan drift | 0.25 rad joint L2 **to the tracker-matched waypoint** | DAgger label freshness (warm-started, §6.2). Defining drift to the *matched waypoint* (not the nearest path point globally) also self-heals a stuck-behind monotonic tracker: if the policy detours and rejoins the path ahead of the window, distance to the stale match grows past threshold → replan → fresh tracker |
| Settle steps | 5 zero-action | pre-reversal on collision triggers (§6.3.2); also goal-hold recording (§5.1) |
| Continuity kernel `σ_c` | 0.02 | MDN route-commitment selection (§5.2) |

**Acceptance criteria (tests/test_rewind.py):**
- Triggering an intervention on a scripted near-collision yields exactly one recovery + one correction segment and a terminated episode.
- All recorded recovery transitions are collision-free; `set_state` restore reproduces the rewind observation exactly.
- After round k ≥ 2, success rate on the fixed eval set strictly exceeds the BC baseline (sanity gate, not a unit test).

---

## 7. Stage 3 — Synthetic Clutter Scaling (curriculum)

Once the policy is stronger, artificially harder scenes force generalization to denser clutter.

- **Levels:** `L0: n_obs ∈ {2,3}` → `L1: {3,4}` → `L2: {4,6}` → `L3: {6,8}`. Obstacle sizes unchanged; scene-validity rules of §3.1 still apply (every scene remains planner-solvable).
- **Promotion rule:** continue running the §6 DAgger+RaC loop, but bump the curriculum level whenever the latest π_k achieves ≥ 70% success on `val_L{current}` (§10.2 tier 2 — probes are the frozen validation sets, not freshly sampled scenes, so promotion decisions are comparable across conditions and seeds). Demote (one level) if success drops below 30% after a promotion (hysteresis prevents thrashing).
- **Dataset balance:** when training at level `Lj`, sample minibatches with at most 50% of frames drawn from levels **other than the current level j** (simple per-level weighted sampler in `data/buffer.py`) — phrased as "other than", not "< j", so a demotion does not invert the cap's meaning — keeping off-level data from drowning current-level data in either direction.
- **Synthetic densification (DMD-inspired, optional flag `synthetic_obs: true`):** for 20% of replayed *clean* demos, inject 1–2 extra obstacles into the scene encoding *only where they do not intersect the recorded trajectory's swept volume* (sphere-bound check on recorded `q` waypoints). The recorded actions remain valid labels, and the policy learns that its route still works amid additional clutter it must merely ignore — a cheap, fully offline analogue of Diffusion-Meets-DAgger's "generate OOD inputs artificially, label them with what you already know."
- **Deliverable:** curriculum run script `scripts/04_run_curriculum.py` = §6 loop + promotion controller; outputs the final stage-3 checkpoint `pi_curriculum.pt`.

---

## 8. Stage 4 — Deep RL with Path-Diversity Reward (stretch goal)

Listed last in the proposal pipeline. **Go/no-go is automated gate G-PPO (§11.7.2), evaluated on the validation tier** — the original draft keyed it to the test-set CI, which contradicted the §10.2 firewall (a go/no-go is a decision; decisions never read test data). GO requires: (a) val_unseen paired CI for (full − RaC-NoReroute) excluding zero, (b) §9.2 validity-gate pass ≥ 60% on val scenes, (c) before the `ppo_deadline` in `configs/pipeline.yaml`. NO-GO is recorded in pipeline state and this section moves to future work verbatim; either way no human decides.

- **Algorithm:** PPO (clip 0.2, GAE λ=0.95, γ=0.99, lr 1e-4, 8 parallel envs), policy initialized from the stage-3 checkpoint; KL-to-anchor regularizer `0.5 · KL(π‖π_stage3)` to prevent catastrophic forgetting of recovery behaviors. **MDN-specific estimators (no closed forms exist for mixtures):** first, a correctness precondition — **PPO rollouts sample from the plain mixture, with the §5.2 continuity kernel disabled.** The kernel conditions component selection on `a_prev`, making the behavior distribution history-dependent and ≠ π(a|s); computing the importance ratio from the plain mixture log-prob while *acting* through the kernel would bias every gradient silently (no error, no NaN — just wrong learning). The kernel returns at evaluation time, identically for all conditions; the resulting train/deploy sampling mismatch is acknowledged and is the lesser evil. With that pinned: log-prob of an executed action is exact (logsumexp over components — use it for the PPO ratio); the entropy bonus uses a single-sample Monte-Carlo estimate `−log π(ã|s)`, `ã ∼ π(·|s)`; the KL anchor uses the same sampled form `log π(ã|s) − log π_stage3(ã|s)` averaged over the batch. These estimators are standard but noisy — halve the entropy coefficient relative to Gaussian-PPO defaults (0.005) and monitor for collapse via the effective-component perplexity already tracked in §12.
- **Training scenes: a finite, frozen pool.** Stage 4 trains on `pool_rl.json` — 256 planner-verified scenes at the current top curriculum level, sampled from the training seed range and cycled uniformly across the 8 envs. This makes "per-scene" exact (each scene accrues ~hundreds of episodes) instead of the unimplementable "coarse layout features" bucketing of the draft; generalization is still measured on the untouched test tiers (§10.2), so the finite pool costs nothing scientifically.
- **Diversity bonus, precisely:** per scene, keep a FIFO archive `A_scene` of the last `M = 16` *successful* EE trajectories. On success: `r_div = β · clip(min_{τ' ∈ A_scene} d_route(τ, τ'), 0, 0.5)` with β = 2.0 and `d_route` from §9.1 (empty archive ⇒ `r_div` = the cap), added at the terminal step, then insert τ. **Variance handling:** the terminal bonus flows through GAE like any terminal reward, but (a) the clip caps its scale at the same order as the success bonus, and (b) use per-batch advantage normalization (standard) plus a value-function head that receives the scene index as an embedding (16-D learned, from the pool index) so the critic can absorb per-scene reward offsets. `r_div` mean/std are monitored per epoch by an in-training callback (gate G-β, §11.7.2) that halves β automatically when std exceeds 2× the success bonus (≥ 3 halvings → freeze at the floor).
- **Expected outcome to test:** diversity-rewarded fine-tuning increases the distinct-route count (§9.2) on *training-distribution* scenes and — the thesis — improves zero-shot success on *unseen* L3+ scenes relative to the stage-3 policy.

---

## 9. Route-Diversity Metrics (`diversity/route_metrics.py`)

The headline metric of the project — defined precisely so all stages share one implementation.

### 9.1 Trajectory distance `d_route(τ_a, τ_b)`

1. Take end-effector position sequences, resample each by arc length to `P = 64` points.
2. `d_route` = discrete Fréchet distance between the resampled polylines (dynamic-programming implementation, O(P²); also expose DTW as an alternative behind a flag — report Fréchet in the paper, DTW in an appendix ablation).
3. Distances are in meters. **Threshold calibration (replaces eyeballing):** generate 200 planner trajectory pairs on the constructed blocking-obstacle scenes of §4.2 — 100 same-homotopy pairs (same seed family, different smoothing noise) and 100 cross-homotopy pairs (left vs right of the blocker) — and set `δ_distinct` to the midpoint of the two distance distributions (expected ≈ 0.15–0.20 m). This runs as the automated `calibrate_delta` pipeline stage (§11.7.3): it asserts distribution separation (gate G-CAL), writes the value into the run's resolved config itself, and archives the distributions — no hand-edited config value exists. Report all `#routes` results at the calibrated threshold **plus a sensitivity sweep at ±33%** in the appendix. Caveats to state in the report: EE-space Fréchet treats elbow-up/elbow-down with identical EE paths as one route (acceptable — task-space diversity is what generalization needs), and metric distance is a proxy for homotopy, not a homotopy test.

### 9.2 Distinct-route count for a (policy, scene) pair

- Roll out the policy in *stochastic* mode (§5.2: sampled mixture component with per-episode Gumbel coherence, temperature 1.0) `n = 20` times on the identical scene (fixed `SceneSpec`; **env/sampling seeds fixed to the list {0..19} per scene, identical across all conditions and checkpoints** — diversity comparisons thereby pair on (scene, rollout-seed), and a re-run reproduces the exact rollout set).
- Keep successful trajectories; **validity gate: if fewer than 8 of 20 rollouts succeed, `#routes` is NA for that (policy, scene)** — a route count over 2 successes is noise. Cluster the successes by complete-linkage agglomerative clustering with cutoff `δ_distinct`.
- **`#routes` = number of clusters.** Because policies with different success rates see different sample sizes, also report **routes-per-success = #clusters / #successes** and **route entropy** (Shannon entropy of cluster occupancy, in nats). Report mean ± std over the eval scene set together with the *fraction of scenes passing the validity gate* (itself an informative robustness number). **Cross-condition comparisons of route diversity are computed on the *intersection* of scenes passing the gate for all compared conditions** — otherwise a weak policy is scored only on its easy scenes while a strong one is scored everywhere, biasing the comparison in an unknown direction. Report the intersection size alongside.

### 9.3 Recovery statistics

- A *recovery attempt* at deployment = event where `min_clearance < ε_danger` at step t and the policy returns to `min_clearance > ε_safe` within 40 steps without collision (events separated by < 10 steps merge into one). Count per episode.
- **Unit of analysis for the proposal's correlation hypothesis:** within a single policy, per *scene* on `test_unseen` — x = mean recoveries/episode over that scene's rollouts, y = that scene's success rate — reported as Spearman r (monotonic, rank-robust) with a scene-difficulty stratification (bin scenes by their planner path length) to partially control for "hard scenes cause both more recoveries and more failures." Also report the cruder across-policy version (one point per condition). **Label both correlational** — the causal version of the claim is tested by the RaC-NoReroute ablation (§10.1), not by this statistic.

**Acceptance criteria (tests/test_diversity_metrics.py):** Fréchet of identical trajectories = 0; mirrored left/right paths around an obstacle exceed `δ_distinct`; clustering of 10 noisy copies of 2 ground-truth routes returns exactly 2 clusters.

---

## 10. Evaluation Protocol (`eval/`)

### 10.1 Conditions (the experiment grid)

| Policy | Trained on | Label |
|---|---|---|
| BC | stage 1 only | baseline |
| BC + DAgger | §6 loop with RaC intervention **disabled** (pure HG-DAgger-style relabeling; on near-failure just terminate) | ablation isolating RaC as a whole |
| **RaC-NoReroute** | full §6 **except** §6.3 step 3's distinct-homotopy rejection is off (correction = first plan from a fresh seed, no `δ_reroute` filter) | **the key ablation: isolates route-*diversity* forcing from recovery/correction per se (tests H2)** |
| BC + DAgger + RaC | full §6 | main method |
| + Curriculum | + §7 | main method @ scale |
| + Curriculum (no densification) | §7 with `synthetic_obs: false` | ablation for the DMD-inspired flag (2 seeds suffice) |
| + RL diversity | + §8 | stretch |

### 10.2 Scene-set hierarchy (frozen before any training, committed as JSON)

**Three strictly separated tiers — this is a leakage firewall:**

1. **Train scenes:** unlimited, sampled on the fly from the training seed range; never reused for any evaluation.
2. **Validation sets** `val_L{0..3}.json` (100 scenes/level) **+** `val_unseen.json` (100): used for *everything decision-shaped* — round-by-round monitoring in §6.1, curriculum promotion/demotion probes in §7 (which therefore stop sampling their own probes), checkpoint selection, and any hyperparameter adjustment. May be looked at freely.
3. **Test sets** `test_L{0..3}.json` (200 scenes/level) **+** `test_unseen.json` (200): evaluated **exactly once per (condition, seed)** after all training and selection decisions are final; results go straight into the report. No decision of any kind may depend on them. The evaluator writes a `test_touched.json` audit stamp per run to make violations visible. **Bug-recovery protocol (decided now, because the first post-test bug otherwise forces a bad improvisation):** seed ranges for `test_v2` and `test_v3` are reserved *today* in `configs/env/panda_reach.yaml`. If a results-invalidating code bug is found after a test touch, the fix is rerun against freshly generated `test_v2` sets; the v1 touch is recorded as burned and v1 numbers are discarded, never re-measured. Test sets are consumables, and the spec budgets spares.

All sets: held-out, mutually disjoint seed ranges; planner-verified solvable.
- The `*_unseen` sets use L3 density with a *shifted obstacle-size distribution* (half-extents `U(0.08, 0.14)`) — the "clutter it has never seen before" condition. **These scenes are planner-verified solvable exactly like training scenes** (otherwise unsolvable scenes masquerade as policy failures), and the rejection rate during their generation is recorded — if > 50% of sampled scenes are unsolvable, the generator deterministically shrinks the size-shift in fixed 10% steps until the rate clears (§11.7.3), logging the chosen step. The re-tune is thereby automated, identical on every run, and incapable of being result-driven.

### 10.3 Reported numbers

For every condition × test set, over **3 training seeds (pinned: {0, 1, 2})**: success rate, collision rate, timeout rate, mean episode length, `#routes` / routes-per-success / route entropy (§9.2) on a 50-scene subset, recoveries/episode, and the recovery↔success correlation (§9.3). **Uncertainty + tests:** all scene-level comparisons between conditions use *scene-paired* statistics on the identical frozen scene sets — report 95% paired bootstrap CIs (10,000 resamples) on the success-rate difference, where the paired resampling unit is the **(scene, seed) cell** — both conditions evaluated on identical scene × seed combinations — plus per-seed numbers in the appendix. Pooling rollouts while pairing only on scenes would mix seed variance into the pairing and quietly widen or bias the CI. A condition is claimed "better" only when the paired CI on the difference excludes zero. Three seeds is a compute-forced floor — present per-seed scatter, never seed-σ error bars alone. **Reference ceiling row:** on the 50-scene `#routes` subset, also compute the *planner's* route diversity — 20 RRT-Connect runs per scene with distinct seeds, clustered identically to §9.2 — and report it as the top row of the route-diversity table. Policy route counts are uninterpretable without knowing how many distinct routes the scene geometry plus expert can even supply (if the planner ceiling on a scene is 1, no policy can score 2). Plots (`eval/plots.py`): success-vs-round curves per condition; #routes bar chart (with planner ceiling marked); recovery/correction data-composition stacked bars per round (mirrors RaC Fig. 12); qualitative top-down trajectory fans for 4 cherry-picked scenes.

### 10.4 Compute budget (sanity)

Single workstation w/ one mid-range GPU (or CPU-only at ~3× wall clock), 8 collection workers. Itemized: stage 1 collection ≈ 2–3 h; per DAgger+RaC round ≈ 1.5 h collect + 0.5 h retrain **+ 0.5 h round-end eval** (200 fixed scenes × 1 deterministic rollout + 50 scenes × 20 stochastic rollouts for `#routes`); curriculum probes (100 scenes/level) ≈ 0.2 h/round. Per condition ≈ 6 rounds × ~2.7 h ≈ 16 h; the §10.1 grid (5 training conditions, of which 4 are non-stretch) × 3 seeds ≈ **8–9 days of wall clock on one box** — i.e., the grid, not training, is the bottleneck. Mitigations, in order: parallelize runs across GPUs/machines with the §10.5 grid launcher (the primary lever — see §10.5.7 for revised totals); reduce the BC-only and DAgger-only ablations to 2 seeds; shrink `#routes` eval to 30 scenes. Decide cuts *before* launching, not per-result.

## 10.5 Multi-GPU parallelization and automatic GPU allocation

### 10.5.1 Workload profile → parallelization strategy (read this before adding GPUs)

The compute shape of this project is unusual and the strategy must match it:

| Workload | Bound by | Right parallelism |
|---|---|---|
| Policy training (BC retrains, §5.3/§6.1) | One GPU is *underutilized* — MDN-MLP, batch 1024, ~500k rows | **Run-level**: many runs, one GPU-slot each |
| Data collection (§5.1, §6.2) | CPU (MuJoCo physics + RRT-Connect) | Process-level on CPU cores; **GPUs irrelevant** |
| Evaluation (§10.3) | CPU rollouts; inference is tiny | Scene-sharded CPU workers |
| Condition grid (§10.1: ~15 runs) | Wall clock — **the actual bottleneck** | **This is what multi-GPU buys down** |
| Stage 4 PPO (§8) | Mixed; envs on CPU, updates on GPU | One GPU per run; DDP optional |

Consequently: **the unit of GPU scheduling is a whole run**, not a gradient step. Within-run DDP exists behind a flag (§10.5.5) but is off by default — sharding a model that doesn't saturate one device adds nondeterminism and complexity for negative speedup. This section therefore specifies (a) an allocator that places runs on GPUs automatically by live utilization/memory, and (b) a grid launcher that drains the §10.1 condition grid across all available devices.

### 10.5.2 `utils/gpu_alloc.py` — discovery, scoring, leasing

```python
@dataclass(frozen=True)
class GpuLease:
    physical_id: int      # NVML index
    lock_path: Path
    def env(self) -> dict:        # {"CUDA_VISIBLE_DEVICES": str(physical_id)}
    def release(self) -> None

class GpuAllocator:
    def __init__(self, lock_dir: Path = Path("~/.robot_routes/gpu_locks"),
                 jobs_per_gpu: int = 2, util_w: float = 0.5, mem_w: float = 0.5): ...
    def acquire(self, mem_required_gb: float = 2.0,
                exclusive: bool = False,
                timeout_s: float | None = None) -> GpuLease | None
```

- **Discovery:** enumerate devices via `pynvml`; if the parent environment already sets `CUDA_VISIBLE_DEVICES` (cluster schedulers, SLURM), restrict the candidate set to those physical ids — the allocator subdivides what it was given, never escapes it. If NVML init fails or no devices exist, return `None` and the caller falls back to CPU (every consumer must handle this; the whole pipeline is CPU-runnable per §10.4).
- **Scoring (automatic allocation):** for each candidate GPU sample utilization and memory **3 times at 200 ms intervals** (instantaneous reads race against jobs that are still warming up) and compute `score = util_w · mean_util + mem_w · mem_used_frac`. Eligible = `free_mem ≥ mem_required_gb + 0.5 GB headroom` **and** `active_leases < jobs_per_gpu` (and `active_leases == 0` if `exclusive`). Acquire the **lowest-scoring eligible** device; if none, block-poll every 10 s until `timeout_s`.
- **Leasing (race prevention):** NVML queries alone cannot stop two processes from picking the same idle GPU simultaneously. Leases are JSON files `{pid, mem_gb, ts}` in `lock_dir/gpu{i}/`, created under an `fcntl.flock` on the directory so acquire-check-write is atomic on one host. **Stale-lease reaping:** on every acquire, delete leases whose `pid` is no longer alive (`os.kill(pid, 0)`); no heartbeat needed on a single box. (Multi-host scheduling is explicitly out of scope — use the cluster's own scheduler per-host and let the allocator subdivide within each node.)
- **Consumption rule:** the lease exports `CUDA_VISIBLE_DEVICES=<physical_id>` into the *child process* environment, so all run code unconditionally uses `cuda:0` internally — no device indices anywhere in library code. Config override: `compute.device: auto | cpu | cuda:N` (where `auto` invokes the allocator; `cuda:N` bypasses it for debugging).
- **OOM backoff:** a run catching `torch.cuda.OutOfMemoryError` releases its lease and re-enqueues itself once with `exclusive=True`; a second OOM is a hard failure (the model is small — a second OOM means a bug, not contention).

### 10.5.3 `scripts/07_launch_grid.py` — draining the condition grid

- Reads `configs/grid.yaml`: the §10.1 conditions × pinned seeds {0,1,2}, ordered by the §10.4 pre-committed priority (main conditions first, 2-seed ablations last) — so if the machine pool shrinks mid-week, what's been completed is what matters most.
- Maintains a FIFO of pending runs; for each, `acquire(mem_required_gb=2.0)` then `subprocess` the appropriate `make reproduce CONDITION=... SEED=...` with the lease's env; on exit, release the lease and start the next. **Concurrency cap honest to CPUs:** each run wants 8 collection workers, so concurrent runs = `min(jobs_per_gpu × n_gpus, floor(n_cores / 8))` — on a typical 32-core/4-GPU box the *cores* bind (4 concurrent runs), and `jobs_per_gpu=2` only pays off on high-core-count machines. The launcher computes and logs this cap at startup.
- **Resume:** runs whose directory contains a `COMPLETED` stamp are skipped, so the launcher is idempotent after crashes/preemption.
- **Telemetry:** per-run log of which physical GPU served it + utilization snapshots every 60 s appended to the run dir (post-hoc evidence the allocator balanced sensibly; also the debugging trail when it didn't).

### 10.5.4 Device placement within a run

- **Training:** the leased GPU (`cuda:0` after env masking); CPU fallback works at ~3× wall clock.
- **Collection workers:** **CPU inference, deliberately** — `torch.set_num_threads(1)` per worker, model in eval mode. The MDN forward is microseconds; routing 8 workers' single-observation queries through a shared GPU would serialize on context switches and couple collection to the trainer's device. The GPU belongs to the optimizer during collection rounds.
- **Evaluation:** scene-sharded CPU workers by default; `06_evaluate.py --device auto` may lease idle GPUs to shard the 20-rollout `#routes` evals across devices when the grid is otherwise drained.
- **Stage 4 PPO:** one leased GPU for policy/value updates; the 8 envs remain CPU subprocesses feeding batched tensors.

### 10.5.5 Optional within-run DDP (off by default — and why)

`train.ddp: true` enables single-host `torch.distributed` (NCCL backend, via `torchrun`, leasing `ddp_world_size` GPUs as one exclusive multi-device lease). Kept because stage 4 PPO with enlarged batches *might* benefit; off by default because (a) the supervised retrains complete in ~30 min on one device — Amdahl says the grid dominates; (b) DDP all-reduce introduces nondeterministic reduction orders that violate §10.5.6 unless deterministic algorithms are forced at further speed cost. Enabling DDP for stages 1–3 is explicitly discouraged in-config with a comment.

### 10.5.6 Determinism under automatic allocation

- **Placement must not change results:** allocation chooses *where*, never *what* — no RNG stream may be derived from device id, hostname, or allocator state. Enforced by `tests/test_gpu_alloc.py`: (a) allocator unit tests against a mocked NVML (scoring, lease atomicity under thread-hammering, stale reaping); (b) a placement-invariance integration test — train the smoke profile twice with the same seed on different devices (or CPU vs GPU) and assert eval metrics agree within `1e-3` (bitwise equality across device architectures is not promised; metric-level equality is).
- Set `torch.use_deterministic_algorithms(True, warn_only=True)` and seed CUDA per §11 conventions; log any nondeterministic-kernel warnings into the run dir.

### 10.5.7 Revised wall-clock with the launcher

Per §10.4, one condition ≈ 16 h. With the launcher on a 32-core/4-GPU box (cap = 4 concurrent runs): full grid ≈ ceil(15/4) × 16 h ≈ **2.7 days** (was 8–9 sequential). Two such boxes (launcher per box, grid split by condition): ≈ 1.5 days. Collection-stage CPU saturation is the binding constraint throughout — adding GPUs beyond `floor(cores/8)` buys nothing, which the launcher's startup log makes visible rather than leaving anyone to discover it from flat utilization graphs.

---

## 11. Implementation Plan for Cursor (ordered work packages)

Implement strictly in this order; each WP ends with its tests green before starting the next. Suggested Cursor workflow: keep this spec open as context, implement one WP per session/branch, and ask the agent to write the tests *first* from the acceptance criteria.

| WP | Content | Spec §§ | Definition of done |
|---|---|---|---|
| 1 | Repo scaffold, configs, `mj_state.py`, Panda asset loads & renders, **`scripts/00_smoke.py`** (env reset→20 random steps→state save/restore→clearance query, < 60 s CPU) and **GitHub Actions CI** running `pytest` + smoke on CPU per push. **`utils/gpu_alloc.py` + mocked-NVML unit tests** (§10.5.2) — built first so every later script takes `--device auto` from day one | 2, 10.5 | CI green; viewer shows arm; `test_gpu_alloc.py` green on a GPU-less runner |
| 2 | `scene_gen.py` + `PandaReachEnv` | 3 | `test_env.py` green |
| 3 | `collision.py`, `rrt_connect.py`, `oracle.py` | 4 | `test_expert.py` green |
| 4 | Data schema/buffer + stage-1 collection + BC training | 5 | BC ≥ 40% on L0 eval |
| 5 | DAgger+RaC loop (`dagger_rac.py`) incl. rewind machinery | 6 | `test_rewind.py` green; success(π_2) > success(BC) on `val_L0` |
| 6 | Curriculum controller + synthetic densification | 7 | promotion/demotion behaves on a scripted success sequence |
| 7 | Diversity metrics + full evaluator + plots | 9, 10 | `test_diversity_metrics.py` green; full eval runs end-to-end on checkpoints |
| 8 | (stretch) PPO + diversity reward | 8 | training runs stably 1M steps; report deltas |

**Cross-cutting conventions for the agent:**
- Determinism first: every random draw flows through a seeded `np.random.Generator` passed explicitly; no global seeds inside library code.
- All scripts accept `--config path.yaml --seed N --out dir/` and dump the resolved config + git hash next to outputs.
- No silent magic numbers: every constant in this spec lives in a YAML config with the same name used here.
- Keep `envs/`, `expert/`, `agents/` import-independent of each other except through the dataclass interfaces (`SceneSpec`, `MjSimState`, `JointPath`, transition schema) so WPs can be developed and tested in isolation.
- **Artifact retention:** every run directory keeps: resolved config, git hash, per-round dataset deltas `Δd_k` (never only the merged aggregate — round-composition analysis and the RaC Fig.-12-style plot need them), all round checkpoints, eval JSONs, and 4 rendered rollout videos per round (fixed showcase scenes). Nothing is overwritten; runs are append-only directories named `{stage}_{seed}_{timestamp}`.

---

## 11.5 Deliverables and reproduction path

**Deliverables checklist:** (1) the repository with CI green; (2) final report (course format) whose results tables are generated by `eval/plots.py` from committed eval JSONs — no hand-transcribed numbers; (3) a 3-minute results video assembled from the per-round showcase renders (§11 artifact rules); (4) the frozen scene-set JSONs and calibrated `δ_distinct`; (5) trained checkpoints for every (condition, seed) in the grid; (6) `hypotheses_verdicts.json` and per-run `pipeline_state.json` files — machine-readable evidence of which gates fired, which branches were taken, and that test sets were touched exactly once.

**Reproduction path:** a top-level `Makefile` chains the numbered scripts: `make reproduce SEED=0 CONDITION=full` runs collect → BC → DAgger+RaC → curriculum → eval end-to-end from a clean checkout, consuming only committed configs. `make grid` invokes the §10.5 launcher to drain all (condition, seed) runs across available GPUs. A `PROFILE=smoke` variable scales every count down (~50 demos, 2 rounds, 2k-transition budgets, 20-scene evals) to finish in < 1 h on CPU — this profile is what CI's nightly job runs, and it is the first thing implemented in WP5 so the full pipeline is exercised long before full-scale runs. The **orchestrator skeleton** (§11.7: state machine, heartbeats, gate interfaces with stub predicates) is built in WP5 alongside it; each gate's real predicate lands with its stage, so by WP7 `make pipeline` is genuinely zero-touch end-to-end.

---

## 11.6 Limitations register (accepted by design — state these in the report)

1. **Observation caps scenes at 8 obstacles** with a fixed slot encoding; generalization beyond the cap is untested (future: set/point-cloud encoder).
2. **EE-space route metric:** elbow-family diversity with identical EE paths is invisible; metric distance is a homotopy proxy, not a homotopy test (§9.1).
3. **The expert bounds the achievable:** policy route diversity is upper-bounded by the planner ceiling (§10.3) — the project measures how much of that ceiling imitation can capture, not diversity ex nihilo.
4. **Automated RaC ≠ human RaC:** our near-failure trigger and reversal-based recovery are simulator constructs; conclusions about Hu et al.'s human-in-the-loop protocol transfer only by analogy.
5. **Floating, static, primitive obstacles** (§3.1); no sensing — ground-truth scene encoding; sim-only, no real-robot claim.
6. **H3 is observational** (§9.3); only the H2 ablation carries causal weight on the diversity claim.

---

## 11.7 Zero-touch orchestration (full-pipeline automation)

**Goal:** one command — `make pipeline` — takes a clean checkout to evaluated checkpoints, hypothesis verdicts, and report-ready tables for the entire §10.1 grid, with no human decision between first demo and final number. Every prose gate elsewhere in this spec is restated here as an executable predicate; where this section and older prose conflict, **this section wins**.

### 11.7.1 The orchestrator: a persisted DAG state machine

`scripts/run_pipeline.py` executes, per (condition, seed), the stage DAG:

```
setup → scene_sets → calibrate_delta → collect_bc → train_bc → [G-BC]
      → dagger_rac_round(k=1..K) with [G-DATA], [G-REGRESS] per round
      → curriculum (promotion controller, §7) → [G-DITHER]
      → ppo [only if G-PPO] → evaluate_test (once) → verdicts → report_assets
```

- **State:** `pipeline_state.json` in each run dir — per stage: `{status: PENDING|RUNNING|COMPLETED|FAILED, config_hash, started, finished, attempts}`. The orchestrator is **idempotent**: re-invocation skips `COMPLETED` stages.
- **Config-hash invalidation:** each stage records the SHA of the resolved config slice it consumed; if a config edit changes a hash, that stage and *all downstream stages* auto-invalidate (no stale-results-from-old-configs class of bug, no human bookkeeping).
- **Grid mode:** `make pipeline` = §10.5.3 launcher running `run_pipeline.py` per (condition, seed) under GPU leases; priority order and resume semantics unchanged.

### 11.7.2 Gates as code (the former human judgment calls)

| Gate | Predicate (all on **validation tier** — §10.2 firewall applies to automated decisions too) | On fail |
|---|---|---|
| **G-CAL** (calibration valid) | cross-homotopy distance p5 > same-homotopy p95 (distributions separated) | ABORT + diagnostics dump — δ_distinct is load-bearing; proceeding on a broken calibration poisons everything downstream |
| **G-BC** | val_L0 success ≥ 0.25 | auto-run the §5.3 diagnostic ladder *as code*: (i) expert-replay check, (ii) overfit-10-episodes check, (iii) automated 2×2 sweep (lr × chunking), retrain best; if still < 0.25 → FAILED + notify (a genuine research problem has been *detected*, which is the automation's job — solving it is ours) |
| **G-DATA** (per collection round) | schema valid; zero NaNs; every recovery segment re-verified collision-free; recovery:correction frame ratio ∈ [0.2, 2.0]; fallback-acceptance rate (§6.3.3) < 0.8 | auto-recollect the round once with a bumped collection seed; second failure → FAILED + notify |
| **G-REGRESS** (per round) | success(π_k) ≥ success(π_{k−1}) − 10 pts on val_L{cur} | auto-retry round once with a new collection seed, **after discarding the failed round's Δd_k from the aggregate** (otherwise the retry silently doubles round-k data and corrupts composition stats); if regression persists, *continue* but flag — DAgger curves legitimately dip; two consecutive regressions > 20 pts abort |
| **G-DITHER** | median mode-switches/episode ≤ 10 (§5.2 diagnostic) after the final stage-2/3 round | **auto-branch**: retrain on the identical aggregated dataset with the IMLE head (pre-identified fallback, §12), evaluate both branches on validation, carry the winner forward; both recorded in state. **Downstream coupling:** an IMLE policy has no tractable likelihood, so PPO's ratio is undefined on it — if the IMLE branch wins, **G-PPO auto-resolves NO-GO** with reason `head_incompatible` (stages 1–3 results stand; bolting a surrogate likelihood onto IMLE is out of scope) |
| **G-PPO** (go/no-go, replaces §8's human decision) | (a) paired-CI on **val_unseen** for (full − RaC-NoReroute) excludes zero, (b) §9.2 validity-gate pass ≥ 60% on val scenes, (c) `now() < ppo_deadline` from `configs/pipeline.yaml` | NO-GO recorded in state; §8 auto-moves to future work; pipeline proceeds to test evaluation. **Cross-run dependency, declared:** predicate (a) reads the RaC-NoReroute condition's validation evals — another run's output. `grid.yaml` declares `ppo.requires: [rac_noreroute.curriculum.eval_val]` per seed, and the launcher implements it as a barrier: the full condition's PPO stage parks (state `WAITING_DEP`, heartbeat maintained) until the dependency's stamps exist, with a `dep_timeout` (default 48 h) after which G-PPO resolves NO-GO with reason `dep_unmet`. Without the declared barrier the DAG either deadlocks or reads missing files, depending on scheduling luck |
| **G-β** (in-training) | std(r_div) ≤ 2 × success bonus, checked per epoch | callback halves β automatically, logs the event; ≥ 3 halvings → freeze β floor and continue |

Note the deliberate correction baked into G-PPO: the earlier §8 prose keyed the go/no-go to the *test*-set CI, which contradicted the §10.2 rule that no decision may consume test data. Automation forces such conflicts into the open: gates can only be coded against data the code is allowed to read.

### 11.7.3 Self-running calibration and scene generation

- The `setup` stage runs **config-invariant validation** before anything else: `margin_plan > ε_danger` (§4.1), `ε_safe > margin_plan`, recorded goal-hold steps == success-sustain steps (5 == 5, §3.2/§5.1), `δ_reroute == calibrated δ_distinct` once calibration lands, and disk preflight (≥ 50 GB free on the run volume; the full grid's data + per-round checkpoints + videos budget ≈ 25 GB, headroom doubles it). Invariant violations ABORT with the named constraint — these couplings were individually load-bearing bugs at some point in this spec's history, and config edits must not be able to silently reintroduce them.
- `calibrate_delta` is a **grid-level singleton**, not a per-run stage: it runs exactly once per grid with a *fixed, committed calibration seed*, generates the §9.1 constructed-scene pairs, computes the midpoint, asserts G-CAL, and writes `calibration/delta.json` (value + distributions + SHA). Every run consumes that one file, and the §11.7.3 invariant check asserts all runs in a grid reference the identical calibration hash. **Why singleton is load-bearing:** per-run calibration (the original design) lets planner RNG produce slightly different `δ_distinct` per run — conditions would then be compared on diversity metrics computed at *different thresholds*, and `δ_reroute` (the treatment itself, §6.3) would differ across conditions. A comparability bug, not a style choice.
- `scene_sets` deterministically regenerates every frozen set from committed seed ranges and **verifies SHA-256 against committed hashes** (drift = ABORT). The `*_unseen` generator's "re-tune if > 50% unsolvable" rule (§10.2) becomes a deterministic ladder: shrink the size-shift in fixed 10% steps until the rejection rate clears, log the step chosen — same outcome every run, zero human tuning.

### 11.7.4 Unattended-operation machinery

- **Watchdog:** every stage touches a heartbeat file ≥ once/min; 30 min silence → SIGKILL, requeue (max 2 attempts), then FAILED + notify. Catches hung planners and deadlocked workers overnight.
- **Auto checkpoint selection:** per round and at stage ends, the checkpoint with best val success is symlinked `best.pt` by the trainer itself; every later stage consumes the symlink. No human ever picks a checkpoint.
- **Notifications:** pluggable `notify(event)` — webhook URL from env (Slack/Discord) if set, else stdout + `events.log`. Fired on: stage FAILED, gate fail-branches taken, grid completion, and a daily digest (stages done, GPU utilization, ETA from per-stage moving averages).
- **Provisioning:** `make setup` builds a pinned container (Dockerfile + `uv.lock`); CI and all machines run the same image — eliminates the "works on the lab box" manual debugging genre. The image sets `MUJOCO_GL=egl` with automatic fallback to `osmesa` (headless servers have no GLX — the classic first-night surprise), and **all rendering is non-fatal**: a failed video write logs a warning and the pipeline continues, because showcase videos are QA sugar, not results. The `setup` stage also checks **host NVIDIA driver ≥ the pinned CUDA runtime's minimum** (parsed from `nvidia-smi`) and aborts with both version strings on mismatch — driver/runtime incompatibility is the single most common container-on-GPU failure and its native error message is famously unhelpful.
- **Filesystem locality rules:** HDF5 writes and the GPU lock dir **must live on a local filesystem** — HDF5 file locking and `fcntl.flock` are both unreliable on NFS (lab home directories are the usual trap). The container exports `HDF5_USE_FILE_LOCKING=FALSE`, collection writes shards to local scratch and the merge step moves the result; `gpu_alloc` refuses an NFS lock dir at startup (detected via `statfs`). Run dirs **rsync to shared storage nightly** (a launcher cron hook) so a single dead disk costs at most one day of one machine's runs.
- **Logging is offline-first:** `wandb` runs in offline mode by default with post-hoc sync — an unattended multi-day grid must not stall or die on the lab network's mood.
- **Pre-registration is timestamped:** before the first full-scale run, the orchestrator requires a git tag `prereg-v1` covering §1.4 and the frozen scene-set hashes, and records the tag in every `pipeline_state.json` — making "hypotheses written before results" verifiable rather than asserted.
- **Auto-verdicts and report assets:** the `verdicts` stage runs the §10.3 paired bootstrap and writes `hypotheses_verdicts.json` — H1/H2/H3 each `CONFIRMED|REFUTED|UNDERPOWERED` with CIs, computed by code, on test data touched exactly once. **Verdict rules, pre-registered (so the labels cannot be argued at write-up):** minimum detectable effect MDE = 10 percentage points of success rate. `CONFIRMED` = CI excludes 0 in the predicted direction; `REFUTED` = CI lies entirely below the MDE (the effect, if any, is demonstrably smaller than the smallest effect we claimed to care about); `UNDERPOWERED` = neither (CI straddles 0 *and* reaches past the MDE — the data cannot distinguish). **Multiple comparisons:** H1 and H2 verdicts are computed at raw α = 0.05 *and* Holm–Bonferroni-adjusted across the two confirmatory hypotheses; both reported, adjusted ones headline. H3 is observational and exempt. `report_assets` regenerates every figure and table (incl. the RaC-Fig.-12-style composition plot and the planner-ceiling row) into `report/` from eval JSONs. The report's numbers section is therefore a build artifact.

### 11.7.5 What stays manual — by design, and it's short

(1) Writing the report's prose and interpreting the verdicts; (2) *optional* QA glances at the per-round showcase videos (the pipeline never blocks on them); (3) the one-time decision to launch `make pipeline`. Everything between that command and `hypotheses_verdicts.json` is unattended.

---

## 12. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| RRT-Connect too slow per label call → DAgger rounds crawl | Med | plan-caching + drift-gated replans (§6.2); 8-process episode parallelism; lower `time_budget_s` at L0 |
| Reversed-rollout "recovery" segments are dynamically infeasible (velocity discontinuities) | Med | actions are position deltas tracked by stiff position actuators, so reversal is kinematically benign; drop last 3 pre-trigger steps; verify clearance on replay |
| Rerouting rarely finds distinct homotopies (workspace too open) | Med | multi-goal IK seeding (§4.1) + reject-below-`δ_reroute` retry loop (§6.3); if still degenerate, raise obstacle density of L0 |
| Distinct-route threshold `δ_distinct` arbitrary | Low | calibrated from planner same/cross-homotopy distance distributions (§9.1) + ±33% sensitivity sweep in appendix |
| Stage 4 RL destabilizes recovery behavior | Med | KL anchor to stage-3 policy; explicit go/no-go gate (§8) — report stands on stages 1–3 |
| MDN component collapse (all weight on one mode) | Med | entropy bonus + log-std clamp (§5.2); monitor per-retrain effective-component count (perplexity of mean mixture weights); if persistently < 2, switch head to IMLE Policy objective (pre-identified fallback) |
| Full condition grid exceeds wall-clock budget | High | pre-committed cut order in §10.4 (parallel seeds → 2-seed ablations → 30-scene #routes eval); never cut conditions after seeing results |
| H2 null result (rerouting adds nothing) | Med | pre-registered as an acceptable outcome (§1.4) — the report's framing does not depend on H2 confirming |
| GPU allocator races / placement-dependent results | Low | flock-atomic leasing + stale-PID reaping (§10.5.2); placement-invariance test (§10.5.6); allocation chooses placement only, never touches RNG streams |
| Packing runs per GPU starves CPU collection workers | Med | launcher caps concurrency at min(jobs_per_gpu × GPUs, cores/8) and logs the binding constraint at startup (§10.5.3) |
| Automated gates mask a real problem by silently branching/retrying | Med | every gate action is an event in `pipeline_state.json` + notification (§11.7.4); retry caps are low (1–2); G-CAL and double G-DATA failures hard-abort rather than route around load-bearing breakage |
| Obs encoding caps scenes at 8 obstacles | Accepted | fixed by design; noted as limitation/future work (point-cloud or set-transformer encoder) |

---

## 12.5 Pre-mortem register (adverse surprises, pre-empted)

Compiled by assuming failure and working backwards; each row names the surprise, the moment it would have struck, and where it is now neutralized.

| # | Adverse surprise (as it would have appeared) | When it would strike | Pre-emption |
|---|---|---|---|
| 1 | "The policy keeps triggering RaC on perfectly good trajectories, and our recovery counts look absurdly high" | mid stage 2 + at eval, corrupting H3 | margin/trigger coupling `margin_plan > ε_danger` + startup invariant check (§4.1, §11.7.3) |
| 2 | "It keeps smashing into the table with no intervention" | stage 2 | table included in clearance/trigger geom set (§3.3) |
| 3 | "Joint 4 grinds at its limit and the arm enters weird states" | first long rollouts | `q_target` clamped to joint limits − margin (§3.2) |
| 4 | "State restore is *almost* deterministic — trajectories diverge after ~20 steps" | WP2, debugged for days | `qacc_warmstart` enumerated in `MjSimState` (§3.3) |
| 5 | "Round-3 retry doubled the dataset and the composition plot makes no sense" | stage 2 analysis | G-REGRESS discards failed Δd_k before retry (§11.7.2) |
| 6 | "IMLE branch won, and now PPO crashes: no log-prob" | stage 4 launch, overnight | G-DITHER→G-PPO coupling: IMLE win ⇒ auto NO-GO `head_incompatible` (§11.7.2) |
| 7 | "We found a bug after touching the test set — re-run and quietly hope, or throw away the project?" | the worst possible week | reserved `test_v2/v3` seed ranges + burn protocol, decided in advance (§10.2) |
| 8 | "Route diversity 'improved' because the weak baseline was only scored on its easiest scenes" | report writing / review | gate-intersection rule for cross-condition #routes comparisons (§9.2) |
| 9 | "The CI is mysteriously tight/wide depending on how we pooled seeds" | verdict computation | pairing unit pinned to (scene, seed) cells (§10.3) |
| 10 | "Rendering crashed the whole grid at 2 a.m. on the headless box" | first unattended night | `MUJOCO_GL=egl→osmesa` + all rendering non-fatal (§11.7.4) |
| 11 | "HDF5 corruption / two runs on one GPU — only on the machine with NFS homes" | lab-machine scale-out | locality rules: local scratch + lock-dir NFS refusal + `HDF5_USE_FILE_LOCKING=FALSE` (§11.7.4) |
| 12 | "wandb outage hung every run at once" | any network blip | offline-first logging with post-hoc sync (§11.7.4) |
| 13 | "A disk died and took a week of runs with it" / "disk filled at hour 60" | mid-grid | nightly rsync of run dirs + 50 GB preflight (§11.7.3–4) |
| 14 | "Reviewer: 'were these hypotheses really pre-registered?'" | grading/review | `prereg-v1` git tag required and recorded in pipeline state (§11.7.4) |
| 15 | "Frozen scenes turn out unplannable under the expert's margin" | stage 1 | scene acceptance runs the planner *with* the full margin (§3.1) |
| 16 | "PPO trained fine but the policy got worse — gradients were silently off-policy" | stage 4, undetectable from losses | continuity kernel disabled during PPO collection; kernel scoped as inference-time wrapper (§5.2, §8) |
| 17 | "Conditions were compared at different δ_distinct values — the diversity table is apples-to-oranges" | report writing / review | calibration is a grid-level singleton with fixed seed + identical-hash invariant (§11.7.3) |
| 18 | "The full condition's PPO stage hung all weekend waiting for files that didn't exist" | grid execution | declared cross-run dependency + launcher barrier + dep_timeout → NO-GO `dep_unmet` (§11.7.2) |
| 19 | "We argued for a week over whether H2 was 'refuted' or just 'underpowered'" | write-up | pre-registered MDE = 10 pts with three-way verdict rule (§11.7.4) |
| 20 | "Reviewer: 'two hypotheses, no multiplicity correction?'" | review | Holm–Bonferroni across H1/H2 reported as headline; H3 exempt as observational (§11.7.4) |

Residual surprises this register cannot pre-empt — named so they are at least *expected*: the empirical unknowns already flagged (MDN dithering in practice, planner route ceiling per scene, H2's true effect size), and any MuJoCo/PyTorch behavior outside the pinned versions. Everything else that bites should now bite a gate, not the schedule.

**Spec freeze (the last pothole is the meta one).** Two consecutive hardening passes have now been run; this second one found real bugs, but every one of them was an interaction *between earlier fixes* — the signature of a document at its useful complexity ceiling, where further review passes manufacture as many seams as they close. **v3.4 is therefore declared frozen.** From here, changes are made only through `ISSUES.md`: implementation-discovered deviations, each with the spec section it amends, a rationale, and sign-off at the next WP boundary. The register's final pre-empted surprise is the project that polishes its specification instead of running it.

---

## 13. References

1. Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* arXiv:1011.0686, 2011.
2. Zhang, Chang, Kumar, Gupta. *Diffusion Meets DAgger: Supercharging Eye-in-Hand Imitation Learning.* arXiv:2402.17768, 2024.
3. Hu, Wu, Enock, et al. *RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction.* arXiv:2509.07953, 2025.
4. Zakka, Tabanpour, et al. *MuJoCo Playground.* 2025.
5. Rana et al. *IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation.* 2025. (Background on sample-efficient multimodal BC; candidate drop-in policy class for future work.)


---

## 14. Agent Implementation Harness (Cursor autonomy layer)

**Status note:** this section is an *additive implementation-enablement layer*. It changes no experimental parameter, gate, metric, or scene definition — the v3.4 freeze on experimental design is intact (logged as Rev 17, amendment class: additive harness). Its purpose: shrink human involvement during the *build* from continuous supervision to (a) eight WP-boundary reviews and (b) responding to structured escalations.

### 14.1 Operating principle

An agent can work unsupervised exactly to the extent that "done" is machine-checkable and "forbidden" is machine-enforced. Everything below converts the spec's prose criteria into commands the agent runs itself, and its implicit norms into lint rules that fail CI. The human is removed from the *loop* and kept at the *boundaries*.

### 14.2 Repository additions

```
robot-routes/
├── .cursor/rules                # §14.4 verbatim — loaded into every agent session
├── .devcontainer/               # same image as CI/pipeline (§11.7.4) — agent env = prod env
├── TASKS/
│   ├── wp1.yaml … wp8.yaml      # task files, schema §14.3
├── src/robot_routes/contracts.py# all cross-module dataclasses & Protocols — written FIRST
├── BLOCKED.md                   # escalation log, format §14.7
├── ISSUES.md                    # spec-deviation log (per §12.5 freeze process)
└── scripts/
    ├── check_constants.py       # numeric-literal lint: no magic numbers outside configs/tests
    └── check_spec_constants.py  # asserts configs == Appendix B golden values
```

### 14.3 Task files: the unit of autonomous work

Each WP decomposes into 4–8 atomic tasks. **Schema** (`TASKS/wp{N}.yaml`):

```yaml
- id: wp2.3
  title: Clearance computation with broad-phase pruning
  spec: ["3.3"]                       # sections the agent must (re)read before coding
  creates: ["src/robot_routes/envs/panda_reach_env.py::min_clearance"]
  depends: [wp2.1, wp2.2]
  done_when: "pytest tests/test_env.py -k clearance -q"   # exit 0 = done, no judgment call
  budget_minutes: 45                  # exceeded → file BLOCKED entry, move on
```

Canonical example — **WP1 fully expanded** (the agent's first session):

```yaml
- {id: wp1.1, title: Scaffold + pyproject + ruff/mypy strict + CI workflow,
   spec: ["2.2","2.3"], done_when: "make lint && make type", budget_minutes: 40}
- {id: wp1.2, title: contracts.py — SceneSpec, MjSimState, JointPath, Transition,
   ExpertOracle/Env Protocols, exactly as typed in §§3–6,
   spec: ["3.3","4.2","5.1"], done_when: "make type && pytest tests/test_contracts.py -q",
   budget_minutes: 60}
- {id: wp1.3, title: mj_state.py save/restore (field list §3.3 incl. qacc_warmstart),
   spec: ["3.3"], done_when: "pytest tests/test_env.py -k roundtrip -q", budget_minutes: 45}
- {id: wp1.4, title: Panda asset vendoring + headless render fallback,
   spec: ["3.1","11.7.4"], done_when: "python scripts/00_smoke.py --render", budget_minutes: 45}
- {id: wp1.5, title: gpu_alloc.py + mocked-NVML tests, spec: ["10.5.2","10.5.6"],
   done_when: "pytest tests/test_gpu_alloc.py -q", budget_minutes: 90}
- {id: wp1.6, title: check_constants.py literal-lint + Appendix B golden config test,
   spec: ["14.5","Appendix B"], done_when: "make check-constants", budget_minutes: 45}
```

**Self-planning rule:** completing WP *k* includes the agent drafting/validating `TASKS/wp{k+1}.yaml` against the spec sections it cites, committed for the human's WP-boundary review — so planning itself needs no separate human session, only sign-off.

### 14.4 `.cursor/rules` (paste verbatim)

```
1. contracts.py is law. Never change a signature there without an ISSUES.md entry;
   all other modules import from it, never redefine shared types.
2. No numeric literal outside configs/, tests/, or contracts defaults.
   `make check-constants` enforces this; do not allowlist around it.
3. Never modify: frozen scene JSONs, calibration/delta.json, anything under
   eval test-tier paths, Appendix B golden values, .cursor/rules itself.
4. Test-first: for each task, write the test from its spec sections' acceptance
   criteria BEFORE the implementation. If the spec's criterion is ambiguous,
   that is a BLOCKED entry, not an interpretation.
5. Inner loop: make lint && make type && targeted pytest. Full `make check`
   before marking any task done. Never mark done on a red command.
6. One task = one commit, message "wpN.M: <title> (spec §X.Y)". CI green
   before starting the next task.
7. When stuck past budget_minutes: append a BLOCKED.md entry (format §14.7),
   pick the next task with satisfied depends. Never hack around a blocker;
   never weaken a test to pass it.
8. Determinism rules of §11 apply to test code too: seeded Generators only.
9. Long-running anything (training, collection) uses PROFILE=smoke in your
   sessions. You never launch full-scale runs; humans do (§11.7.5).
10. If implementation reveals the spec is wrong: ISSUES.md entry with the
    section, the contradiction, and a proposed amendment. Do not silently fix.
```

### 14.5 Verification ladder (the agent's feedback loop)

| Tier | Command | Wall time | Agent uses it |
|---|---|---|---|
| 0 | `make lint && make type` (ruff, mypy --strict on src/) | seconds | every edit |
| 1 | targeted `pytest -k <task>` | < 1 min | inner loop |
| 2 | `make check` = tiers 0–1 full + `check-constants` + unit suite (no sim rollouts > 5 s) | < 3 min | before marking a task done |
| 3 | `make check-wp WP=N` = tier 2 + that WP's integration tests + `00_smoke.py` | < 10 min | before declaring a WP done |
| 4 | `make pipeline PROFILE=smoke` (end-to-end, CPU) | < 1 h | WP5+ boundaries; nightly CI |

Tests are tiered by pytest markers so tier 2 stays under 3 minutes *forever* — a slow inner loop is the main cause of agents (and humans) skipping verification.

### 14.6 Self-verifiable correctness: golden + property tests

Human code review cannot scale to every line; these make correctness checkable without eyes. The agent implements them as part of the relevant WP — they are listed here so their *expected values* come from the spec, not from the implementation under test:

- **Fréchet golden cases (§9.1):** identical curves → 0; two parallel straight segments offset by d → exactly d; a curve vs itself reversed → known analytic value for a semicircle fixture. DTW flag cross-checked on the same fixtures.
- **IK property (§4.1):** ∀ sampled reachable targets: `‖fk(ik(x)) − x‖ < 1e-3`, all solutions within joint limits − margin; elbow stratification yields ≥ 2 solution families on a centered target.
- **Reversal algebra (§6.3):** executing actions `a_1..a_T` then their reversal in a *frictionless, obstacle-free* fixture returns within 0.02 rad of start (loose bound — dynamics aren't exactly reversible; the test catches sign/order bugs, not physics).
- **Bootstrap calibration (§10.3):** on synthetic paired data with a planted 15-pt effect and known variance, the CI machinery achieves nominal coverage within Monte-Carlo error across 500 repetitions, and the MDE verdict rule returns CONFIRMED/REFUTED/UNDERPOWERED correctly on three planted effect sizes (0, 5, 15 pts).
- **Allocator hammer (§10.5.2):** 32 threads × 200 acquire/release cycles against mocked NVML → zero double-leases, zero orphaned locks.
- **Scene determinism (§3.1):** regenerating any frozen set from its seed range reproduces the committed SHA-256 (this test is also the drift alarm).

### 14.7 Escalation: BLOCKED.md (the remaining human interrupt, made cheap)

```
## [wp5.4] 2026-06-12T03:41Z
Tried: (1) ..., (2) ...      # concrete attempts, with commands
Error/symptom: <paste, trimmed>
Hypothesis: ...
Spec sections consulted: 6.3, 11.7.2
Smallest question whose answer unblocks me: <one question>
```

The contract: the agent never waits on a human — it files and moves to the next satisfiable task; the human answers asynchronously. A well-formed entry converts a 30-minute synchronous debugging session into a 2-minute async answer. CI posts new BLOCKED entries through the §11.7.4 notification hook.

### 14.8 What this layer cannot remove (so it is expected, not a surprise)

(1) WP-boundary reviews — tests prove code does what *tests* say, and §6's algorithmic core (WP5) deserves human eyes against the spec precisely because a coherent misreading passes its own tests; (2) answering BLOCKED entries; (3) hardware-touching steps (§"one-time setup": drivers, boxes, secrets, the prereg tag); (4) launching full-scale runs — rule 9 reserves this for humans deliberately, since an agent that can start multi-day GPU jobs can also start the wrong ones at scale. Net human surface for the build: ~8 reviews + async unblocking — down from continuous supervision.

---

## 15. Code-Level Reference (anti-hallucination layer)

**Status:** additive, like §14 — no experimental parameter changes; logged as Rev 18. **Purpose:** agents deviate at two places — *interface boundaries* (inventing signatures) and *external API calls* (plausible-but-wrong library usage, e.g. mixing `mujoco`, `mujoco-py`, and `dm_control` idioms, or regressing to pre-Gymnasium `step()` conventions). This section removes both surfaces: contracts are given verbatim, every permitted external call is enumerated in a ledger, and the tricky small algorithms are provided as reference implementations to copy, not recall.

**Self-defense rule:** a spec that pins APIs can itself hallucinate. Therefore `tests/test_api_assumptions.py` (§15.4) validates every pinned call at tier 0 on day one — a wrong pin in this section fails loudly in WP1.1, never silently mid-build.

### 15.1 Dependency pinning discipline

The spec pins **floors** (§2.3); **exact versions live only in the committed `uv.lock`**, created once in WP1.1 and never upgraded by the agent (rule 14 below). This avoids the spec inventing version numbers while guaranteeing the agent, CI, and the pipeline resolve identical trees. Any upgrade is an `ISSUES.md` entry with the motivating bug.

### 15.2 `contracts.py` — verbatim (WP1.2 creates exactly this)

```python
"""Single source of truth for all cross-module types.
Spec §15.2. Changing any signature here requires an ISSUES.md entry (§14.4 rule 1)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol
import numpy as np

Vec3 = np.ndarray  # shape (3,), float64
Vec7 = np.ndarray  # shape (7,), float64

SegmentType = Literal["full_demo", "dagger_label", "clean_rollout",
                      "recovery", "correction"]
SEGMENT_CODE: dict[SegmentType, int] = {
    "full_demo": 0, "dagger_label": 1, "clean_rollout": 2,
    "recovery": 3, "correction": 4}

@dataclass(frozen=True)
class Obstacle:
    kind: Literal["box", "sphere"]
    center: tuple[float, float, float]
    size: tuple[float, float, float]   # box: half-extents; sphere: (r, 0.0, 0.0)

@dataclass(frozen=True)
class SceneSpec:
    obstacles: tuple[Obstacle, ...]    # tuples, not lists: hashable + JSON-stable
    goal: tuple[float, float, float]
    q_start: tuple[float, ...]         # length 7
    level: int
    seed: int
    def to_json(self) -> str: ...      # json.dumps(asdict, sort_keys=True) — sorted
    @staticmethod                      # keys make the SHA-256 of §11.7.3 stable
    def from_json(text: str) -> "SceneSpec": ...

@dataclass(frozen=True)
class MjSimState:
    state: np.ndarray                  # mj_getState(..., mjSTATE_INTEGRATION), float64
    step_idx: int                      # control step at capture (ring-buffer key, §6.2)

@dataclass
class JointPath:
    waypoints: np.ndarray              # (M, 7) float64; per-joint spacing ≤ 0.04 rad (§4.1)

@dataclass
class PathTracker:
    idx: int = 0                       # monotonic matched-waypoint index (§4.2)

@dataclass(frozen=True)
class Transition:                      # one HDF5 row (§15.6)
    obs: np.ndarray                    # (79,) float32
    action: np.ndarray                 # (7,)  float32 — expert label, not executed action
    q: np.ndarray                      # (7,)  float32
    ee_pos: np.ndarray                 # (3,)  float32
    done: bool
    segment: SegmentType
    episode_id: int
    level: int

class Expert(Protocol):
    def plan(self, q_start: Vec7, scene: SceneSpec, rng_seed: int, *,
             time_budget_s: float,
             forbid_similar_to: np.ndarray | None = None,   # EE path (P,3); §6.3.3
             warm_start: np.ndarray | None = None,          # remaining waypoints; §6.2 (Rev 19)
             ) -> JointPath | None: ...
    def label(self, q: Vec7, path: JointPath, tracker: PathTracker) -> Vec7: ...
    def ee_path(self, path: JointPath) -> np.ndarray: ...   # (M,3) FK polyline; §6.3.3 (Rev 19)

class Env(Protocol):
    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]: ...
    def step(self, action: Vec7
             ) -> tuple[np.ndarray, float, bool, bool, dict]: ...   # Gymnasium 5-tuple
    def get_state(self) -> MjSimState: ...
    def set_state(self, s: MjSimState) -> np.ndarray: ...           # returns obs
    def min_clearance(self) -> float: ...
    @property
    def scene(self) -> SceneSpec: ...

class Policy(Protocol):
    def act(self, obs: np.ndarray, *, stochastic: bool,
            a_prev: Vec7 | None = None) -> Vec7: ...   # a_prev → §5.2 continuity kernel
    def nll(self, obs_b: "torch.Tensor", act_b: "torch.Tensor") -> "torch.Tensor": ...
```

### 15.3 External API ledger (the ONLY permitted third-party calls)

| Library | Permitted calls (exact) | Notes |
|---|---|---|
| `mujoco` | `MjModel.from_xml_path(p)`, `MjData(m)`, `mj_step(m,d)`, `mj_forward(m,d)`, `mj_resetData(m,d)`, `mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)` (and `mjOBJ_GEOM`/`mjOBJ_BODY`), `d.site_xpos[sid]`, `d.ctrl[:]`, `d.qpos/qvel`, `d.ncon`, `d.contact[i].geom1/.geom2`, `mj_jacSite(m, d, jacp, jacr, sid)` with `jacp = np.zeros((3, m.nv))`, `mj_geomDistance(m, d, g1, g2, distmax, fromto)` → signed float, `fromto = np.zeros(6)` or `None` | **State snapshot — the one true way:** `n = mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)`; `buf = np.empty(n)`; `mj_getState(m, d, buf, mjSTATE_INTEGRATION)`; restore via `mj_setState(...)` then `mj_forward(m, d)`. `mjSTATE_INTEGRATION` is the composite that includes qpos, qvel, act, time, ctrl, mocap, **and `qacc_warmstart`** — do NOT hand-roll snapshots from qpos/qvel (§3.3, pre-mortem #4) |
| `gymnasium` | subclass `gymnasium.Env`; `reset → (obs, info)`; `step → (obs, reward, terminated, truncated, info)` | never the legacy 4-tuple / `done` API |
| `torch` | `nn.Linear/LayerNorm/GELU`, `torch.logsumexp`, `log_softmax`, `distributions.Categorical(logits=…)`, `optim.AdamW`, `optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)`, `nn.utils.clip_grad_norm_` | MDN math hand-rolled per §15.5.2 — do not reach for `MixtureSameFamily` (its event-shape conventions are a classic deviation trap) |
| `h5py` | `File`, `create_dataset(..., maxshape=(None, …), chunks=(4096, …))`, `ds.resize` | schema §15.6; one writer per file (§5.1) |
| `pynvml` | `nvmlInit`, `nvmlDeviceGetCount`, `nvmlDeviceGetHandleByIndex`, `nvmlDeviceGetUtilizationRates(h).gpu`, `nvmlDeviceGetMemoryInfo(h).free/.total` | wrapped solely inside `gpu_alloc.py` |
| `scipy` | `scipy.spatial.cKDTree` (RRT nearest-neighbor) | IK uses the MuJoCo Jacobian directly; no `scipy.optimize` |

**Ledger rule (also rule 11, §15.7):** a call not in this table does not exist. Needing a new one = add it to `test_api_assumptions.py` *first* (an `hasattr`/smoke assertion), then to this table via `ISSUES.md`, then use it.

### 15.4 `tests/test_api_assumptions.py` (tier 0; WP1.1)

```python
def test_mujoco_surface():
    import mujoco, numpy as np
    for fn in ("mj_step", "mj_forward", "mj_geomDistance", "mj_stateSize",
               "mj_getState", "mj_setState", "mj_name2id", "mj_jacSite"):
        assert hasattr(mujoco, fn), fn
    assert hasattr(mujoco.mjtState, "mjSTATE_INTEGRATION")
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
    assert n >= 1                       # signature + composite actually callable

def test_gymnasium_is_five_tuple(): ...  # instantiate a trivial Env subclass, assert arity
def test_torch_surface(): ...            # logsumexp, Categorical(logits=), CosineAnnealingLR
def test_pynvml_optional(): ...          # import-or-skip; assert names if present
```

### 15.5 Reference implementations (copy verbatim, then extend; do not reinvent)

**15.5.1 Discrete Fréchet (§9.1)** — the exact DP; off-by-one variants are the common silent corruption:
```python
def frechet(p: np.ndarray, q: np.ndarray) -> float:
    """p: (N,3), q: (M,3) — arc-length-resampled EE polylines (P=64)."""
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    ca = np.full(d.shape, np.inf)
    ca[0, 0] = d[0, 0]
    for i in range(1, len(p)):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, len(q)):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, len(p)):
        for j in range(1, len(q)):
            ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d[i, j])
    return float(ca[-1, -1])
```

**15.5.2 MDN NLL + kernel sampling (§5.2)** — D = 7, so the constant is 3.5·log 2π:
```python
def mdn_nll(logits, mu, log_std, a):
    """logits (B,K); mu, log_std (B,K,7); a (B,7). Unbounded density — §5.2."""
    log_std = log_std.clamp(-4.0, 0.0)
    z = (a[:, None, :] - mu) / log_std.exp()
    log_comp = -0.5 * (z ** 2).sum(-1) - log_std.sum(-1) - 3.5 * math.log(2 * math.pi)
    return -(torch.logsumexp(torch.log_softmax(logits, -1) + log_comp, -1)).mean()

def mdn_sample(logits, mu, log_std, a_prev=None, sigma_c=0.02, clip=0.05):
    """Single obs: logits (K,), mu/log_std (K,7). a_prev=None ⇒ plain mixture
    (MANDATORY during PPO collection, §8); a_prev given ⇒ §5.2 continuity kernel."""
    log_w = torch.log_softmax(logits, -1)
    if a_prev is not None:
        log_w = log_w - ((mu - a_prev) ** 2).sum(-1) / (2 * sigma_c ** 2)
    k = torch.distributions.Categorical(logits=log_w).sample()
    a = mu[k] + log_std[k].clamp(-4.0, 0.0).exp() * torch.randn(7)
    return a.clamp(-clip, clip)        # clipping at EXECUTION only (§5.2)
```

**15.5.3 Reversal actions (§6.3.2):**
```python
def reversal_actions(q_hist: np.ndarray, clip: float = 0.05) -> np.ndarray:
    """q_hist (T,7): forward-order joints, rewind step → trigger step.
    Returns (T-1,7) actions retracing trigger → rewind when executed in order."""
    return np.clip(np.diff(q_hist[::-1], axis=0), -clip, clip)
```

**15.5.4 Pure-pursuit label (§4.2):**
```python
def label(q, path, tracker, lookahead=3, window=10, clip=0.05):
    wp = path.waypoints
    seg = wp[tracker.idx : tracker.idx + window + 1]
    tracker.idx += int(np.argmin(np.linalg.norm(seg - q, axis=1)))  # monotonic
    if tracker.idx >= len(wp) - 1:
        return np.zeros(7)                                          # terminal hold
    tgt = wp[min(tracker.idx + lookahead, len(wp) - 1)]
    return np.clip(tgt - q, -clip, clip)
```

**15.5.5 Scene-paired bootstrap (§10.3):**
```python
def paired_bootstrap_ci(a, b, rng, n=10_000, alpha=0.05):
    """a, b: (N,) success rates per (scene, seed) cell, identically aligned."""
    diffs = a - b
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    return tuple(np.quantile(diffs[idx].mean(axis=1), [alpha / 2, 1 - alpha / 2]))
```

### 15.6 HDF5 schema — exact layout (§5.1, `data/schema.py`)

| Dataset | dtype | shape | Notes |
|---|---|---|---|
| `obs` | `f4` | `(N, 79)` | normalization applied at train time, never at storage |
| `action` | `f4` | `(N, 7)` | expert label (§6.2) |
| `q`, `ee_pos` | `f4` | `(N, 7)`, `(N, 3)` | |
| `done` | `bool` | `(N,)` | |
| `segment` | `u1` | `(N,)` | `SEGMENT_CODE` (§15.2) |
| `episode_id`, `level` | `i4` | `(N,)` | episode_id globally re-indexed at merge (§5.1) |
| `episodes/scene_json` | `vlen str` | `(E,)` | index = episode_id |
| file attrs | — | — | `schema_version=1`, `created`, `git_hash`, `worker_id` |

Chunking `(4096, dim)`; no compression (local-disk speed > size at ~GB scale); growth via `maxshape=(None, …)` + `resize`. A `schema_version` bump is an `ISSUES.md` event.

### 15.7 Extended reference implementations (the deviation-prone core, written out)

**Why this exists:** §14.8 reserved WP5 for human review because "a coherent misreading passes its own tests." The strongest mitigation is to remove the *reading* step: the orchestration core below is the spec's algorithm transcribed into runnable form. Rule 12 applies — copy, then wrap; never re-derive. Code comments cite the governing section for every non-obvious choice; where reference code simplifies (noted inline), production code keeps the semantics and adds only plumbing.

**Contract amendments (Rev 19, logged):** writing §15.7.7 exposed two missing members in §15.2 — `Expert.plan` gains `warm_start: np.ndarray | None = None` (remaining waypoints for §6.2 homotopy-preserving replans) and `Expert` gains `ee_path(path: JointPath) -> np.ndarray  # (M,3)` (forward-kinematics polyline, needed by §6.3.3's suffix-vs-candidate comparison). The §15.2 block is updated in place.

#### 15.7.1 Scene generation + frozen-slot observation encoding (§3.1, §3.2)

```python
def point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))

def sample_scene(level: int, rng: np.random.Generator, cfg, start_ee: np.ndarray,
                 penetrates_start: "Callable[[SceneSpec], bool]") -> SceneSpec | None:
    lo, hi = cfg.curriculum.levels[level]
    n_obs = int(rng.integers(lo, hi + 1))
    for _ in range(200):                                   # rejection budget per call
        obstacles: list[Obstacle] = []
        while len(obstacles) < n_obs:
            r = rng.uniform(cfg.scene.r_min, cfg.scene.r_max)
            th = rng.uniform(-np.pi, np.pi)
            c = np.array([r * np.cos(th), r * np.sin(th),
                          rng.uniform(cfg.scene.z_min, cfg.scene.z_max)])
            if c[0] <= 0.1:                                # §3.1: in front of robot
                continue
            if rng.random() < 0.5:
                size = tuple(rng.uniform(cfg.scene.box_half_min,
                                         cfg.scene.box_half_max, 3))
                kind = "box"
            else:
                size = (float(rng.uniform(cfg.scene.sph_r_min,
                                          cfg.scene.sph_r_max)), 0.0, 0.0)
                kind = "sphere"
            # §3.1 overlap heuristic: bounding-sphere distance check
            br = max(size) if kind == "box" else size[0]
            ok = all(np.linalg.norm(c - np.array(o.center))
                     > 0.8 * (br + (max(o.size) if o.kind == "box" else o.size[0]))
                     for o in obstacles)
            if ok:
                obstacles.append(Obstacle(kind, tuple(c), size))
        goal = np.array([rng.uniform(cfg.scene.r_min, cfg.scene.r_max), 0.0, 0.0])
        th = rng.uniform(-np.pi, np.pi)
        goal = np.array([abs(goal[0]) * np.cos(th), abs(goal[0]) * np.sin(th),
                         rng.uniform(cfg.scene.z_min, cfg.scene.z_max)])
        if goal[0] <= 0.1 or np.linalg.norm(goal - start_ee) < cfg.scene.goal_min_dist:
            continue
        if any(_inside(goal, o) for o in obstacles):       # §3.1: goal not in obstacle
            continue
        spec = SceneSpec(tuple(obstacles), tuple(goal),
                         q_start=_noisy_home(rng, cfg), level=level,
                         seed=int(rng.integers(2**31)))
        if penetrates_start(spec):                         # §3.1: collision-checked start
            continue
        return spec        # planner-solvability check (§3.1) happens in validate_scene()
    return None

def make_slots(scene: SceneSpec, start_ee: np.ndarray) -> np.ndarray:
    """(8,7) slot array, §3.2. Computed ONCE at reset; order FROZEN for the episode."""
    goal = np.asarray(scene.goal)
    rows = []
    for o in scene.obstacles:
        c = np.asarray(o.center)
        rows.append((point_seg_dist(c, start_ee, goal),
                     np.array([0.0 if o.kind == "box" else 1.0, *c, *o.size])))
    rows.sort(key=lambda t: t[0])                          # nearest-first, once
    slots = np.zeros((8, 7), dtype=np.float64)
    for i, (_, row) in enumerate(rows[:8]):
        slots[i] = row
    return slots

def encode_obs(q, qvel, ee, goal, slots) -> np.ndarray:
    return np.concatenate([q, qvel, ee, goal, goal - ee,
                           slots.ravel()]).astype(np.float32)   # 7+7+3+3+3+56 = 79
```

#### 15.7.2 `PandaReachEnv.step` essentials (§3.2)

```python
def step(self, action: np.ndarray):
    a = np.clip(np.asarray(action, dtype=np.float64), -self.cfg.action_clip,
                self.cfg.action_clip)
    # §3.2: target from MEASURED q, clamped inside joint limits − margin
    q = self.data.qpos[:7].copy()
    q_target = np.clip(q + a, self._jnt_lo + self.cfg.jlimit_margin_rad,
                              self._jnt_hi - self.cfg.jlimit_margin_rad)
    self.data.ctrl[:7] = q_target
    for _ in range(self.cfg.n_substeps):                   # 25 × dt=0.002 → 20 Hz
        mujoco.mj_step(self.model, self.data)
    self._t += 1
    collision = self._contact_violation()                  # exclusion list, §3.2
    clearance = self.min_clearance()                       # incl. table, §3.3
    ee = self.data.site_xpos[self._ee_sid].copy()
    at_goal = np.linalg.norm(ee - self._goal) < self.cfg.success_tol_m
    self._hold = self._hold + 1 if at_goal else 0          # §3.2 sustained-success
    success = self._hold >= self.cfg.success_hold_steps
    terminated = bool(collision or success)
    truncated = bool(self._t >= self.cfg.horizon and not terminated)
    reward = (-0.1 * np.linalg.norm(ee - self._goal) + 10.0 * success
              - 25.0 * collision - 1e-3 * float(a @ a))    # §3.2 (logging in stages 1–3)
    info = {"collision": collision, "min_clearance": clearance, "ee_pos": ee,
            "q": self.data.qpos[:7].copy(), "success": success}
    return self._encode(), reward, terminated, truncated, info

def _contact_violation(self) -> bool:
    for i in range(self.data.ncon):
        c = self.data.contact[i]
        pair = (int(c.geom1), int(c.geom2))
        if pair in self._excluded or pair[::-1] in self._excluded:   # §3.2
            continue
        if pair[0] in self._robot_geoms or pair[1] in self._robot_geoms:
            return True
    return False
```

#### 15.7.3 RRT-Connect core (§4.1)

```python
def edge_free(cc, qa: np.ndarray, qb: np.ndarray) -> bool:
    """Per-joint max |Δq| ≤ edge_check_rad between checks (§4.1)."""
    n = int(np.ceil(np.abs(qb - qa).max() / 0.03)) + 1
    for q in np.linspace(qa, qb, n):
        if not cc.free(q):                  # cc.free: clearance > margin_plan (§4.1)
            return False
    return True

def rrt_connect(q_start, goal_roots, cc, rng, cfg, deadline):
    """goal_roots: list of IK solutions (§15.7.4) — multi-root seeding, §4.1.
    Trees as (nodes: list[np.ndarray], parent: list[int]). Linear vectorized
    nearest is adequate at max_iters=20k (reference simplification, noted §15.7)."""
    Ta = ([q_start.copy()], [-1])
    Tb = ([r.copy() for r in goal_roots], [-1] * len(goal_roots))
    for _ in range(cfg.max_iters):
        if time.monotonic() > deadline:
            return None
        q_rand = (goal_roots[rng.integers(len(goal_roots))]
                  if rng.random() < cfg.goal_bias else _sample_limits(rng, cc))
        q_new = _extend(Ta, q_rand, cc, cfg.step_size)             # one step
        if q_new is not None:
            q_conn = _connect(Tb, q_new, cc, cfg.step_size)        # repeat until blocked
            if q_conn is not None and np.abs(q_conn - q_new).max() < 1e-9:
                path = _trace(Ta, -1)[::-1] + _trace(Tb, -1)       # join at q_new
                path = _shortcut(path, cc, iters=cfg.shortcut_iters, rng=rng)  # §4.1
                return JointPath(_resample(np.array(path), cfg.waypoint_rad))  # ≤0.04 rad
        Ta, Tb = Tb, Ta                                            # swap (bidirectional)
    return None

def _extend(tree, q_to, cc, step):
    nodes, parent = tree
    arr = np.asarray(nodes)
    i = int(np.argmin(((arr - q_to) ** 2).sum(axis=1)))            # vectorized nearest
    d = q_to - nodes[i]
    q_new = nodes[i] + d * min(1.0, step / (np.abs(d).max() + 1e-12))
    if edge_free(cc, nodes[i], q_new):
        nodes.append(q_new); parent.append(i)
        return q_new
    return None
```

#### 15.7.4 Damped-least-squares IK with elbow stratification (§4.1)

```python
def ik_dls(model, data, target, sid, q0, jlo, jhi, iters=100, damp=1e-2):
    q = q0.copy()
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        data.qpos[:7] = q; mujoco.mj_forward(model, data)
        err = target - data.site_xpos[sid]
        if np.linalg.norm(err) < 1e-3:
            inside = np.all(q > jlo + 0.05) and np.all(q < jhi - 0.05)  # §4.1 margin
            return q if inside else None                  # reject on-boundary converge
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), err)
        q = np.clip(q + dq, jlo + 0.05, jhi - 0.05)
    return None

def goal_roots(model, data, target, sid, jlo, jhi, rng, cc, n=16, dedup=0.2):
    """§4.1: stratified elbow seeding — 8 seeds per elbow family (joint-4 nominal
    values −0.6 / −2.4 inside the Panda's joint-4 range), ± uniform noise."""
    sols = []
    for fam_q4 in (-0.6, -2.4):
        for _ in range(n // 2):
            seed = Q_HOME.copy() + rng.uniform(-0.4, 0.4, 7)
            seed[3] = fam_q4 + rng.uniform(-0.2, 0.2)
            q = ik_dls(model, data, target, sid, seed, jlo, jhi)
            if q is None or not cc.free(q):
                continue
            if all(np.linalg.norm(q - s) >= dedup for s in sols):   # §4.1 dedup
                sols.append(q)
    return sols          # log len(sols): the diversity seed must be observable (§4.1)
```

#### 15.7.5 `MDNPolicy` module (§5.2; uses §15.5.2 verbatim)

```python
class MDNPolicy(nn.Module):
    def __init__(self, obs_dim=79, act_dim=7, k=5, hidden=(512, 512, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]; d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, k * (1 + 2 * act_dim))
        self.k, self.act_dim = k, act_dim
        # normalization stats as buffers → persisted inside state_dict (§6.1)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, obs):
        x = (obs - self.obs_mean) / self.obs_std
        out = self.head(self.trunk(x))
        logits, rest = out[..., :self.k], out[..., self.k:]
        mu, log_std = rest.reshape(*out.shape[:-1], self.k, 2, self.act_dim).unbind(-2)
        return logits, mu, log_std

    def nll(self, obs_b, act_b):
        return mdn_nll(*self(obs_b), act_b)                # §15.5.2 — copied, not re-derived

    @torch.no_grad()
    def act(self, obs_np, *, stochastic: bool, a_prev=None) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        logits, mu, log_std = self(obs)
        if stochastic:
            ap = None if a_prev is None else torch.as_tensor(a_prev, dtype=torch.float32)
            a = mdn_sample(logits[0], mu[0], log_std[0], a_prev=ap)   # §15.5.2
        else:                                              # §5.2 deterministic mode
            a = mu[0, int(torch.argmax(logits[0]))].clamp(-0.05, 0.05)
        return a.numpy()
```

#### 15.7.6 BC training loop with normalization + weighting (§5.3, §6.1, §6.2)

```python
def train_bc(ds, cfg, device, rng) -> MDNPolicy:
    policy = MDNPolicy().to(device)
    # §6.1: UNWEIGHTED stats over the raw aggregate, stored as buffers
    policy.obs_mean.copy_(torch.as_tensor(ds.obs.mean(0)))
    policy.obs_std.copy_(torch.as_tensor(ds.obs.std(0) + 1e-6))
    w = np.array([cfg.weights[SEG_NAME[c]] for c in ds.segment])     # §6.2 sampler
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(w))
    loader = DataLoader(ds, batch_size=cfg.batch, sampler=sampler, drop_last=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = (np.inf, None)
    for epoch in range(cfg.epochs):
        for obs_b, act_b in loader:
            loss = policy.nll(obs_b.to(device), act_b.to(device))
            if epoch < cfg.entropy_bonus_epochs:                     # §5.2 stabilizer
                logits = policy(obs_b.to(device))[0]
                p = torch.log_softmax(logits, -1)
                loss = loss - cfg.entropy_bonus * (-(p.exp() * p).sum(-1).mean())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        v = eval_nll(policy, ds.val_loader(), device)                # §5.3 held-out 5%
        if v < best[0]:
            best = (v, copy.deepcopy(policy.state_dict()))
    policy.load_state_dict(best[1])
    return policy                          # checkpoint = state_dict incl. norm buffers
```

#### 15.7.7 THE core: `collect_round` + `run_rac_intervention` (§6.2, §6.3)

```python
def collect_round(env, policy, expert, cfg, rng) -> list[Transition]:
    out, b = [], 0
    while b < cfg.budget:
        scene = sample_valid_scene(env, expert, rng, cfg)            # planner-verified
        obs, info = env.reset(options={"scene": scene})
        path = expert.plan(info["q"], scene, int(rng.integers(2**31)),
                           time_budget_s=cfg.t_label)
        if path is None:
            continue                       # §6.2 fallback: resample scene, NO budget charge
        tracker, ring, rows = PathTracker(), deque(maxlen=cfg.ring_buffer), []
        a_prev, ee_hist, last_replan, steps = None, [], -99, 0
        intervened = False
        for t in range(cfg.horizon):
            q = info["q"]
            ring.append((env.get_state(), info.get("min_clearance", np.inf), q.copy()))
            # §6.2: drift to MATCHED waypoint → warm-start, seed-stable replan
            if (np.linalg.norm(q - path.waypoints[tracker.idx]) > cfg.replan_drift
                    and t - last_replan >= 20):
                new = expert.plan(q, scene, seed=scene.seed,         # same seed (§6.2)
                                  time_budget_s=cfg.t_label,
                                  warm_start=path.waypoints[tracker.idx:])
                last_replan = t
                if new is not None:
                    path, tracker = new, PathTracker()               # else: stale plan, retry in 20
            a_star = expert.label(q, path, tracker)
            rows.append(Transition(obs, a_star.astype(np.float32), q.astype(np.float32),
                                   info.get("ee_pos", np.zeros(3)).astype(np.float32),
                                   False, "dagger_label", episode_id=-1, level=scene.level))
            a = policy.act(obs, stochastic=True, a_prev=a_prev); a_prev = a
            obs, _, term, trunc, info = env.step(a); steps += 1
            ee_hist.append(info["ee_pos"])
            stuck = (len(ee_hist) > cfg.stuck_steps and
                     np.linalg.norm(ee_hist[-1] - ee_hist[-cfg.stuck_steps])
                     < cfg.stuck_eps_m)                              # §6.2 secondary trigger
            if info["collision"] or info["min_clearance"] < cfg.eps_danger or stuck:
                extra, n = run_rac_intervention(env, expert, ring, path, tracker,
                                                scene, cfg, rng, info)
                rows += extra; steps += n; intervened = True
                break                                                # §6.3 Rule 2
            if term or trunc:
                break
        if not intervened:                 # §6.2 dedup: RE-TAG, never re-append
            rows = [dataclasses.replace(r, segment="clean_rollout") for r in rows]
        out += rows
        b += steps                         # §6.1: ALL simulated steps charge the budget
    return out

def run_rac_intervention(env, expert, ring, path, tracker, scene, cfg, rng, info):
    rows, n_steps = [], 0
    snaps = list(ring)
    # §6.3.2 reversal start: collision trigger → restore last PRE-CONTACT snapshot + settle
    if info["collision"]:
        pre = next((s for s in reversed(snaps) if s[1] > 0.0), snaps[0])
        env.set_state(pre[0])
        for _ in range(cfg.settle_steps):
            env.step(np.zeros(7)); n_steps += 1
    # §6.3.1 rewind target: most recent snapshot with clearance ≥ eps_safe, else oldest
    ti = next((i for i in range(len(snaps) - 1, -1, -1)
               if snaps[i][1] >= cfg.eps_safe), 0)
    q_hist = np.stack([s[2] for s in snaps[ti:]])
    q_hist = q_hist[: max(2, len(q_hist) - 3)]             # drop last 3 pre-trigger (§6.3.2)
    obs = env.observe()
    q_now = env.q()
    bridge = np.clip(q_hist[-1] - q_now, -0.05, 0.05)[None]
    rec, ok = [], True
    for a in np.vstack([bridge, reversal_actions(q_hist)]):  # §15.5.3 — EXECUTED LIVE
        rec.append(Transition(obs, a.astype(np.float32), env.q().astype(np.float32),
                              env.ee().astype(np.float32), False, "recovery", -1, scene.level))
        obs, _, term, _, info = env.step(a); n_steps += 1
        if info["collision"]:
            ok = False; break                              # §6.3.2: discard recovery only
    if ok:
        rows += rec
    # §6.3.3 correction: suffix-vs-candidate, fresh seeds, δ_reroute = δ_distinct
    suffix_ee = resample64(expert.ee_path(JointPath(path.waypoints[tracker.idx:])))
    best, best_d = None, -np.inf
    for _ in range(cfg.reroute_attempts):
        cand = expert.plan(info["q"], scene, int(rng.integers(2**31)),
                           time_budget_s=cfg.t_label)
        if cand is None:
            continue
        d = frechet(suffix_ee, resample64(expert.ee_path(cand)))     # §15.5.1
        if d > best_d:
            best, best_d = cand, d
        if d >= cfg.delta_reroute:
            break                                          # distinct homotopy accepted
    if best is None:                                       # §6.2 fallback: recovery-only
        return rows, n_steps                               # logged correction_failed upstream
    tr2 = PathTracker()
    for _ in range(cfg.horizon):
        a_c = expert.label(info["q"], best, tr2)
        rows.append(Transition(obs, a_c.astype(np.float32), info["q"].astype(np.float32),
                               info["ee_pos"].astype(np.float32), False, "correction",
                               -1, scene.level))
        obs, _, term, trunc, info = env.step(a_c); n_steps += 1
        if info["success"]:                                # §5.1: record 5 hold steps
            for _ in range(cfg.settle_steps):
                rows.append(dataclasses.replace(rows[-1], obs=obs,
                                                action=np.zeros(7, np.float32)))
                obs, *_, info = env.step(np.zeros(7)); n_steps += 1
            break
        if info["collision"] or trunc or term:
            break
    return rows, n_steps                                   # caller terminates (Rule 2)
```

#### 15.7.8 Route clustering — hand-rolled complete linkage (§9.2; stays inside the §15.3 ledger)

```python
def count_routes(trajs: list[np.ndarray], delta: float) -> int:
    """trajs: successful EE paths, each resampled to (64,3). n ≤ 20 → O(n³) is fine."""
    n = len(trajs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = frechet(trajs[i], trajs[j])          # §15.5.1
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, bi, bj = np.inf, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = max(D[a, b] for a in clusters[i] for b in clusters[j])  # complete link
                if d < best:
                    best, bi, bj = d, i, j
        if best >= delta:
            break
        clusters[bi] += clusters.pop(bj)
    return len(clusters)
```

#### 15.7.9 `GpuAllocator.acquire` (§10.5.2)

```python
def acquire(self, mem_required_gb=2.0, exclusive=False, timeout_s=None):
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        with open(self.lock_dir / "dir.lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)                 # §10.5.2: atomic acquire-check-write
            self._reap_stale()                             # delete leases of dead PIDs
            scores = []
            for gid in self._candidates():                 # honors CUDA_VISIBLE_DEVICES
                h = pynvml.nvmlDeviceGetHandleByIndex(gid)
                util = np.mean([pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                                for _ in range(3) if not time.sleep(0.2)])  # 3×200 ms
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                leases = len(list((self.lock_dir / f"gpu{gid}").glob("*.json")))
                if mem.free / 2**30 < mem_required_gb + 0.5:        continue
                if leases >= (1 if exclusive else self.jobs_per_gpu): continue
                if exclusive and leases > 0:                          continue
                scores.append((self.util_w * util / 100
                               + self.mem_w * (1 - mem.free / mem.total), gid))
            if scores:
                gid = min(scores)[1]
                lease = self.lock_dir / f"gpu{gid}" / f"{os.getpid()}.json"
                lease.parent.mkdir(exist_ok=True)
                lease.write_text(json.dumps({"pid": os.getpid(),
                                             "mem_gb": mem_required_gb,
                                             "ts": time.time()}))
                return GpuLease(physical_id=gid, lock_path=lease)
        if deadline and time.monotonic() > deadline:
            return None
        time.sleep(10)                                     # §10.5.2 block-poll
```

### 15.8 Additional `.cursor/rules` (append to §14.4)

```
11. External calls: §15.3 ledger or it does not exist. New call ⇒ assumptions
    test first, ledger via ISSUES.md second, usage third.
12. §15.5 reference implementations are copied verbatim as the starting point;
    extensions wrap them, never re-derive them.
13. If §15 code and body prose conflict, STOP: ISSUES.md entry. Neither wins
    by default — a contradiction means one of them is wrong.
14. uv.lock is immutable to agents. Dependency changes are human acts via
    ISSUES.md (§15.1).
```

---

## Appendix B — Golden configuration values (machine-checked source of truth)

`scripts/check_spec_constants.py` asserts the live configs equal these values; CI fails on divergence. (Transcribed from the body; section refs in comments.)

```yaml
env:        {ctrl_hz: 20, n_substeps: 25, horizon: 300, action_clip: 0.05,    # §3.2
             success_tol_m: 0.05, success_hold_steps: 5, jlimit_margin_rad: 0.02,
             max_obstacles: 8, obs_dim: 79}
scene:      {r_min: 0.25, r_max: 0.75, z_min: 0.05, z_max: 0.7,               # §3.1
             box_half_min: 0.03, box_half_max: 0.10, sph_r_min: 0.04, sph_r_max: 0.10,
             goal_min_dist: 0.35, start_noise_rad: 0.05}
expert:     {step_size: 0.15, goal_bias: 0.10, max_iters: 20000,              # §4.1
             t_validate_s: 10.0, t_label_s: 3.0, margin_plan_m: 0.03,
             edge_check_rad: 0.03, ik_restarts: 16, ik_dedup_rad: 0.2,
             shortcut_iters: 100, waypoint_rad: 0.04, lookahead: 3}
policy:     {trunk: [512, 512, 256], k_mix: 5, logstd_clamp: [-4, 0],         # §5.2
             entropy_bonus: 1.0e-3, entropy_bonus_epochs: 20, sigma_c: 0.02,
             dither_escalate_median: 10, chunk_k: 4}
bc:         {n_demos: 2000, lr: 3.0e-4, batch: 1024, epochs: 200,             # §5.1/5.3
             grad_clip: 1.0, val_frac: 0.05, gate_soft: 0.25, gate_target: 0.40}
dagger_rac: {rounds: 6, budget: 40000, eps_danger_m: 0.02, eps_safe_m: 0.10,  # §6.4
             ring_buffer: 40, stuck_steps: 60, stuck_eps_m: 0.01,
             replan_drift_rad: 0.25, settle_steps: 5, retrain_epochs: 100,
             reroute_attempts: 5,
             weights: {full_demo: 1.0, dagger_label: 1.0, clean_rollout: 0.5,
                       recovery: 1.5, correction: 1.5}}
curriculum: {levels: [[2,3],[3,4],[4,6],[6,8]], promote: 0.70, demote: 0.30,  # §7
             offlevel_cap: 0.5, synth_frac: 0.2}
ppo:        {clip: 0.2, gae_lambda: 0.95, gamma: 0.99, lr: 1.0e-4,            # §8
             n_envs: 8, kl_anchor: 0.5, entropy_coef: 0.005,
             pool_scenes: 256, archive_m: 16, beta: 2.0, rdiv_cap: 0.5,
             beta_halvings_max: 3}
diversity:  {resample_pts: 64, n_rollouts: 20, validity_min: 8,               # §9
             rollout_seeds: "0..19", recovery_merge_steps: 10}
eval:       {val_per_level: 100, test_per_level: 200, routes_scenes: 50,      # §10
             bootstrap_n: 10000, seeds: [0, 1, 2], mde_pts: 10,
             unseen_half_min: 0.08, unseen_half_max: 0.14, unseen_reject_max: 0.5}
compute:    {jobs_per_gpu: 2, mem_required_gb: 2.0, util_w: 0.5, mem_w: 0.5,  # §10.5
             nvml_samples: 3, nvml_interval_ms: 200, watchdog_min: 30,
             dep_timeout_h: 48, disk_min_gb: 50}
# Invariants asserted at setup (§11.7.3): margin_plan_m > eps_danger_m;
# eps_safe_m > margin_plan_m; success_hold_steps == settle_steps == 5;
# delta_reroute == calibrated delta_distinct; identical calibration hash grid-wide.
```

---

## Appendix A — Revision Changelog

### Rev 1 — correctness of core mechanics
- §6.3: recovery segments are now generated by *executing* the reversed action sequence in sim from the trigger state (genuine observations incl. velocities); replay-in-reverse of forward snapshots is explicitly forbidden (velocity-sign mismatch). Correction now starts from the executed reversal's endpoint.
- §3.2: success redefined as goal tolerance sustained 5 steps before termination; removed redundant zero-collision clause (collision is terminal).
- §4.2: expert labeler given a monotonic PathTracker to prevent backwards nearest-point jumps on self-proximal paths.
- §3.3: clearance computation given an explicit cost-bounding strategy (geom budget, broad-phase prune, distmax, ≤0.5 ms/step perf assertion).
- §5.3: BC exit criterion converted to a soft gate (proceed ≥25%) with a three-step diagnostic ladder to prevent WP deadlock.

### Rev 2 — multimodality made structural
- §5.2: policy head changed from unimodal Gaussian to a 5-component MDN with mixture NLL, stabilizers, per-episode component-coherent sampling, and defined stochastic/deterministic modes. A unimodal head mode-averages across route homotopies — directly contradicting the research question — so it is demoted to an explicit ablation (`head: gaussian`) whose failure becomes a reported result.
- §3.2: obstacle slot ordering frozen at reset (was per-step nearest-first, which swapped slot contents mid-episode and made observations discontinuous).
- §6.2: clean-rollout double-counting removed (re-tag, don't re-append); per-segment-type sampling weights defined (recovery/correction up-weighted 1.5×, clean rollouts 0.5×) with per-retrain composition logging.
- §9.2: route-diversity rollouts explicitly use the MDN stochastic mode.

### Rev 3 — expert oracle made precise
- §4.1: edge-collision discretization pinned to max single-joint |Δq| ≤ 0.03 rad; planning uses a 5 mm inflated clearance margin.
- §4.1: IK made joint-limit aware with stratified elbow-up/elbow-down seeding and solution dedup; number of distinct goal roots logged (this is the diversity seed — its collapse must be observable).
- §4.1: planner time budget split into validate (10 s, one-off) vs label (3 s, in-loop) profiles.
- §4.2 acceptance: vague random-scene diversity test replaced by a constructed blocking-obstacle scene family with an 80% pass bar.

### Rev 4 — evaluation rigor
- §9.1: `δ_distinct` now calibrated from planner-generated same/cross-homotopy pair distributions on constructed scenes, with a ±33% sensitivity sweep; EE-space metric caveats made explicit.
- §9.2: validity gate (≥8/20 successes else NA) added; routes-per-success added to deconfound route counts from success rates; gate pass-rate itself reported.
- §9.3: correlation given a unit of analysis (per-scene, within policy, Spearman, difficulty-stratified) and explicitly labeled observational, with the causal claim deferred to the RaC-NoReroute ablation.
- §10.2: eval_unseen scenes planner-verified; pre-registered re-tune rule if >50% rejection.
- §10.3: scene-paired bootstrap CIs (10k resamples) required for any "better than" claim; per-seed scatter mandated.

### Rev 5 — engineering and compute honesty
- §5.1: shard-per-worker HDF5 + single-process merge mandated for all collection stages (HDF5 concurrent writes corrupt).
- §6.1: normalization stats pinned (unweighted over raw aggregate, serialized in checkpoint, never recomputed at eval); RaC budget accounting restated for the automated setting (all sim steps charge the budget; storage keeps everything).
- §10.4: wall clock re-budgeted including round-end and curriculum evals — honest estimate is 8–9 days for the full grid, with pre-committed mitigation order.
- §11: WP1 now includes a 60-second smoke script and CPU CI; artifact-retention rules added (keep per-round Δd_k deltas, append-only run dirs, showcase videos).

### Rev 6 — scientific framing
- §1.4 (new): pre-registered hypotheses H1–H3 with decision rules bound to the scene-paired CI machinery; explicit acceptable-null framing for H2.
- §10.1: added **RaC-NoReroute** — the ablation that separates the project's novel claim (route-diversity forcing) from RaC's known recovery effect — and a densification-off curriculum ablation; resolves the forward reference introduced in Rev 4.
- §8: Stage 4 given a three-condition go/no-go gate instead of a vibe ("after stages 1–3 are solid").
- §12: risk table refreshed — MDN component collapse (with IMLE fallback), grid-overrun with pre-committed cut order, H2-null risk; δ_distinct row updated to the calibrated procedure.

### Rev 7 — leakage firewall
- §10.2 restructured into a three-tier scene hierarchy (train / validation / test) with the rule that all monitoring, curriculum promotion, and checkpoint selection consume validation sets only, and test sets are touched exactly once per (condition, seed) with an audit stamp.
- §6.1 and §7 rewired to validation sets; promotion probes are now frozen sets (comparable across conditions/seeds).
- Training seeds pinned to {0, 1, 2}; hypothesis and correlation sections renamed to test_unseen.

### Rev 8 — RaC machinery consistency
- §6.3.3: fixed the δ_reroute units bug (0.5 m → calibrated δ_distinct) and replaced the ill-posed plan-vs-failed-prefix Fréchet with a shared-endpoint suffix-vs-candidate comparison; fallback-acceptance rate now logged.
- §6.3.2: collision-triggered reversals now restore the last pre-contact snapshot + settle steps before executing (reversal from an in-contact, forward-momentum state was ill-posed).
- §6.2: drift replans are warm-started and seed-stable to preserve homotopy (DAgger label whipsaw prevention), with a flip detector; planner-failure fallbacks specified for all three call sites.

### Rev 9 — MDN honesty
- §5.2: the per-episode Gumbel coherence trick removed — it silently assumed component-identity stability across states, which MDNs do not have. Replaced with action-continuity component selection (continuity kernel on component means vs previous action) plus a logged dithering diagnostic with an escalation threshold to the IMLE fallback.
- §5.2: likelihood semantics pinned — unbounded-space NLL, clipping at execution only (tanh-in-likelihood + clipping was incoherent).
- §8: PPO on a mixture head specified properly — exact log-prob for the ratio, sampled estimators for entropy bonus and the KL anchor, with adjusted coefficients.

### Rev 10 — termination semantics in the data
- §4.2: expert labeler given a terminal-hold rule (zero action once the tracker saturates).
- §5.1/§6.3: every successful demo and correction now records 5 zero-action hold steps at the goal — the sustained-success criterion of Rev 1 was otherwise unlearnable from data that ends at arrival.
- §3.2: contact-exclusion list added (base↔table and Menagerie adjacent-link pairs), defined once in config and shared by env + planner collision checker; without it, every episode terminates at t=0.
- §3.1: floating obstacles acknowledged as a deliberate abstraction with the homotopy rationale.

### Rev 11 — stage 4 made implementable
- §8: training moved to a frozen 256-scene pool, making per-scene archives exact (replaces the unimplementable "coarse layout features" bucketing); diversity bonus clipped and given an explicit variance-handling recipe (scene-index critic embedding, β back-off rule, empty-archive case defined).
- §10.3: planner route-diversity ceiling added as a mandatory reference row — policy #routes numbers are meaningless without the geometric/expert upper bound.

### Rev 12 — wrapper and consistency sweep
- §11.5 (new): deliverables checklist and a `make reproduce` chain with a < 1 h CPU smoke profile that CI runs nightly and that WP5 implements first.
- §11.6 (new): consolidated limitations register (obstacle cap, EE-metric blindness, planner ceiling, automated-vs-human RaC, abstraction list, observational H3) for verbatim inclusion in the report.
- §6.4: settle-step and continuity-kernel constants swept into the hyperparameter table (per the spec's own no-silent-magic-numbers rule); WP5 gate pinned to `val_L0`.

### Rev 13 — multi-GPU parallelization (feature addition, user-requested)
- New §10.5: workload-profile analysis concluding the correct unit of GPU scheduling is the whole run (small MDN-MLP cannot saturate one device; the 15-run grid is the bottleneck); NVML-based `GpuAllocator` with utilization+memory scoring (3×200 ms sampling), flock-atomic JSON leasing with stale-PID reaping, CUDA_VISIBLE_DEVICES masking so library code is device-index-free, OOM re-enqueue; grid launcher with cores-honest concurrency cap min(jobs_per_gpu×GPUs, cores/8), priority-ordered drain, idempotent resume, per-run telemetry; explicit device-placement table (collection stays on CPU deliberately); optional single-host DDP off by default with stated rationale; determinism contract (placement never affects RNG; placement-invariance test at metric tolerance); revised wall clock ≈ 2.7 days on a 32-core/4-GPU box.
- §2 layout, dependency table (pynvml), WP1, §10.4 mitigations, §11.5 reproduce path, and the risk table updated accordingly.

### Rev 14 — zero-touch orchestration (feature addition, user-requested)
- New §11.7: persisted DAG state machine (`make pipeline`) with idempotent resume and config-hash downstream invalidation; all prose judgment calls converted to executable gates (G-CAL, G-BC, G-DATA, G-REGRESS, G-DITHER, G-PPO, G-β) evaluated on the validation tier; self-running δ_distinct calibration and hash-verified scene-set generation with a deterministic unseen-set re-tune ladder; watchdog + retry caps + notifications + containerized provisioning; automatic checkpoint selection; auto-computed hypothesis verdicts and report assets. Residual manual surface enumerated (report prose, optional video QA, pressing go).
- Bug found by the automation pass itself: §8's go/no-go was keyed to the test-set CI, violating the Rev-7 leakage firewall — gates can only be coded against data code may read; G-PPO now uses val_unseen.
- §5.3, §5.2, §8, §9.1, §10.2 rewired from human runbooks to gate references; WP5 builds the orchestrator skeleton; deliverables and risk table extended.

### Rev 15 — pre-mortem hardening (user-requested bulletproofing)
- Found and fixed a load-bearing inconsistency: planning margin (0.005 m) below the near-failure trigger (0.02 m) would have caused spurious RaC interventions throughout training and inflated the §9.3 recovery statistic underpinning H3; margin raised to 0.03 m with an asserted coupling constraint, and scene acceptance now planner-verifies under the full margin.
- Table added to the clearance/trigger geom set; q_target clamped to joint limits; MjSimState fields enumerated (incl. qacc_warmstart).
- Gate couplings hardened: G-REGRESS discards failed-round data before retry; IMLE branch win auto-resolves G-PPO to NO-GO (no tractable likelihood).
- Statistics hardened: bootstrap pairing unit pinned to (scene, seed); route-diversity comparisons restricted to gate-passing scene intersections; test-set burn protocol with reserved v2/v3 seed ranges.
- Ops hardened: config-invariant validation + disk preflight in setup; EGL→OSMesa with non-fatal rendering; NFS locality rules for HDF5 and GPU locks; offline-first logging; nightly run-dir sync; mandatory prereg git tag recorded in pipeline state.
- New §12.5: pre-mortem register mapping 15 pre-empted surprises to their neutralizations, plus the named residual unknowns.

### Rev 16 — second pre-mortem pass (interaction seams) + spec freeze
- Catch 1: §5.2 continuity kernel scoped as an inference-time wrapper and disabled during PPO data collection — acting through the kernel while computing ratios from the plain mixture density would silently bias every PPO gradient (history-dependent behavior distribution ≠ π(a|s)).
- Catch 2: δ_distinct calibration converted from per-run stage to grid-level singleton with fixed seed and identical-hash invariant — per-run RNG would have compared conditions at different thresholds and varied δ_reroute (the treatment) across conditions.
- Catch 3: G-PPO's read of the RaC-NoReroute condition's validation results declared as a cross-run dependency with a launcher barrier and dep_timeout→NO-GO — previously an undeclared inter-DAG edge that deadlocks or reads missing files.
- Also: pre-registered MDE (10 pts) with three-way CONFIRMED/REFUTED/UNDERPOWERED rule; Holm–Bonferroni across H1/H2; drift defined to the tracker-matched waypoint (self-heals stuck trackers); fixed rollout-seed lists for #routes (pairs on (scene, rollout-seed)); curriculum level-cap rephrased to survive demotions; host-driver/CUDA-runtime compat check in setup.
- Pre-mortem register extended to 20 rows; **spec frozen at v3.4** with an ISSUES.md deviation-log process — further hardening passes are declared net-negative.

### Rev 17 — agent implementation harness (additive; experimental design freeze intact)
- New §14: task-file schema with executable done_when commands and per-task budgets (WP1 fully expanded as canonical example); self-planning rule (WP k drafts wp{k+1}.yaml for boundary sign-off); verbatim .cursor/rules with ten guardrails (contracts-as-law, literal-lint, frozen-asset protection, test-first, never-weaken-tests, BLOCKED-don't-hack, smoke-profile-only for agents, ISSUES.md for spec contradictions); five-tier verification ladder keeping the agent's inner loop < 3 min; golden/property test suite making correctness self-verifiable without human review (Fréchet analytic cases, fk∘ik round-trip, reversal algebra, bootstrap coverage with planted effects, allocator hammer, scene-hash determinism); BLOCKED.md escalation format converting synchronous debugging into async answers; honest residual list (~8 WP reviews + unblocking + hardware + launch authority reserved to humans).
- New Appendix B: golden configuration values as machine-checked source of truth (check_spec_constants.py), with the §11.7.3 invariants restated.
- ISSUES.md instituted with ISSUE-001 (cross-host dependency stamps) per the §12.5 freeze process.

### Rev 18 — code-level anti-hallucination layer (additive)
- New §15: verbatim `contracts.py` (all cross-module dataclasses + Protocols, Gymnasium 5-tuple pinned, kernel-aware Policy.act signature); external API ledger enumerating every permitted third-party call with exact signatures — notably the one-true-way MuJoCo state snapshot via mjSTATE_INTEGRATION (covers qacc_warmstart, closing pre-mortem #4 at the API level) and an explicit ban on torch MixtureSameFamily (event-shape deviation trap); `test_api_assumptions.py` at tier 0 so any wrong pin in the spec itself fails on day one rather than becoming agent improvisation mid-build; reference implementations to copy verbatim (discrete Fréchet DP, MDN NLL with the 3.5·log2π constant + kernel-aware sampling with the PPO plain-mixture mandate inline, reversal actions as a one-liner, monotonic pure-pursuit with terminal hold, paired bootstrap); exact HDF5 schema (dtypes, shapes, chunking, attrs, segment enum); dependency-pinning discipline (spec pins floors, committed uv.lock pins exacts, agents may not upgrade); rules 11–14 appended to .cursor/rules (ledger-or-it-doesn't-exist, copy-don't-rederive, code/prose conflicts STOP to ISSUES.md, lockfile immutability).

### Rev 19 — extended reference implementations (additive)
- New §15.7 (rules section renumbered to §15.8): full reference code for the nine deviation-prone modules — scene generation + frozen-slot encoding, env.step with clamping/hold/exclusion semantics, RRT-Connect core with per-joint edge discretization and multi-root goal trees, DLS IK with elbow-stratified seeding and boundary rejection, the MDNPolicy module with norm-stat buffers persisted in state_dict, the BC loop with unweighted stats + segment-weighted sampler + entropy-bonus window, and — the centerpiece — collect_round + run_rac_intervention transcribing §6.2/§6.3 end to end (warm-start drift replans, budget accounting, retag-don't-reappend, pre-contact restore + settle, bridge + live reversal execution with discard-on-collision, suffix-vs-candidate δ_reroute rejection, hold-step recording, Rule-2 termination at the caller); hand-rolled complete-linkage route clustering (stays inside the API ledger); GpuAllocator.acquire with flock-atomic lease writes.
- Contract amendments exposed by writing the core: Expert.plan gains warm_start; Expert gains ee_path (FK polyline). §15.2 updated in place.
