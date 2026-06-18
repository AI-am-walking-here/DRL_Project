# Spec deviation log (§12.5 freeze process)

## ISSUE-001: Cross-host dependency stamps
- **Section:** §11.7.2 G-PPO
- **Status:** open (by design)
- **Note:** Single-host grid launcher implements dependency waits; multi-host stamp sync deferred per spec.

## ISSUE-002: uv.lock not committed
- **Section:** §15.1
- **Status:** open (human act)
- **Note:** Agents must not modify lockfile. Run `uv lock` locally and commit when pinning is finalized.

## ISSUE-003: Full-scale scene sets
- **Section:** §10.2
- **Status:** smoke committed; full pending human launch
- **Note:** Run `make scene-sets PROFILE=full` before research grid (~hours CPU). Not automated here.

## ISSUE-004: Live calibration at full scale
- **Section:** §11.7.3
- **Status:** precommitted `calibration/delta.json`; live via `make calibrate`
- **Note:** Re-run live calibration before full grid if environment changes.

## ISSUE-005: check_constants warn-only
- **Section:** §14.4 rule 2
- **Status:** accepted pragmatic deviation

## ISSUE-006: Residual deferrals
- **Section:** §11.7.4
- **Status:** closed for code; open for ops
- **Resolved in code:** G-BC ladder, G-DATA/G-REGRESS, G-DITHER→IMLE, G-PPO, watchdog, notify, prereg gate, shard-per-worker, cross-condition verdicts, eval ckpt selection, showcase videos, state merge/reconcile, placement-invariance test (`tests/test_integration.py`), WANDB offline + HDF5 locking in Makefile
- **Ops-only (not code):** nightly rsync to shared storage, `uv.lock` commit, `make scene-sets PROFILE=full`, `git tag prereg-v1` before full grid
