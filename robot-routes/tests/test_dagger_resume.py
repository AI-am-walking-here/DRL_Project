"""DAgger round resume tests."""

from __future__ import annotations

from pathlib import Path

from robot_routes.pipeline.dagger_resume import detect_dagger_resume, save_round_stats


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_fresh_start(tmp_path: Path) -> None:
    out = tmp_path / "dagger"
    out.mkdir()
    bc = tmp_path / "demos.h5"
    ckpt = tmp_path / "bc.pt"
    bc.write_bytes(b"")
    ckpt.write_bytes(b"")
    st = detect_dagger_resume(out, bc, ckpt, 6)
    assert st.start_round == 0
    assert not st.resumed
    assert st.policy_ckpt == ckpt


def test_resume_after_round_3(tmp_path: Path) -> None:
    out = tmp_path / "dagger"
    out.mkdir()
    bc = tmp_path / "demos.h5"
    bc.write_bytes(b"")
    for k in range(4):
        _touch(out / f"ckpt_{k}" / "best.pt")
        _touch(out / f"merged_{k}.h5")
    save_round_stats(out, [0.1, 0.2, 0.25, 0.3])
    st = detect_dagger_resume(out, bc, tmp_path / "bc.pt", 6)
    assert st.start_round == 4
    assert st.resumed
    assert st.completed_through == 3
    assert st.prev_sr == 0.3
    assert st.policy_ckpt == out / "ckpt_3" / "best.pt"
    assert st.merged_base == out / "merged_3.h5"
    assert len(st.round_stats) == 4


def test_incomplete_round_keeps_shard_for_reuse(tmp_path: Path) -> None:
    out = tmp_path / "dagger"
    out.mkdir()
    bc = tmp_path / "demos.h5"
    bc.write_bytes(b"")
    _touch(out / "ckpt_2" / "best.pt")
    _touch(out / "merged_2.h5")
    _touch(out / "round_3.h5")
    save_round_stats(out, [0.1, 0.2, 0.25])
    st = detect_dagger_resume(out, bc, tmp_path / "bc.pt", 6)
    assert st.start_round == 3
    assert (out / "round_3.h5").exists()
    assert st.reuse_shard
    assert st.policy_ckpt == out / "ckpt_2" / "best.pt"


def test_round0_shard_reuse(tmp_path: Path) -> None:
    out = tmp_path / "dagger"
    out.mkdir()
    bc = tmp_path / "demos.h5"
    bc.write_bytes(b"")
    ckpt = tmp_path / "bc.pt"
    ckpt.write_bytes(b"x")
    _touch(out / "round_0.h5")
    st = detect_dagger_resume(out, bc, ckpt, 6)
    assert st.start_round == 0
    assert st.resumed
    assert st.reuse_shard
    assert (out / "round_0.h5").exists()
    assert st.policy_ckpt == ckpt
    assert st.merged_base == bc


def test_all_complete(tmp_path: Path) -> None:
    out = tmp_path / "dagger"
    out.mkdir()
    bc = tmp_path / "demos.h5"
    bc.write_bytes(b"")
    for k in range(6):
        _touch(out / f"ckpt_{k}" / "best.pt")
        _touch(out / f"merged_{k}.h5")
    st = detect_dagger_resume(out, bc, tmp_path / "bc.pt", 6)
    assert st.start_round == 6
    assert st.completed_through == 5
