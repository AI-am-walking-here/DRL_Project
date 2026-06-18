"""Standalone shard merge (§5.1)."""

from __future__ import annotations

from pathlib import Path

from robot_routes.data.schema import merge_shards

__all__ = ["merge_shards"]


def main(shards: list[Path], out: Path) -> None:
    merge_shards(shards, out)
