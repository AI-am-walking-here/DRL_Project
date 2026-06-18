"""Pipeline notifications (§11.7.4)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def notify(run_dir: Path | None, event: str, **kwargs: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    line = json.dumps(payload)
    print(f"[notify] {line}")
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "events.log").open("a") as f:
            f.write(line + "\n")
    url = os.environ.get("PIPELINE_WEBHOOK_URL", "")
    if not url:
        return
    req = urllib.request.Request(
        url,
        data=json.dumps({"text": line}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except (urllib.error.URLError, TimeoutError):
        pass
