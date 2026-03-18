"""
Logging Component — records all test executions in structured JSONL.

Each record includes: timestamp, run_id, event_type, and event-specific
payload (prompts, responses, metadata). Enables reproducibility and
post-hoc oracle evaluation.

Output format: one JSON object per line (JSONL), UTF-8 encoded.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunLogger:
    run_id: str
    path: str

    def write(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "ts": utc_now_iso(),
            "run_id": self.run_id,
            "event_type": event_type,
            **payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def new_run_logger( out_dir: str = "logs", prefix: str = "run", meta: Optional[Dict[str, Any]] = None,) -> RunLogger:
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path = os.path.join(out_dir, f"{run_id}.jsonl")

    logger = RunLogger(run_id=run_id, path=path)
    logger.write("run_start", {"meta": meta or {}})
    return logger


def safe_preview(obj: Any, max_len: int = 600) -> str:
    s = str(obj)
    return s if len(s) <= max_len else s[:max_len] + "…"

