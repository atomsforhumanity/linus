"""The ROBUST molecule records for the reproducers: bench/robust/molecules/*.jsonl, plus
molecules/internal/*.jsonl where present (GMTKN55 and tmQM, not in the public release)."""

from __future__ import annotations

import json
from pathlib import Path

MOLDIR = Path(__file__).resolve().parents[1] / "molecules"


def load() -> list[dict]:
    recs = []
    for f in sorted(MOLDIR.glob("*.jsonl")) + sorted((MOLDIR / "internal").glob("*.jsonl")):
        recs += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    return recs
