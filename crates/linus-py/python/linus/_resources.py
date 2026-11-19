"""What a request needs from the machine: qccompute's ``Plan.resources`` from the plan's
estimates (made by the library: peak host memory and device memory, each with its error band;
``linus --plan`` prints the same), and the measured peak host memory of a request
(``ExecutionInfo.peak_memory_gib``).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")
_lock = threading.Lock()
_active = 0
_started = 0


def _vm_kib(field: str) -> int | None:
    try:
        m = re.search(rf"^{field}:\s+(\d+) kB", _STATUS.read_text(), re.M)
    except OSError:
        return None
    return int(m.group(1)) if m else None


class PeakMemory:
    """The process's peak resident memory (GiB) during a block, as ``gib`` after it: Linux's
    high-water mark (``VmHWM``), reset to the current resident size at the block's start
    (``/proc/self/clear_refs``). None when it cannot be measured: another request of this
    process overlapped the block (sessions running in threads), or the platform has no such
    counters."""

    def __init__(self) -> None:
        self.gib: float | None = None
        self._alone = False
        self._ticket = 0

    def __enter__(self) -> PeakMemory:
        global _active, _started
        with _lock:
            _active += 1
            _started += 1
            self._ticket = _started
            self._alone = _active == 1
            if self._alone:
                try:
                    _CLEAR_REFS.write_text("5")
                except OSError:
                    self._alone = False
        return self

    def __exit__(self, *exc: Any) -> None:
        global _active
        with _lock:
            alone = self._alone and _started == self._ticket
            _active -= 1
            if alone:
                kib = _vm_kib("VmHWM")
                self.gib = None if kib is None else kib / (1 << 20)


def resources(estimates: dict[str, Any]) -> dict[str, Any]:
    """qccompute's ``Plan.resources`` of a plan's `estimates` (the plan report's, made by the
    library: ``linus --plan`` prints the same): the peak host memory, and with a device one GPU
    and its peak device memory. No threads: the runner gives a calculation its count, and linus
    caps a count given without the ``threads`` keyword itself (`linus.session.apply_grant`; qc*
    decision 16)."""
    device = estimates.get("device_memory_gib")
    return {"memory_gib": estimates["memory_gib"]["value"], "gpus": 1 if device else 0,
            "device_memory_gib": device["value"] if device else None}
