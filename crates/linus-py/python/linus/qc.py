"""linus as a program for runners (qccompute's Program protocol, structurally: linus never
imports qccompute). `program` is the object the package registers as the entry point ``linus``
of `ENTRY_POINT_GROUP`, so qccompute finds it by name, and its serve loop hosts it in a worker
process: ``python -m qccompute.serve linus`` (docs/python.md, "Many calculations").

It adds nothing to the calculations: its ``compute`` is `linus.compute` and its
``open_session`` opens a `linus.Session`, with the runner's ``ctx`` (the calculation's share of
threads, memory and GPUs). What it adds is what a runner needs before it runs anything: the
capabilities, the settings and record schemas, the plan, and `LinusProgram.numerics`, the
numerics identity a result store records with every run of linus (decision 29).

>>> from linus.qc import program
>>> program.capabilities()["methods"]["b3lyp"]
['energy', 'gradient', 'hessian']
>>> out = program.compute(inp, ctx)                  # = linus.compute(inp, ctx); raises on failure
>>> with program.open_session(inp, ctx) as s:        # = linus.Session(inp, ctx=ctx)
...     s.compute(x)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from qcdata import CalcType, ProgramInput, ProgramOutput

from . import _linus
from .session import CALCTYPES, PROGRAM, Session, compute

__all__ = ["ENTRY_POINT_GROUP", "LinusProgram", "program"]

#: The entry point group runners find AFH's programs in (qccompute's registry); the package's
#: entry point ``linus`` there is `program`.
ENTRY_POINT_GROUP = "qccompute.programs"


class LinusProgram:
    """linus for a runner: capabilities, settings and record schemas, plans, calculations and
    sessions.

    Attributes:
        name: ``linus``.
    """

    name = PROGRAM

    def numerics(self, spec: Any = None) -> str:
        """The numerics identity of linus's results (decision 29): `linus.NUMERICS_EPOCH` as text,
        the same for every input and method. Results of one input are bitwise the same while it
        stays (on one device class; a runner's grant may change a resolved choice at round-off,
        which the output's ``resolved`` records), and it changes with any release that can move
        them.

        Args:
            spec: The input or subprogram spec asked about; linus's epoch covers every method.
        """
        return str(_linus.NUMERICS_EPOCH)

    def version(self) -> str:
        """linus's version."""
        return _linus.__version__

    def capabilities(self) -> dict[str, Any]:
        """What linus computes and how it may run, as a plain document (the worker's hello line
        carries the same): every method with its calctypes, sessions, in process, separate
        sessions concurrent in threads; the numerics epoch in ``extras``. Whether a calculation
        uses a GPU is its plan's (``resources.gpus``), not a capability."""
        calctypes = sorted(c.value for c in CALCTYPES)
        return {
            "calctypes": calctypes,
            "methods": {m: calctypes for m in _linus.available_functionals()} | {"hf": calctypes},
            "sessions": True,
            "batches": False,
            "in_process": True,
            "thread_safe": True,
            "extras": {"numerics_epoch": _linus.NUMERICS_EPOCH},
        }

    def settings_schema(self, calctype: CalcType | str | None = None) -> dict[str, Any]:
        """The JSON Schema of linus's keywords (and of ``resolved.settings``) for a calctype."""
        return json.loads(_linus.settings_schema(None if calctype is None else CalcType(calctype).value))

    def record_schema(self, calctype: CalcType | str | None = None) -> dict[str, Any]:
        """The JSON Schema of a run's record (diagnostic: its keys may change between versions)."""
        return json.loads(_linus.record_schema())

    def plan(self, inp: ProgramInput) -> dict[str, Any]:
        """The input checked and planned, nothing computed: ``resolved``, ``resources``,
        ``warnings``, ``record`` (a plain document)."""
        with Session(inp, calctypes=[inp.calctype]) as s:
            return s.plan()

    def compute(self, inp: ProgramInput, ctx: Any = None) -> ProgramOutput:
        """One calculation (`linus.compute`); a failure raises `linus.LinusError`, a
        `qcdata.ProgramError` with the failed output."""
        return compute(inp, ctx)

    def open_session(self, inp: ProgramInput, ctx: Any = None,
                     calctypes: Iterable[CalcType | str] | None = None) -> Session:
        """A session at the input's atoms (`linus.Session`) under the runner's grant."""
        return Session(inp, calctypes=calctypes, ctx=ctx)


#: The program object (an instance) for runners' registries.
program = LinusProgram()
