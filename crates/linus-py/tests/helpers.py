"""One calculation in a test, through the public API: a qcdata ProgramInput, a session declared
for its calctype, the frame of its first request."""

from __future__ import annotations

from typing import Any

import linus
import pytest
from qcdata import Frame, ProgramError, ProgramInput, ProgramOutput, Structure


def program_input(structure: Structure, calctype: str = "energy", *, method: str = "hf", basis: str = "def2-svp",
                  **keywords: Any) -> ProgramInput:
    """A linus input: `method` and `basis` in the model, every other setting a keyword."""
    return ProgramInput(program="linus", calctype=calctype, structure=structure,
                        model={"method": method, "basis": basis}, keywords=keywords)


def run(structure: Structure, calctype: str = "energy", *, outputs: tuple[str, ...] = (), method: str = "hf",
        basis: str = "def2-svp", **keywords: Any) -> Frame:
    """The frame of one request at the structure's geometry (raises the request's error)."""
    with linus.Session(program_input(structure, calctype, method=method, basis=basis, **keywords),
                       calctypes=[calctype]) as s:
        return s.compute(outputs=outputs)


def failure(inp: ProgramInput) -> ProgramOutput:
    """The failed output `linus.compute` raises for `inp` (decision 9: a `qcdata.ProgramError`,
    here linus's, with the whole failed output)."""
    with pytest.raises(ProgramError) as info:
        linus.compute(inp)
    assert isinstance(info.value, linus.LinusError) and info.value.output is not None
    return info.value.output


def moved(structure: Structure, positions: Any) -> Structure:
    """The structure at other positions (bohr)."""
    return structure.model_copy(update={"geometry": positions})
