"""linus's exceptions: `qcdata.ProgramError` with linus's kinds, raised by the library.

Every exception is a `LinusError`, a `qcdata.ProgramError` (so a program with subprograms
catches one class for every program) and the matching builtin: `InputError` is a `ValueError`, `UnsupportedError` a
`NotImplementedError`, `ResourceError` a `MemoryError`, `NumericalError` an `ArithmeticError`.

- `kind`: linus's error kind, the CLI's ``error_kind`` (input, basis, convergence, unsupported,
  resource, backend, numerical, internal; ``stopped``: a runner's stop request,
  ``ctx.stop_requested()``, ended the calculation at its next step, with the results it reached);
- `error`: the qcdata `ErrorInfo` (a basis set error is kind ``input`` there, code
  ``basis_set``); `code` its code: a settings error's (``unknown_key``, ``not_applicable``,
  ``no_effect``), a memory error's (``host_memory``, ``device_memory``, with ``error.details``
  ``required_gib`` and ``available_gib``: the memory budget that holds what failed and the
  budget it had, so a dispatcher resubmits without parsing the message), ``device_lost`` for a
  GPU fault that poisoned the CUDA context (nothing else can run on the device in this process),
  ``session_closed``;
- `frame`: the failed `qcdata.Frame` of a session request (partial results, warnings, the
  record); `output`: the failed `qcdata.ProgramOutput` of `linus.compute` (the same, with the
  input); `results`: the partial results of either;
- `failure`: the CLI's failure JSON as a dict (the ``scf`` history of an SCF that did not
  converge, the ``response``, the ``timings``); `result`: the report of what the calculation
  reached, when it reached one.
"""

from __future__ import annotations

from typing import Any

from qcdata import ErrorInfo, ErrorKind, ProgramError

__all__ = [
    "LinusError",
    "InputError",
    "UnsupportedError",
    "ResourceError",
    "ConvergenceError",
    "NumericalError",
    "BackendError",
]

# linus's kinds that qcdata names otherwise, with the code that keeps them apart
_QC_KIND = {"basis": (ErrorKind.input, "basis_set")}


class LinusError(ProgramError, RuntimeError):
    """A linus calculation failed (also a `qcdata.ProgramError` and a RuntimeError)."""

    default_kind = "internal"

    def __init__(self, message: str, *, kind: str | None = None, code: str | None = None,
                 details: dict[str, Any] | None = None, frame: Any = None, output: Any = None) -> None:
        kind = kind or type(self).default_kind
        qk, qcode = _QC_KIND.get(kind, (ErrorKind(kind), None))
        super().__init__(ErrorInfo(kind=qk, message=message, code=code or qcode, details=details or {}), frame=frame,
                         output=output)
        self.kind = kind
        self.failure: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None

    @property
    def code(self) -> str | None:
        """The error's code (`error.code`), if it has one."""
        return self.error.code

    def __str__(self) -> str:
        return self.error.message

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with everything it carries, for worker processes."""
        return (_rebuild, (type(self), self.error.message, self.kind, self.__dict__))


def _rebuild(cls: type[LinusError], message: str, kind: str, state: dict[str, Any]) -> LinusError:
    e = cls.__new__(cls)
    Exception.__init__(e, message)
    e.__dict__.update(state)
    e.kind = kind
    return e


class InputError(LinusError, ValueError):
    """Settings or a molecule linus cannot run as given (also a ValueError)."""

    default_kind = "input"


class UnsupportedError(LinusError, NotImplementedError):
    """A valid request for something linus does not implement (also a NotImplementedError)."""

    default_kind = "unsupported"


class ResourceError(LinusError, MemoryError):
    """Not enough memory for the request as set up (also a MemoryError)."""

    default_kind = "resource"


class ConvergenceError(LinusError):
    """An SCF or a response did not converge; `failure` has its history."""

    default_kind = "convergence"


class NumericalError(LinusError, ArithmeticError):
    """A numerical failure: non-finite energies, singular matrices (also an ArithmeticError)."""

    default_kind = "numerical"


class BackendError(LinusError):
    """The GPU backend failed: no usable device for backend gpu, or a device error in a
    calculation (code ``device_lost`` when it poisoned the CUDA context)."""

    default_kind = "backend"
