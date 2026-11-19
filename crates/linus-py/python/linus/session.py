"""`linus.Session` and `linus.compute`: Hartree-Fock and Kohn-Sham DFT on qcdata objects.

A session is one molecule (atoms, charge, multiplicity), one model and one set of settings,
opened from a `qcdata.ProgramInput`, at many geometries: an optimization, an IRC, the images of
a path, finite differences. It keeps, per lineage (any JSON value; None is the default one),
what makes the next step faster and steadier: the last converged orbitals (the next SCF starts
from them, made orthonormal at the new positions, within 1 bohr) and the stability probes'
state (they run at a lineage's first SCF and again only when the HOMO-LUMO gap falls). Every
result is a `qcdata.Frame`; its record says where the SCF started (``warm_start``,
``scf.guess``) and which probes ran.

>>> import qcdata, linus
>>> inp = qcdata.ProgramInput(program="linus", calctype="gradient", structure=water,
...                           model={"method": "b3lyp-d3bj", "basis": "def2-svp"},
...                           keywords={"threads": 8})
>>> out = linus.compute(inp)                        # one calculation: a ProgramOutput (raises on failure)
>>> with linus.Session(inp) as s:
...     f = s.compute()                             # the input's geometry and calctype
...     f = s.compute(x2)                           # warm start from the last step
...     h = s.compute(x2, calctype="hessian")       # with frequencies and thermochemistry
...     b = s.compute(y, lineage=("irc", "forward"), seed=[None])   # a chain of its own

These two are linus's whole Python interface for calculations (decision 7). A failed calculation
raises a `LinusError`, a `qcdata.ProgramError` with its partial results (decision 9). Under a
runner, ``ctx`` is the calculation's share of the machine (decisions 10 and 16): its threads,
host memory, GPUs and device memory, each used as the smaller of the keyword and the grant, and
threads without the keyword at most 8 (`apply_grant`). The session is qccompute's protocol object, structurally (linus never imports
qccompute): `linus.qc.program.open_session` returns one.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any

import numpy as np
from qcdata import (
    Accuracy,
    AOConvention,
    BasisSet,
    CalcType,
    Ecp,
    EcpTerm,
    ExecutionInfo,
    Frame,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramOutput,
    ProgramWarning,
    Request,
    Shell,
    SinglePointData,
    Wavefunction,
    lineage_key,
)

from . import _frames, _linus, _resources
from .errors import InputError, LinusError, UnsupportedError

__all__ = ["Session", "compute", "PROGRAM", "CALCTYPES", "OUTPUTS"]

PROGRAM = "linus"
#: What a linus session can run, with every method: energies, gradients and Hessians (with
#: their harmonic analysis and thermochemistry).
CALCTYPES = frozenset({CalcType.energy, CalcType.gradient, CalcType.hessian})
#: A request's optional outputs (``results.wavefunction`` always has the orbital levels): the
#: orbitals' coefficients (``orbitals``); the AO matrices: the Fock (Kohn-Sham) matrices of the
#: SCF's last build, with the overlap and the orbitals that are their eigenvectors
#: (``hamiltonian``), the overlap (``overlap``) and the core Hamiltonian (``core_hamiltonian``);
#: each comes with the basis set they share (``Wavefunction.basis``); with a Hessian the IR intensities, dipole derivatives and the
#: polarizability (``ir`` or ``polarizability``: one response); ``dipole_moment`` is always
#: there. None changes another number. The keywords' ``output`` group (the same names) asks for
#: them in every request, and ``output.molden`` adds the Molden file
#: (``results.files["orbitals.molden"]``).
OUTPUTS = frozenset({"orbitals", "hamiltonian", "overlap", "core_hamiltonian", "ir", "polarizability",
                     "dipole_moment"})
#: The keywords' output settings, which add the request outputs of the same names.
_OUTPUT_SETTINGS = ("orbitals", "hamiltonian", "overlap", "core_hamiltonian")
_RANK = {CalcType.energy: 0, CalcType.gradient: 1, CalcType.hessian: 2}


def _calctype(c: Any) -> CalcType:
    """A calctype linus computes: energy, gradient or hessian."""
    try:
        ct = CalcType(c)
    except ValueError:
        raise InputError(f"unknown calctype {c!r}; linus computes the calctypes energy, gradient and hessian") from None
    if ct not in _RANK:
        raise UnsupportedError(f"linus computes the calctypes energy, gradient and hessian, not {ct.value} (programs "
                               "with subprograms, such as qcopt, run optimizations, scans and paths over a session)")
    return ct


def _declared(calctypes: Iterable[Any] | None) -> frozenset[CalcType] | None:
    """The calctypes a session may run: the declared ones with the lower derivatives they
    imply (a Hessian session also returns gradients and energies); None: not declared (every
    one the method supports)."""
    if calctypes is None:
        return None
    declared = [_calctype(c) for c in calctypes]
    if not declared:
        raise InputError("declare at least one calctype (calctypes=None: all)")
    top = max(_RANK[c] for c in declared)
    return frozenset(c for c, r in _RANK.items() if r <= top)


def _grant(asked: Any, granted: Any, path: str, warnings: list[ProgramWarning]) -> Any:
    """A resource as used: the smaller of the request (the keyword; ``"auto"`` is no request)
    and the runner's grant, or the grant without a request; a request above the grant is capped
    (``resource_capped``)."""
    if asked == "auto":
        asked = None
    if granted is None:
        return asked
    if asked is None:
        return granted
    if asked > granted:
        warnings.append(ProgramWarning(code="resource_capped", path=path,
                                       message=f"{path} {asked} is above the {granted} granted; it runs with {granted}"))
        return granted
    return asked


def _device_ordinals(ctx: Any) -> list[int]:
    """The CUDA ordinals of a grant's GPUs in this process: the runner's own mapping
    (``RunContext.device_ordinals``, qc* decision 16), so every program opens the devices a grant
    names alike; a granted GPU this process cannot see fails with kind ``environment``."""
    try:
        return [int(i) for i in ctx.device_ordinals()]
    except ProgramError as e:
        raise LinusError(e.error.message, kind=e.error.kind.value, code=e.error.code) from None


def apply_grant(keywords: Mapping[str, Any], ctx: Any, warnings: list[ProgramWarning]) -> dict[str, Any]:
    """linus's settings under a runner's grant (``ctx``, decisions 10 and 16; None: the keywords
    as they are).

    - ``threads``, ``memory_gib`` and ``device_memory_gib`` (qc*'s shared resource keywords, left
      out of the calculation key): the smaller of the keyword and the grant, or the grant when
      the keyword is unset or ``auto``, except that threads without the keyword are at most
      linus's own cap, the 8 it uses well (``estimates.threads``; qc* decision 16: a program may
      use fewer threads where more don't help). A runner grants a memory keyword in place of its share,
      at most what it has, so the grant is the keyword unless the runner capped it; a capped
      keyword warns (``resource_capped``). A memory grant comes before linus's own default (the
      cgroup limit or the RAM). Applied on every call: each `compute` and each session opens
      with its own call's grant (a worker serves calls of different threads).
    - GPUs: with GPUs granted, ``backend`` ``gpu`` or ``auto`` runs on the first of them
      (``backend.devices``: its CUDA ordinal, ``ctx.device_ordinals()``); with none granted it
      runs on the CPU, and ``backend: gpu`` warns (``resource_capped``). The resolved settings
      record every choice.
    """
    s = dict(keywords)
    if ctx is None:
        return s
    for k in ("threads", "memory_gib", "device_memory_gib"):
        v = _grant(s.get(k), getattr(ctx, k, None), k, warnings)
        if k == "threads" and v is not None and s.get(k) in (None, "auto"):
            v = min(v, _linus.THREADS_USED_WELL)
        if v is not None:
            s[k] = v
    backend = s.get("backend", {})
    backend = {"kind": backend} if isinstance(backend, str) else dict(backend)
    kind = backend.get("kind", "cpu")
    if kind not in ("gpu", "auto"):
        return s
    gpus = tuple(getattr(ctx, "gpus", ()) or ())
    if not gpus:
        if kind == "gpu":
            warnings.append(ProgramWarning(code="resource_capped", path="backend.kind",
                                           message="backend gpu, but the runner granted no GPU: it runs on the CPU"))
        backend.pop("devices", None)
        s["backend"] = {**backend, "kind": "cpu"}
        return s
    ids = _device_ordinals(ctx)
    asked = backend.get("devices")
    asked = [asked] if isinstance(asked, int) else asked
    if asked and asked[0] not in ids:
        warnings.append(ProgramWarning(code="resource_capped", path="backend.devices",
                                       message=f"backend.devices {asked} is not among the GPUs granted ({list(gpus)}); "
                                               f"it runs on device {ids[0]}"))
        asked = None
    s["backend"] = {**backend, "devices": asked or [ids[0]]}
    return s


class Session:
    """One molecule, model and set of settings at many geometries (see the module docs).

    Args:
        inp: The program (``linus``), model (method and basis set), keywords (linus's settings
            tree: docs/cli.md, ``linus --list-settings``), structure (atoms, charge,
            multiplicity, and the default geometry) and calctype (the default of a request).
            ``inp.given.wavefunction.orbitals`` start the default lineage (its other given
            fields are not used).
        calctypes: The calctypes the session may run, each with its lower derivatives
            (default: every one the method supports: energy, gradient and Hessian, less the
            derivatives the method lacks); a setting that applies to none of them is an input
            error here.
        ctx: The runner's grant (a qccompute ``RunContext``: ``threads``, ``memory_gib``,
            ``gpus``, ``device_memory_gib``), applied by `apply_grant`; its ``progress``: when
            it is not None, every calculation of the session reports its progress events there
            (docs/python.md, "Progress events"); and its ``stop_requested()``, asked before each
            stage and between the SCF's and the responses' iterations: a calculation it stops
            fails with kind ``stopped`` and the results it reached (docs/python.md, "Stopping").
            None: a direct call, the keywords alone.

    Raises:
        InputError, UnsupportedError: The input cannot run (``qcdata.ProgramError``s; codes
            ``unknown_key``, ``not_applicable``, ``no_effect``).
        TypeError: `inp` is not a ProgramInput.
    """

    def __init__(self, inp: ProgramInput, *, calctypes: Iterable[CalcType | str] | None = None,
                 ctx: Any = None) -> None:
        if not isinstance(inp, ProgramInput):
            raise TypeError(f"a Session opens from a qcdata ProgramInput, not {type(inp).__name__}")
        self._engine: Any = None
        self._inp = inp
        self._structure = inp.structure
        declared = _declared(calctypes)
        if inp.program.lower() != PROGRAM:
            raise InputError(f"this input is for {inp.program!r}, not linus")
        if inp.model is None or not inp.model.method or not inp.model.basis:
            raise InputError("linus needs a model with a method and a basis set: "
                             "model={'method': 'b3lyp', 'basis': 'def2-svp'}")
        if inp.structure is None:
            raise InputError("linus needs a structure")
        if inp.cmdline_args or inp.files:
            raise InputError("linus runs in process: it takes no cmdline_args or files", code="not_applicable")
        settings = dict(inp.keywords)
        for k in ("method", "basis"):
            if k in settings:
                raise InputError(f"{k} comes from the model, not the keywords", code="unknown_key")
        try:
            # the shared keyword, read as every program reads it (a bare number is refused)
            Accuracy.from_keywords(settings)
        except ProgramError as e:
            raise InputError(e.error.message, code=e.error.code) from None
        self._warnings: list[ProgramWarning] = []
        settings = apply_grant(settings, ctx, self._warnings)
        settings = {"method": inp.model.method, "basis": inp.model.basis, **settings}
        s = inp.structure
        self._structure = s
        self._positions = np.ascontiguousarray(np.asarray(s.geometry, dtype=np.float64).reshape(-1, 3))
        self._engine = _linus.Session(list(s.atomic_numbers), self._positions, charge=int(s.charge),
                                      multiplicity=int(s.multiplicity), options=json.dumps(settings),
                                      calctypes=None if declared is None else
                                      [c.value for c in sorted(declared, key=_RANK.__getitem__)],
                                      progress=getattr(ctx, "progress", None) if ctx is not None else None,
                                      stop=getattr(ctx, "stop_requested", None) if ctx is not None else None)
        #: The calctypes the session may run.
        self.calctypes: frozenset[CalcType] = frozenset(CalcType(c) for c in self._engine.calctypes)
        options = self._engine.options
        self._threads = int(options.get("threads") or _linus.default_options()["threads"])
        output = options.get("output") or {}
        self._outputs = frozenset(k for k in _OUTPUT_SETTINGS if output.get(k))
        self._shells = tuple(self._engine.basis_shells())
        self._ecp = tuple(self._engine.ecp_centers())
        self._pure = not bool(options.get("cartesian", False))
        if inp.given is not None:
            self._start_from_given(inp.given)

    # -- the protocol's attributes ------------------------------------------------------

    @property
    def program(self) -> str:
        """The program's name, ``linus``."""
        return PROGRAM

    @property
    def program_version(self) -> str:
        """linus's version."""
        return _linus.__version__

    @property
    def input_data(self) -> ProgramInput:
        """The input the session was opened with: its atoms, model, keywords and defaults (the
        geometry and calctype of a request that gives none)."""
        return self._inp

    # -- requests ---------------------------------------------------------------------

    def compute(self, positions: Any = None, calctype: CalcType | str | None = None, *,
                accuracy: Accuracy | float | None = None, lineage: Any = None,
                outputs: Iterable[str] = (), seed: Iterable[Any] = (), commit: bool = True) -> Frame:
        """Evaluate one geometry.

        Args:
            positions: Atomic positions (bohr), n_atoms x 3; None: the input's.
            calctype: ``energy``, ``gradient`` or ``hessian``; None: the input's.
            accuracy: This request's accuracy targets (`qcdata.Accuracy`; a bare number targets
                the calctype's highest derivative), replacing the input's ``accuracy``; linus
                turns them into its thresholds and records ``record["accuracy"]``.
            lineage: The chain this request continues (any JSON value; None: the default).
            outputs: Optional outputs (`OUTPUTS`).
            seed: Lineages to start from, in order, when `lineage` has no state yet.
            commit: Whether a success becomes `lineage`'s state.

        Returns:
            The frame: the structure evaluated, its results, warnings, record and execution.

        Raises:
            LinusError: The request failed (a `qcdata.ProgramError`; `frame` has the partial
                results and the record). The session and every lineage stay as they were.
        """
        frame = self._step(Request(positions=positions, calctype=calctype, accuracy=accuracy, lineage=lineage,
                                   outputs=frozenset(outputs), seed=list(seed), commit=commit))
        if not frame.success:
            raise _error_of(frame)
        return frame

    def compute_many(self, requests: Sequence[Request | Mapping[str, Any]]) -> list[Frame]:
        """Evaluate many requests in list order, each reading its lineage as the requests before
        it left it. Every request runs; then failures raise together (decision 9).

        Args:
            requests: `qcdata.Request` objects, or mappings of their fields.

        Returns:
            One frame per request, in order, when all succeeded.

        Raises:
            qcdata.ProgramErrors: Some requests failed: one `LinusError` per failed request (with
                its failed frame), and ``results``, every request's frame in order (the
                successful ones too).
        """
        frames = [self._step(r if isinstance(r, Request) else Request.model_validate(r)) for r in requests]
        failed = [_error_of(f) for f in frames if not f.success]
        if failed:
            raise ProgramErrors(f"{len(failed)} of {len(frames)} failed", failed, frames)
        return frames

    def reset(self, lineage: Any = ...) -> None:
        """Forget a lineage's state (None: the default lineage); omitted: every lineage."""
        self._open()
        self._engine.reset(None if lineage is ... else lineage_key(lineage))

    def lineages(self) -> list[Any]:
        """The lineages with a state (their JSON values)."""
        self._open()
        return [json.loads(k) for k in self._engine.lineages()]

    def plan(self, calctype: CalcType | str | None = None) -> dict[str, Any]:
        """What a request will do, decided before it runs (qccompute's Plan, as a plain dict).

        - ``resolved``: the settings and why each ``auto`` took its value; ``deferred`` lists the
          settings left ``auto`` because the run decides them (the DF tensors' storage from the
          free memory at the SCF's start);
        - ``resources``: estimates of its peak host memory (``memory_gib``) and, with a device,
          of its device memory (``gpus``, ``device_memory_gib``); no threads, which the runner
          gives (qc* decision 16);
        - ``warnings``;
        - ``record``: linus's plan (``linus --plan``, docs/cli.md "Plan"): where each family of
          kernels runs, in which arithmetic, the memory plan, the sizes, the responses, and
          ``estimates``: peak host memory, device memory and the threads it uses well, each
          with its error band (the largest relative error over the cases it was fitted on).

        Args:
            calctype: The request's calctype; None: the session's highest.
        """
        self._open()
        top = max(self.calctypes, key=_RANK.__getitem__) if calctype is None else _calctype(calctype)
        if top not in self.calctypes:
            raise InputError(f"this session was opened for {sorted(c.value for c in self.calctypes)}, not "
                             f"{top.value}", code="not_applicable")
        p = self._engine.plan(top.value)
        record = {k: v for k, v in p.items() if k not in ("settings", "reasons", "deferred")}
        return {"resolved": {"settings": _frames.keywords(p.get("settings", {})), "reasons": p.get("reasons", {}),
                             "deferred": list(p.get("deferred", []))},
                "resources": _resources.resources(p["estimates"]),
                "warnings": [w.model_dump() for w in self._warnings],
                "record": record}

    def close(self) -> None:
        """Free the session's engine: its threads, memory and device."""
        if self._engine is not None:
            self._engine.close()

    @property
    def closed(self) -> bool:
        """Whether the session is closed (by `close`, or after an internal error)."""
        return self._engine is None or self._engine.closed

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    def __repr__(self) -> str:
        m = self._inp.model
        state = ", closed" if self.closed else ""
        return (f"linus.Session(method={m.method!r}, basis={m.basis!r}, atoms={len(self._structure.symbols)}, "
                f"calctypes={sorted(c.value for c in self.calctypes)}{state})")

    # -- internals --------------------------------------------------------------------

    def _open(self) -> None:
        if self.closed:
            raise InputError("this session is closed (by close(), or after an internal error); open a new one",
                             code="session_closed")

    def _start_from_given(self, given: SinglePointData) -> None:
        """Start the default lineage from the given orbitals (record ``scf.guess.source``
        ``given``); without orbitals, or with orbitals of another basis set, the input is
        refused (``not_applicable``)."""
        wfn = given.wavefunction
        if wfn is None or wfn.basis is None or wfn.orbitals.coefficients_a is None:
            raise InputError("given results without orbital coefficients (given.wavefunction.orbitals, with the "
                             "request output orbitals): nothing linus can start an SCF from", code="not_applicable")
        c_a, c_b, restricted = _guess_orbitals(wfn, self._shells, self._nalpha_nbeta())
        self._engine.set_guess_orbitals(c_a, c_b, restricted)

    def _nalpha_nbeta(self) -> tuple[int, int]:
        z = sum(self._engine.numbers) - sum(self._engine.ecp_core_electrons)
        n = z - int(self._structure.charge)
        unpaired = int(self._structure.multiplicity) - 1
        return (n + unpaired) // 2, (n - unpaired) // 2

    def _basis(self) -> BasisSet:
        """The basis set as placed on the atoms, in linus's AO order (qcdata's default
        convention), with the ECPs in qcdata's layout (the power of r: Basis Set Exchange's
        n - 2)."""
        def terms(ts: Any) -> list[EcpTerm]:
            return [EcpTerm(power=p, exponent=z, coefficient=c) for p, z, c in ts]

        return BasisSet(
            name=self._inp.model.basis,
            shells=[Shell(center=sh["atom"], angular_momentum=sh["l"], pure=sh["pure"], exponents=sh["exponents"],
                          coefficients=sh["coefficients"]) for sh in self._shells],
            ecp=[Ecp(center=e["atom"], core_electrons=int(e["core_electrons"]), local=terms(e["local"]),
                     semilocal=[terms(ch) for ch in e["semilocal"]]) for e in self._ecp])

    def _step(self, r: Request) -> Frame:
        """One request: its frame, failed or not (a failure leaves every lineage as it was)."""
        return self._run(r)[0]

    def _run(self, r: Request) -> tuple[Frame, Mapping[str, Any] | None]:
        """One request: its frame and, when it succeeded, linus's report."""
        t0 = time.perf_counter()
        positions = self._positions if r.positions is None else np.ascontiguousarray(r.positions, dtype=np.float64)
        structure = self._structure.model_copy(update={"geometry": positions.reshape(-1, 3)})
        outputs = set(r.outputs)
        peak = _resources.PeakMemory()
        try:
            self._open()
            calctype = _calctype(self._inp.calctype if r.calctype is None else r.calctype)
            if calctype not in self.calctypes:
                raise InputError(f"this session was opened for {sorted(c.value for c in self.calctypes)}, not "
                                 f"{calctype.value}; open it with calctypes that include it", code="not_applicable")
            unknown = sorted(outputs - OUTPUTS)
            if unknown:
                raise UnsupportedError(f"linus has no outputs {unknown}; it has {sorted(OUTPUTS)}")
            outputs |= self._outputs
            if "hamiltonian" in outputs:
                # the Fock matrices come with what they imply: the overlap, and the orbitals
                # that are their fresh eigenvectors (qc* decision 25a)
                outputs |= {"overlap", "orbitals"}
            ir = bool(outputs & {"ir", "polarizability"})
            if ir and calctype != CalcType.hessian:
                raise UnsupportedError("ir and polarizability come with a Hessian (calctype hessian)")
            request: dict[str, Any] = {"lineage": lineage_key(r.lineage), "seed": [lineage_key(s) for s in r.seed],
                                       "commit": r.commit, **{k: k in outputs for k in _frames.AO_OUTPUTS}}
            accuracy = Accuracy.of(r.accuracy, calctype)
            if accuracy is not None:
                request["accuracy"] = {k: v for k in ("energy", "gradient", "hessian")
                                       if (v := getattr(accuracy, k)) is not None}
            if ir:
                request["freq"] = {"ir": True}
            with peak:
                report = self._engine.compute(None if r.positions is None else positions, calctype.value,
                                              json.dumps(request))
        except ProgramError as e:
            return self._failed(structure, e, t0, peak.gib), None
        frame = _frames.frame(structure, report, outputs, self._threads, time.perf_counter() - t0,
                              list(self._warnings), self._basis, peak.gib)
        return frame, report

    def _failed(self, structure: Any, e: ProgramError, t0: float, peak_gib: float | None = None) -> Frame:
        """The failed frame of a request: the error, the results the calculation reached (an
        SCF that did not converge: its last iterate) and its record."""
        report = getattr(e, "result", None)
        failure = getattr(e, "failure", None) or {}
        if report is not None:
            results = _frames.single_point(report)
            record = _frames.record(report)
        else:
            results = SinglePointData(provenance=_frames.provenance())
            record = {k: failure[k] for k in ("scf", "response") if failure.get(k) is not None}
        return Frame(structure=structure, results=results, success=False, traceback=e.error.message,
                     error=e.error, record=record, warnings=list(self._warnings), logs=failure.get("logs"),
                     execution=_frames.execution(failure, self._threads, time.perf_counter() - t0, peak_gib,
                                                 ran=report or {}))


def _error_of(frame: Frame) -> LinusError:
    """The exception of a failed frame: linus's class of its kind, carrying the frame."""
    from . import errors

    by_kind = {"input": errors.InputError, "unsupported": errors.UnsupportedError,
               "resource": errors.ResourceError, "convergence": errors.ConvergenceError,
               "numerical": errors.NumericalError, "backend": errors.BackendError}
    cls = by_kind.get(frame.error.kind.value, errors.LinusError)
    kind = "basis" if frame.error.code == "basis_set" else frame.error.kind.value
    return cls(frame.error.message, kind=kind, code=frame.error.code, details=dict(frame.error.details), frame=frame)


def _guess_orbitals(wfn: Wavefunction, shells: Sequence[Mapping[str, Any]],
                    nab: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, bool]:
    """The occupied orbitals of a given wavefunction (with coefficients) in linus's AO order,
    checked against the session's basis set `shells` (the same shells on the same atoms: else
    ``not_applicable``)."""
    if wfn.basis.ao_convention != AOConvention():
        # the orbitals alone: the given AO matrices are not used
        wfn = Wavefunction(orbitals=wfn.orbitals, basis=wfn.basis).with_convention(AOConvention())
    orbitals = wfn.orbitals
    given = wfn.basis.shells
    same = len(given) == len(shells) and all(
        g.center == s["atom"] and g.angular_momentum == s["l"] and bool(g.pure) == bool(s["pure"])
        and np.allclose(g.exponents, s["exponents"], rtol=1e-12, atol=0.0)
        and np.allclose(g.coefficients, s["coefficients"], rtol=1e-10, atol=1e-14)
        for g, s in zip(given, shells))
    if not same:
        raise InputError("the given orbitals are in another basis set than the session's (shells, centers, "
                         "exponents or contractions differ)", code="not_applicable")
    na, nb = nab

    def occupied(c: np.ndarray, occ: Any, n: int, spin: str) -> np.ndarray:
        if occ is not None:
            idx = np.flatnonzero(np.asarray(occ) > 0.5)
            if len(idx) != n:
                raise InputError(f"the given {spin} orbitals have {len(idx)} occupied, the session's molecule {n}",
                                 code="not_applicable")
            return np.ascontiguousarray(c[:, idx])
        return np.ascontiguousarray(c[:, :n])

    restricted = orbitals.restricted
    c_a = occupied(np.asarray(orbitals.coefficients_a), orbitals.occupations_a, na, "alpha")
    c_b = occupied(np.asarray(orbitals.coefficients_a if restricted else orbitals.coefficients_b),
                   orbitals.occupations_b, nb, "beta")
    return c_a, c_b, restricted and na == nb


def compute(inp: ProgramInput, ctx: Any = None) -> ProgramOutput:
    """One calculation: a `qcdata.ProgramInput` (program ``linus``) in this process, as a
    `qcdata.ProgramOutput`. The input's calctype is the only one declared, so a setting of
    another calctype is an error (D41), as in the CLI.

    Args:
        inp: The input; ``inp.given.wavefunction.orbitals`` start the SCF.
        ctx: A runner's grant (`apply_grant`); None: a direct call.

    Returns:
        The successful output.

    Raises:
        LinusError: The calculation failed or the input cannot run (a `qcdata.ProgramError`
            with the failed output in ``output``: the error, the partial results, warnings,
            record and execution; decision 9).
        TypeError: `inp` is not a ProgramInput.
    """
    if not isinstance(inp, ProgramInput):
        raise TypeError(f"linus.compute takes a qcdata ProgramInput, not {type(inp).__name__}")
    t0 = time.perf_counter()
    try:
        session = Session(inp, calctypes=[inp.calctype], ctx=ctx)
    except LinusError as e:
        # qcdata's empty results of the input's data type (an optimization's are OptimizationData)
        e.output = ProgramOutput.failure(inp, e.error, execution=ExecutionInfo(wall_time=time.perf_counter() - t0))
        e.results = e.output.results
        raise
    with session:
        frame, report = session._run(Request(calctype=inp.calctype))
    if not frame.success:
        e = _error_of(frame)
        e.output = ProgramOutput.failure(inp, frame.error, results=frame.results, traceback=frame.traceback,
                                         warnings=frame.warnings, record=frame.record, execution=frame.execution,
                                         logs=frame.logs)
        raise e
    return ProgramOutput(input_data=inp, results=frame.results, success=True, warnings=frame.warnings,
                         record=frame.record, resolved=_frames.resolved(report), execution=frame.execution,
                         logs=frame.logs)
