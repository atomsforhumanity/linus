# linus in Python

The Python package `linus` (`pip install afh-linus`) runs Hartree-Fock and Kohn-Sham DFT in your
process, on [qcdata](https://github.com/atomsforhumanity/qcdata) objects: a calculation is a
`qcdata.ProgramInput`, and its results are qcdata's `ProgramOutput` and `Frame`. Everything is in
atomic units (bohr, Eh, e·bohr). Every setting has the same name as in the command line and the
JSON request ([cli.md](cli.md)); recipes for common calculations are in [usage.md](usage.md).

linus is a **program** in the sense of the qc* stack: it maps one `ProgramInput` to one
`ProgramOutput`. It has no **subprograms** (other programs it runs as part of its own calculation);
programs that do, such as qcopt's optimizations, run linus as their subprogram. linus's Python
interface is two names, the same as every AFH program's: `linus.compute(inp)` for one calculation
and `linus.Session(inp)` for one molecule at many geometries. Many calculations (lists, a result
store, worker processes, a share of the machine for each) go through
[qccompute](https://github.com/atomsforhumanity/qccompute), which runs linus by name. This page
follows that order: the command line, `compute` and `Session`, then many calculations.

## Installing

Two distributions of the same version install the package `linus`; install one of them, not
both (as with CuPy's per-CUDA packages):

```bash
pip install afh-linus             # the CPU build
pip install afh-linus-cuda13      # the same with the GPU backend (NVIDIA GPUs, CUDA 13)
pip install qccompute             # runners: lists, a result store, workers
```

- **`afh-linus`** runs everything on the CPU. Pick it when the machine has no NVIDIA GPU.
- **`afh-linus-cuda13`** runs everything `afh-linus` does, and runs calculations with
  `backend: "gpu"` (or `"auto"`) on an NVIDIA GPU ([gpu.md](gpu.md)). Nothing CUDA is needed to
  install it; at run time the GPU needs an NVIDIA driver with CUDA 13 support (580 or newer) and
  the CUDA 13 libraries NVRTC, cuBLAS and cuSOLVER where the dynamic loader finds them (a CUDA 13
  toolkit's `lib64` on the loader path, or `LD_LIBRARY_PATH`). Without them, `backend: "cpu"`
  (the default) and `"auto"` still run on the CPU, and `"gpu"` fails with a `backend` error that
  names what is missing.
- `linus.GPU_BUILD` says which one is installed: True in `afh-linus-cuda13`, False in
  `afh-linus` (`linus.__version__` is the same in both).
- Both are wheels for Linux x86-64 with glibc 2.28 or newer (manylinux_2_28) and CPython 3.11 or
  newer (abi3), with the basis sets inside. They need numpy, pydantic and qcdata 0.24; qccompute
  0.23 runs linus as the program `linus`.

## The command line

The same calculation from a shell, with the result as JSON ([cli.md](cli.md) has every option and
result key):

```bash
linus water.xyz --method b3lyp-d3bj --basis def2-svp --jk df --grad --threads 8 --json
```

```python
import numpy as np
import qcdata, linus

water = qcdata.Structure.open("water.xyz")        # Å in the file, bohr in memory
inp = qcdata.ProgramInput(program="linus", calctype="gradient", structure=water,
                          model={"method": "b3lyp-d3bj", "basis": "def2-svp"},
                          keywords={"jk": "df", "threads": 8})
```

The model holds the method and the basis set; the keywords hold every other setting, with
groups as dicts (`{"scf": {"convergence": "tight"}, "freq": {"thermo": "qrrho"}}`). A setting
linus does not know is an error with a suggestion, and so is a setting that cannot take effect
(a `grid` for Hartree-Fock, `freq` settings without a Hessian).

## One calculation: `linus.compute`

```python
out = linus.compute(inp)                 # a qcdata.ProgramOutput
out.results.energy, out.results.gradient # Eh, Eh/bohr (natoms x 3)
out.results.energy_components            # nuclear repulsion, one-electron, Coulomb, exchange, XC, dispersion, solvation
out.resolved.settings                    # every setting the run used, `auto`s resolved (`out.resolved.reasons`: why)
out.record["scf"]["history"]             # the run record: SCF iterations, probes, memory plan, backend, sizes
out.execution.timings, out.execution.threads, out.execution.peak_memory_gib
out.execution.cpu_model, out.execution.libc, out.execution.devices   # where it ran (the GPU's name, if any)
out.logs                                 # with output.verbose: the SCF's progress (never printed in process)
```

`keywords={"accuracy": {"gradient": 1e-6}}` sets accuracy targets: linus picks its SCF and
response thresholds for them and records them in `out.record["accuracy"]` with its error
estimates ([usage.md, section 11](usage.md#11-accuracy-targets)).

A failed calculation raises a `linus.LinusError`, a `qcdata.ProgramError`, as every AFH program
does: `e.error` is what went wrong (`kind`, `code`, `message`), `e.output` the failed
`ProgramOutput` with what the calculation reached (an SCF that did not converge: its last energy in
`e.results` and its history in `e.output.record["scf"]`):

```python
try:
    out = linus.compute(inp)
except linus.LinusError as e:            # or qcdata.ProgramError, for any program
    e.error.kind, e.error.code, e.output.record["scf"]["history"]
```

A `hessian` carries its harmonic analysis and thermochemistry (`freq` settings):
`out.results.vibrational_analysis` (`qcdata.VibrationalAnalysis`: frequencies in cm⁻¹, imaginary
ones negative, normal modes, reduced masses, `n_imaginary()`, ...) and `out.results.thermochemistry`
(a list of `qcdata.Thermochemistry`, one per `freq.temperature`). Optional outputs come from the
`output` settings or a request's `outputs` (below). `results.wavefunction.orbitals` (a
`qcdata.Orbitals`) always has the orbital energies and the occupations per spin;
`output.orbitals` adds the coefficients and the basis set (with its ECPs, in qcdata's layout),
`output.hamiltonian` the Fock matrices of the SCF's last build (`results.wavefunction.hamiltonian_a`,
`_b` when unrestricted) with the `overlap` (H C = S C diag(ε) to rounding), `output.overlap` and
`output.core_hamiltonian` those matrices; `output.molden` adds `results.files["orbitals.molden"]`.
None of them changes another number. There is no density output: the SCF density is
`orbitals.density()` (C n Cᵀ of the returned orbitals).

## Sessions

A **session** is one molecule (its atoms, charge and multiplicity), one model and one set of
settings, computed at many geometries: an optimization, the images of a path, an IRC, finite
differences. It keeps what makes the next step faster and steadier: per **lineage** (a chain of
geometries, named by any JSON value; `None` is the default lineage) the last converged orbitals,
from which the next SCF starts (made orthonormal at the new positions, when every atom moved less
than 1 bohr), and the stability probes' state (they run at a lineage's first SCF, and again only
when the HOMO-LUMO gap falls). On a 39-atom anion's optimization that is 15 SCF iterations per step
from scratch against 11 from the last step, and the closed-shell probe once instead of every step
([bench/session](../bench/session/README.md)).

```python
with linus.Session(inp) as s:                # opened from the ProgramInput: atoms, model, settings
    f = s.compute()                          # the input's geometry and calctype: a qcdata.Frame
    f = s.compute(x2)                        # new positions (bohr, natoms x 3): starts from f's orbitals
    f.results.energy, f.results.gradient
    f.record["warm_start"], f.record["scf"]["guess"]   # where this SCF started, and from which lineage
```

- `Session(inp, *, calctypes=None, ctx=None)`. `calctypes` are the calculations the session may
  run, each with its lower derivatives (`["hessian"]` also gives gradients and energies); the
  default is every one the method supports. A setting that applies to none of them is an error
  here. `ctx` is passed only by a runner (below); you leave it out.
- `compute(positions=None, calctype=None, *, accuracy=None, lineage=None, outputs=(), seed=(),
  commit=True)` evaluates one geometry and returns a `qcdata.Frame`: the structure, `results`,
  `warnings`, `record` (with qcdata's shared `warm_start` at its top) and `execution`. `accuracy`
  (a `qcdata.Accuracy`, or a number for the calctype's highest derivative) replaces the input's
  accuracy targets for this request. `outputs` adds optional
  results to this request: `"orbitals"`, `"hamiltonian"` (the Fock matrices, with the overlap),
  `"overlap"`, `"core_hamiltonian"`, and with a Hessian `"ir"` (IR intensities, dipole derivatives
  and the polarizability).
- A failed request raises a `linus.LinusError`, which is a `qcdata.ProgramError` with the failed
  frame in `.frame` (the partial results and the record); the session stays open and every lineage
  stays as it was: a lineage's state changes only when a request succeeds.
- `compute_many(requests)` runs a list of `qcdata.Request`s (or dicts of their fields) in order and
  returns one frame per request. Every request runs; if any failed, it then raises
  `qcdata.ProgramErrors`, an `ExceptionGroup` of each failed request's `LinusError`, with every
  request's frame, the successful ones included, in `.results`.
- `reset()` forgets every lineage, `reset(None)` the default one, `reset(name)` that one;
  `lineages()` lists the lineages with a state.
- `plan(calctype=None)` says what a request will do before it runs: the settings it will have
  (`resolved`; `deferred` lists the ones only the run decides), the resources it needs
  (`resources`: estimated peak host memory and device memory, each with its error band in
  `record["estimates"]`; one GPU when it runs on the device; no threads, which the runner gives),
  where each family of kernels runs,
  and the memory plan (`record["memory"]`: `required_gib`, the smallest `memory_gib` that runs
  the calculation in this process, which counts the Python process itself, and each phase's
  parts; the `required_gib` of the `host_memory` failure below it).
- `close()` (or the end of the `with` block) frees the session's threads, memory and GPU at once.

A session runs one request at a time; a call from a second thread while one runs raises
`InputError` (open a session per thread). The GIL is released while linus computes.

### An optimization loop

Any optimizer can drive a session: each step continues the default lineage. With SciPy's BFGS:

```python
from scipy.optimize import minimize

with linus.Session(inp) as s:
    def energy_and_gradient(x):
        f = s.compute(x.reshape(-1, 3))
        return f.results.energy, f.results.gradient.ravel()

    opt = minimize(energy_and_gradient, water.geometry.ravel(), jac=True, method="BFGS",
                   options={"gtol": 3e-4})
x_opt = opt.x.reshape(-1, 3)
```

Line-search trials and accepted steps alike start from the previous request's orbitals, which is
what you want: the trial geometries are close. For geometry optimizations, transition states and
IRCs use qcopt ([usage.md, section 4](usage.md#4-transition-states-and-geometry-optimizations)),
which opens linus sessions itself.

### Paths: one lineage per image

The images of a nudged-elastic-band path each follow their own lineage, so each image's SCF starts
from that image's previous iteration, not from its neighbour's. The first iteration seeds each
image from the one before it (`seed`: the lineages to start from, in order, when the request's own
lineage has no state yet):

```python
from qcdata import Request

def band_gradients(s, images):              # images: a list of (natoms, 3) arrays, bohr
    requests = [Request(positions=x, lineage=("image", k), seed=[("image", k - 1)] if k else [])
                for k, x in enumerate(images)]
    frames = s.compute_many(requests)       # raises qcdata.ProgramErrors if an image failed
    return [f.results.energy for f in frames], [f.results.gradient for f in frames]
```

A request's own lineage wins over its seed, and a seed naming a lineage without state is a fresh
start, not an error. A request seeded from a lineage sees every earlier request of that lineage in
the same list. Both branches of an IRC work the same way: `lineage=("irc", "forward")` and
`("irc", "backward")`, both seeded from the transition state's lineage.

### Frequencies after an optimization

Open the session for Hessians from the start, and the Hessian at the optimized geometry starts from
the optimization's last orbitals:

```python
inp_h = inp.model_copy(update={"keywords": {**inp.keywords, "freq": {"thermo": "qrrho", "temperature": [298.15, 310.15]}}})
with linus.Session(inp_h, calctypes=["hessian"]) as s:
    ...                                       # the optimization, s.compute(x) per step
    h = s.compute(x_opt, calctype="hessian", outputs=["ir"])
va, th = h.results.vibrational_analysis, h.results.thermochemistry
va.frequencies, va.n_imaginary(), va.ir_intensities       # cm^-1, km/mol
th[0].zero_point_energy, th[0].total.gibbs, th[1].gibbs_correction
h.record["vibrational_analysis"]["warnings"]              # what to check (approximate symmetry, ...)
```

The `freq` settings are fixed when the session opens (temperatures, thermochemistry model,
`freq.ir` for every Hessian). Finite differences fan out from one state without keeping one per
displacement: run each displacement in the center's lineage with `commit=False`, and each starts
from the center's orbitals and leaves the lineage as it was:

```python
center = s.compute(x0, calctype="gradient")
frames = s.compute_many([Request(positions=x0 + d, calctype="gradient", commit=False) for d in displacements])
```

### Orbitals from elsewhere

`ProgramInput.given` holds results already known at the input's structure; linus starts the default
lineage's first SCF from `given.wavefunction.orbitals` (the record says `scf.guess.source:
"given"`), for instance a frame's results with the `orbitals` output, from another session or an
earlier run of the same molecule and basis set. Orbitals of another basis set, or no orbitals, are
refused (`not_applicable`).

```python
first = linus.compute(inp.model_copy(update={"keywords": {**inp.keywords, "output": {"orbitals": True}}}))
warm = linus.compute(inp.model_copy(update={"structure": moved, "given": first.results}))
```

## Many calculations: qccompute

qccompute runs AFH programs by name: the package installs the `qccompute.programs` entry point
`linus` (`linus.qc.program`: capabilities, the settings and record schemas, plans, the numerics
identity, and `compute` and `Session` themselves). A runner never changes a number by itself; it
gives each calculation its share of the machine, and linus decides within it:

```python
import qccompute

out = qccompute.compute(inp)                     # the default runner: linus.compute in this process
outs = qccompute.compute([inp1, inp2, inp3])     # a list, sharing the runner's threads; failures
                                                 # raise together after every input ran
with qccompute.Runner(threads=16, gpus=["0"]) as runner:
    outs = runner.compute(inputs, raise_exc=False)   # failed outputs in place instead
    with runner.session(inp) as s:               # linus's own Session, under the runner's grant
        f = s.compute(x2)
```

- **Threads**: a calculation runs on its `threads` keyword, else on the count its runner gives it
  capped at 8 (one CCD; a direct `linus.compute` without the keyword runs on 1). Results are
  bitwise the same at every thread count. How a runner sets the count (a list's share, a session's,
  1 for a calculation given GPUs) is qccompute's
  [Threads](https://qccompute.docs.afh.org/next/threads/) page. A GPU calculation still does
  host-side work (XC setup, parts of gradients and Hessians): set `threads` on GPU inputs to give
  it more than 1.
  The GPUs a calculation asks for are one when it runs on the device (`backend` `gpu`, or `auto`
  with a GPU), and its memory figures are linus's estimates.
- **The grant** (`ctx`, a `qccompute.program.RunContext`), applied on every call (each
  `compute`, and each session when it opens; a worker serves calls of different threads): linus
  uses the smaller of each resource keyword and the grant, or the grant when the keyword is unset
  (threads: at most 8, as above): `threads`, `memory_gib` and `device_memory_gib` (qc*'s shared resource keywords, left out of the
  calculation key; a runner grants a memory keyword in place of its share, at most what it has, and
  a memory grant comes before linus's own default, the cgroup limit or the RAM), and the GPUs
  (`backend: gpu` or `auto` runs on the first GPU granted, opened by its CUDA ordinal in the process,
  `ctx.device_ordinals()`; with none granted, on the CPU). A keyword the grant caps warns
  (`resource_capped`), and `resolved.settings` records what ran. The memory and device can change a resolved choice, at
  round-off (the one-sided DF coefficients where the tensor does not fit; the GPU's arithmetic):
  pin `jk.df_storage` and `backend` in inputs whose results must be bitwise across machines.
- **Programs with subprograms** (qcopt's optimizations, transition states and IRCs) run linus
  through qccompute, as sessions, so every step after the first starts from the last one's
  orbitals (`record["warm_start"]`) without anything linus-specific.
- **A result store**: `compute(inputs, store=...)` looks every input up first and runs only the
  misses. A store keeps each run with the numerics identity of linus that made it
  (`linus.qc.program.numerics()`, the numerics epoch), and never stores a session's warm-started
  frames: a warm start can reach a different SCF solution.
- **The run's environment**: the runner records the machine's part on every output (`hostname`,
  `cpu_model`, `libc`); linus records what only it knows: the GPU its backend opened (`devices`,
  empty when it ran on the CPU, the device's name as the driver gives it, which is also
  `nvidia-smi`'s) and the request's `peak_device_memory_gib`. A direct `linus.compute` has no runner,
  so no machine fields.

### Progress events

With a progress listener (`progress=` on `qccompute.compute`, `submit`, `session` or a `Runner`'s
methods; qccompute hands it to linus as `ctx.progress`), every calculation reports what it does
while it runs, as small JSON objects with an `event` name (the runner adds `path`, here
`["linus"]`, or the program names down to linus when it runs as a subprogram):

| event | fields | when |
|---|---|---|
| `stage_start` | `stage` | a stage starts: `scf`, `gradient`, `hessian` (its response and assembly), `polarizability` (an IR run's electric-field response) |
| `stage_end` | `stage`, `seconds`, `success` | the stage ends (`success` false when it failed: an SCF that did not converge, an error) |
| `scf_iteration` | `iteration`, `energy` (Eh), `delta_energy` (Eh), `gradient` (max \|F_ai\|) | after each SCF iteration's Fock build, on the CPU and on the device; the numbers are those of the record's `scf.history` |
| `scf_escalation` | `iteration`, `trigger` (`stagnation`, `two_thirds`), `from_iteration` | the SCF restarts within its budget from its lowest-energy orbitals with quasi-Newton steps (the record's `scf.escalation`) |
| `response_iteration` | `solve` (`hessian`, `polarizability`), `iteration`, `residual`, `unconverged`, `total` | after each iteration of a CPHF/CPKS solve: the largest residual left and the right-hand sides not yet converged of `total` |

```python
def show(e):
    if e["event"] == "scf_iteration":
        print(f"SCF {e['iteration']:3d}  E = {e['energy']:.10f}  |g| = {e['gradient']:.1e}")
    elif e["event"] == "stage_end":
        print(f"{e['stage']} done in {e['seconds']:.1f} s")

out = qccompute.compute(inp, progress=show)
```

Events are not results: no number depends on whether anyone listens, nothing in them is stored,
and without a listener linus makes none. A listener runs on the calculation's threads while it
runs (the GIL taken for the call): it should return quickly and never call back into the same
session; an exception it raises is dropped. They cost nothing measurable: on water B3LYP/def2-SVP
(48 ms, 11 events) and HF/STO-3G (10 ms) a no-op listener changed the median wall time by under
0.2%. In a worker (`isolation="process"`) the events cross the pipe as they happen.

### Stopping

`qccompute.stop(future)` (a `submit`ted calculation; qccompute hands the request to linus as
`ctx.stop_requested()`) ends a calculation at its next step and keeps what it reached. linus asks
before each stage (`scf`, `gradient`, `hessian`, `polarizability`) and between the iterations of
its SCF and of its CPHF/CPKS solves, at most every 50 ms there, on the CPU loop and the device
loop alike. The calculation then fails with kind `stopped` (`qcdata.ProgramError`, its output
attached; the message says where: `stopped on request in the SCF after iteration 7`, `stopped on
request before the gradient`) and holds what it reached: stopped in the SCF, the last iterate
(its energy, orbitals and `scf.history`, whose last `step` is `stopped`), as for an SCF out of
iterations; later, the converged SCF with the gradient (and an IR run's Hessian) computed before
the stop. A runner never stores a stopped output. A direct call (no `ctx`) never asks; under a
runner a check costs a GIL acquisition: nothing measurable alone (water B3LYP/def2-SVP gradient,
49 ms, 2 checks), up to a GIL switch interval each beside a busy Python thread.

```python
fut = qccompute.submit(inp)
...
qccompute.stop(fut)                     # ends at its next step
try:
    fut.result()
except qcdata.ProgramError as e:
    assert e.error.kind == "stopped"
    print(e.output.results.energy)      # what it reached
```

### The worker: linus in its own process

In process, a crash of the calculation (a segfault, a GPU fault that poisons the device, an
out-of-memory kill) ends your Python too. A **worker** hosts sessions in a separate process and
answers over a pipe, so the caller survives: the client sees the worker's end, reports it
(`environment`, `worker_died`) and can start a new one; the lineages on the dead worker are lost
and the next step starts fresh, which its record shows. `qccompute.Runner(isolation="process")`
runs every calculation and session in workers. qccompute's serve loop is the worker for every AFH
program; it finds linus by the `qccompute.programs` entry point:

```bash
python -m qccompute.serve linus                  # JSON lines on stdin/stdout, one client
python -m qccompute.serve linus --socket PATH    # a Unix socket, one client at a time
```

Run each worker under its own memory cap (a cgroup: `systemd-run --user --scope -p
MemoryMax=32G ...`, or the pod's), never an address-space limit (CUDA reserves huge virtual
ranges), and give it its GPU when you start it (`CUDA_VISIBLE_DEVICES`): a worker's device is
fixed for its life. Spawn workers; never fork a process whose GPU has started.

The protocol, one JSON object per line:

- The worker's first line is its hello: `protocol` (1), `program`, `version` and `capabilities`
  (every method with its calctypes, and in `extras` the `numerics_epoch`), so a scheduler can
  refuse a request, or a store its results, without importing linus.
- Every message carries an `id`; the reply is `{"id", "ok": true, "result"}` or `{"id", "ok":
  false, "error"}` (an ErrorInfo). `ok: false` is for the message itself (bad JSON, an unknown op or
  session, an open or a plan linus refuses); a calculation that fails is a result, a failed Frame
  or ProgramOutput with its `error`.
- The ops mirror the session: `open {input, calctypes, ctx}` -> `{session}`; `compute {session,
  positions, calctype, accuracy, lineage, outputs, seed, commit}` -> a Frame; `compute_many
  {session, requests}` -> Frames; `reset {session, lineage}` (absent: every lineage; null: the
  default one); `lineages {session}`; `plan {session}`, `{session, calctype}` or `{input}`; `close
  {session}`; and `compute {input, ctx}`, a one-off, -> a ProgramOutput. `ctx` is `{threads,
  memory_gib, device_memory_gib, scratch_dir, gpus}` (`gpus` empty or the worker's own).
- Floats cross as shortest round-trip decimals, so results are bitwise those of the same requests
  in process. NaN and Infinity never cross: a message holding one is refused, and a result holding
  one is a failed frame (kind `numerical`, code `non_finite`).
- End of input closes every session and the worker exits 0. After answering a request whose error
  is `device_lost` (a GPU fault after which nothing can run on that device in the process), the
  worker exits 3.
- On a socket, the sessions outlive a client's connection: the next client continues them.
- stdout carries the protocol only; whatever libraries print goes to stderr, with the logs.

qccompute's runner is the client (`Runner(isolation="process")`): its worker sessions have the
API of `linus.Session`, so code moves between in process and a worker by one argument. By hand, a
client is a few lines:

```python
import json, subprocess, qcdata

w = subprocess.Popen(["python", "-m", "qccompute.serve", "linus"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, text=True)
hello = json.loads(w.stdout.readline())          # {"protocol": 1, "program": "linus", ...}

def ask(**msg):
    w.stdin.write(json.dumps(msg) + "\n")
    w.stdin.flush()
    reply = json.loads(w.stdout.readline())
    if not reply["ok"]:
        raise qcdata.ProgramError(qcdata.ErrorInfo.model_validate(reply["error"]))
    return reply["result"]

sid = ask(id=1, op="open", input=inp.model_dump(mode="json"), ctx={"threads": 8})["session"]
f = qcdata.Frame.model_validate(ask(id=2, op="compute", session=sid, positions=x2.tolist()))
f.success, f.results.energy, f.record["warm_start"]
w.stdin.close(); w.wait()                        # end of input: the sessions close, exit 0
```

Use a worker for GPU runs and long ones; in process for short CPU work. Per request the worker adds
the frame's serialization and the pipe, 0.25 ms on a water B3LYP/STO-3G gradient (a 7.8 kB frame, 36 ms
in process); over whole optimizations a worker's session is as fast as one in process, and on a GPU 1.5-3.8×
faster than a new process per gradient ([bench/session/README.md](../bench/session/README.md#rerun-through-the-session-api-2026-11-13)).

## Analyses and converters

- `linus.vib.harmonic_analysis(numbers, positions, hessian, dipole_derivatives=None, **freq)`: the
  harmonic analysis of any program's Cartesian Hessian, as a `qcdata.VibrationalAnalysis`;
  `linus.vib.thermochemistry(analysis, energy, multiplicity, **freq)`: its ideal-gas
  thermochemistry, one `qcdata.Thermochemistry` per temperature; `linus.vib.atomic_mass(z, table)`.
- `linus.convert`: `to_pyscf(structure, wavefunction)` (a PySCF `Mole`, `mo_coeff`, `mo_energy`,
  `mo_occ` in PySCF's AO order and normalization), `from_pyscf(mol, mo_coeff, mo_energy, mo_occ)`
  (a structure and a `Wavefunction`), `to_pyscf_mol(structure, basis)`, `from_pyscf_mol`:
  functions on qcdata's `Wavefunction` (its orbitals' coefficients with its `basis`) and
  `BasisSet` (with its ECPs and the core electrons each replaces), for any program's orbitals.
  linus's AO order is qcdata's default convention (PySCF's).
- Molden files of a wavefunction: qcinf's `qcinf.molden.write(wavefunction, structure)` (and
  `qcinf.molden.read`), which writes ECP atoms with their cores (`[core]`, afh-stack#67) as linus's
  own file does (`output.molden`, `results.files["orbitals.molden"]`).

## Errors

Every exception linus raises for a calculation is a `linus.LinusError`, also a
`qcdata.ProgramError` (so a program with subprograms catches one class for every program) and a
`RuntimeError`, with `error` (qcdata's `ErrorInfo`: `kind`, `code`, `message`), `kind` (the CLI's
`error_kind`), `results` (what the calculation reached), and the failed `frame` of a session
request or the failed `output` of `linus.compute`. `Session.compute_many` raises
`qcdata.ProgramErrors`, an `ExceptionGroup` of them with every request's frame in `results`:

| class | also a | kind | when |
|---|---|---|---|
| `InputError` | `ValueError` | `input`, `basis` | settings, a molecule or a request linus cannot run as given; codes `unknown_key`, `not_applicable`, `no_effect`, `session_closed` |
| `UnsupportedError` | `NotImplementedError` | `unsupported` | a valid request linus does not implement (an RIJCOSX Hessian, ROHF, a calctype other than energy, gradient, hessian) |
| `ConvergenceError` | | `convergence` | an SCF or a response did not converge (the frame has the SCF history) |
| `ResourceError` | `MemoryError` | `resource` | not enough memory for the request as set up; codes `host_memory`, `device_memory`, with `error.details` `required_gib` and `available_gib` (both as the memory budget, `memory_gib` or `device_memory_gib`: resubmit with `required_gib`) |
| `NumericalError` | `ArithmeticError` | `numerical` | non-finite energies, singular matrices |
| `BackendError` | | `backend` | the GPU backend failed; code `device_lost` when it poisoned the device context |

A bug inside linus (a Rust panic) raises an internal error and closes that session; the process
and its other sessions go on.

## The API at a glance

| name | what |
|---|---|
| `linus.compute(inp, ctx=None)` | one calculation: a `qcdata.ProgramOutput`; a failure raises `LinusError` |
| `linus.Session(inp, *, calctypes=None, ctx=None)` | one molecule, model and settings at many geometries (above) |
| `Session.compute`, `.compute_many`, `.reset`, `.lineages`, `.plan`, `.close`, `.calctypes`, `.closed`, `.input_data` | the session's requests and state (`input_data`: the ProgramInput it was opened with) |
| `linus.qc.program` | linus for qccompute's runners (the `qccompute.programs` entry point; `numerics(spec)`; `python -m qccompute.serve linus` hosts it in a worker) |
| `linus.vib`, `linus.convert` | analyses of any Hessian; PySCF and Molden converters on qcdata's orbitals |
| `linus.settings_schema(calctype=None)`, `linus.settings.Settings` | the settings' JSON Schema; typed stubs for editors |
| `linus.default_options()`, `linus.normalize_options(d)` | every setting at its default; settings as the library reads them |
| `linus.available_functionals()`, `available_basis_sets()`, `available_solvents()`, `basis_file(name)`, `basis_search_path()`, `solvent_table()` | what this build knows |
| `linus.capabilities()`, `linus.__version__`, `linus.NUMERICS_EPOCH` | calctypes, version, the numerics epoch results are keyed on |
| `linus.LinusError` and its subclasses | the errors above (`qcdata.ProgramErrors` groups them) |
