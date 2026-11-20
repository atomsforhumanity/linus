# Using linus: recipes

Each recipe comes in two forms that run the same calculation:

- **CLI**: `linus mol.xyz [flags]` reads an XYZ file (Å) and prints a summary, or JSON with
  `--json`. `linus request.json` or `linus --json < request.json` reads a JSON request instead:
  the molecule as the `structure` group (`xyz`, the text of an XYZ file in Å, or `symbols`/`numbers`
  and `geometry` in bohr, with `charge` and `multiplicity`), plus linus's settings
  (`method`, `basis`, `jk`, `grid`, `solvent`, `scf`, ...). `--set KEY=VALUE`
  sets any setting from the command line (dotted for nested ones: `--set scf.soscf=true`),
  `--options FILE` reads them from a JSON file; `--list-functionals`, `--list-basis` and
  `--list-solvents` print what is available. Every setting, the request and the result:
  [cli.md](cli.md).
- **Python**: `linus.compute(ProgramInput(program="linus", ...))` runs one calculation in this
  process and returns a qcdata `ProgramOutput`; `linus.Session(ProgramInput(...))` computes one
  molecule at many geometries, each step from the last one's orbitals ([python.md](python.md)).
  The method and basis set go in `model`, every other setting in `keywords`. A qcdata
  `Structure` (`Structure.open("mol.xyz")`) carries the positions (bohr), charge and
  multiplicity. `qccompute.compute(...)` runs the same input, or a list of them (linus is a
  qccompute program; [python.md](python.md#many-calculations-qccompute)).

Everything is in atomic units (Eh, bohr, e·bohr). Python needs the package in a venv
(`pip install afh-linus`, which brings qcdata, or `maturin develop --release` in
`crates/linus-py`). The Python snippets assume

```python
import linus
from qcdata import ProgramInput, Structure
s = Structure.open("mol.xyz")
```

The settings have the same names in both forms (`linus::CalcOptions`; the JSON request's keys,
qcdata `keywords`). Three apply to every recipe:

- **`memory_gib`** (CLI `--memory GIB`): the memory of the whole process: its peak resident
  memory stays within it. Every budget derives from it: the SCF's DF tensors up to 0.75 of it
  (less the PCM's matrices), the H-bar blocks of RI Hessians a quarter, the XC-kernel and response
  buffers an eighth (at most 1 GiB each). A calculation whose phases do not fit fails before any
  work with a `host_memory` error giving the `memory_gib` that holds it; the plan states it first
  (`memory.required_gib`, [Memory and storage modes](#memory-and-storage-modes)).
  Unset: from the machine's total RAM, or the cgroup limit of a memory-capped scope if lower;
  only the DF tensors also stay within the memory free when the SCF starts. Set it when sharing a
  node, and for the same DF storage, and the same bits, on every machine. Memory never selects a
  compressed DF tensor: one that does not fit in f64 is a `host_memory` error naming the
  `memory_gib` needed.
- **`resolved`** in every result (Python: `out.resolved.settings` and the run record
  `out.record`): what the calculation actually used, with every `auto` resolved: grid, auxiliary basis, NLC grid,
  SCF thresholds and dropped functions, the dispersion implied by the method, exact-integral
  and DF storage, memory budgets and their source, the response settings with the CPKS grid and
  the residuals reached, PCM preset and SMD smoothing, the backend that ran each kernel family,
  and any `LINUS_*` debug variables that were set. Sizes and peak memory are in `resources` (Python:
  `out.record["sizes"]`, `out.execution.peak_memory_gib`), seconds in `timings` (`out.execution.timings`;
  `total` is the whole call).
- **Errors** are typed, with one vocabulary of kinds ([conventions.md](conventions.md#errors)).
  Python: `linus.InputError` (also a `ValueError`: bad settings, unknown basis set or
  functional), `UnsupportedError` (a `NotImplementedError`), `ResourceError` (a `MemoryError`),
  `ConvergenceError` (an SCF or CPHF/CPKS that did not converge), `BackendError` (the GPU backend
  failed) and `NumericalError` (an `ArithmeticError`), all `linus.LinusError` (a
  `qcdata.ProgramError`) with `.error` (`kind`, `code`, `message`), the partial `.results`, and
  the failed `.frame` of a session request (with the SCF history) or the failed `.output` of
  `linus.compute`.
  CLI: exit code 2 for bad input, 1 for a failed calculation; with `--json` a failure prints
  `{"success": false, "error": ..., "error_kind": ..., "scf": ...}`.

## 1. ωB97X-D3BJ/def2-TZVP in THF (C-PCM)

A production single point: ωB97X-D3BJ (the ωB97X-V semilocal part with VV10 replaced by
two-body D3(BJ), as ORCA's `wB97X-D3BJ`) with RIJ+COSX exchange, in THF.

```bash
linus mol.xyz --method wb97x-d3bj --basis def2-tzvp --jk rij-snk --solvent thf --solvent-model cpcm --threads 8 --json > out.json
# JSON request: {"structure": {"xyz": "..."}, "method": "wb97x-d3bj", "basis": "def2-tzvp", "jk": "rij_snk",
#                "solvent": {"name": "thf", "model": "cpcm"}, "threads": 8}
```

```python
inp = ProgramInput(program="linus", calctype="energy", structure=s,
                   model={"method": "wb97x-d3bj", "basis": "def2-tzvp"},
                   keywords={"jk": "rij_snk", "solvent": {"name": "thf", "model": "cpcm"}, "threads": 8})
out = linus.compute(inp)
c = out.results.energy_components
out.results.energy, c["solvation_electrostatic"], c["dispersion"], out.record["solvation"]
```

`--jk df` (RI-JK) instead of `rij-snk` is faster below about 1100 basis functions and more
accurate (section 7). The C-PCM preset is PySCF 2.15's `pyscf.solvent.pcm` (modified Bondi
radii x 1.2, SWIG, 302 Lebedev points per sphere); `--solvent-model iefpcm|cosmo|ssvpe|smd`,
`--eps X` for a dielectric constant, `--set solvent.preset=orca` for ORCA-like settings. SMD adds the
CDS term (`energy_components["solvation_cds"]`). Every model, preset and option, with the validation
against PySCF, Psi4 and ORCA: [solvation.md](solvation.md).

## 2. A gradient

```bash
linus mol.xyz --method b3lyp --dispersion d3bj --basis def2-svp --jk df --grad --threads 8 --json
# JSON: "calctype": "gradient"
```

```python
inp = ProgramInput(program="linus", calctype="gradient", structure=s,
                   model={"method": "b3lyp-d3bj", "basis": "def2-svp"}, keywords={"jk": "df", "threads": 8})
out = linus.compute(inp)
out.results.energy, out.results.gradient          # gradient: (natoms, 3), Eh/bohr
```

The gradient's parts (nuclear repulsion, one-electron, two-electron, XC, dispersion) are the CLI's
`gradient_parts` and Python's `out.results.extras["gradient_components"]`. In every form a dispersion suffix on the method (`b3lyp-d3bj`, `-d3(bj)`,
`-d3bj-atm`, `-d4`) sets the dispersion, unless linus knows the whole name as a functional
(`wb97x-d3bj`). Bare `d3` (`--dispersion d3`, `b3lyp-d3`) is rejected as ambiguous (Psi4 and
qcdata mean zero damping, ORCA Becke-Johnson): write `d3bj`. D3 with zero damping is not
implemented, so `wb97x-d3` is rejected before the SCF; use `wb97x-d3bj` or `wb97x-v`.
Gradients default to tighter SCF thresholds than energies (max |F_ai| < 1e-7)
and include the XC grid-weight derivatives (`grid.response`, default on).

## 3. Frequencies with quasi-RRHO thermochemistry

An analytic Hessian carries the harmonic analysis and thermochemistry (the settings group
`freq`; there is no separate frequencies calculation), here with Grimme's quasi-RRHO entropy and
Head-Gordon's quasi-RRHO enthalpy (`freq.thermo = qrrho-hg`, cutoff 100 cm⁻¹) and IR
intensities.

```bash
linus mol.xyz --method b3lyp --basis def2-svp --jk df --thermo qrrho-hg --ir --temperature 298.15 --threads 8 --json
# JSON: "calctype": "hessian", "freq": {"thermo": "qrrho-hg", "ir": true, "temperature": 298.15}
```

```python
inp = ProgramInput(program="linus", calctype="hessian", structure=s,
                   model={"method": "b3lyp", "basis": "def2-svp"},
                   keywords={"jk": "df", "threads": 8, "freq": {"thermo": "qrrho-hg", "ir": True, "temperature": 298.15}})
res = linus.compute(inp).results
va, th = res.vibrational_analysis, res.thermochemistry[0]           # one thermochemistry per temperature
va.frequencies, va.n_imaginary(), va.ir_intensities                 # cm^-1 (imaginary negative), km/mol
th.zero_point_energy, th.total.gibbs, th.gibbs_correction           # Eh
```

Hessians use linus's second-derivative defaults unless set: grid `linus:3` (`auto`), the CPKS
kernel on `linus:1` (meta-GGAs: `linus:2`), first-order XC grid response (as ORCA's analytic Hessian), RI-JK with the
`auto` auxiliary basis (cc-pVTZ-jkfit, def2-universal-jkfit for elements it lacks), SCF to max
|F_ai| < 1e-9. Hessians need `jk` exact or df. The `freq` settings: `freq.analysis`
(`harmonic`, `off`), `freq.thermo` (`rrho` default, `qrrho`, `qrrho-hg`, `quasiharmonic`, `off`),
`freq.temperature` (a list gives one thermochemistry per temperature), `freq.pressure`,
`freq.concentration`, `freq.scale`, and the rest by `--set` (`--set freq.symmetry_number=2`,
`freq.mass_table: isotope|average`, `freq.projection`); the CLI has `--ir`, `--thermo`,
`-T/--temperature`, `--pressure`, `--concentration` and `--freq-scale` for the common ones. A setting
that cannot take effect (`freq.rotor_cutoff` with `rrho`) is an error. Any program's Hessian goes through the same analysis
with `linus.vib.harmonic_analysis(numbers, positions, hessian)` (a `qcdata.VibrationalAnalysis`).

## 4. Transition states and geometry optimizations

The CLI computes single points only; optimizations, transition states and IRCs are programs
with subprograms. qcopt runs linus as its subprograms, through qccompute, as a session per task
(linus is the `qccompute.programs` entry point `linus`), so every step starts from the previous
step's orbitals, the stability probes run at a task's first SCF (and again only when the
HOMO-LUMO gap falls), and finite-difference displacements start from their center without
changing it. Nothing linus-specific to pass:

```python
import qcopt

lin = {"program": "linus", "model": {"method": "b3lyp-d3bj", "basis": "def2-svp"}, "keywords": {"jk": "df", "threads": 8}}
sub = [{**lin, "calctype": "gradient"}, {**lin, "calctype": "hessian"}]   # linus's analytic Hessians
ts = qcopt.compute(ProgramInput(program="qcopt", calctype="transition_state", structure=guess,
                                keywords={"freq": {"check": True}}, subprograms=sub))
last = ts.results.frames[-1]
path = qcopt.compute(ProgramInput(program="qcopt", calctype="irc", structure=last.structure,
                                  given=last.results, keywords={"ends": {"minimize": True}}, subprograms=sub))
```

The `hessian` subprogram gives the search exact Hessians (it may be another program);
`freq.check` computes the Hessian at the final structure and reports its frequencies (a minimum
has none imaginary, a transition state one), and the IRC given that frame's results starts from
its Hessian instead of computing it again. `calctype="optimization"` minimizes. The frames keep
linus's records (`frame.record["warm_start"]`, `["scf"]["guess"]`).

For your own loops, a `linus.Session` does the same: each `compute(x)` starts from the last
step's orbitals ([python.md](python.md#an-optimization-loop)).

## 5. Effective core potentials

def2 basis sets put ECPs on Rb-Rn (ECP28MWB, ECP46MWB, ECP60MWB, ...). linus applies them
automatically, for energies, gradients and Hessians, closed and open shells, with every `jk`.
Nothing to set:

```bash
linus pdcl2.xyz --method b3lyp --basis def2-svp --jk df --grad --threads 8 --json
```

```python
out = linus.compute(ProgramInput(program="linus", calctype="gradient", structure=s,
                                 model={"method": "b3lyp", "basis": "def2-svp"}, keywords={"jk": "df", "threads": 8}))
out.results.wavefunction.orbitals.core_electrons   # core electrons an ECP replaces, per atom
```

Validated against PySCF (agreement ~1e-12 Eh), Psi4 and ORCA 6.0.0 for HI, CH3I, CH2I, AgCl,
PtCl4²⁻, Au2/Au2⁺, CsCl and PdCl2(PH3)2 (reference/validation/README.md). Only def2-type
(semi-local, BSE) ECPs are supported.

## 6. VV10 functionals (ωB97M-V, ωB97X-V, B97M-V)

Energies, gradients, Hessians and frequencies, with the VV10 nonlocal correlation evaluated
self-consistently on its own grid (`grid.nlc`, default `linus:1`; with `grid.xc` `pyscf:N` it is PySCF's NLC grid, with
`orca:N` ORCA 6's NL grid, with `psi4` Psi4's VV10 grid):

```bash
linus mol.xyz --method wb97m-v --basis def2-tzvp --jk df --grad --threads 8 --json      # result: "vv10" block
```

```python
out = linus.compute(ProgramInput(program="linus", calctype="gradient", structure=s,
                                 model={"method": "wb97m-v", "basis": "def2-tzvp"}, keywords={"jk": "df", "threads": 8}))
out.record["vv10"]        # VV10 energy (included in energy_components["xc"]), NLC grid and point counts
```

Hessians, frequencies, IR intensities and polarizabilities with VV10 include its CPKS kernel and
its explicit second-derivative terms on the NLC grid, with the NLC grid's first-order response
in the default convention (`grid.response`, as the XC grid's). The kernel is a double
sum over the NLC grid's points for every perturbation, so with ωB97M-V it is most of a CPU
Hessian's response time; with `backend: "gpu"` its pair sums run on the device. The -D3BJ variants
(`wb97m-d3bj`, `wb97x-d3bj`, `b97m-d3bj`) drop VV10 for D3(BJ). NLC grid accuracy:
[grids.md](grids.md#vv10-nonlocal-correlation-grids).

## 7. Choosing `jk` and the grid

| `jk` | What | When | Measured (linus, 8 cores) |
|---|---|---|---|
| `df` (default) | RI-JK, one auxiliary basis (`aux`, default `auto`: def2-universal-jkfit; cc-pVTZ-jkfit for Hessians) | production up to ~1100 basis functions; Hessians | ωB97X-D3BJ energy / gradient (s): L-lactide SVP 0.9 / 1.6, 39-atom anion SVP 11.8 / 21.4, 61-atom TS SVP 37.6 / 62.5, L-lactide TZVP 3.3 / 5.5 |
| `exact` | integral-direct four-center J and K | small molecules, references, validation (RI-JK's error: energies 1e-5 to 1e-4 Eh, gradients ~1e-5 Eh/bohr, frequencies below 1 cm⁻¹) | 61-atom TS, B3LYP/def2-TZVP gradient: 777 s (an untimed reference run on another CCD), against 103-153 s with RI-JK |
| `rij_snk` (`--jk rij-snk`) | RI-J (def2-universal-jfit) + seminumerical (COSX) exchange on ORCA-like grids (`jk.k_grid` `cosx:2`) | large molecules (faster than RI-JK from about 1100-1200 basis functions), tight memory | same cases: 3.0 / 4.6, 19.2 / 26.4, 50.4 / 80.3, 10.4 / 14.9; 113-atom TS ωB97X-D3BJ/def2-TZVP energy 619 s against 1511 s for RI-JK |
| `incore` | exact four-center integrals held in memory (`jk: "exact"` with `jk.exact_storage: "incore"`) | tests, tiny molecules | |

Accuracy and memory. RI-JK is 2-3 times more accurate than RIJ+COSX: RIJ+COSX gradients are
within 4e-5 to 2.4e-4 Eh/bohr of exact J/K (B3LYP, L-lactide to 61 atoms), the COSX part alone
8e-6 to 2.4e-4. RI-JK keeps a 3-index tensor in memory (up to 0.75 of the memory limit; beyond
that, for range-separated hybrids, the one-sided coefficients, else a `host_memory` error unless a
compressed storage is set: [Memory and storage modes](#memory-and-storage-modes)), at
def2-TZVP 6-12 GB for the 39-atom anion (B3LYP / ωB97X-D3BJ) and 51 GiB per f64 tensor for the
113-atom TS (ωB97X-D3BJ: 102 GiB for B and B^ω, 51 GiB one-sided, 36 GB one-sided in F40);
RIJ+COSX needs 4-11 GB for the 39-, 61- and 113-atom cases. RIJ+COSX has no
Hessians. For Hessians the `auto` auxiliary basis matters: its RI error in the lactide
frequencies is 0.18-0.26 cm⁻¹ against 0.61-0.78 with def2-universal-jkfit.

The 3-index storage follows the memory budget
([Memory and storage modes](#memory-and-storage-modes)) and the `jk` settings: `jk.df_storage`
(`auto`: `f64`; the compressed `f48`, `f40`, `f32` only when set) and `jk.df_long_range` for range-separated hybrids (`auto`, `stored`, `one_sided`).
`resolved.record.df` reports the storage chosen and its size (on the GPU backend also where the
three-center integrals were computed, `integrals`). For example
`linus big.xyz --method wb97x-d3bj --basis def2-tzvp --jk df --memory 32 --set jk.df_storage=f48`.

| `grid` | Use |
|---|---|
| `auto` (default) | PySCF level 3 for energies and gradients, `linus:3` for Hessians and frequencies |
| `linus:2` | about half the points of PySCF level 3: 5-30% less wall time at twice the gradient error (39-atom anion: 38-44 µEh from the converged grid against 5-16 µEh) |
| `linus:3`, `linus:4` | tight geometry optimizations (smallest gradient errors), Hessians |
| `pyscf:N`, `psi4`, `orca:N` | the same points as PySCF, Psi4 or ORCA (DefGrid1-3), to compare with those programs |

Do not use `orca:2` for Hessians: it is too coarse for the lowest modes (off by more than
1 cm⁻¹ even with grid response). Details and measurements: [grids.md](grids.md).

### Memory and storage modes

`memory_gib` limits the whole process: each phase of a calculation (the SCF, the gradient, the
Hessian, the polarizability of IR intensities) holds the process itself (0.03 GiB for the
command line, 0.09 GiB for a Python process, 0.6 GiB more with a GPU), the large arrays and
workspaces its engines size from the problem alone (the DF tensors and the exchange's blocks,
the DF gradient's blocks, a response's arrays), its shares of the limit (the H-bar blocks, the
response and XC-kernel buffers) and a measured allowance for the matrices, the XC buffers, each
thread's buffers and the allocator. The block sizes never follow the limit, so neither do the
results (only the DF storage below does, at round-off, and `resolved.record.df` says which). The
plan (`linus --plan`, Python's `Session.plan`, qccompute's plan) gives `memory.required_gib`, the
smallest `memory_gib` that runs the calculation (with f64 DF storage), and `memory.phases`, what
each phase holds at the limit (`parts`: `what`, `gib`) and needs (`required_gib`); below it the
run fails before any work with a `host_memory` error whose `required_gib` is that number. Measured
on the command line at one thread under a cgroup limit equal to `memory_gib`, ωB97X-D3BJ/def2-TZVP
(def2-universal-jkfit, grid `linus:2`): water 0.2 GiB, caffeine 1.8 GiB for the energy and
2.3 GiB for the gradient, lidocaine 3.3 GiB for the energy; the peak resident memory stayed below
it in every case (bench/memfit/README.md). Without `memory_gib` the same check runs against the
memory the calculation may hold now (the machine's limit, at most 90% of the memory free), so a
calculation the node cannot hold fails before any work instead of being killed; it is a check only,
and no storage or number follows it.

The SCF's 3-index tensors (B, and B^ω for range-separated hybrids; the raw integrals of RI-J;
the in-core exact integrals) may take 75% of the memory limit, less the memory reserved for the
large allocations held with them: the PCM's dense surface matrices while they are built (2n²
doubles for C-PCM and COSMO, 6n² for IEF-PCM, SS(V)PE and SMD, for n surface points). The limit
is `memory_gib`, else the machine's total RAM or the cgroup limit of a memory-capped container or
systemd scope, whichever is lower, with no fixed ceiling. With `memory_gib` unset a guard also
keeps the tensors within the memory free when the SCF starts (MemAvailable, within a capped
cgroup's headroom): at most 90% of it, less the reservations. On a dedicated node or pod the
limit's 75% binds; the guard binds only on a node too full for it, where a range-separated
hybrid takes its one-sided storage instead of risking an out-of-memory kill, and says so on
stderr. The tensors live only during the SCF; the budgets of gradients and Hessians start after
them.

The storage (`jk.df_storage` `auto`) is always f64: B (and B^ω) when they fit; for
range-separated hybrids beyond that, the fitted coefficients with the long-range side's integrals
computed per build (`coef`, one-sided, f64); without exchange (pure functionals) the raw RI-J
integrals, else integral-direct J. With `memory_gib` a storage is taken only if, with its
exchange workspace, it also fits what the rest of the process leaves it. A budget, a runner's
memory grant included, never selects a compressed tensor: when no f64 storage of the exchange
tensor fits, the calculation fails with a resource error of code `host_memory` whose message gives
the size needed and the `memory_gib` that holds it. The compressed storages (F48, F40, F32: `B48`, `coef48`, ...) run only when
`jk.df_storage` sets them. `resolved.record.df` reports the storage, its size and the budget;
`resolved.record.memory` the limit and its source, what set the DF budget
(`df_tensor_budget_source`: `available memory` when the guard lowered it), the free memory read
(`available_gib`) and the `reservations`.

Reproducibility. The storage modes agree to round-off, not bit for bit (they factor the fit
differently; F48 and F40 round the stored values), measured against B in f64 on ten DF cases (seven range-separated: CH3SH, NH3+ and la18
Hessians, CH2, water ωB97M-V, lidocaine and the 39-atom anion/def2-TZVP gradients; three B3LYP),
with the same SCF iterations in every mode:

| storage | max \|ΔE\| (Eh) | max \|Δgradient\| (Eh/bohr) | max \|ΔHessian\| (Eh/bohr²) | max \|Δfrequency\| (cm⁻¹) |
|---|---|---|---|---|
| `coef` (one-sided, f64) | 9e-13 | 1e-12 | 5e-8 | 8e-5 |
| `B48`, `coef48` | 9e-11 | 3e-11 | 6e-8 | 1e-4 |
| `B40`, `coef40` | 4e-9 | 2e-10 | 4e-8 | 9e-5 |

The Hessians' differences come through the single-precision response exchange of RI Hessians,
which carries any round-off of the SCF to its own level. For scale, linus and PySCF agree to
1e-12-3e-12 Eh and 4e-11-4e-10 Eh/bohr on the global-hybrid DF validation cases: the one-sided f64
storage differs from B by less, F48 by about as much.

The compressed rows apply only where `jk.df_storage` sets them. Under `auto` the memory can choose
only between B (and B^ω) and the one-sided coefficients of a range-separated hybrid, which agree
to f64 round-off: with `memory_gib` unset, one binary can store such a molecule's tensors
differently on machines with different memory (and, through the guard, on a full node), and its
results then differ at that level. For the same storage, and the same bits, on every machine, set
`memory_gib` (or `jk.df_long_range`).

What the memory buys: at the 113-atom TS (ωB97X-D3BJ/def2-TZVP RI-JK, 2155 basis functions) B
and B^ω take 102 GiB, the one-sided coefficients 51 GiB, and the energies agree to 1.3e-11 Eh. In
a 160 GiB pod the default now keeps B and B^ω: with 32 threads (Xeon 8470) a J/K build takes 30 s
against 52-58 s one-sided, the energy 577 s against 857-950 s. With 8 threads (Zen 5) the gain is
small (41-43 s against 48-49 s per build): the one-sided exchange's integrals per build are what
stop scaling with threads.

## 8. A hard SCF

The SCF uses DIIS with a level shift of 0.25 Eh until max |F_ai| < 1e-2 and incremental Fock
builds. For an SCF that oscillates or converges to the wrong state, raise the shift and keep it
longer, damp the first iterations, switch to second-order steps near convergence, and allow
more iterations; for UHF/UKS, check stability:

```bash
linus mol.xyz --method b3lyp --basis def2-svp --charge 1 --mult 2 --max-iter 300 --guess core \
    --set scf.level_shift=0.5 --set scf.level_shift_until=1e-4 --set scf.damping=0.5 --set scf.damping_until=1e-2 \
    --set scf.soscf=true --set scf.stability=always --json
# or --options scf.json with {"scf": {"level_shift": 0.5, ...}}, or the same keys in a JSON request
```

```python
keywords = {"scf": {"max_iter": 300, "guess": "core", "level_shift": 0.5, "level_shift_until": 1e-4,
                    "damping": 0.5, "damping_until": 1e-2, "soscf": True, "stability": "always"}}
```

The `scf` settings ([cli.md](cli.md)) hold the thresholds, iterations and guess (`convergence`
presets `loose` ... `extreme`, `e_tol`, `g_tol`, `max_iter`, `guess`: `sad` default, `core`) and
the algorithm (`crates/linus-scf/src/settings.rs`, `ScfAlgorithm`): `diis_space` (10),
`diis_start`, `damping` and `damping_until`, `level_shift` (0.25) and `level_shift_until`
(1e-2), `adiis` and `adiis_until` (ADIIS while max |[F,D]| is large), `soscf` and
`soscf_from` (quasi-Newton orbital rotations below max |F_ai| = 3.3e-3), `stability` (`auto`, `always`, `off`) and
`stability_max` (stability analysis of every converged UHF/UKS determinant; by default only
when the SCF looks like it stopped on a saddle point; `stability_probe`: a cheap check of
every converged one, two Hessian-vector products by default at a lineage's first SCF
(single points, the first SCF of Hessians and optimizations), none at later geometries, `0`
turns it off; on a minimum the result is unchanged, on a saddle it runs the full analysis and
steps down. Every search runs once per irreducible representation of the converged state's
abelian point group (D2h and its subgroups) whose lowest orbital pair lies near the lowest one,
so a negative mode of another symmetry than the lowest pair's is found however the arithmetic
rounds; each record says which group, irrep and how many irreps (`symmetry`, `irrep`,
`irrep_index`, `irreps_searched`);
`resolved.record.scf.stability_probe`; the GPU's device SCF loop, `backend.scf_loop: "device"`, runs
it on the device), `stability_closed_probe` and `stability_closed_gap` (a closed-shell RKS solution can
lie above a broken-symmetry UKS one: stretched bonds, singlet diradicals, some transition-metal
complexes. When the RKS HOMO-LUMO gap is below `stability_closed_gap`, 0.15 Eh, two Hessian-vector
products of the UKS orbital Hessian look for it, by default for KS-DFT at a lineage's first
SCF; Hartree-Fock is opt-in with `stability_closed_probe: 2`, `0` turns it off. A negative mode
prints a warning and is recorded in `resolved.record.scf.closed_shell_probe`, with `skipped` saying why
the probe did not run otherwise; the SCF never switches and its result is unchanged. Rerun with
`reference: "unrestricted"` (and `scf.stability: "always"`) to get the lower solution. Both default probes
run at a session lineage's first SCF (a one-off calculation's only SCF); a later request that
continues the lineage (an optimization, IRC or finite-difference step, [python.md](python.md#sessions))
runs them only when the HOMO-LUMO gap (open shells: the smaller alpha
or beta gap) has fallen below `scf.stability_closed_gap` since the lineage's last probe, or more than
0.02 Eh below that probe's gap: `resolved.record.scf.probe_trigger` holds `last_gap`, `gap` and
`trigger: "gap_dropped"`), `two_cycle` (quasi-Newton steps once the energy repeats every other
iteration; `resolved.record.scf.two_cycle_at`), `plateau_window` (an SCF whose energy is stationary
and whose orbital gradient is small but stuck, as for open-shell atoms in a PCM, is accepted when
its escalation comes due or at `max_iter`, with qcdata's shared warning `loosely_converged`, the residual in `resolved.record.scf.plateau`; under
accuracy targets only when the error estimates of its final orbital gradient meet them),
`escalate` (on: an SCF whose DIIS stagnates, its lowest energy not improved for
`stagnation_window` iterations, or that has spent 2/3 of `max_iter`, restarts once from the
orbitals of its lowest-energy iteration with quasi-Newton steps and no level shift, for the rest
of `max_iter`, unless the plateau rule takes it there; `resolved.record.scf.escalation`
says why it started, or why it did not run in an SCF that used its budget), and `jk.incremental`.
`max_iter` is the SCF's whole budget, the escalation's iterations included, and its last
iteration is always a full Fock build. Also: `reference: "unrestricted"`
(UHF/UKS for a closed shell, e.g. to break symmetry). The result's `scf.history`
has every iteration (energy and its change, max and RMS |F_ai|, density change, the kind of
step, timings) and `scf.stability` the stability analyses, for diagnosis. An SCF that does not
converge raises `linus.ConvergenceError` from a session request (`.frame` holds the unconverged
result and its history; the message names the last |dE| and max |F_ai| and these remedies;
`linus.compute` returns it as a failed output); the CLI exits
with code 1 and, with `--json`, prints `{"success": false, ..., "scf": ...}` with the history.

## 9. On a GPU

A build with the `gpu` feature ([gpu.md](gpu.md)) runs the density-fitted SCF (an implicit
solvent's surface terms included), the gradient's two-electron, XC and PCM terms and the
response equations of Hessians on an NVIDIA GPU; the rest stays on the CPU, each part with its
reason in `resolved.record.backend`:

```bash
linus mol.xyz --method b3lyp --basis def2-tzvp --jk df --backend gpu --threads 8 --hess --json
```

```python
out = linus.compute(ProgramInput(program="linus", calctype="gradient", structure=s,
                                 model={"method": "b3lyp", "basis": "def2-tzvp"},
                                 keywords={"jk": "df", "backend": "gpu", "threads": 8}))
out.results.gradient, out.record["backend"]["fallbacks"]     # where each part ran, and why on the CPU
```

For many geometries on a GPU, keep one session (the device context, its memory pool and kernel
modules are set up once: 0.3-0.8 s per new process), in a worker process for isolation
([python.md](python.md#the-worker-linus-in-its-own-process)).

`backend: "auto"` takes the GPU when there is one. Results agree with `backend: "cpu"` to
round-off and are reproducible bit for bit on one device; a failed device stops the calculation
with error kind `backend` (`linus.BackendError`) rather than continuing on the CPU.

## 10. Reproducibility

On the CPU, one linus binary running with one C math library gives the same bits for the same
input:

- at any thread count (`threads`);
- run to run and machine to machine, across CPU vendors: the validation matrix (234
  calculations, at 1, 2, 3, 4, 8, 16 and 32 threads) and the merge gate's 48-case reference suite
  (Hessians, frequencies and polarizabilities of up to 39 atoms included) give identical bits on
  AMD Zen 5 and Intel Sapphire Rapids;
- on AVX2 and AVX-512 CPUs alike: linus's kernels and faer's matrix products run AVX-512 code on
  AVX-512 CPUs and AVX2 code otherwise, with the same summation order per element, so the bits
  are the same (the reference suite on an emulated AVX2-only CPU, and with `LINUS_ISA=avx2`, is
  identical to the AVX-512 runs);
- for any number of the DF Hessian's exchange passes, of the response exchange's density groups
  and of the VV10 kernel's perturbation groups, which `memory_gib` (and for VV10 the thread count)
  sizes: a matrix product cut into such parts takes the kernel of the whole product (since 0.1.1;
  in 0.1.0 these results could change in their last bits with the grouping).

Four things can change the last bits, never more than round-off:

- **The C math library.** linus, and libxc inside it, take `erf`, `erfc`, `exp`, `pow` and the rest
  from the system's C library (glibc's libm), so the same binary on another glibc can round
  differently. glibc 2.36 (Debian 12, the linus container images) and 2.43 (Ubuntu 26.04) differ in
  `erf` (2% of arguments, at most 1 ulp) and `erfc` (31%, at most 2 ulp); `exp`, `log`, `pow`,
  `cbrt`, `expm1`, `atan`, `sin`, `cos` and `tanh` agree. That moves the results that call
  them, range-separated functionals (ωB97X, ωB97X-D3BJ, CAM-B3LYP, ωB97M-V), implicit solvent and
  D4: in the reference suite energies by up to 2e-12 Eh and gradients by 3e-13 Eh/bohr (39 atoms),
  and, through the CPKS iterations, Hessians by up to 5e-7 Eh/bohr² and harmonic frequencies by
  1e-3 cm⁻¹ (caffeine, ωB97X-D3BJ); small molecules move far less (~1e-13 Eh, ~1e-10 Eh/bohr²).
  The others keep every bit. For bit-identical results on several machines, run the same
  container image (or the same OS release) everywhere.
- **CPUs without AVX2 and FMA** (and non-x86 builds) run linus's generic kernels, which round
  differently: `LINUS_ISA=generic` on a Sapphire Rapids moved energies by up to 9e-13 Eh and
  gradients by 5e-13 Eh/bohr, and, through the CPKS iterations, Hessians by up to 1e-6 Eh/bohr²
  (frequencies 2e-3 cm⁻¹).
- **Another build** (compiler version, build options) is another binary; compare results of one
  build.
- **The memory.** With `memory_gib` unset, a range-separated hybrid's DF tensors follow the
  machine's memory (and, on a full node, the free memory): B and B^ω, or the one-sided
  coefficients, which agree to f64 round-off (9e-13 Eh); set `memory_gib` (or `jk.df_long_range`)
  for the same storage everywhere. Memory never selects a compressed (F48, F40) tensor: those run
  only when `jk.df_storage` sets them, and a budget too small for f64 is a `host_memory` error
  ([Memory and storage modes](#memory-and-storage-modes)).

GPU results are reproducible bit for bit on one device and agree with the CPU to round-off
(section 9, [gpu.md](gpu.md)).

## 11. Accuracy targets

`accuracy` (qc*'s shared keyword, an object: `{"energy": 1e-8, "gradient": 1e-6}`) states the
error from convergence a caller can tolerate: in the energy (Eh), the largest gradient component
(Eh/bohr) and the largest Hessian element (Eh/bohr²). linus turns it into its SCF and response
thresholds and records what it did; the basis set, grids and density fitting are approximations
it does not cover. A session request's own `accuracy` (an object, or a number for its calctype's
highest derivative) replaces the input's for that request; qcopt sends them.

```python
out = linus.compute(ProgramInput(..., calctype="gradient", keywords={"accuracy": {"gradient": 1e-5}}))
out.record["accuracy"]
# {"target": {"gradient": 1e-05}, "thresholds": {"scf.e_tol": 1.5e-11, "scf.g_tol": 2.8e-07},
#  "estimate": {"gradient": 7.8e-07}, "met": true}         (water B3LYP/def2-SVP: 9 iterations, 10 by default)
with linus.Session(inp) as s:
    f = s.compute(x, accuracy=1e-3)          # a loose gradient for an early optimization step
```

The errors follow the SCF's orbital gradient and the response's residual. Measured against
references converged far tighter (max |F_ai| 1e-10, response residual 1e-10) for HF, B3LYP and
PBE0, closed and open shells (water, NH3⁺, CH3, triplet p-benzyne, caffeine and its cation;
def2-SVP), with g the max |F_ai| the SCF converges and gn PySCF's norm of the orbital gradient:

| error | in g | in gn |
|---|---|---|
| energy | ≤ 153 g² | ≤ 3.4 gn² |
| largest gradient component | ≤ 10.5 g | ≤ 0.90 gn |
| largest Hessian element, from the SCF | ≤ 11.2 g | ≤ 0.57 gn |
| largest Hessian element, from the response (residual r) | ≤ 41 r² (r ≤ 1e-3), ≤ 120 r² (r ≤ 1e-2) | |

The factors in g grow with the molecule (3-5 atoms: 15 g², 0.6 g; caffeine the largest) and
those in gn do not. The SCF tests g, so the targets map to thresholds with the bounds in g and a
factor 3 of margin:

| target | `scf.e_tol` | `scf.g_tol` | `response.tol` |
|---|---|---|---|
| energy E | E | √(E/600) | |
| gradient G | 200 g_tol² | G/36 | |
| Hessian H | 200 g_tol² | H/36 | √(H/360); below H = 1e-7 under 1e-6, which keeps the response exchange in double precision |

- A request takes the tightest thresholds of its targets that apply to it: an energy target to
  every request, a gradient target to gradients and Hessians, a Hessian target to Hessians. A
  target of a calctype the input cannot run is an error (`not_applicable`), like any setting.
- Given `scf.e_tol`, `scf.g_tol`, `scf.convergence` and `response.tol` are floors: a target
  tightens them, never loosens them. A target replaces linus's defaults, so a loose one runs a
  cheaper SCF (water B3LYP/def2-SVP: 9 iterations for a gradient target of 1e-5, 10 by default).
- `record.accuracy` holds the `target`, the `thresholds` used, the `estimate` of each targeted
  error from the final orbital gradient (the bounds in gn, which do not grow with the molecule)
  and response residual, and `met`: every estimate within its target. `resolved.settings` has
  the thresholds, with the targets as their reason.
- Without `accuracy` nothing changes: the defaults and every number are as before.

## 12. The Python API

`linus.compute` and `linus.Session`, lineages, many calculations through qccompute (lists, grants,
a result store, workers), the converters and the API at a glance: [python.md](python.md).
