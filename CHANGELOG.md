# Changelog

All notable changes to linus. The supported interfaces are the command line, the JSON request
and result (`schema_version`) and the Python package; before 1.0 any minor release may change
them, and every change is listed here. The Rust crates are unstable and not published. Versions follow [Semantic Versioning](https://semver.org); the format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.3] - 2026-11-26

A bug-fix release. Update if you compute gradients, Hessians or IR spectra. Results can differ
from 0.1.2's where an entry below says so (numerics epoch 13).

**Wrong derivatives after some hard SCFs (0.1.0 to 0.1.2).** An SCF can converge through
quasi-Newton steps: the escalation after DIIS stagnates or blows up, the two-cycle rule, or a
stability step. Such an SCF can end in a state that leaves an orbital empty below an occupied one.
Earlier releases then gave the derivatives the wrong determinant: the gradient, the Hessian, the
polarizability (and with it the IR intensities) and the closed-shell stability probe were computed
for a different state than the one whose energy they reported. The energy was always correct. In
most such cases a gradient was off by 0.01 to 0.13 Eh/bohr, occasionally more. In a C2 singlet
(PBE/def2-SVP), dE/dz was -0.1667 instead of -0.0383 Eh/bohr. The probe reported eigenvalues of -18
to -142 Eh. Closed and open shells are affected, mostly stretched bonds and transition-metal
complexes. In our SCF robustness sweep of 49,847 calculations, 1,768 gradient and Hessian jobs
escalated, and 162 of them (on 37 molecules) ended in such a state. SCFs that took no quasi-Newton
steps are unaffected. linus now gives the derivatives the orbitals of the state the SCF converged
to. It reports such a state with the warning `non_aufbau` and `resolved.record.scf.non_aufbau`.
**With an earlier release:** a gradient, Hessian or IR result is suspect if its `scf.history` has a
step `soscf`, `stability` or `escalation` (for example, `resolved.record.scf.escalation.ran` is
true). Rerun it with 0.1.3. If the rerun warns `non_aufbau`, the earlier derivatives were wrong.

**IR runs and the coupled-perturbed solver.** The CPHF/CPKS solver treats several right-hand
sides at once. It stopped once each one had been below its tolerance at some iteration. But the
directions added for one right-hand side change all the solutions, so one that had converged could
rise above the tolerance again. IR runs then failed just above `response.field_tol`. The solver now
stops only when all of them are below the tolerance at the same time. IR runs that failed this way
now succeed, after zero to two more iterations. Other results move only where a solve needed that
extra step, and they stay within the tolerance. Sometimes the electric-field response has no
finite solution at the SCF's state: at a saddle point, or at a broken-symmetry state that rotates
freely, such as a linear radical's about its axis. An IR run then keeps its Hessian, frequencies
and thermochemistry, leaves out only the IR intensities, dipole derivatives and polarizability, and
warns `ir_declined`. Until now the whole run failed.

**Also since 0.1.2** (details in the entries below):

- **Faster on GPUs, above all the H100 and H200** (Changed): the XC functional is evaluated on the
  device. Density-fitted exchange runs on FP64 tensor cores, more than twice as fast for a 113-atom
  ωB97X-D3BJ/def2-TZVP molecule on an H200 and in less device memory. The Coulomb gradient of
  functionals without exact exchange is computed in Hermite space (113 atoms at def2-TZVP: 0.077 s
  instead of 1.8 s). Less host time is spent around the kernels.
- **Large molecules on a GPU** (Added, Changed): when the three-index tensor would not fit on the
  device, the density-fitted J of pure functionals and of `rij_cgk` is computed without it, chosen
  by the new expert setting `backend.df_j` (`auto`, `stored` or `direct`). The nuclear attraction of
  large molecules is computed on the device (`backend.nuclear_attraction`).
- **The SCF escalation also starts when DIIS blows up** (Changed; `scf.blowup_window`, default 4).
  Such SCFs converge sooner: for example, a nearly gapless 304-atom protein (PBE/def2-SVP) in 65
  iterations instead of 125.
- **GPU Hessians on H100 and H200 with f and g functions** were off by up to about 5e-4
  Eh/bohr^2, because the CUDA compiler built four integral kernels wrongly (Fixed).
- **`jk: "rij_cgk"`**: density-fitted J with compensated grid-based exchange, opt-in (Added).

### Added

- **The warning `non_aufbau` and `resolved.record.scf.non_aufbau`:** a converged SCF state that
  leaves an orbital empty below an occupied one is reported with its holes, particles and the size
  of the inversion (`homo_lumo_gap` is then negative). The result is that state's; a lower aufbau
  solution may exist.
- **`jk: "rij_cgk"`: density-fitted J with compensated grid-based exchange** (Hohenstein,
  Alkan, Wu and Parrish, 2026), opt-in; the default stays `df`. The exchange of the SAD
  reference density is exact (four-center integrals, once per geometry); the rest goes through a
  seminumerical exchange on one COSX grid (`jk.k_grid`: `gridx:N`, default `gridx:2`) with a
  symmetric overlap fit. Energies and analytic gradients, closed and open shells, Hartree-Fock,
  global and range-separated hybrids, on the CPU and the GPU (the SCF on the device; gradients
  on the CPU). Hessians and a `k_grid` with different levels per stage are input errors. Its
  absolute exchange energies are several times more accurate than density-fitted exchange; its
  relative energies and gradients are not, and it is slower than `df`.
- **Density-fitted J for large molecules on the GPU without the stored three-index tensor.**
  When the tensor would not fit the device beside the other engines, the SCF's J of pure
  functionals (`jk: "df"`, e.g. PBE, r2SCAN) and of `rij_cgk` is computed in every build without
  it, instead of streaming the tensor from host memory or running on the CPU: on data-center
  GPUs (H100, H200, A100) by Hermite J-engine contractions (several times faster), elsewhere by
  recomputed three-center integrals. The new expert setting `backend.df_j` (`auto`, `stored` or
  `direct`; [docs/cli.md](docs/cli.md)) chooses; `auto` takes `direct` only when the tensor would
  not fit. The result records it as `resolved.record.backend.j`
  (`{"engine": "direct", "reason", "kernels", "kernels_reason"}`; absent otherwise). Hybrids
  with `df` and calculations with an implicit solvent keep the stored tensor.

### Fixed

- **IR runs that failed with the polarizability's CPHF/CPKS just above `response.field_tol`.**
  The response solver (all right-hand sides in one shared subspace) stopped once each right-hand
  side had been below the tolerance at some iteration, but the directions it added for the others
  change every solution, so one that converged early could rise above the tolerance again: the
  solve then ended at up to 1.3 times the tolerance after a dozen iterations, and an IR
  frequency run failed as unconverged. The solver now gives every right-hand side above the
  tolerance new directions after each iteration, a converged one too, and stops only when all
  of them are below it. Results change only where a right-hand side rose above the tolerance
  after it had converged (a Hessian's or a polarizability's solve then takes another direction
  or iteration, within its tolerance).
- **Analytic gradients, Hessians, polarizabilities and the closed-shell probe after the SCF
  escalation.** When DIIS stagnated or blew up, the escalation's quasi-Newton steps could converge
  to a state whose occupied orbitals are not the lowest ones of its Fock matrix (an empty orbital
  below an occupied one; stretched bonds and transition-metal complexes with PBE, closed and open
  shells). The energy was that state's, but the orbitals handed to the derivatives were the lowest
  eigenvectors, another determinant: in most such cases gradients were off by 0.01 to 0.13 Eh/bohr,
  occasionally more (C2 singlet PBE/def2-SVP: dE/dz -0.1667 instead of -0.0383), and the
  closed-shell probe reported eigenvalues of -18 to -142 Eh. The SCF now hands on the orbitals of
  the occupied space it converged (the same after the two-cycle rule's steps and a stability step).
  Releases 0.1.0 to 0.1.2 are affected; SCFs that did not take quasi-Newton steps are unchanged.
- **GPU Hessians on H100/H200 with f and g shells.** ptxas (CUDA 13.0 and 13.1) at its default
  optimization level built four classes of the three-center second-derivative kernels wrongly for
  these cards ((ff|g), (gg|d), (gd|g), (gf|f)): def2-TZVP Hessians of molecules with bonded
  heavy atoms (f shells) and def2-universal-jkfit (g functions) were off by up to about 5e-4
  Eh/bohr^2 (0.2 cm^-1 in the frequencies of a 39-atom anion). Those kernels are now built with
  ptxas -O1 on compute capability 9.x; other cards and kernels are unchanged.

### Changed

- **Numerics epoch 13** (0.1.2 was 10). Results of the same input can differ from 0.1.2's where an
  entry below says so: the SCF escalation's blow-up trigger (CPU and GPU, only in SCFs where it
  fires), the coupled-perturbed solver's per-iteration convergence rule (only response solves where
  a column rose back above its tolerance, at round-off within it), derivatives after an SCF that
  ends in a non-aufbau state (now correct; SCFs reached by orbital rotations that end aufbau move at
  convergence level), and the GPU changes listed here that move results at round-off on a device
  (within [docs/gpu.md](docs/gpu.md)'s tolerances). Within epoch 13 a result is again bitwise the
  same at any thread count and per device.
- **An IR run keeps its Hessian and frequencies when the electric-field response fails.** The
  Hessian and the frequencies do not depend on the polarizability's CPHF/CPKS. When that solve
  stops above `response.field_tol`, the result has the Hessian, the frequencies and the
  thermochemistry without the IR intensities, dipole derivatives and polarizability. It carries
  the warning `ir_declined` with the reason, and the solve in `resolved.record.response.ir_declined`.
  Until now the whole calculation failed. A broken-symmetry state that rotates freely, such as a
  linear radical's about its axis, has no finite polarizability.
- **A CPHF/CPKS failure says how the solve ended.** The failure's `response` record has `end`:
  `iterations` (at the limit), `exhausted` (no direction left to add) or `indefinite` (an
  orbital-Hessian eigenvalue at or below zero), or `non_finite`. With `exhausted` or `indefinite`
  the response has no finite value at that SCF state (a saddle point, or a broken-symmetry state
  that rotates freely, such as a linear radical's about its axis), and the message says so
  instead of suggesting a looser tolerance or another precision.
- **The SCF escalation also starts when DIIS blows up** (`scf.blowup_window`, new expert setting,
  default 4; 0: off). Until now the restart from the lowest-energy orbitals with quasi-Newton steps
  (`scf.escalate`) started only once the lowest energy had not improved for 64 iterations, or at 2/3 of
  `max_iter`. It now also starts after 4 iterations in a row whose energy is more than 1 Eh above the
  lowest visited while max |F_ai| is more than 10 times the smallest visited; the record's
  `scf.escalation.trigger` is then `blowup`. The restart is the same, only earlier: a nearly gapless
  Trp-cage (PBE/def2-SVP, 304 atoms) converges in 65 iterations instead of 125, LiF at 7 Å in 33
  instead of 88, to the same bits. Results change only for SCFs whose DIIS blows up. On the robustness
  sets every SCF that converged without escalating is unchanged; in the cases where the trigger
  fires, Fock builds fall from a median of 133 to 39 and two more SCFs converge.
- **GPU backend: the nuclear attraction of large molecules on the GPU.** Beside the device's
  density-fitting setup, the SCF's nuclear attraction V of molecules with at least 2e8 kept shell
  pairs x point charges (ubiquitin, 1231 atoms; not the validation's molecules) is computed by the
  device PCM engine's pair x point integrals instead of on the host (ubiquitin/def2-SVP: 15.7 s on
  16 cores; the device's values within 1e-15 of the CPU's, relative to the largest element; bitwise
  run to run); `resolved.record.backend.nuclear_attraction` says where it ran and why, and the
  new expert setting `backend.nuclear_attraction` (`auto`, `gpu` or `cpu`) chooses. The overlap,
  kinetic energy and ECPs stay on the host, and so does V on any device failure.
- **GPU backend: the exchange-correlation functional is evaluated on the GPU.** libxc's own
  functional code (its Maple-generated derivatives, MPL-2.0) is compiled for the device at run
  time for B3LYP, PBE, PBE0, r2SCAN, SCAN, TPSS, ωB97X-V/-D3BJ, ωB97M-V, B97M-V, SVWN and the
  other functionals built from the same libxc components; it is checked against libxc when the
  calculation starts and agrees with it to the last few bits (energies within about 1e-14 Eh at a
  fixed density). Functionals without device code, or that fail the check, are evaluated by
  libxc on the host as before; `resolved.record.backend.functional` says where each XC part ran
  and why. GPU builds (the `afh-linus-cuda13` wheel, the GPU image) embed that libxc source;
  THIRD_PARTY_NOTICES.md lists it.
- **GPU backend: faster XC.** The collocation goes straight into each grid batch's compressed
  values, and the Becke partition of PySCF's grids is computed on the device, per point only for
  the atoms whose cell function can matter (the others, each below 2^-100 of the sum by a
  rigorous bound, are left out; the weights equal the CPU's in every test). On an H200 a V_xc
  build of the 39-atom anion with B3LYP/def2-TZVP takes 30 ms instead of 94, of the 113-atom
  ts113 170 instead of 323, and ts113's grid 0.19 s instead of 1.44; the grid of a 376-atom urea
  cluster (def2-SVP) takes 0.74 s instead of 70 s on the CPU, and of a 1231-atom protein 7.3 s
  (the partition in tiles of points that share their atoms' cell functions). The XC setup of
  those two takes 0.20 and 0.56 s instead of 0.55 and 1.4 (the screening of each grid batch's
  functions on the device), and the XC gradient's grid response above about 110 atoms runs on a
  faster kernel (ts113: 193 ms instead of 229). A guess density (no orbitals) goes to the device
  XC as it is, where the host SCF loop factorized it on the CPU (99 s of a 1231-atom protein's
  first XC call on a build whose J/K had fallen back). GPU energies change only in their last
  bits (within 4e-12 Eh, the same iterations). The `grid` timer has the parts `grid_partition`
  and `grid_batches` on the GPU.
- **GPU backend: VV10's pair sums in FP64 on FP64-rich cards** (H100, H200, A100) with
  `precision.fp32: "auto"`, where they were double-float on every card: 3.2x faster there and at
  least as accurate. Results are now bitwise identical between cards of one class (as the
  exchange's already were), not across classes; docs/gpu.md's Reproducibility section says
  which. `precision.fp32: "off"` and `"on"` are unchanged.
- **Faster density-fitted exchange on the GPU, in less device memory.** The exchange now
  multiplies only the shell pairs the screening keeps (the dense blocks it multiplied before were
  47-60 % zeros at def2-TZVP). On NVIDIA H100/H200 an FP64 tensor-core kernel reads them
  straight from the stored tensor, so the resident dense planes (20 GiB for a 39-atom molecule
  at def2-TZVP) are gone. On an H200, per exchange build: 39 atoms, B3LYP/def2-TZVP, 28 to 21 ms;
  113 atoms, ωB97X-D3BJ/def2-TZVP, 1.39 to 0.75 s. GPU results move at round-off, within
  [docs/gpu.md](docs/gpu.md)'s tolerances. On H100/H200 the kernel then got a dedicated copy
  stage (its own warps copy the operands ahead while the others multiply): 113 atoms,
  ωB97X-D3BJ/def2-TZVP, 0.76 to 0.61 s per J/K build, B3LYP 0.40 to 0.32 s; 39 atoms, B3LYP,
  21.6 to 18.3 ms. Results are bitwise unchanged. `resolved.record.df` of GPU runs records the
  exchange kernel as `k_kernel` (`"mma_sm90a"` on H100/H200, `"mma"` or `"strided"` elsewhere).
- **Much faster density-fitted gradients without exact exchange on the GPU** (PBE, r2SCAN,
  B97M-V and other functionals without exact exchange): the Coulomb part of the gradient is now
  contracted in Hermite space instead of through the general derivative integrals. On an H200,
  the two-electron gradient at def2-TZVP: 39 atoms 0.25 to 0.033 s, 113 atoms 1.8 to 0.077 s.
  GPU results move at round-off, within [docs/gpu.md](docs/gpu.md)'s tolerances.
- **Faster density-fitted gradients on FP64-rich GPUs** (H100/H200): the two-electron gradient
  stores its derivative integrals and contracts them instead of contracting inside the
  integral kernels, whose fixed-point atomics bound them on these cards, and batches its MO
  products. 1.3-1.4x on the two-electron gradient (113 atoms, B3LYP/def2-TZVP: 4.0 to 2.8 s).
- **Less host time around the GPU's kernels** (no number changes on the CPU; on the GPU,
  energies and gradients bitwise as before, Hessians within round-off):
  - the host's SCF setup (one-electron matrices, the molecular grid, the SAD guess) runs beside
    the device's density-fitting setup; the SAD guess computes its elements' atomic densities
    side by side (the same densities);
  - a GPU Hessian's CPHF/CPKS solver runs its own algebra (the subspace projections, the
    orthonormalization, the residuals) on the device when its subspace fits beside the response's
    engines; `resolved.record.backend.response[].solver` says where it ran (`gpu`, or `cpu` with
    `solver_note`);
  - a GPU Hessian's one-electron explicit terms run on the host while its two-electron and XC
    terms run on the device;
  - the device PCM's setup reuses the SCF's shell pairs; the dipole integrals are built while the
    device iterates, and the dipole's sum and the Hessian's assembly no longer read their matrices
    element by element (the same results);
  - a GPU Hessian's assembly GEMMs run on the device;
  - the command line no longer tears its session down before it exits.
  - the shell pairs (with their Schwarz factors) and SAD's atomic densities are built while the
    GPU opens, and the device's density-fitting setup screens those pairs instead of building
    its own (the same layout, bit for bit); the host's setup beside it takes half the threads;
  - a GPU calculation whose density-fitting tensors fit neither the device nor the host memory
    is refused before any setup (it was refused after both the device's and the CPU's setup:
    ubiquitin/def2-SVP on an H200 in 3 s instead of 295 s).

## [0.1.2] - 2026-11-20

### Fixed

- **Builds from source on Ubuntu 26.04 with its x86-64-v3 packages** (the amd64v3 variant, whose
  gcc generates FMA instructions by default) compiled libxc with fused multiply-adds, so every
  density functional's results differed from the wheels' in their last bits, and at a few
  near-empty spin channels by more. libxc is now compiled without floating-point contraction on
  every compiler: such a build gives the wheels' numbers. The wheels themselves are unchanged
  (the same extension module, byte for byte).
- **The test suite on GitHub's ubuntu-26.04 runner** (Ubuntu's amd64v3 packages): the comparison
  of linus's functionals with PySCF's, run with the same libxc, demanded agreement to 1e-13 at
  every grid point, which only holds with the C math library it was made with. Its tolerances now
  follow how far the C library's rounding can move each kind of point (measured on four Linux
  systems): 1e-11 at ordinary points, 5e-9 where one spin channel is almost empty, and 2e-2 plus
  the same screened points where a channel is empty or vanishing. The CI names its runner
  (ubuntu-26.04) and runs every test binary even after a failure.

### Changed

- **The reproducibility statement** (README, [docs/usage.md](docs/usage.md#10-reproducibility))
  now says what it covers: bitwise for the same wheel on the same glibc version built the same
  way. Ubuntu 26.04's x86-64-v3 build of glibc rounds `cbrt` differently from its x86-64 build;
  there the same wheel moved energies by up to 2.5e-12 Eh and gradients by about 1e-14 Eh/bohr,
  and an SCF can take a different number of iterations (M06-2X on the OH radical: 16 instead of
  14) and, rarely, change whether a hard SCF converges within `scf.max_iter`. The documented `erf`/`erfc` differences of glibc 2.36 and 2.39 are unchanged.

## [0.1.1] - 2026-11-20

A bug-fix release: CPU results that could differ in their last bits with `memory_gib` or the
thread count no longer do. The numbers of 0.1.0 are kept wherever they did not depend on them
(numerics epoch 10).

### Fixed

- **Results independent of `memory_gib` and the thread count in three more places.** Some steps
  of a calculation split their work into blocks whose number follows the memory budget
  (`memory_gib`, or the machine's memory without it) or the thread count, and a matrix product
  can round differently with its size. So these results could differ in their last bits (round-
  off, far below any accuracy that matters) between two runs with a different `memory_gib` or
  thread count, or on another machine; each now has the bits of the run with a single block:
  - **DF Hessians and frequencies with exact exchange** (Hartree-Fock and hybrid functionals,
    `jk: "df"`): large molecules, or small `memory_gib`, compute the exchange term in several
    passes over the atoms. Results with one pass are unchanged (except as noted below).
  - **Hessians, frequencies and polarizabilities with the RI exchange in the CPHF/CPKS
    equations** for an open shell with a single electron in one spin (e.g. a doublet with one
    beta electron), where the response's exchange groups its densities by memory.
  - **Hessians, frequencies and polarizabilities with VV10 functionals** (ωB97M-V, ωB97X-V,
    B97M-V and the other `-V` functionals), whose nonlocal kernel groups its perturbations by
    memory and thread count: large molecules, small `memory_gib`, many threads, and in
    particular an open shell with a single electron in one spin. A single group (small
    molecules at the default memory) is unchanged.

  One result that did not depend on them changes in its last bits: the DF Hessian of a hybrid or
  Hartree-Fock calculation whose 3 × (basis functions) × (occupied orbitals, both spins counted)
  is one more than a multiple of 256, in the Hessian's row and column of the last atom's z
  coordinate. The bitwise reference suite is unchanged.

### Changed

- The Python package's classifier is "Development Status :: 3 - Alpha" (it said "5 -
  Production/Stable"): linus is a 0.x release whose interfaces may still change.
- A GitHub release marked as a pre-release no longer uploads its wheels to PyPI.
- The benchmark harness `bench/run_bench.py` (with its summary and run scripts) is no longer in
  the public repository: it ran only with tooling that is not public.

## [0.1.0] - 2026-11-19

The first public release: linus feature-complete. The command line, the JSON request and result
(`schema_version` 1) and the Python package are the supported interfaces; as a 0.x release they
may still change. The Python package is on PyPI as two distributions of the same version, both
imported as `linus` (install one): **`afh-linus`**, the CPU build, and **`afh-linus-cuda13`**, the
same with the GPU backend for NVIDIA GPUs with CUDA 13 (`linus.GPU_BUILD` tells them apart). Both
are manylinux_2_28 x86-64 abi3 wheels for CPython 3.11 or newer, with the basis sets inside. The
package runs on the qc* stack's releases: it needs qcdata 0.24.0, and qccompute 0.23.0 (qccodec
0.22.0, qcstore 0.1.0, qcinf 0.6.0) runs it as the program `linus`. Numerics epoch 10.

Against the internal baseline of 2026-11-08: "Added" lists what is new, "Changed: behaviour"
every change of numerical results (with its decision, Dnn, in AFH's design records), "Changed:
interfaces" what users of the internal baseline must do, "Performance" the speed-ups with their conditions, and
"Known limitations" what linus does not do. CPU results are bitwise those of the internal baseline, at any thread
count, except where "Changed: behaviour" says otherwise.

Pre-releases of 0.1.0, distributed as wheels and not on PyPI, followed the qc* stack's
pre-releases: **0.1.0a1** (its alpha-3 trials), **0.1.0a2** (alpha 5), **0.1.0a3** (alpha 6) and
**0.1.0a4** (its release candidate rc1, qcdata 0.24.0a8 and qccompute 0.23.0rc1), with the GPU
wheel as the local version `0.1.0aN+cuda13`. From 0.1.0a2 on they compute the numbers of 0.1.0
(numerics epoch 10); 0.1.0 differs from 0.1.0a4 only in the qc* versions it requires and in its
distributions. Every front end reports the PEP 440 version: the command line's `--version` and
JSON `version`, `linus.__version__` and qccompute's program version (`linus::version()` in Rust).

### Added

- **Wheels on PyPI, a CPU and a GPU build**: `pip install afh-linus` or `pip install
  afh-linus-cuda13` (the GPU backend; at run time an NVIDIA driver >= 580 and the CUDA 13
  libraries NVRTC, cuBLAS and cuSOLVER on the loader path, else it runs on the CPU). Both have the
  same version and import `linus`; `linus.GPU_BUILD` (Rust `linus::GPU_BUILD`) is True in the GPU
  build. Release wheels are uploaded from GitHub releases by trusted publishing
  (`.github/workflows/publish.yml`; CONTRIBUTING.md, "Releasing").
- **The ORCA-compatible presets ship in every build**: the `orca:N` grids (ORCA 6.0.0's
  DefGrid1-3, reproduced from ORCA's printed output) with ORCA's NL grids for VV10, the `orca` PCM
  preset (`solvent.preset: "orca"`, ORCA 6.0.0's C-PCM) and ORCA 6.0.0's solvent table (its
  permittivities, and the names it accepts: `isooctane`, `butanol` and others resolve) are
  compiled in from `data/orca`, so the CLI, the Python package and the containers have them with
  no extra files. The ORCA comparisons in the documentation, the validation matrix's ORCA 6.0.0
  reference runs and the tests against ORCA's numbers are public too. All of it is included with
  permission of FAccTs GmbH (THIRD_PARTY_NOTICES.md); linus contains no ORCA code or binaries.
  Numbers are unchanged.

- **The plan states the smallest `memory_gib` that runs a calculation**: `memory.required_gib`
  (f64 DF storage) with `memory.phases`, what each phase (`scf`, `gradient`, `hessian`,
  `polarizability`) holds at the limit and needs; the same from `linus --plan`, Python's
  `Session.plan` (`record["memory"]`) and qccompute's plan, and the `required_gib` of the
  `host_memory` failure below it.
- **Progress events** (qc* decision 14a): under a runner that listens (`ctx.progress`: qccompute's
  `progress=`), every calculation reports `stage_start`/`stage_end` (`scf`, `gradient`,
  `hessian`, `polarizability`; with `seconds` and `success`), `scf_iteration` (`iteration`,
  `energy`, `delta_energy`, `gradient`: the record's SCF history as it happens, on the CPU and on
  the device), `scf_escalation` (`iteration`, `trigger`, `from_iteration`) and `response_iteration` (`solve`, `iteration`, `residual`, `unconverged`,
  `total`), in process and from a worker (docs/python.md, "Progress events"). No number depends on
  them, none is made without a listener, and on small SCFs (10-60 ms) a listener costs under 0.2%
  of the wall time.

- **Accuracy targets** (`accuracy`, qc*'s shared keyword, decisions 22 and 25b): `{"energy",
  "gradient", "hessian"}` in Eh, Eh/bohr and Eh/bohr², the error from convergence a caller can
  tolerate. linus maps them to `scf.e_tol`, `scf.g_tol` and `response.tol` from measured bounds
  (energy E: e_tol E, g_tol √(E/600); gradient G: g_tol G/36; Hessian H: g_tol H/36,
  response.tol √(H/360), under 1e-6 below H = 1e-7), the tightest of the targets a request
  computes; given thresholds are floors a target tightens, never loosens. `record.accuracy`
  holds the `target`, the `thresholds`, the `estimate` of each targeted error and `met`; a
  session request's own `accuracy` (Python `Session.compute(..., accuracy=)`) replaces the
  input's for that request. Without targets nothing changes. docs/usage.md, section 11.
- **The run's environment in Python outputs** (decision 29): `execution.cpu_model`,
  `execution.libc` (`glibc 2.39`), `execution.devices` (the GPU the backend opened) and
  `execution.peak_device_memory_gib` (the calculation's engines' peak device memory).
- **Error codes**: a failure's code within its kind, qcdata's shared codes, as `error_code` in the
  CLI's failure JSON (absent when there is none) and `code` on Python's exceptions: a setting's
  `unknown_key`, `no_effect` or `not_applicable`, and `host_memory` (not enough host memory: the
  DF tensors, the in-core integrals, the solvent's matrices, a Hessian's arrays, page-locked
  memory a GPU engine cannot pin) or `device_memory` (a GPU allocation beyond the device's memory
  or its grant) for resource errors (qcdata's shared codes, decision 23); `scf_not_converged` (an
  SCF out of iterations, the code every qc* program gives it) and `response_not_converged` (a
  CPHF/CPKS solve above its tolerance) for kind `convergence`.
- **Peak device memory per calculation**: `resources.mem_device_peak_gib` (GiB), the most device
  memory a GPU calculation's engines held at once, their buffers and grants (linus's own
  accounting, not the driver's context or the libraries' workspaces), next to the host's
  `mem_hwm_gb_*` checkpoints. Rust: `linus_gpu::Gpu::reset_peak` and `Gpu::peak`.
- **GPU memory grants**: an engine's planned device memory is set aside at its decision
  (`linus_gpu::Gpu::grant`), so a later allocation cannot invalidate the decision; the
  response's stored exchange operator is granted at the response's setup (it used to re-check
  the free memory at its first product and could fall back then) and the DF tensor at its
  setup's decision. `resolved.record.backend.response[].operator_granted_gib`; an allocation
  beyond a grant is a `plan_violation` (in the record and as a warning). The workspaces of the
  DF gradient and Hessian, of the XC gradient and Hessian and the PCM's surface matrices are
  granted too, at their setups (the XC gradient's before its call): their allocations draw
  from the grant and give the bytes back when freed, so a violation is an engine holding more
  at once than its plan, recorded per family in `resolved.record.backend.plan_violations`
  (absent when none). The XC gradient keeps the density pass's X for the contraction only when
  the grant holds it (else it recomputes X; the same bits): it used to keep up to 8 GB without
  checking the free memory, so under a 1 GB cap la18's XC gradient call did not fit and fell
  back to the CPU; it now runs on the device.

- **`linus --plan`** and **`Session.plan(calctype)`** (Rust: `Calculator::plan`): what a
  calculation will do, decided before it runs: the settings it will have and why, where each
  family of kernels runs and in which arithmetic on a GPU, the memory plan, a Hessian's
  responses, and what only the run decides, with the rule ([docs/cli.md](docs/cli.md#plan)).
  The plan also gives the `sizes` (atoms, basis and auxiliary functions, occupied orbitals),
  `deferred` (the settings it leaves `auto` because the run decides them, as dotted paths:
  qcdata's `Resolved.deferred`) and `estimates`: the peak host memory, the device memory and
  the threads the calculation uses well, each an estimate with its error band (the largest
  relative error over the cases its model was fitted on).

- **`linus.Session` and `linus.compute` on qcdata objects** (the shared session interface of
  AFH's programs): a session opens from a `qcdata.ProgramInput` (one molecule, model and
  settings) and computes it at many geometries, each request a `qcdata.Frame`, continuing a
  lineage (any JSON value) from that lineage's last converged orbitals and probe state;
  failures raise `qcdata.ProgramError` subclasses carrying the failed frame, and leave every
  lineage as it was. `calctypes` default to every calctype the method supports. `plan()` is
  qccompute's plan (`resolved` with `deferred`, `resources` from the estimates, `record`);
  frames record the threads and the process's peak memory (`execution.threads`,
  `execution.peak_memory_gib`). `linus.compute(inp)` runs one calculation and returns a
  `qcdata.ProgramOutput` (a failure raises, as from a session); its keywords are linus's settings
  tree (`output.orbitals`, `output.hamiltonian`, `output.overlap`, `output.core_hamiltonian` and
  `output.molden` add the orbitals' coefficients, the AO matrices and a Molden file), and
  `given.wavefunction.orbitals` start its SCF. `linus.qc.program` is linus for qccompute's
  runners (capabilities, schemas, plans, one-off calculations and sessions), installed as the
  `qccompute.programs` entry point `linus`, so `qccompute.compute` and `qccompute.session` run
  linus in process, qcopt opens linus sessions, and qccompute's serve loop hosts linus's
  sessions in a worker process (`python -m qccompute.serve linus`; a session's `input_data` is
  the input it was opened with). Frames keep the report's run facts in `record`
  (`numerics_epoch`, `homo_lumo_gap`, `vv10`, the response iterations, the solvation) and their
  `extras` are plain JSON values, so every frame serializes.

- **VV10 Hessians, frequencies, IR intensities and polarizabilities** (ωB97M-V, ωB97X-V,
  B97M-V): VV10's CPKS kernel and explicit second-derivative terms on the NLC grid, with the NLC
  grid's first-order response in the default `response.grid_response` convention (the derivative
  of the gradient without grid response, the grid moving with the atoms). They were refused as
  `unsupported`. Timers `hess_vv10` (with `_potential_terms`, `_pairs`, `_grid`),
  `hess_vv10_setup`, `hess_vv10_device_setup`, `hess_cphf_vv10` (with `_pairs`, `_grid`) and the
  resource `hess_cphf_vv10_perturbations`, and the same `pol_` ones for polarizabilities.
- **The GPU backend** ([docs/gpu.md](docs/gpu.md)): the `gpu` cargo feature of linus, linus-cli
  and linus-py (off by default; CUDA 13 is loaded at run time, nothing CUDA is needed to build)
  and the `backend` setting (CLI `--backend`): `"cpu"` (the default), `"gpu"`, `"auto"` (the GPU
  when the build has it and a device opens), or `{"kind", "precision": "auto" | "double" |
  "emulated" | "mixed", "devices": [n], "scf_loop": "auto" | "host" | "device"}`. With
  `jk: "df"` the device runs:
  - the SCF's J/K (range-separated hybrids included, the long-range exchange in Psi4's wK
    scheme as on the CPU), XC (LDA, GGA, meta-GGA), VV10, an implicit solvent's surface terms
    (C-PCM, COSMO, IEF-PCM, SS(V)PE and SMD, both presets, the `exact` and `predicted` modes),
    the density-fitting setup (three-center integrals and the fit), and the SCF loop itself
    (orbitals, densities, Fock matrices and DIIS on the device, closed and open shells; open
    shells take the host loop's stability decisions, the probe and mode check running on the
    device and the rest resumed on the host loop from the device's state);
  - the gradient's two-electron, XC, VV10 and PCM terms;
  - the Hessian's explicit two-electron and XC terms (LDA, GGA and meta-GGA, any grid
    partition), and the CPHF/CPKS J/K and XC kernel of Hessians, IR spectra and polarizabilities
    (VV10's pair sums too), the response's exchange on a stored operator where it pays (built
    once per spin, applied to all perturbations; streamed from page-locked host memory when it
    outgrows the card), and an IR run's electric-field response on the Hessian's operator.

  Everything else (exact and seminumerical J/K, one-electron and ECP integrals, dispersion, the
  cavity and SMD's CDS term, the PCM's Hessian terms, VV10's NLC-grid passes) runs on the CPU,
  each family with its reason. Results agree with the CPU backend to round-off
  (`reference/validate_gpu.py`: energies within 1e-10 Eh with the same SCF iterations, gradients
  1e-9 Eh/bohr, Hessians 1e-10 Eh/bohr², frequencies 1e-6 cm⁻¹, polarizabilities, dipole
  derivatives and IR intensities 1e-8) and are bitwise reproducible run to run on one device.
  Also:
  - `precision: "emulated"`: FP64-quality density-fitted exchange GEMMs on INT8 tensor cores
    (linus's Ozaki scheme, 7 slices of 55-bit operands, the tensor sliced once where it fits;
    cuBLAS's emulation, then FP64, as fallbacks), the density-fitting fit and the FP64 CPKS
    kernel's large GEMMs likewise; `resolved.backend.emulation` reports the tier per family.
    `"mixed"` (opt-in): the XC gradient's, the CPKS kernel's, the Hessian XC terms' and VV10's
    gradient products in FP32, and the stored response operator in FP32 where the response runs
    single precision (errors at least 10x below the default grids' own);
  - `LINUS_GPU_MEMORY_GB`: a per-process cap of the device memory, for processes sharing a card
    ([docs/gpu.md](docs/gpu.md#sharing-a-gpu));
  - `LINUS_GPU_KERNEL_DUMP`: a directory that receives each compiled kernel module's source and
    compile options (a debugging aid; off unless set); `LINUS_SCF_TRACE` (CPU and GPU): a file
    the SCF loops append their per-iteration DIIS, ADIIS and SOSCF state to (likewise);
  - `resolved.backend`: where each family of kernels ran (`jk`, `xc`, `vv10`, `gradient_jk`,
    `gradient_xc`, `hessian_jk`, `hessian_xc`, `response_jk`, `response_xc`, `scf_loop`,
    `one_electron`, `pcm`, `solvent`, `dispersion`), the `fallbacks` with their reasons,
    `device` (with `memory_limit_gib` and `memory_cap_gb`), `precision`, `precision_auto`,
    `emulation`, `response` (per response: J/K and kernel arithmetic, the stored operator),
    `vv10_arithmetic` and `no_device`; `resolved.df.storage` `device`/`host_streamed` and
    `resolved.df.integrals`;
  - `error_kind: "backend"` (Python `linus.BackendError`): no usable device for
    `backend: "gpu"`, or a device error, which stops the calculation (a calculation never
    continues on the CPU midway); with `backend: "gpu"` a stderr line when the SCF's `jk` or `xc`
    runs on the CPU after all;
  - container images ([docker/](docker/README.md)): `Dockerfile.gpu` (CUDA 13.0 runtime) and
    `Dockerfile.cpu`, sharing one build stage, built by `docker/build.sh` from a commit.
- **SCF robustness**, on both backends:
  - the two-cycle reaction (`scf.two_cycle`, on): quasi-Newton steps take over when the energy
    repeats every other iteration while the SCF is not converging (`resolved.scf.two_cycle_at`;
    UHF CN/def2-SVP, which failed after 300 iterations, converges in 134);
  - the soft-mode plateau rule (`scf.plateau_window`, 50; 0 off): an SCF with a stationary energy
    and a small, stuck orbital gradient when its escalation comes due, or at `max_iter`, is accepted
    with the warning `loosely_converged` (`resolved.scf.plateau`; the F atom in C-PCM); under accuracy
    targets only when its final orbital gradient's error estimates meet them;
  - the cheap stability probe (`scf.stability_probe`; on by default for open shells, D25 below):
    Hessian-vector products for the lowest orbital-Hessian mode, the full analysis and descent
    only on a negative mode; `resolved.scf.stability_probe`, `probe: true` in `scf.stability`;
  - escalation within `max_iter` (`scf.escalate`, on, D30; numerics epoch 10): an SCF whose DIIS
    stagnates (`scf.stagnation_window`, 64 iterations without a lower energy) or that has spent 2/3
    of `max_iter` restarts once from the occupied orbitals of its lowest-energy iteration, with a
    new DIIS subspace, no level shift or damping, and quasi-Newton steps for the rest of
    `max_iter`; `resolved.scf.escalation` (`ran`, `trigger`, `at`, `from_iter`, `from_energy`,
    `iterations`, `converged`, or `ran: false` with the `reason` in an SCF that used its budget
    without it), the step `"escalation"` in the history. On the device loop it continues on the
    host loop from the device's state after the step it came due at. On the robustness sweep's
    hard set (740 cases, extreme preset): converged 417 at `max_iter` 128 + 128 (epoch 9), 434
    at `max_iter` 288 (epoch 10); on the open-shell set (542 molecules) 522 and 522;
  - the stability searches per irrep (D45, numerics epoch 7 below): the probe, the full analysis,
    the saddle check's analysis, the zero-mode check and the closed-shell probe search each irrep of
    the converged state's abelian point group on its own; `symmetry`, `irrep`, `irrep_index` and
    `irreps_searched` in their records;
  - the zero-mode check of open-shell Hessians (`scf.stability_hessian_check`, 6; 0 off): since
    numerics epoch 8 the stability probe of every converged point of a Hessian SCF, with these
    products per irrep; a negative mode is followed, and a degenerate solution (or a negative mode
    no step lowers) is flagged, its analytic Hessian not the energy surface's
    (`resolved.record.response.mode_check`, the `mode_check_flagged` warning);
  - `scf.lindep` (1e-8), with `resolved.scf.dropped_functions`.
- **Interfaces:** `schema_version: 1` in the result and failure JSON; `error_kind: "internal"`
  (a panic exits with code 1 and the failure JSON, not 101); `exact.storage` (`auto`, `direct`,
  `incore`) with `resolved.exact`; `$LINUS_SOLVENT_TABLE` and `linus.solvent_table()` (qcref's
  full CRC permittivity table, `resolved.solvent.eps_table`); Python `linus.NumericalError`,
  `.kind` on every exception, `linus.basis_search_path()`, `linus.basis_file()` and `py.typed`;
  `timings` and `resources` keys
  for the new phases (the GPU setups `*_setup*`, `hess_2e_*`, `hess_xc_*`, `vv10*`,
  `cphf_solver`, `cphf_orth`, `cphf_vo`, `pol_cphf_jk`, `pol_cphf_xc`, ...), the others keeping
  their names.
- **Build and documentation:** the `dispersion` cargo feature of linus-cli and linus-py (on;
  `--no-default-features` builds an MIT-only CLI or wheel without D3/D4); `[profile.dist]` for
  release artifacts; `[profile.test-fast]` for the tests (`cargo test --profile test-fast`: the
  release profile without thin LTO, the same numbers, a test build in a fraction of the time;
  README.md, "Testing"); `THIRD_PARTY_NOTICES.md` and `LICENSES/` (also in the wheel);
  [docs/conventions.md](docs/conventions.md) (units, AO order, spin layout, error kinds, the
  glossary); [docs/usage.md](docs/usage.md#10-reproducibility) "Reproducibility" (what is bitwise
  across machines and what moves the last bits) and section 11, the Python API.

### Changed: behaviour

Numerical results change only as listed here.

- **Numerics epoch 10: `scf.max_iter` is the SCF's whole budget, and the escalation runs within
  it.** The escalation (a restart from the lowest-energy iterate's orbitals with quasi-Newton steps)
  used to start at `max_iter` and add up to another `max_iter` iterations; now it starts when DIIS
  stagnates (the lowest energy not improved for `scf.stagnation_window` iterations, new, default 64)
  or after 2/3 of `max_iter`, whichever comes first, and spends the rest of the budget. When it
  comes due the plateau rule decides first: an SCF on a soft-mode plateau is accepted there, on one
  more full build, instead of escalating (an open shell's plateau is analysed first; a negative mode
  is followed as the escalation). The plateau rule warns (`loosely_converged`: residual, window
  and `g_tol` in the message) and, under accuracy targets, accepts only a final orbital gradient whose
  estimates meet them. The last iteration of the budget is always a full Fock build. The default
  `max_iter` is 288, the smallest that converges at least as many SCFs as today's 128 + 128 on both
  of the robustness sweep's sets: on the hard set (740 cases, extreme preset) 434 (396 at the lowest
  energy any run reached) against today's 417 (375), with the default window of 64 iterations
  (max_iter 128/160/192/224/256/288 converge 296/343/371/396/417/434; a window of 48 converges 418
  at 256 but fails 8 SCFs today converges, and one of 32 405), and on the open-shell set (542
  molecules) 522 (522 at the lowest) against 522 (520); 256 matched the hard set but lost the boron
  atom, whose DIIS converges at iteration 86 while the stagnation window fires at 78. Iterations on
  the hard set 1.05x of today's (only SCFs that escalate or fail run longer). Only SCFs that reach
  the escalation's trigger move: of the hard-set SCFs today converges without escalating, none
  fails; of the robustness sweep's 69,738 SCF histories, 868 converged ones reach the trigger. Converged SCFs that never reach it, which is every SCF of the
  validation set, the golden CLI JSON and the bitwise suite, are bitwise unchanged; the escalation
  record changes shape (`resolved.record.scf.escalation`: "Changed: interfaces").
- **Numerics epoch 9: the SCF ends on a full Fock build.** An incremental build (integral-direct
  exact J/K builds J and K from the change of the density) that meets the convergence criteria
  is now always rebuilt in full at the same density and tested again; it used to be only when
  `scf.e_tol` was below 1e-8, so energies with exact J/K at the default thresholds could end on
  an incremental build. The returned Fock matrix (`output.hamiltonian`) is then the operator of
  one build with the returned orbitals as its eigenvectors, and requesting it changes no other
  number. Only exact J/K SCFs at `scf.e_tol` of 1e-8 and above move (energies at the default
  thresholds; gradients and Hessians already ended on a full build): B3LYP/def2-SVP energies of
  caffeine and a 39-atom anion move by 1.5e-11 and 2.9e-11 Eh, orbital energies by up to
  3.4e-9 Eh, and the SCF takes one more build (`iterations` and `scf.fock_builds` + 1, about 6%
  of the SCF's time). Density-fitted J/K and the GPU's exact J/K never build incrementally.
  The same holds for the other terms built from their history: a converged iterate whose PCM
  charges came from the `predicted` mode (the default with density-fitted J) or whose COSX
  exchange was incremental is confirmed by one full build at the same density (the exact PCM's
  two passes, a full K), on the host and in the device loop. Energies are unchanged (COSX within
  2.3e-13 Eh); orbital energies move by up to 7e-10 Eh, gradients by up to 2.4e-10 Eh/bohr and
  Hessians by 2e-11 Eh/bohr²; it costs one Fock build (caffeine B3LYP/def2-SVP: 4-9% of the
  SCF). The SCF history's `fock` says `incremental` for those builds (it said `full`), and
  `scf.incremental_builds` counts them.
- **Numerics epoch 8: the zero-mode check steps, a plateau is analysed, long searches start wide,
  and the DF Hessian's exchange cross term no longer depends on memory.** Results stay bitwise at
  1, 3 and 8 threads and run to run on one card; on the 48-case CPU suite only the density-fitted
  Hessians with exchange move (the last item).
  - **The zero-mode check of a Hessian SCF is the probe of its converged points**
    (`scf.stability_hessian_check`, 6). It ran after the SCF and only warned, so a negative mode
    it found left the Hessian of a saddle (FeH, B3LYP/def2-SVP RI-JK: -5.0e-3, 7.5 mEh up). Now
    every converged point of a Hessian SCF is probed with max(`scf.stability_probe`,
    `scf.stability_hessian_check`) products per irrep, also where the probe is otherwise off (a
    later geometry): a negative mode takes the probe's path (the full analysis, the step,
    reconvergence; on the device loop the probe's hand-over), and the search at the accepted point
    is the check's record (`resolved.record.response.mode_check`). Near-zero modes, and a negative
    one no step lowers, stay a warning. One search where a first-geometry Hessian ran two. On the
    robustness sweep's 377 open-shell Hessians (B3LYP/def2-SVP RI-JK, pyscf:3, extreme): every
    SCF ends at the lowest state any variant reached (epoch 7: 370), 7 of them lower (Fe(O)Cl4²⁻
    48 mEh, FeH 7.5 mEh, the Fe²⁺ ion, the V atom, TiO, CH, a G21EA anion: 6e-6 to 9e-5 Eh), none
    higher; the searches' Fock builds per Hessian SCF median 0.78x, p90 1.00x (max 6.3x: C2's
    plateau analysis), all its builds median 0.93x; flags 52 -> 51. With the probe off, as at a
    later geometry: 13 lower states (Fe(O)Cl4²⁻ 48 mEh, triplet O3 20, CoCl4²⁻ 14, a G21IP
    radical 9.9, a W4-11 radical 9.8, FeH 7.5, an RC21 radical 4.8, V(CO)6 0.95, IrCl6²⁻ 0.42,
    ...), none higher;
    builds median 1.00x, p90 1.25x (max 8x: I2⁺'s zero mode, searched in every irrep), all its
    builds median 1.00x, p90 1.09x; flags 58 -> 51.
  - **A plateau is analysed before it is accepted.** A UHF/UKS SCF can crawl down a negative mode
    at the plateau rule's pace (FeH's Hessian path: 1e-12 Eh per iteration down a -1.1e-3 mode,
    accepted at `max_iter`). An open-shell plateau is now a converged point for the stability
    trigger; a step continues the SCF on the escalation's budget while it has one (recorded as an
    escalation from that iteration); the device loop hands such a plateau to the host loop. On the
    sweep's 542 open shells 14 SCFs end on the plateau rule, all now analysed; 2 step: Fe(O)Cl4²⁻
    (-3.6e-2) to a state 48 mEh lower (the one epoch 6 reached), the Fe²⁺ ion (-3.9e-4) to one
    9.3e-5 Eh lower.
  - **Searches of six or more products start from an irrep's three lowest pairs** (the full
    analysis, the zero-mode check): the normalized sum of the irrep's three lowest-gap pairs (the
    whole space's without symmetry), and they end early on a negative Ritz value only once its
    residual is below a tenth of it. The gap can misrank a pair's curvature by 0.2 Eh in a d shell:
    FeH converged on a C2v state whose -1.1e-3 mode (PySCF's symmetry-blind analysis) is one beta
    dxy -> sigma pair, third of its irrep by gap, which the search from the lowest pair never
    reached (+2.5e-2), and its Hessian was that state's; it now ends at PySCF's stable UKS solution
    (-1264.0916459966 Eh; PySCF's lowest internal eigenvalue there +1.09e-3), its frequency matching
    central differences of gradients (1044.31 cm⁻¹, max |H - H_fd| 3.5e-8). The tighter stop costs
    builds only where a negative mode exists (the L-lactide cation's saddle: 19 instead of 15; with
    the old stop the sum's first negative Ritz value, a blend, sent the step to the minimum 1.85e-5
    Eh higher; at the default settings the converged mode's step now reaches that minimum, the one
    epoch 6 reached, on the CPU and the GPU alike: known limitations). The two-product probe keeps the single pair, which a sum dilutes. On the 542 open
    shells the full analysis misses a lower state 10 times instead of 17 (8 lower, 1 higher: the
    glutamine radical cation, known limitations); at the default settings 6 SCFs end lower
    (Fe(O)Cl4²⁻ above, the Fe²⁺ ion, the V atom, a Cu and a Sc complex, Ru(NH3)6³⁺: 3e-6 to 9e-5
    Eh), 2 higher (the glutamine radical cation 1.8 mEh, a Re complex's cation 3e-6); Fock builds
    median 1.00x, p90 1.00x. Candidates measured and rejected (docs/internal/robustness.md item 8):
    the six lowest pairs, the lowest exact diagonal element, a block of three pairs, the pairs
    within 0.1 Eh.
  - **The DF Hessian's exchange cross term is taken in an orientation fixed by the atoms**, so H-bar
    atom blocks (set by `memory_gib` or the machine's memory) no longer change its bits; Hessians
    move by ≤ 2.4e-14 relative.
- **Numerics epoch 7: the stability searches per irrep (D45), and the SAD guess on the four-center
  J/K engine.** Results stay bitwise at 1, 3 and 8 threads and run to run on one card.
  - **Every stability search runs per irrep (D45, extending D25).** At a symmetric point the
    orbital Hessian is block-diagonal by irreducible representation, and a Davidson search started
    on one orbital pair reached another block only through what broke the symmetry in its
    arithmetic: whether a negative mode of another irrep was found depended on the last bits.
    Triplet O3 (B3LYP/def2-SVP RI-JK, pyscf:3, extreme preset) stopped on a C2v saddle 20 mEh above
    the minimum for 3 of 10 one-ulp perturbations of its start on the CPU. **Fixed bug:** the GPU's
    device loop stopped on that saddle as it was and with a one-ulp change of a coordinate. Each
    search (the probe, the full analysis, the saddle check's analysis, the zero-mode check of
    open-shell Hessians and the closed-shell probe) now runs once per irrep of the converged
    state's abelian point group (D2h and its subgroups, elements the state breaks dropped) whose
    lowest orbital pair lies within 0.1 Eh of the lowest, from that irrep's lowest pair, on the
    irrep's block of the orbital Hessian, with its full products per irrep (probe 2, closed-shell
    probe 2, zero-mode check 6), stopping at the first negative irrep. Triplet O3 reaches the
    minimum under all 10 perturbations on the CPU and the device loop, by the same path every time
    (the C2v saddle's mode in B2, then a Cs saddle's in A'', which the old start's block never held).
    On their own, the per-irrep searches leave molecules without symmetry bit for bit unchanged,
    and SCFs whose probes stay positive keep their results: only the probes' records (eigenvalue,
    Fock builds) change (the 48-case suite: all bitwise).
    On ROBUST's 542 open shells (B3LYP/def2-SVP RI-JK, pyscf:3, extreme convergence), the probe
    misses a lower state that the full analysis at every converged point finds 5 times instead of
    6, by more than 0.1 mEh twice instead of 4 times (FeH, on the path from this epoch's guess, and
    a Ti complex; with the old guess, 3 times and once). A one-ulp change of the starting guess no
    longer moves any converged energy by more than 0.1 mEh (two atoms' SCFs then fail to converge,
    known limitations; on epoch 6 it moved triplet O3 by 20 mEh, an RC21 radical by 4.8 and
    Fe(CN)6³⁻ by 1.3). Fock builds per SCF: median 1.00x, p90 1.27x (symmetric molecules median
    1.09x, p90 1.38x, max 2.1x; wall median 1.06x, p90 1.27x); the probe-triggered analyses that
    find nothing are as many as before (28). The closed-shell probe flags 3 more broken-symmetry
    instabilities among 888 closed shells, the textbook ones (singlet O3 at two geometries, -0.032
    and -0.034 in B2; water stretched to 1.8 Å, -0.058 in A1), at +0.8% of the set's time (on the
    139 probed molecules: median +0.8%, p90 +25%). The zero-mode check flags 3 more near-zero modes
    among 377 open-shell Hessians (NaO -7.7e-7, CrO -1.39e-6, Fe(O)Cl4²⁻ +2.35e-4; PySCF's UKS
    stability analysis at the same states: -7.8e-7, -1.40e-6, +2.35e-4), at a larger share of a
    symmetric open-shell Hessian (median 4.6% -> 7.8%, p90 6.3% -> 16.1%, max 25%; molecules
    without symmetry unchanged). Their analytic frequencies agree with finite differences of
    gradients to 0.02 cm⁻¹: on this set these modes are harmless. D32's multi-pair start, which
    broke a symmetric saddle only through a probe-triggered step, goes with its setting.
  - **The SAD guess's atomic SCF runs on the four-center J/K engine** (linus-ints' `JkPlan`, as the
    exact J/K builder runs it; its own builder is gone, 171 lines): every SCF starts from a guess
    that differs in the last bits, and converged results move at round-off (on the 48-case suite:
    energies by at most 1.8e-12 Eh, gradients 3.6e-13 Eh/bohr, Hessians 7.1e-7 Eh/bohr², largest
    in the wB97X-D3BJ cases, frequencies 9.0e-4 cm⁻¹, dipole derivatives 6.5e-6 e). Knife-edge SCF
    paths move with it: HF⁺ r²SCAN on the bond axis no longer passes its saddle (the CPU converges
    to the same minimum after the escalation, the device loop on the hole orientation's other
    minimum, 1.8e-7 Eh higher, the one aarch64 reaches), and on the open set some
    transition-metal SCFs end on other states (known limitations).
- **Numerics epoch 6: the architecture refactor's numerics-changing items, as one set (D38).** The
  refactor's other steps kept results bitwise (the CPU at any thread count, the GPU per card class);
  these change numbers on purpose, each by one mechanism. linus now reports the epoch next to its
  version, so a store keyed by input and version can tell results apart: `linus --version`
  (`linus <version> (numerics epoch 6)`), the result JSON's `numerics_epoch` and Python's
  `linus.NUMERICS_EPOCH`. Any later change beyond the bitwise guarantee raises it. Results stay bitwise
  at 1, 3 and 8 threads and run to run on one card. The sizes are each item's largest change over linus's
  48-case CPU suite at 8 threads (relative to the quantity's largest magnitude in parentheses).
  Hessians whose response runs in single precision (`precision.response` auto on large systems, `fp32`,
  two-level responses) are listed apart: FP32 amplifies a round-off change of their inputs to about
  1e-6 relative, within that mode's accuracy.
  - **`jk` defaults to `df`** (decision 6): RI-JK with the calculation's auxiliary basis
    (def2-universal-jkfit for energies and gradients, cc-pVTZ-jkfit|def2-universal-jkfit for
    Hessians). Set `jk: "exact"` (`--jk exact`) for four-center integrals. A calculation that leaves
    `jk` unset now carries RI-JK's error and runs faster. The GPU backend runs its J/K on the device
    without `jk: "df"`, and a PCM's SCF terms take their RI-JK default (`solvent.scf_terms`
    predicted). The default against exact (def2-SVP unless noted; 8 threads, one run each):

    | molecule | calculation | E(df) - E(exact) (Eh) | max abs d gradient (Eh/bohr) | max abs d frequency (cm⁻¹) | wall exact / df (s) | speedup |
    |---|---|---|---|---|---|---|
    | water | HF/def2-TZVP frequencies | +7.6e-6 | 5.7e-6 | 0.05 | 1.2 / 0.6 | 2.0x |
    | water | B3LYP frequencies | -1.9e-5 | 5.4e-6 | 0.09 | 0.9 / 1.0 | 0.9x |
    | NH3⁺ (doublet) | B3LYP frequencies | -1.9e-5 | 2.2e-5 | 0.62 | 0.6 / 0.4 | 1.5x |
    | CH3SH | B3LYP frequencies | -5.4e-5 | 9.5e-6 | 0.68 | 3.1 / 2.2 | 1.4x |
    | CH3SH | ωB97X-D3BJ/def2-TZVP frequencies | -2.4e-5 | 1.0e-5 | 0.52 | 11.6 / 4.0 | 2.9x |
    | L-lactide (la18) | B3LYP gradient | -4.7e-5 | 1.4e-5 | | 4.1 / 1.7 | 2.4x |
    | caffeine | B3LYP gradient | -5.2e-5 | 1.6e-5 | | 10.1 / 3.5 | 2.9x |
    | caffeine | PBE0/def2-TZVP gradient | -8.5e-5 | 4.6e-5 | | 83.9 / 10.7 | 7.8x |

    Speedups run from 1.4x to 7.8x from CH3SH up. Water, with 24 basis functions, gains nothing.
    The frequency changes (up to 0.68 cm⁻¹) are RI-JK's own error.
  - **`grid.kernel: auto` follows the XC grid's family** (input review F5: an `auto` reads resolved
    values, never whether another key was given). With linus grids, the default `linus:3` or an
    explicit `linus:N`, the CPKS kernel runs on linus:1 (meta-GGAs: linus:2). With PySCF, ORCA and
    Psi4 grids it runs on the XC grid. Before, any explicit `grid.xc`, even `linus:3` (the value
    auto resolves to), moved the kernel onto the XC grid. Default grids are unchanged.
    Frequencies with an explicit linus grid, before (the kernel on the XC grid) and after (def2-SVP
    RI-JK, 8 threads):

    | molecule | functional, grid | kernel before / after | max abs d frequency (cm⁻¹) | wall before / after (s) |
    |---|---|---|---|---|
    | water | B3LYP, linus:3 | linus:3 / linus:1 | 0.0015 | 0.2 / 0.1 |
    | water | B3LYP, linus:4 | linus:4 / linus:1 | 0.0015 | 0.2 / 0.2 |
    | CH3SH | B3LYP, linus:4 | linus:4 / linus:1 | 0.027 | 1.0 / 0.8 |
    | L-lactide (la18) | B3LYP, linus:4 | linus:4 / linus:1 | 0.019 | 17.6 / 11.3 |
    | CH3SH | r2SCAN, linus:3 | linus:3 / linus:2 | 1.48 | 0.9 / 0.7 |
    | CH3SH | r2SCAN, linus:4 | linus:4 / linus:2 | 0.87 | 1.3 / 1.0 |

    **Meta-GGA frequencies with an explicit linus grid move by up to about 1.5 cm⁻¹.** CH3SH r2SCAN
    with `grid: "linus:3"` moved 1.48 cm⁻¹: its kernel went from linus:3 to linus:2, and the result is
    now bitwise the default run (whose XC grid is linus:3). This is the meta-GGA kernel's known grid
    sensitivity ([docs/grids.md](docs/grids.md)). Old and new explicit-grid meta-GGA frequencies
    therefore differ at this level. Set `grid.kernel` to the XC grid to reproduce the old ones. GGAs
    and hybrids move by at most 0.03 cm⁻¹.
  - *One summation order instead of several (round-off):*
    - the potential's derivative matrices (d/dA) sum the centers' Hermite tensors before one
      contraction, as the potential matrix does: one `matrix_block` for point charges, ECP local
      channels and surface charges. Energies and gradients bitwise; Hessians 1.1e-11 Eh/bohr² (2e-11),
      frequencies 4.7e-8 cm⁻¹ (single-precision responses: Hessians 8e-7 relative, frequencies 5e-4 cm⁻¹,
      APT 1e-6);
    - the stationary correction of response properties is one Gram product (`Correction::add_gram`)
      and the dipole derivatives one second-order assembly. APT 1.3e-15 e (5e-15) and α 4.3e-14 au
      (1e-15); everything else bitwise;
    - the Davidson solver's dot products are fixed-chunk reductions (`det::dot`): bitwise on the
      suite. An SCF that steps along an orbital-Hessian mode moves at round-off: the L-lactide
      cation, ωB97X-D3BJ UKS, whose saddle check steps at iteration 31, by 1.6e-12 Eh (3e-15) in the
      same 57 iterations;
    - one XC batch layout (one stride for the basis values) and one basis-motion routine for the XC
      Hessian and VV10's response: bitwise on the suite (VV10 Hessians 4e-13 relative);
    - VV10: one pair primitive (R², g_i, g_j over centred coordinates) for the SCF, the gradient and
      the response, one local block, and the response's ground potential computed as the SCF's.
      SCF energies bitwise; gradients 1.8e-14 Eh/bohr (2e-12); VV10 Hessians (water and CH3SH
      ωB97M-V, CH2 B97M-V) 2e-10 Eh/bohr² (4e-10), frequencies 1.6e-7 cm⁻¹, IR 2e-8 km/mol;
    - one orientation of the orbital shell pairs (by shell index); the batched four-center ERIs
      transpose their l_a < l_b pairs instead. Energies 2.3e-13 Eh (1e-15), gradients 6.8e-14 Eh/bohr
      (4e-12), Hessians 8.6e-11 (1e-10), frequencies 1.3e-7 cm⁻¹, APT 8e-10 e (2e-9).
  - *Consistent thresholds:*
    - the range-separated operator's screening factor applies in every three-center path (the DF
      gradient and Hessian blocks): bitwise on the suite and the GPU set;
    - `DfThresholds` owns the three-center derivative thresholds (pair 1e-18, weighted 1e-14, block
      1e-15, metric 1e-14; formerly constants in four files). Double-precision cases bitwise; two
      single-precision Hessians 1.3e-6 relative (1.2e-3 cm⁻¹); the GPU set bitwise;
    - one COSX overlap fit S_num, on the basis cut of the SCF (`rij_snk`): energies 6.8e-13 Eh
      (9e-16), gradients 1.4e-13 Eh/bohr (6e-12). The gradient's distance to finite differences
      went from 2.6e-10 to 1.8e-10 (energy consistency 7e-15 -> 4e-14 Eh);
    - PCM gradients take the Rys integral derivatives in every SCF mode (they did in the predicted
      mode only): one gradient 1.4e-16 Eh/bohr; caffeine B3LYP exact-mode C-PCM `grad_solvent_ints`
      -76%.
  - *Robustness:* the GPU's fixed-point accumulators flag an invalid value (NaN, infinity, beyond
    2^32) and read the whole accumulator as NaN, as the CPU's do, instead of a clamped finite
    number. Finite results are unchanged (61 of 61 GPU cases bitwise).
  - *Policy:* the GPU's DF storage ladder orders its rungs below the device-resident tensor by
    measured cost (decision D11; docs/gpu.md). A shrunk K block on the device comes first unless
    even its smallest block costs more than streaming the tensor from host memory. u1an39
    B3LYP/def2-TZVP at 10 and 12 GB caps now keeps the tensor on the device (blocks of 40 and 136,
    2.5-2.7 s for 16 builds) instead of streaming it (3.9 s). The K block orders the exchange's sums,
    so the round-off depends on the cap: every rung's energy agrees to 1e-12 Eh.
  - The whole epoch on the CPU suite (every case sets `jk` and its grids, so the two defaults move none):
    energies 6.8e-13 Eh (2e-15) in 14 of 39 cases, gradients 1.4e-13 Eh/bohr (6e-12), Hessians 3e-11
    Eh/bohr² (5e-11), frequencies 1.1e-7 cm⁻¹, APT 3.2e-10 e (9e-10), α 6.3e-13 au (5e-14). Single-precision
    responses: Hessians 5e-7 relative, frequencies 8e-4 cm⁻¹. No SCF or response iteration count
    changed. Results at 1 and 3 threads are bitwise those at 8. On the 61-case GPU set (RTX PRO 6000), energies move only in the five L-lactide-cation UKS cases whose SCF steps along an orbital-Hessian mode (A2; 3e-15 relative). Hessians move by 2e-12 relative with double-precision responses and by 2e-9 with FP32 ones (frequencies 3.4e-6 cm⁻¹, IR 9e-9 relative). Two runs on one card are bitwise. CPU timings are within
    noise (8 threads, against the commit before the epoch: L-lactide B3LYP Hessian -1.2%, ωB97X-D3BJ
    Hessian +0.3%, ωB97M-V gradient -0.1%, RIJCOSX gradient +0.3%, exact-J/K energy +0.8%, the 39-atom
    anion's gradient -1.1%). The exception is caffeine's exact-mode C-PCM gradient, -3.8%: its solvent
    integrals are 77% faster on the Rys kernels.
- **The stability probe is on by default (D25)** for UHF/UKS at a calculator's first geometry
  (single points, the first SCF of Hessians and optimizations): `scf.stability_probe` unset means
  2 one-sided Hessian-vector products there and none at later geometries; `0` turns it off, a
  number applies at every geometry. On 500 open-shell molecules (B3LYP/def2-SVP) it costs a median
  1.08x the SCF's wall time and catches 10 of the 13 saddles the full analysis finds (triplet O3,
  FeH, V(H2O)6³⁺, ...): those results change, to the lower state. Minima are bitwise unchanged
  (gas phase and PCM); closed shells have their own probe (D36, next).
- **Closed-shell KS-DFT is probed for a lower broken-symmetry solution (D36)**, warning only: at
  a converged RKS point whose HOMO-LUMO gap lies below `scf.closed_shell_gap` (0.15 Eh), 2
  one-sided products of the UKS orbital Hessian (UKS-type Fock builds inside the restricted SCF)
  from the lowest-gap pair with alpha and beta rotating in opposite senses. A negative mode is
  recorded in `resolved.scf.closed_shell_probe` (`gap`, `eigenvalue`, `kind`, `unstable`,
  `products`, `fock_builds`, `seconds`; when it does not run, `skipped`: `gap_above_threshold`,
  `hf_default`, `off`, `later_geometry`, ...) with one stderr warning ("RKS is unstable towards a
  broken-symmetry UKS solution ...; rerun with restricted: false"); the SCF never switches, and
  energies, gradients and Hessians are bitwise those without the probe (with a PCM in the
  predicted mode, the solvation summary to round-off, as after D25's probe). `scf.closed_shell_probe` unset is 2 for
  KS-DFT at the calculator's first geometry and 0 for Hartree-Fock (opt-in, at least 2: RHF's
  triplet instabilities are common and rarely matter for an RHF reference) and at later
  geometries; `0` turns it off. On the ROBUST closed-shell set (888 molecules, B3LYP/def2-SVP
  RI-JK, extreme preset), 30 RKS solutions have a lower UKS one (median 2.7e-2 Eh lower). The gate
  probes 139 molecules (15.7%) and the probe flags 27 of the 30 (SCFROB's alpha-only start: 21;
  probing every molecule: 29), with no false positives. It costs 2.5% of the RKS SCF on average
  (a probed molecule: 14%, median), and all 1776 SCFs (B3LYP and HF) keep their energy bits and
  iterations. With HF, opted in, it flags 78 of the 197 RHF solutions that lie above a UHF one
  (bench/results/2026-11-11-closedprobe). Both backends; the GPU's device loop runs it on the
  device (hexacene B3LYP/def2-SVP on an RTX PRO 6000: 0.5 s of a 2.3 s SCF), and its Ritz value
  agrees with the CPU's to 1e-11 Eh.
- **The default probes skip continued SCFs unless the gap drops (D36).** The open-shell probe
  (D25) and the closed-shell probe ran at every step of an optimization, IRC or
  finite-difference calculation that made a new calculation per step. A session's lineage now
  continues: its later requests run the unset probes as at a later geometry (`skipped:
  "later_geometry"`). A gap trigger keeps the safety net: when the HOMO-LUMO gap (open shells: the
  smaller alpha or beta gap) has fallen below `scf.closed_shell_gap` since the lineage's last
  probe, or more than 0.02 Eh below that probe's gap, the probes run
  (`resolved.scf.probe_trigger`: `last_gap`, `gap`, `trigger: "gap_dropped"`). Explicit probe
  settings apply at every step, as before.
- **The probe's start direction is fixed by the orbitals' phases (D29)**, not by the
  eigensolver's arbitrary signs, so the CPU and both GPU loops agree (4e-12): on the robustness
  sweep's open-shell set the probe's eigenvalue moved in 219 of 498 SCFs (median 2.4e-7), no decision
  changed, and 4 SCFs reach the same minimum within 3.5e-11 Eh by another step.
- **Escalation at `max_iter` (D30) and the multi-pair stability start (D32, replaced in numerics
  epoch 7 by the per-irrep searches)** (Added): only SCFs that failed, or whose probe triggered a
  full analysis, change. SCFs that converged before are bitwise unchanged except a few whose
  probe-triggered analysis now steps differently (12 of the robustness sets, each on the same state
  within 1.4e-10 Eh; triplet O3 and PtH reach lower states).
- **linus:3 adjusts H-X atom pairs (D26)**: the default Hessian and frequency grid keeps
  Treutler's atomic size adjustment for the pairs of H with an atom beyond neon. B3LYP/def2-SVP
  gradients of PH3, AsH3, H2S, H2Se, HCl, HBr and GaH3 on linus:3 were 1.1-8.7e-4 Eh/bohr from
  pyscf:8 (now 2.7e-6 to 1.3e-5) and their frequencies move closer to linus:4. Molecules without
  such a pair are bitwise unchanged; with one, energies, gradients and Hessians change at the
  grid-error level (frequency jobs' energies by up to ~4e-5 Eh).
- **The meta-GGA CPKS kernel grid is linus:2 (D33)** (was linus:1) with the default grids:
  meta-GGA frequencies move by up to several cm⁻¹ toward the converged grid (80 molecules against
  linus:4: r2SCAN MAD/max 3.27/14.4 -> 2.92/9.1 cm⁻¹, TPSS 1.01/13.3 -> 0.84/7.6) for a median
  15% more Hessian time. Other functionals keep linus:1.
- **The DF tensors' memory budget scales with the memory (D34)**: 75% of the limit (`memory_gb`,
  else the total RAM or the cgroup limit) less the PCM's dense matrices, with no 64 GiB ceiling,
  and with `memory_gb` unset a guard within 90% of the memory free when the SCF starts. Large
  jobs on large machines keep B (and B^ω) instead of the one-sided or compressed storage, and
  their results move at round-off (9e-13 Eh; F48 9e-11 Eh); jobs whose tensors fitted before are
  bitwise unchanged. Set `memory_gb` (or `df.precision`, `df.long_range`) to keep one storage
  everywhere. `resolved.memory` gains `df_tensor_budget_source`, `available_gib` and
  `reservations`.
- **IR dipole derivatives in a stationary form (D35)**: they add the electric-field response
  times the final CPHF/CPKS residuals of the Hessian, so their error is quadratic in the residual
  like the Hessian's (L-lactide B3LYP/def2-SVP at the default `cphf_tol`: 1.2e-6 au -> 4.9e-11 au
  from converged with an FP64 response; two-level responses up to 2e-4 au -> 4e-6 au). Only
  dipole derivatives and IR intensities change, by their former error; the field response runs
  where it already ran for IR intensities, at no extra cost.
- **`response.pol_cphf_tol` 1e-8 -> 1e-6 (D28)**: the polarizability's error is quadratic in the
  field CPHF's residual; polarizabilities move by at most 1.9e-10 au (7e-13 relative), and the
  field CPHF takes 13-22% fewer iterations.
- **The CPHF/CPKS subspace solver's algebra runs in BLAS-3** (block Gram-Schmidt, an
  incrementally grown projected matrix): Hessians, IR intensities and polarizabilities move at
  round-off (FP64 responses within 2e-12 Eh/bohr², the same iterations; the CPU's default
  two-level single-precision response within 7e-7 Eh/bohr² and, in frequencies, ~2e-3 cm⁻¹ at
  la18/def2-SVP and up to ~1e-2 cm⁻¹ at def2-TZVP (u1an39/def2-TZVP: 7.7e-3 cm⁻¹), that mode's
  own error against FP64, which `response.precision: "double"` removes).
- **`k_grid: "auto"`** is the documented default `cosx:2`; it ran PySCF's level-3 grid without
  overlap fitting.
- **GPU defaults per card (D27)**: `backend.precision: "auto"` is native FP64 on FP64-rich cards
  (A100, H100/H200, V100, P100) and the emulated exchange on the others (RTX and RTX PRO,
  L40/L4, A40/A10, T4, B200, B300), by compute capability, reported in
  `resolved.backend.precision_auto`; explicit settings are used as set. At the default
  `response.precision: auto`, where the CPU's rule runs single precision (la18-size molecules and
  up), GPU Hessians run the response exchange on the stored FP64 operator and, on FP64-poor
  cards, the CPKS kernel (and VV10's CPKS pair sums) in FP32: frequencies within ~1e-4 cm⁻¹ of an
  FP64 response.
- **The GPU response's exchange path by estimated cost (D37)**: at the default
  `response.precision: auto`, where the CPU's rule runs single precision, Hessians on cards with
  little FP64 throughput (RTX and RTX PRO, L40/L4, A40/A10, T4, B300) run the response exchange
  on the stored FP64 operator or in FP32 through the tensor, whichever a fixed cost model
  estimates cheaper (before: the operator whenever it fitted the device). The model counts the
  operator's setup and products, the tensor path's products, a streamed operator's PCIe copies
  and, for IR runs, the field solve that the operator also serves, at rates scaled per card from
  the RTX PRO 6000's; `resolved.backend.response` gains `estimate_s` with the two estimates, and
  its `reason` names the choice. On the RTX PRO 6000 plain Hessians take the FP32 exchange
  (u1an39/def2-SVP 28.4 -> 27.0 s, u1an39/def2-TZVP 87.2 -> 79.1 s; frequencies move by 5.6e-5
  and 6.9e-4 cm⁻¹), IR runs keep the operator, and under `auto` the operator may now stream from
  host memory where that is the cheaper path. A B200 (FP32 only twice its FP64) keeps the
  operator; FP64-rich cards and the CPU are bitwise unchanged.
- **The GPU's SCF loop runs on the device by default (D27, D31)**: `backend.scf_loop: "auto"`
  takes the device loop for closed and open shells where its Fock terms run on the device and
  the algorithm is the default one; `"host"` keeps the CPU loop.
- GPU results moved at round-off with this release's engines (the Ozaki exchange and fit,
  per-batch XC screening, the stored response operator, the sm_90 kernel schedules, the field
  response on the Hessian's operator: energies within ~3e-12 Eh with the same iterations); none
  of them changes CPU results.

### Changed: interfaces (what users of the internal baseline must do)

- **`memory_gib` limits the whole process**: the peak resident memory of a calculation stays
  within it (measured under a cgroup limit equal to it). Each phase's need counts the process
  itself (command line 0.03 GiB, Python 0.09 GiB, 0.6 GiB more with a GPU), the engines' blocks
  sized from the problem (the DF exchange's, the DF gradient's, a response's arrays), the shares of
  the limit and a measured allowance for matrices, XC, per-thread buffers and the allocator;
  before, only the DF tensors (75%) and a Hessian's arrays were counted, and peaks overshot it (by
  0.1-0.2 GiB for ωB97X-D3BJ/def2-TZVP energies at one thread, 0.4 GiB at eight, 0.7 GiB for
  gradients, up to 2-3 times for Hessians). A calculation that does not fit fails before any work
  (`host_memory`, with `required_gib` the `memory_gib` that holds every phase), so the smallest
  `memory_gib` that runs rises: caffeine ωB97X-D3BJ/def2-TZVP 1.4 to 1.8 GiB (energy) and 2.3 GiB
  (gradient). With `memory_gib` set, a DF storage is taken only if it also fits beside the rest of
  the process (a range-separated hybrid's one-sided coefficients instead of B and B^ω, at
  round-off, recorded in `resolved.record.df`). Without `memory_gib` the same check runs against
  the machine's limit within 90% of the free memory (a check only: no number or storage follows it).
  Rust: `Session::set_process` declares the front end's process (`linus::memory::Process`).
- **`scf.max_iter` is the SCF's whole budget** (numerics epoch 10), the escalation's iterations
  included (it allowed up to twice as many), and its default is 288 (was 128). `max_iter: 2`
  runs 2 iterations; the escalation needs 2 left after it starts, and an SCF that used its budget
  without it says why (`resolved.record.scf.escalation: {"ran": false, "reason": "off" | "max_iter"}`;
  the failure message too). New expert setting `scf.stagnation_window` (64; 0: the escalation only
  at 2/3 of `max_iter`). The escalation record is `{"ran": true, "trigger": "stagnation" |
  "two_thirds", "at", "from_iter", "from_energy", "iterations", "converged"}`.
- **A plateau is accepted with the warning `loosely_converged`** (it was silent), qcdata's shared
  warning for a result accepted under a relaxed rule (qc* alpha 6; linus's 0.1.0a2 called it
  `scf_plateau`, a name that is gone), when the escalation comes due or at `max_iter`; under
  accuracy targets only within them. Its `path` is the threshold missed, `scf.g_tol`; the message
  gives the residual reached and the rule (`scf.plateau_window`).
- **The qc* releases**: the Python package needs qcdata 0.24.0 (and qccompute 0.23.0 for its
  runner); the container images and CI install that set. qcdata renamed the dipole moment:
  `results.dipole_moment` (was `scf_dipole_moment`), and the request output of that name.
- **The basis set is `Wavefunction.basis`** (qc* alpha 4): it comes with the orbitals'
  coefficients or any AO matrix, and only then; an overlap or a core Hamiltonian alone has the
  basis set and no coefficients (they used to bring the coefficients); `hamiltonian` brings the
  overlap and the orbitals that are its eigenvectors. Each ECP carries the core electrons it
  replaces (`Ecp.core_electrons`), agreeing with `Orbitals.core_electrons`. Given orbitals
  are read from `given.wavefunction` with its basis set. `linus.convert` follows:
  `to_pyscf(structure, wavefunction)`, `from_pyscf` returns a structure and a `Wavefunction`,
  `to_pyscf_mol(structure, basis)` and `from_pyscf_mol` (a structure and a basis set) carry the
  cores in the ECPs (Molden files: `qcinf.molden.write`, below).
- **Runners (qc* alphas 4 and 5)**: `linus.qc.program`'s capabilities drop `gpu` (whether a
  calculation uses a GPU is its plan's); the granted GPUs open by their CUDA ordinal from
  `ctx.device_ordinals()` (linus's own `CUDA_VISIBLE_DEVICES` mapping is gone; a granted GPU the
  process cannot see fails with kind `environment`); the grant applies on every call, in process
  and in a worker, whose calls may differ in threads; a memory keyword replaces the share in the
  grant (qccompute's rule; linus uses the smaller of keyword and grant as before). The machine's
  part of the run's environment (`cpu_model`, `libc`) is the runner's to record: linus no
  longer fills it, and keeps `devices` (the GPU its backend opened) and
  `peak_device_memory_gib`. The worker's hello carries the numerics epoch in
  `capabilities.extras`.
- **A runner's stop request ends a calculation at its next step** (qc* alpha 6):
  `qccompute.stop(future)` reaches linus as `ctx.stop_requested()`, which linus asks before each
  stage and between its SCF's and its CPHF/CPKS solves' iterations (at most every 50 ms there),
  on the host and the device SCF loops. The calculation fails with the new kind `stopped` (the
  CLI's `error_kind` list gains it; the CLI itself never stops) and the results it reached: the
  SCF's last iterate (`scf.history`'s last `step` is `stopped`), or the converged SCF with the
  gradient (and an IR run's Hessian) computed before the stop. Without a `ctx` nothing is asked;
  numbers are unchanged. Rust: `Session::stop_when`, `LinusError::Stopped`.
- **`execution.devices` of a failure that reached nothing is left to the runner** (qc* alpha 6:
  a runner keeps a program's explicit `devices`, and an explicit `[]` means no device): a
  failure with partial results names the device its run opened, as a success does (`[]` on the
  CPU); one that reached nothing leaves `devices` unset, and the runner fills in its grant.
- **Molden files of a wavefunction are qcinf's** (`qcinf.molden.write(wavefunction, structure)`,
  qcinf 0.6.0a2, which writes ECP atoms with their cores in `[core]`): `linus.convert.write_molden`
  is gone; `output.molden` (`results.files["orbitals.molden"]`, the CLI's `--molden`) stays
  linus's own file.
- **Threads under a runner (qc* decision 16)**: a calculation runs on its `threads` keyword (at
  most the runner's count), else on the count the runner gives it capped at 8 (linus's measured
  threads-used-well, one CCD's 8 cores: `estimates.threads`), with the memory need of the threads
  it runs on. A plan asks for no threads: `plan().resources` has no `threads` (qccompute
  0.23.0 refuses a plan that sets it). A direct `linus.compute` without the keyword still runs
  on 1 thread. A calculation given GPUs and no `threads` gets 1 from the runner (qccompute
  0.23.0), and its host-side work (XC setup, parts of gradients and Hessians) runs on it: on
  the RTX PRO 6000, caffeine B3LYP/def2-SVP's gradient takes 2.51 s on 1 thread against 1.57 s on
  8, its Hessian 11.24 s against 5.93 s (the same bits). Set `threads` on GPU inputs.
- **`output.verbose` in Python goes to `logs`** (decision 24: a program does not print in its
  caller's process): the SCF's progress lines are the output's (or the failed output's) `logs`,
  not stderr. The command line still writes them to stderr.

- **`jk.final_full_build` is gone**: the SCF always ends on a full Fock build ("Changed:
  behaviour", numerics epoch 9); a request that sets it is refused (`unknown_key`).
- **The Python interface is `linus.compute(inp, ctx=None)` and `linus.Session(inp)` (decisions 7,
  9, 10, 16).** `linus.compute` raises a failed calculation, as sessions do: a `linus.LinusError`
  (a `qcdata.ProgramError`) with the failed `qcdata.ProgramOutput` in `output` (the error, partial
  results, warnings, record) and its results in `results`; it no longer returns `success=False`.
  `Session.compute_many` runs every request and then raises `qcdata.ProgramErrors` (an
  `ExceptionGroup` of each failed request's error, with every frame in `results`) when any failed.
  Under a runner, `ctx` is the calculation's grant: threads, host memory (the share comes before
  linus's own default of the cgroup limit or the RAM), the GPUs and the device memory, each used
  as the smaller of the keyword and the grant (`resource_capped` warns); no GPU granted runs on
  the CPU. The qccodec codec (`linus.qccodec`, the `qccodec.codecs` entry point and the `qc`
  extra) is gone: qccompute runs linus as a program (`linus.qc.program`, which drops the `driver`
  capability and gains `numerics(spec)`, the numerics identity a result store records with each run, decision 29: the numerics epoch as text). The
  warning `cpu_fallback` is the shared `backend_fallback`.
- **The wavefunction outputs follow qcdata's wavefunction design (decision 25).** `output.density`
  and the CLI's `--density` are gone: the SCF density is C n Cᵀ of the returned orbitals
  (`qcdata.Orbitals.density()`), which the stored matrix (the last build's input density, one
  step behind) differed from by the SCF residual. `output.fock` is `output.hamiltonian` (the CLI
  JSON's `fock_alpha`/`_beta` are `hamiltonian_alpha`/`_beta`) and brings the overlap. New:
  `output.overlap` and `output.core_hamiltonian` (CLI JSON `overlap`, `core_hamiltonian`), computed
  after the calculation at its geometry. In Python every frame's `results.wavefunction.orbitals`
  has the orbital levels (energies, occupations per spin, `kind` canonical, and `core_electrons`
  of ECP atoms, which leave `extras`); the coefficients and the basis set, now with its ECPs
  (`BasisSet.ecp`, each term's power of r as Basis Set Exchange's n - 2), come with the output
  `orbitals`, and the AO matrices (`hamiltonian_a`/`_b`, `overlap`, `core_hamiltonian`) with
  theirs (the basis set in `Wavefunction.basis`, below). Requesting any of them changes no other number (bitwise,
  tested for energies, gradients and Hessians). `extras["density_a"]`, `["hamiltonian_a"]` and
  `["core_electrons"]` are gone. `linus.convert` follows: `from_pyscf_mol` returns the core
  electrons too and reads ECPs, `to_pyscf_mol` takes them, `to_pyscf` takes them from the
  orbitals, and the PySCF export's
  ECPs come from the basis set, not from the basis set's file.
- **Out of memory is kind `resource`**: a GPU allocation that runs out of device memory after its
  setup (and page-locked host memory that cannot be pinned) stops the calculation with kind
  `resource` and code `device_memory` (`host_memory`); it was kind `backend` (Python
  `ResourceError`, was `BackendError`). Setups still fall back to the CPU out of device memory.
  Every `host_memory` and `device_memory` failure gives its numbers: `required_gib` and
  `available_gib` in `ErrorInfo.details` (the CLI's failure JSON: `error_details`), both as the
  memory budget (`memory_gib`, `device_memory_gib`) that holds what failed and the one it had, or
  for a device's allocation what it held plus the request against its pool.
- **`jk.df_storage` `auto` is f64**: the memory budget (a runner's memory grant, `memory_gib`, the
  machine's or the free memory) never selects a compressed DF exchange tensor. Under `auto` the
  storage is B (and B^omega) in f64, else, for range-separated hybrids, the one-sided f64
  coefficients; when neither fits, the calculation fails with a resource error of code
  `host_memory` whose message gives the size needed and the `memory_gib` that holds it (it used
  to fall to F48, then F40). The compressed formats (`f48`, `f40`, `f32`) run only when set. The
  plan resolves `jk.df_storage` before the run (`resolved.settings.jk.df_storage` `"f64"`, no
  longer in `deferred`; a device's J/K shows `"f64"` for its CPU fallback instead of `auto`). No
  stored result changes: every golden, validation and reference-store record used f64 (B or the
  raw RI-J integrals).
- **The Python package is the session interface on qcdata objects** (decision D46): `linus.Session`
  and `linus.compute` replace `linus.Calculator` and `CalcResult`. A calculation is a
  `qcdata.ProgramInput` (the model's method and basis set, linus's settings as keywords); results
  are qcdata's `Frame` and `ProgramOutput`: the energy, gradient and Hessian in `results`, the
  orbital levels always, the orbitals and the AO matrices on request (`outputs`, or the
  `output` settings), the SCF history, probes, sizes and linus's extra analysis
  fields in `record`, timings, threads and peak memory in `execution`. What was a calculator's
  next SCF from its last orbitals (`set_positions`, `set_guess`, `reset_guess`, guess files from
  `CalcResult.guess_bytes`) is a session's lineage (`compute(x, lineage=...)`, `Request.seed`
  and `commit`, `reset`), or `ProgramInput.given` for orbitals from elsewhere.
  `linus.evaluator.QcoptEvaluator` and its Hessian cache are gone: qcopt 0.3 opens linus sessions
  itself (linus is the `qccompute.programs` entry point `linus`, `linus.qc.program`) and passes a
  transition state's Hessian to the IRC. The qccodec codec is gone (see below): its `molden`
  keyword and the `molden.input` file are `output.molden` (`results.files["orbitals.molden"]`),
  and its wavefunction propagation (`propagate_wfn`, `linus.orbitals.npz`) is a session's
  lineage. `linus.models` is gone: the basis set and orbital
  models are qcdata's (`qcdata.Orbitals`, `BasisSet`, `AOConvention`), and the converters are
  functions on them in `linus.convert` (`to_pyscf`, `from_pyscf`, `to_pyscf_mol`,
  `from_pyscf_mol`; Molden files: `qcinf.molden.write`). `linus.vib` returns qcdata's `VibrationalAnalysis` and
  `Thermochemistry` (`n_imaginary()` is a method, `thermochemistry` returns a list and takes the
  constants as `codata`). qcdata is a dependency of the package.
- **Rust (unstable): `linus::Calculator` is `linus::Session`**, the name of the shared session interface
  (decision D46): one molecule, model and settings at many geometries, with its lineages (`compute`,
  `reset`, `lineages`, `declare`, `plan`); the Python package's `linus.Session` wraps it. The plan's
  reasons for the probes' `auto` say "at a lineage's first SCF", and the error of a request outside the
  declared calctypes says "the session was declared for ... only". The calls only the old Python API
  used are gone: `set_continuation` (a continued task is a lineage), `last_probe_gap`,
  `set_probe_warning` (the reports always carry `closed_shell_unstable`), `reset_guess` and
  `has_guess` (`reset`, `reset_all` and `lineages` cover them).
- **`scf.stability_start_pairs` is gone** with D32's multi-pair start (numerics epoch 7: the
  stability searches run per irrep), and with it `start_pairs` in `scf.stability`. The records of
  the stability analyses (`scf.stability`), the zero-mode check (`resolved.record.response.mode_check`)
  and the closed-shell probe (`resolved.record.scf.closed_shell_probe`) carry `symmetry` (the
  converged state's abelian point group, `C1` without symmetry), `irrep` (its Mulliken symbol) and
  `irrep_index` of the reported eigenvalue's search, and `irreps_searched`.
- **The precision settings are one group** (decision D38; input review IN6): `precision.fp64`
  (`auto`, `native`, `emulated`: how the GPU reaches FP64 accuracy in the density-fitted exchange
  GEMMs; auto native on FP64-rich cards, emulated on the others, D27), `precision.fp32` (`auto`,
  `off`, `on`: where FP32 may run outside the SCF loop; auto linus's validated uses, off FP64
  throughout, on also the GPU's XC gradient, CPKS kernel, Hessian XC and VV10 gradient products)
  and `precision.response` (`auto`, `fp32`, `fp64`, synonyms `single`/`double`: the RI response
  exchange, was `response.precision`). `backend.precision` is gone: `double` is now
  `precision: "fp64"` ({fp64: native, fp32: off}), `emulated` `precision.fp64: emulated`, `mixed`
  `precision: "mixed"` ({fp32: on}, the exchange now as `precision.fp64` says). VV10's pair sums
  run in double-float unless `precision.fp32` is off (then FP64), and its CPKS products in FP32
  only where fp32 is not off and the response runs single (or fp32 is on). `auto` on an FP64-rich
  card now resolves to `precision.fp64: native` in `resolved.settings` (the round trip's interim
  exception is gone). CLI: `--precision`. Defaults are bitwise unchanged; one old combination is
  not: `backend.precision: double` with the response on auto in a VV10 Hessian ran FP64 VV10 sums
  beside a possibly single-precision response, and maps to the coherent
  `{fp64: native, fp32: auto}` (double-float VV10 sums).
- **Device memory and the kernel cache are settings** (input review F12): `device_memory_gib`
  (GiB; auto `$LINUS_GPU_MEMORY_GIB`, else 95% of the memory free when the device opens) replaces
  `LINUS_GPU_MEMORY_GB` (GB), with `resolved.record.backend.device.memory_cap_gib`;
  `backend.kernel_cache` (`auto`, `off` or a directory; auto `$LINUS_GPU_CACHE`, else
  `$XDG_CACHE_HOME/linus/gpu-kernels`). The GPU library takes both as arguments
  (`linus_gpu::Gpu::open(ordinal, cache_dir, memory_cap)`) instead of reading the environment.
- **One `hessian` calctype carries the vibrational analysis** (decision D42): the `frequencies`
  calctype is gone. The settings group `freq` tunes it: `freq.analysis` (`harmonic`, `off`),
  `freq.thermo` (`rrho`, `qrrho`: Grimme 2012's quasi-RRHO entropy, `qrrho-hg`: with Li and
  Head-Gordon 2015's enthalpy, `quasiharmonic`: Truhlar 2011, `off`), `freq.ir`,
  `freq.temperature` (a number or a list: one thermochemistry each), `freq.pressure`,
  `freq.concentration`, `freq.scale` (was `frequency_scale`), `freq.rotor_cutoff` (was
  `qrrho_frequency`; the quasi-RRHO models), `freq.raise_below` (was `raise_low_frequencies`;
  quasiharmonic), `freq.symmetry_tol` (was `symmetry_tolerance`) and the analysis's masses,
  projection, symmetry number and constants; a key that cannot take effect under the chosen
  model is an input error (D41). The quasi-RRHO enthalpy without the entropy is gone. The result's
  `vibrational_analysis` and `thermochemistry` (now a list, one per temperature) take qcdata's
  names: `normal_modes_cartesian`, `normal_modes_mass_weighted`, `point_group`, `symmetry_number`,
  `symmetry_number_source` (the `symmetry` object is flattened), `zero_point_energy`,
  `temperature`, `pressure`, `entropy_model`, `enthalpy_model`, `qrrho_frequency`,
  `frequency_scale_factor` (the `options` object is gone). CLI: `--hess` carries the analysis
  (`--freq` is gone), `--thermo` replaces `--qrrho`, `--ir` implies `--hess`. Python: a
  `hessian` request carries `vibrational_analysis` and `thermochemistry` (qcdata's models, one
  thermochemistry per temperature), and with `freq.ir` (or the request output `ir`) the dipole
  derivatives and the polarizability.
- **A warning for SCF presets looser than the default** (input review F2): the presets are
  absolute while the defaults depend on the calctype, so `--conv tight` loosens a gradient (g_tol
  5e-7 against 1e-7) and every preset but `extreme` loosens a Hessian. Such a run now carries
  the warning `convergence_looser_than_default` (path `scf.convergence`); numbers unchanged.
- **`output.molden`** (`off`, `spherical`, `cartesian`): the Molden file of the orbitals in the
  result's new `files` map (`files["orbitals.molden"]`), from every front end; `cartesian` also writes a spherical basis set's orbitals with
  Cartesian functions. CLI: `--molden FILE` writes it. A `spherical` file of a Cartesian basis
  set, or a basis set beyond g functions, is an input error before the calculation.

- **One settings tree** (decisions D38, D39; the input review's schema): every setting is a
  top-level key or `group.key` in one of the groups `jk`, `grid`, `solvent`, `scf`, `response`,
  `backend`, `output`, spelled the same in the CLI's `--set`, a JSON request, Python's keyword
  arguments and qcdata's `keywords`; keys and enum values ignore case and `-` equals `_`;
  `"auto"` means linus decides. Renamed (no aliases: an old key is an `unknown_key` error with a
  suggestion): `restricted` -> `reference` (`auto`, `restricted`, `unrestricted`; synonyms `rhf`,
  `uhf`, ...), `pure` -> `cartesian` (inverted), `memory_gb` -> `memory_gib`, `verbose` ->
  `output.verbose`, `int_thresh` -> `jk.screening`, top-level `aux`/`k_grid` -> `jk.aux`/`jk.k_grid`,
  `exact.storage` -> `jk.exact_storage`, `df.precision`/`df.long_range` -> `jk.df_storage`/
  `jk.df_long_range`, `grid` -> `grid.xc` (`grid: "linus:3"` stays a shorthand), `nlc_grid` ->
  `grid.nlc`, `response.grid` -> `grid.kernel`, `grid_response` and `response.grid_response` ->
  one `grid.response`, `scf.e_conv`/`g_conv` -> `scf.e_tol`/`g_tol`, `scf.incremental`,
  `incremental_reset`, `final_full_build`, `grid_switch` -> `jk.*`, `scf.adiis_switch` ->
  `scf.adiis_until`, `scf.soscf_start` -> `scf.soscf_from`, `scf.saddle_check` ->
  `scf.stability_saddle_check`, `scf.closed_shell_probe`/`closed_shell_gap` ->
  `scf.stability_closed_probe`/`stability_closed_gap`, `response.mode_check` ->
  `scf.stability_hessian_check`, `response.cphf_tol`/`pol_cphf_tol` -> `response.tol`/`field_tol`,
  `solvent.solvent` -> `solvent.name`, `solvent.scf` -> `solvent.scf_terms` (`exact`, `predicted`)
  and `solvent.df: true` (the density-fitted PCM), `solvent.int_thresh` -> `solvent.screening`,
  top-level `pcm` -> `solvent.model`. `scf.stability` is `auto`, `off` (no probe or analysis) or
  `always` (was `false`/`true`). `solvent.lagged_fock` is removed, with its code path.
- **The request's structure is a group**, as qcdata's `Structure`: `structure: {symbols | numbers,
  geometry (bohr), charge, multiplicity}` or `structure: {xyz}`; `orbitals` is `output.orbitals`
  (the matrices: `output.hamiltonian`, `output.overlap`, `output.core_hamiltonian`). The result's
  `molecule` is `structure`
  (`symbols`, `geometry`); `options` echoes only the settings given, in their canonical form; grids
  are reported as their strings (`"linus:3"`); `scf.convergence` holds `e_tol`/`g_tol`.
- **The command line**: each flag is one key (`--help` names it); a request file is a positional
  argument (`linus request.json --threads 8`); flags, `--options` and `--set` apply in order, the
  last one winning. New: `--reference`, `--cartesian`, `--solvent-model`, `--e-tol`, `--g-tol`,
  `--stability`, `--list-settings [GROUP] [--all]`, `linus schema settings|record`. Removed (use
  `--set`): `--aux`, `--k-grid`, `--int-thresh`, `--cpks-grid`, `--nlc-grid`, `--no-grid-response`,
  `--fixed-grid`, `--pcm-preset`, `--pcm-lebedev`, `--pcm-discretization`, `--pcm-scf`,
  `--qrrho-entropy`, `--qrrho-enthalpy`, `--qrrho-freq`, `--symmetry-number`, `--mass-table`,
  `--projection`; renamed: `--uhf` -> `--reference unrestricted`, `--cart` -> `--cartesian`,
  `--pcm` -> `--solvent-model`, `--e-conv`/`--g-conv` -> `--e-tol`/`--g-tol`; one spelling each
  (`--gradient`, `--hessian`, `--frequencies`, `--chrg`, `--multiplicity`, `--disp`, `--epsilon`,
  `--convergence` are gone).
- **Python**: the settings are qcdata keywords under the setting keys (groups as dicts; the
  model holds the method and basis set); `linus.jk_option`, `grid_option`, `solvent_option` are
  removed. New: `linus.settings_schema()`, the `linus.settings` `TypedDict` stubs, the frames'
  `warnings`.

- **A setting that cannot take effect is an input error** (decision D41), before any work:
  `grid` for Hartree-Fock, `int_thresh` with `jk: "df"`, `df.precision` without exact
  exchange, `scf.stability` for a closed shell, `response.*` in a calculation without a
  response, ... The error (kind `input`, CLI exit code 2) names the setting and why, with code
  `no_effect` (no calculation of these settings reads it) or `not_applicable` (only calctypes
  the calculator was not declared for read it). The CLI and `linus.compute` declare their one
  calculation; a Python session covers the calctypes its method supports unless declared
  (`calctypes=`; Rust `Calculator::declare`), and a request outside its declared calctypes is an
  input error too. Settings that take
  effect only when a run-time condition fires (`scf.level_shift`, `scf.plateau_window`, the
  `df` settings of a GPU run, ...) are accepted. In Rust, the settings given are those that
  differ from the defaults. `resolved.declared_calctypes` records the session's.

Settings:

- **SCF settings are one group**: `convergence`, `e_conv`, `g_conv`, `max_iter` and `guess`
  moved under `scf`, next to the algorithm settings: write `{"scf": {"convergence": "tight",
  "max_iter": 200}}` (Python: `scf={"convergence": "tight"}`; qcdata: the same `scf` keyword).
  The command-line flags (`--conv`, `--e-conv`, `--g-conv`, `--max-iter`, `--guess`) are
  unchanged. The old top-level keys are an error.
- **Response settings are one group**: `cpks_grid` is `response.grid` and
  `hess_grid_response` is `response.grid_response` (flags `--cpks-grid`, `--fixed-grid`
  unchanged).
- **`jk` defaults to `df`** (decision 6; the numbers: "Changed: behaviour", numerics epoch 6). A
  request that leaves `jk` unset runs RI-JK. `jk: "exact"` (`--jk exact`) restores the old default.
  `jk.screening` and `jk.exact_storage` apply to exact integrals only, so without `jk: "exact"` they
  are input errors (D41).
- **`numerics_epoch`** in the result JSON (next to `version`), `linus.NUMERICS_EPOCH` in Python, and
  `linus --version` prints it: which arithmetic produced the numbers (numerics epoch 6, "Changed:
  behaviour").
- **`jk: "incore"`** is exact integrals held in memory: `jk` becomes `exact` with
  `exact.storage: "incore"` (the input `"incore"` is still accepted and sets both).
- **`solvent.scf: "df"`** is the name of the density-fitted PCM (it was `"ri"`, which, like
  `"ri-pcm"`, stays accepted as input); `resolved.solvent.scf`, `solvation.scf` and fallback
  reasons report `df`.
- **`basis_dir` is a search path** (a string or a list) searched first; then
  `$LINUS_BASIS_DIR` (now also a list, separated as in `PATH`), the source tree of a
  development build and the installation's sets (the Python package's copy, or
  `<prefix>/share/linus/basis` beside `<prefix>/bin/linus`). Unset, it is `[]` in `options`
  (was `null`). `--basis-dir` is repeatable and `--list-basis` lists the whole path. Builds
  made with `LINUS_DIST` set (the release wheels and archives) carry no build-machine path.
  Python: `linus.basis_dir()` is gone (use `basis_search_path()`), and
  `available_basis_sets(basis_dir=None)` lists the search path.
- **Solvent table**: linus ships the CRC permittivities of the SMD solvents, the other solvents
  it names and about 30 common liquids (207 of qcref's 1303 rows, values unchanged). The other
  CRC liquids resolve through qcref's full table (`LINUS_SOLVENT_TABLE`), with the same rules
  and values; without it such a name is an input error that says how to enable it.
  `linus --list-solvents` and `linus.available_solvents()` list the shipped solvents.
- **Study switches**: `LINUS_KERNEL`, `LINUS_NO_GEN` and `LINUS_ECP_LEVEL` act only with the
  cargo feature `dev-switches` (off by default) and are otherwise ignored with a warning;
  invalid values of `LINUS_*` switches are ignored with the same warning everywhere.
- **The CLI's `--masses` alias is gone**: use `--mass-table` (in JSON and Python, `masses` is
  the per-atom mass list).
- **Basis set file names** escape `(`, `)` and `,` as `_lp_`, `_rp_` and `_cm_`, beside `*`,
  `+`, `/` as `_st_`, `_pl_`, `_sl_`: `6-311+G(2d,p)` is `6-311_pl_g_lp_2d_cm_p_rp_.json` (it
  was `6-311_pl_g(2d,p).json`), so the wheel can ship it. Rename such files in your own
  `basis_dir`.

Checks and errors:

- **Input checks**: `threads: 0`, SCF algorithm values out of range (`damping` outside [0, 1),
  negative shifts or switches, `diis_start`, `diis_space` or `incremental_reset` 0), Python
  `multiplicity` 0, negative or fractional and fractional charges are input errors (0 used to
  become 1). Atoms closer than 0.05 bohr are rejected. More electrons of a spin than basis
  functions is an input error (it panicked). An explicit `response.iteration_aux` must cover
  every element (it fell back silently). Frequency options are checked before the Hessian.
  qcdata `molden` with h functions fails before the calculation. A basis set with a zero,
  negative or non-finite exponent is a `basis` error when it is read.
- **Memory checks**: `memory_gb` (or the machine's memory) now also bounds the fixed arrays of
  Hessians and polarizabilities (first-derivative matrices, response vectors, density-fitting
  tensors), checked before the SCF, and the in-core exact integrals (`exact.storage:
  "incore"`), checked before they are allocated: a `resource` error that names the need and
  the limit instead of running out of memory.
- **Unconverged CPHF/CPKS**: a Hessian, frequency or polarizability whose response stops above
  its tolerance is a convergence error (`error_kind: "convergence"`) instead of a silent
  result; the failure JSON carries `response` (`what`, `iterations`, `max_residual`, `tol`),
  and Python's `ConvergenceError` the `result` reached (with the unconverged Hessian).
- **CLI `error_kind`**: errors found while setting up a calculation report the library's kind
  (`unsupported`, `resource`, `basis`, ...) instead of `"error"`; command-line and request
  errors stay `"input"`. An auxiliary metric that is not positive definite is `numerical`
  (it was `resource`, with the advice to add memory). Scripts that matched `"error"` should
  match the kinds in [docs/cli.md](docs/cli.md).
- A failing SCF can run up to 2 x `max_iter` iterations (the escalation; `scf.escalate: false`
  restores one `max_iter`), and its error message names the escalation.

Results:

- **`resolved` is `{calctypes, settings, reasons, record}`** (decisions D38, D39, D41; package
  PLAN): `settings` is the run's canonical settings, only the keys that take effect, every `auto`
  as the value it took and grids as their strings (`"linus:3"`), a valid request that reproduces
  the run (`--options`); `reasons` says why each `auto` took its value; `record` is what
  happened, diagnostic. Moved: `method`, `dispersion`, `convergence`, `jk`, `aux`, `k_grid`,
  `exact.storage`, `grid`, `nlc_grid`, `response.{cphf_tol, iteration_aux, grid, pol_cphf_tol,
  precision_setting}`, `solvent.{model, preset, scf}` and `backend.kind` into `settings`
  (`precision_setting` is `options.response.precision`; `convergence`'s thresholds are
  `scf.e_conv` and `scf.g_conv`, its `g_norm` follows from `scf.convergence`: PySCF's norm for
  `"pyscf"`, else max |F_ai|); `scf`, `df`, `memory`, `response`
  (`max_residual`, `single_precision`, `two_level`, `pol_max_residual`, `mode_check`),
  `solvent` (`eps`, `eps_table`, `cds_contact_smoothing`), `backend`, `debug_env` and `dft` into
  `record`, with `scf_purpose` as `record.purpose`. `record.memory` is in GiB throughout:
  `limit_gib` (was `gib`), `hbar_gib` (was `hbar_mb`, 10^6 bytes), `xc_kernel_gib`,
  `response_slot_gib` (were MiB); `memory_gb` is `settings.memory_gb`.
  `record.backend.precision_auto` is gone: `reasons["backend.precision"]` says why, and
  `record.backend.arithmetic` gives every kernel family's arithmetic with its reason.
- **`warnings`**: a calculation that succeeded reports what the user should know as
  `[{code, message, path}]` (`backend_fallback`, `df_storage_by_free_memory`,
  `closed_shell_unstable`, `mode_check_flagged`). The library no longer prints them: the CLI
  prints each on stderr (`linus: warning: ...`), Python returns them in the frame's
  `warnings`.
  `Calculator::set_probe_warning(false)` leaves `closed_shell_unstable` out of the reports.
- **Device fallback reasons**: an out-of-memory density-fitting setup reports the bytes in use and
  the limit of the budget it was checked against (the caller's allowance, or the device's limit),
  not a mix of the two.
- **`resolved`**: the CPKS kernel grid moved from `resolved.cpks_grid` to
  `resolved.response.grid`; new `resolved.scf.dropped_functions`, `two_cycle_at`, `plateau`,
  `stability_probe`, `escalation` and `closed_shell_probe`, `resolved.exact.storage` and
  `resolved.backend`;
  `resolved.response` gains `precision_setting`, `max_residual` (the residual the CPHF/CPKS
  reached), `pol_max_residual` and `mode_check`. The other keys of the result and failure JSON
  are unchanged, except as listed below.
- **Energy components and timings**: `components.xc` is the XC energy alone (the solvent's
  energy is no longer added into it and taken out again); `timings.grad_vv10` is part of
  `timings.grad_xc`; the PCM surface timing is `timings.solvent`.

Python and qcdata:

- **The Python distribution is `afh-linus`** (`pip install afh-linus`, extras `qc`, `pyscf`,
  `test`); the import package stays `linus`. The PyPI name `linus` belongs to another project.
- **Python >= 3.11** (the wheel is abi3 for CPython 3.11 and later). 3.10 never worked: qcdata
  needs 3.11.
- A session called from a second thread while busy raises `InputError` (it was pyo3's
  `RuntimeError`).
- **qcdata**: input errors are `linus.InputError` and unsupported calculation types
  `linus.UnsupportedError` (the private `UnsupportedCalculation` is gone); a failed output's
  `error` has qcdata's kind and code.

Build and Rust:

- **Rust 1.89** is the minimum supported version (`rust-version`).
- `LICENSE` holds the MIT license alone; third-party notices moved to `THIRD_PARTY_NOTICES.md`.
  The wheel declares the SPDX expression of what it contains.
- **Rust crates** (unstable, not published): restructured by concept (ARCHITECTURE.md has the
  module map) behind a small documented facade of the `linus` crate (`Calculator`,
  `CalcOptions`, `Molecule`, the result types, `LinusError`, `vib`, and the `JkBuilder` and
  `ResponseJk` traits); everything else is crate-private or hidden, so Rust module paths
  changed. Developer examples moved to the unpublished `linus-dev` crate.
- Rust (unstable): `Molecule::new` returns a `Result`; `Calculator::set_positions` (was
  `set_coords`) checks its input; `Molecule::ecp_core` and `Calculator::timers` are private
  (accessors); `CalcOptions` is regrouped and typed (`ScfSettings`, `ResponseSettings`,
  `DispersionSetting`, `AuxBasis`, `IterationAux`, `ConvergencePreset`, `ExactSettings`,
  `Backend`); `LinusError::Internal` for broken invariants. The result is a typed `Report`
  (with `Resolved`, `EnergyComponents`, `ScfSummary`, ...) that serializes to the result JSON;
  `Calculator::run` takes a `Calculation` (`Energy`, `Gradient`, `Hessian`,
  `Frequencies(FrequencyOptions)`), and `Calculator::failure` builds the typed `Failure` the
  CLI prints and Python attaches. `HessianResult.polarizability` carries the field CPHF's
  polarizability, and `hessian_with` with `dipole_derivatives` solves the field CPHF itself.
- Rust (unstable), the glossary's names ([docs/conventions.md](docs/conventions.md)): J/K types
  are `Jk`-cased (`JkBuilder`, `JkRequest`, `JkOutput`, `ResponseJk`, `DenseResponseJk`,
  `FastJk`, ...), the density-fitting builder is `DfJk` (was `RiJK`) and its response
  counterpart `DfResponseJk` (was `DfRespJK`); `FreqOptions`, `FreqResult` and `VibAnalysis`
  are `FrequencyOptions`, `FrequencyResult` and `VibrationalAnalysis`; settings types follow
  one rule (`*Options` of an entry point, `*Settings` groups of `CalcOptions`, `*Config`
  internal parameters, `Resolved*` what a run used): `SolventOptions` is `SolventSettings`,
  `ScfOptions`, `DfOptions` and `CosxOptions` are `ScfConfig`, `DfConfig` and `CosxConfig`,
  `PcmSettings` and `DfResolved` are `ResolvedPcm` and `ResolvedDf`; `Molecule::coords` is
  `Molecule::positions`; the exact exchange travels as one `Option<Operator>` (`xop`) instead
  of an `(a_x, rs)` pair (`DfJk::exchange_rs` is `exchange_long_range`); the one- and
  two-spin-channel implementations are named `_restricted`/`_unrestricted`.
- Rust (unstable), for backends: `JkBuilder::compute`/`compute_delta`, `XcPotential::compute`
  and `ResponseJk::fock_occ` return `Result` (a device fails with `LinusError::Backend`), and
  so do `scf::run_scf` and `subspace_solve`; the CPKS XC kernel is the `xc::ResponseKernel`
  trait (`Response::xc` takes a `&dyn ResponseKernel`).
- Rust (unstable), the shared vocabulary crates: `Molecule` is `Structure` (qcdata's name: a
  computed structure need not be one molecule), and the module `molecule` is `structure`. It
  lives, with its XYZ reader, the element table and the CODATA constants (`vib::units`), in the
  new bottom crate linus-core, re-exported as `linus::Structure` and `linus::structure`;
  `Structure::new` and `Structure::from_xyz_str` return linus-core's `Error` (an input error;
  `LinusError` converts it with `From`, so `?` in a function returning `LinusError` is
  unchanged), and the calculator sets the ECP core electrons with
  `Structure::set_ecp_core_electrons`. `LinusError::error_kind` gives the kind as the typed
  `ErrorKind`. Deterministic accumulation and the SIMD layer moved from `linus_ints::core`
  (`simd`, `detred`) to the new leaf crate linus-num (`simd`, `det`).
- Rust (unstable), linus-gpu: the engines no longer read the card. What they chose from it is
  an engine option the caller sets, and linus sets it from its device policy
  (`driver/policy.rs`, one owner of every kernel family's arithmetic):
  `GpuDfJkOptions::k_dense` is a `bool`, off by default (was `Option<bool>`, unset: the dense
  exchange planes on FP64-rich cards), and the XC, gradient and Hessian engines take
  `products: ProductEngine` (`Vgemm`, the default, or `Grouped`: cuBLAS's grouped DGEMM, which
  they picked themselves on FP64-rich cards). An engine built with default options on an
  FP64-rich card now runs the vgemm products without dense planes; linus's results are
  unchanged.
- Rust (unstable), the ground state: a calculator's derivatives and properties read their SCF
  from the ground state the SCF stage hands them, not from "the last SCF", so their results no
  longer depend on the order of calls. `Calculator::polarizability` reuses the last SCF only
  when it is at the current geometry and was converged for a response (as frequencies'
  Hessians are); after `energy()` it runs its own SCF converged as for Hessians (before, it
  took the energy's grid, auxiliary basis and thresholds). `aux_basis` and `aux_set` take the
  purpose; `scf_grid()` returns the grid by value; `df_config` takes the memory budget;
  `df_tensor_bytes` is gone (the budget is the SCF's); `solvent_gradient`, `solvent_hessian`,
  `solvent_response` and `scf_report` are crate-private. A failed SCF (an error, not only an
  unconverged one) leaves the warm start the next SCF starts from as it was;
  `vv10()`, `solvation()` and `scf_purpose()` report the last converged SCF.
- Rust (unstable), density fitting: the fits take the Coulomb metric of their auxiliary basis
  set (`two_electron::df::CoulombMetric`, the one builder of the factor of (P|Q); a
  calculator builds one per geometry and auxiliary basis set for the SCF and the derivatives):
  `DfJk::try_new` takes it instead of the thread count, and `df_gradient`, `df_hessian_2e`
  and `DfResponseJk::new_op` take it after the auxiliary basis set. `DfJk::new` builds its own.
- Rust (unstable), linus-ints: the surface-charge integrals of continuum solvation are plans
  writing into the caller's buffers over caller-owned workspaces, like the rest of the integral
  API: `SurfacePlan` (was `SurfaceInts` and the free functions `potential_points`,
  `potential_matrices`, `potential_grad`; it owns its pair data and points), its
  `SurfaceDerivPlan` (`potential_points_deriv`, `potential_matrices_deriv`, `potential_hess`)
  and `AuxSurfacePlan` (`aux_point_matrix`, `aux_point_grad`); the McMurchie-Davidson
  references are in `reference::pcm`. `EcpPlan`'s contractions take an `EcpWorkspace`.
- Rust (unstable), the crate split: the `linus` crate's method code moved into concept crates
  under a facade (ARCHITECTURE.md, "Crates": the graph, which `tools/check_layering.py`
  enforces): linus-base (`LinusError` and the failures, `linalg`, `memory`, and the method
  families' terms: `JkBuilder`, `ResponseJk`, `CoulombHook`, `XcPotential`, `ResponseKernel`,
  `EnvironmentTerm`, `EnvironmentResponse`, `exchange_parts`), linus-jk (`two_electron`'s
  `exact`, `df`, `cosx` and the simple builders), linus-dft (`xc`), linus-scf (`scf`,
  `response`, `reference`), linus-solvent (the continuum models; the `solvent` settings group
  and the dispatch stay in `linus::solvent`) and linus-vib (`vib`; `linus::vib` re-exports it
  with `FrequencyResult`). `linus` keeps the driver, the plan, the session, the backends, the
  assembly of derivatives and properties and the families' dispatch; its documented facade is
  unchanged. Paths: `linus::util::linalg` is `linus_base::linalg`, `linus::util::timer` and
  `vec3` are linus-core's `timing` and `gradient`; `linus::two_electron::{exact, df, cosx}` are
  `linus_jk::{exact, df, cosx}` (`IncoreJk`, `CompositeJk`, `DenseResponseJk` at linus-jk's
  root); `linus::xc` is `linus_dft`; `linus::scf` is `linus_scf`, and `scf::XcPotential`,
  `XcDensity`, `EnvironmentTerm` and `exchange_parts` are `linus_base::terms`'s, as are
  `xc::ResponseKernel` and `response::operator::EnvironmentResponse`; `scf::Convergence` and
  `GradNorm` are `linus_base::convergence`'s; `ScfFailure::of(&r)` is `r.failure()`;
  `scf::Guess` is also `linus::driver::options::Guess`; the RIJCOSX gradient takes its grid
  (`rijcosx_gradient(.., (GridSpec, fitting), ..)`, from `linus::two_electron::gradient_grid`
  of the `k_grid` setting). The device loop's controller moved from `scf::device` to the
  facade's `backend::gpu::scf_control`, so no crate below the facade has a `gpu` feature;
  linus-disp (LGPL) stays a dependency of the facade alone.

### Performance

Wall times before -> after, with the case, the hardware and the threads; GPU timings on one
NVIDIA H200 or RTX PRO 6000 Blackwell (Max-Q) with 8 host threads unless noted, B3LYP unless
noted. Molecules: la18 (L-lactide, 18 atoms), u1an39 (a 39-atom anion), ts61 and ts113
(transition states of 61 and 113 atoms).

- **Small molecules at many threads** (CPU, B3LYP/def2-SVP gradients, `jk: df`, a Zen 5
  Threadripper with 8 cores per CCD, bitwise): a matrix product with less than about 20 µs of work
  per thread runs on one thread, since faer's parallel GEMM synchronizes all its threads on every
  call. Water at 32 threads (four CCDs) 0.32-0.37 -> 0.028 s (1 thread: 0.142 s, 8: 0.028 s), at
  64 (with SMT) 0.21-0.22 -> 0.035 s; caffeine at 32 threads 2.04-2.47 -> 1.28-1.34 s, at 16
  1.94-1.97 -> 1.77-1.88 s, at 64 1.70-1.73 -> 1.26-1.35 s; caffeine's Hessian at 32 threads
  25.6 -> 19.4 s. At 1 and 8 threads unchanged (water 0.142 and 0.028 s, caffeine 15.2-15.3 and
  2.55-2.66 s, its Hessian at 8 threads 19.3-19.4 s).
- **Per-geometry setups shared by the phases, and the DF exchange's scratch kept across the
  SCF's builds** (CPU, 8 threads, def2-SVP, bitwise): the gradient and the Hessian reuse the
  SCF's XC grid and one shell-pair set, and the DF J/K builder reuses its Z panels from one
  Fock build to the next. u1an39 energy 9.68 -> 9.21 s (peak RSS 2.27 -> 2.19 GB, minor page
  faults 851k -> 678k), u1an39 gradient 13.35 -> 12.52 s, la18 gradient 1.30 -> 1.27 s.
- **ECP gradients and Hessians**: the semi-local channels' projections are written into
  buffers reused level after level (they were allocated and faulted in afresh) and transformed
  to the shells' functions over contiguous columns: ch3i HF/def2-SVP, the gradient's ECP term
  19.1 -> 15.8 ms at 1 thread (7.0 -> 6.5 ms at 8) and the Hessian's 110 -> 82 ms (40 -> 27 ms
  at 8), the same bits.
- **Card-aware GPU defaults (D27)**: energies on the H200, u1an39/def2-SVP 3.0 -> 1.4 s and
  ts61/def2-TZVP 17.5 -> 6.2 s; on the RTX PRO 6000 5.6 -> 2.2 s and 50.8 -> 14.3 s.
- **Open shells on the device loop (D31)**: 1.41-1.46x faster than the host loop on minima and
  1.14-1.18x on a saddle point (UKS doublets, RTX PRO 6000).
- **GPU exchange**: the Ozaki-scheme emulated exchange is 2-5x faster than cuBLAS's emulation on
  def2-TZVP molecules of 1000-2200 functions; on FP64-rich cards native K keeps its dense GEMM
  operand on the device (H200 K per call, def2-TZVP: u1an39 102.9 -> 28.8 ms, ts61 143 -> 85 ms),
  and the XC engines' FP64 products run as grouped DGEMMs (1.5-1.7x).
- **The GPU's density-fitting setup on the device**: ts113 ωB97X-D3BJ/def2-TZVP 72 -> 16.5 s
  with `emulated`; AFH's production job (that energy in C-PCM(THF)) 188.6 -> 118.0 s; page-locked
  host memory set up 5x faster.
- **GPU XC** about 2x faster than the first GPU XC engines (basis functions screened per grid
  batch).
- **GPU gradients and Hessians reuse the SCF's setup** (H200, 16 threads, device loop):
  def2-TZVP gradients u1an39 9.8 -> 5.7 s, ts61 16.1 -> 9.7 s; u1an39/def2-SVP Hessian
  19.8 -> 16.9 s.
- **GPU response exchange on a stored operator**: 35-75x less per CPKS iteration (u1an39/def2-SVP,
  117 perturbations: 20.4 -> 0.57 s; ts61/def2-SVP 100.8 -> 2.7 s); streamed from host memory in
  FP64 when it outgrows the card (ts61/def2-TZVP FP64 Hessian 2548 -> 602 s); in FP32 under
  `mixed` (1.25 s per product of 183 perturbations instead of 13.6 s); the FP64 CPKS kernel's
  GEMMs in the Ozaki scheme 1.5-1.6x per product under `emulated`.
- **GPU Hessians' explicit terms on the device**: B3LYP/def2-SVP u1an39 105.6 -> 47.3 s (8 Zen 5
  cores alone: 196.2 s), ts61 416.7 -> 168.6 s (850.4 s); the meta-GGA XC terms too (u1an39
  frequencies, r2SCAN 51.8 -> 30.5 s, ωB97M-D3BJ 104.4 -> 82.8 s).
- **IR spectra on the GPU**: the electric-field response on the Hessian's operator with a field
  kernel for three perturbations, 2.5-3.9x faster (RTX PRO 6000, frequencies with IR,
  def2-SVP, FP64: L-lactide 6.05 -> 4.08 s, u1an39 63.0 -> 50.2 s; at the card's default,
  emulated, L-lactide 5.61 -> 3.51 s).
- **VV10 on the GPU**: 4.3-5.3x faster than on 8 CPU cores beside the device (ωB97M-V
  u1an39/def2-SVP energy 15.5 -> 10.7 s, gradient 19.6 -> 13.9 s); VV10's FP32 CPKS pair sums
  3.50 -> 0.83 s (la18 ωB97M-V frequencies).
- **The CPHF/CPKS solver in BLAS-3**: its own work 29 -> 8.6 s per ts61/def2-TZVP Hessian (8 cores).
- **The field CPHF at `pol_cphf_tol` 1e-6 (D28)**: 16-19% less time (u1an39 35.9 -> 30.2 s, 8 cores).
- **The DF memory budget (D34)**: where B and B^ω now fit, the 113-atom TS ωB97X-D3BJ/def2-TZVP's
  J/K builds take 30 s instead of 52-58 s (32 threads, Xeon 8470).
- **sm_90 schedules** of the generated three-center kernels: 1.07x on an H200.

### Fixed

- **Molden files name their ECP cores.** `output.molden` writes a
  `[core]` section (`atom : n_core`, as PySCF writes and reads it) for atoms with an ECP: PySCF
  read an ECP atom of a Molden file as an all-electron atom of its effective charge (iodine in
  CH3I/def2-SVP: 28 electrons more than its orbitals hold).
- **A response whose residual turns NaN fails instead of passing as converged.** The
  CPHF/CPKS residual's maximum was a fold with `f64::max`, which skips a NaN, so a residual that
  became NaN read as 0 and the analytic Hessian came back non-finite as if converged (HF/STO-3G
  water, warm-started, `response.tol` 1e-300: 9 iterations, "max residual 0", then the harmonic
  analysis failed on the non-finite Hessian). Every maximum over residuals and norms in the SCF,
  the response and the analysis now propagates a NaN (`linus_base::linalg::max_nan`,
  `max_abs`, `max_abs_slice`): the response raises `ConvergenceError`, and a non-finite Hessian
  under a converged residual is a `NumericalError`. Finite results are bitwise unchanged.
- **A `scf.lindep` that leaves fewer orbitals than occupied ones is an input error** naming the
  counts ("scf.lindep 9e-1 drops the basis to 4 orbitals, fewer than the 5 occupied"), on the
  host and the device loop. The host loop stopped with an internal error (a failed assertion in
  the density), the device loop with an internal "contract" error.
- **GPU: the density-fitting tensor streamed from host memory is reachable on capped
  devices.** The exchange's K block ignored the device allowance, so the streamed rung needed
  as much device memory as the device-resident one at caps below it (u1an39 def2-TZVP: 7.6 GB at
  every cap from 2 to 6 GB) and the SCF's J/K went to the CPU (at 6 GB now 12.3 s instead of
  36.1 s). The rungs are device-resident and streamed with the automatic K block, then both with
  the block shrunk to fit beside the other engines' reserve (at least 32), each counting the
  exchange arithmetic's workspace; `resolved.record.df` reports `k_block` and the `rung`. The
  Ozaki exchange's per-build estimate counts its second slices slot (X in descending digit
  order, or W), which an SCF build met as an out-of-memory error after the setup (u1an39
  def2-TZVP at a 10 GB cap, `emulated`; it now runs on the GPU).

- **`--set` (and `--options`) refine a shorthand instead of dropping it.** A dotted path below a
  setting given in short form replaced it with an object of the path alone: `--set backend=auto
  --set backend.precision=double` ran on the CPU and exited 0; `--jk df --set jk.aux=X` failed
  with "missing field kind", `--set solvent=thf --set solvent.model=smd` with "a solvent needs eps
  or a solvent name", `--set grid=linus:2 --set grid.level=3` with "missing field kind". The
  shorthand now becomes its object form first (`backend`, `jk`: `{"kind": ...}`; a grid: its
  tagged object; `solvent`: `{"solvent": name}` or `{"eps": number}`; `CalcOptions::expand_shorthand`),
  the same for an `--options` object over a flag's shorthand; setting `kind` starts a new object, and
  a path below a value without settings (`--set method.x=1`) is an input error.
- **Hessian and frequency results report the gradient's timers.** The `grad_*` timings of the
  gradient that runs inside every Hessian were dropped (water B3LYP with `jk: "df"`: 17 `grad_*`
  keys with `--grad`, none with `--freq`); they are now in `timings`, and the gradient's resources
  (`grad_df_3c_blocks`) in `resources`. Results are unchanged.
- **A GPU implicit solvent's device error stops the SCF at once with its `backend` error.** The
  environment term of the SCF (`EnvironmentTerm::compute`) returned no error: the device PCM
  returned a NaN energy, which ended the SCF as unconverged, and its error was returned after the
  SCF. It now returns a `Result`, and the SCF stops at the failing step with that error.
- **Only out-of-memory and uncovered cases fall back to the CPU at a GPU setup (D38).**
  cuSOLVER's numerical failures (a matrix not positive definite, singular, an eigensolver that did
  not converge) and its other statuses were typed as "not supported", so a setup that met one
  (a density-fitting metric that is not positive definite, say) ran on the CPU with a "device
  setup" note. They are now error kind `numerical` (the numerical failures) and `backend` (the
  library's other statuses), and `GpuError::falls_back_at_setup` is the one rule.
- **GPU errors carry their kind of failure, and three cases now report the right one.** The
  device layer's errors are typed by what failed (not covered, out of memory, host memory, a
  numerical failure, a library's status, a broken internal invariant, invalid input), and both
  policies read the kind: the CPU fallback at a setup (out of memory and not covered, as before)
  and the calculation's error kind. A broken invariant of the device code (a buffer too small
  for its operation, a missing setup step: a bug) is now error kind `internal` (was `backend`).
  The device SCF loop's non-finite core-Hamiltonian orbitals are kind `numerical` (were
  `backend`). Under the `predicted` PCM, a numerical or library failure of the device's Coulomb
  fit setup is now the calculation's error; it was taken for "cannot take the hook", which failed
  with "the J builder is not density-fitted" or moved the surface terms to the CPU. A hook that
  is not taken now says why. cuSOLVER's internal error counts as out of memory when linus's own
  device budget is nearly spent, no longer when the whole card is (another process's memory on
  a shared card decided it). Results are unchanged.
- **The GPU's explicit XC Hessian terms size their accumulators to the device memory left.** They
  took a fixed 24 GiB budget, so under `LINUS_GPU_MEMORY_GB` the whole call had to fit or the terms
  ran on the CPU (ts61/def2-TZVP needs about 7.9 GB of accumulators). They now run in more passes
  over the grid, one per group of atoms, with the same bits. Uncapped runs are unchanged.
- **A GPU XC gradient whose call does not fit falls back to the CPU instead of failing.** Its
  workspaces are sized at the call, not at the setup: la18 B3LYP frequencies under
  `LINUS_GPU_MEMORY_GB=1` failed with "gradient_xc: device memory ... rerun with backend cpu". The
  CPU now computes the XC gradient, with the reason (`device call: ...`) in
  `resolved.backend.fallbacks`.
- **GPU SCFs whose J/K builds do not fit the device are placed before the first iteration.** The
  J/K engine's setup checked only its tensor and a native build's workspace, before the XC engine
  took its memory: under `LINUS_GPU_MEMORY_GB` the first build could then run out of device
  memory (la18 B3LYP/def2-SVP with the emulated exchange at 1.4-1.6 GB, u1an39 at 4 GB: "GPU
  backend, jk (or scf_loop): device memory ... rerun with backend cpu"). Once every SCF engine is
  set up, the SCF's known J/K builds (the iterations', a closed shell's probe builds whenever
  the probe may run, the SAD guess's) are sized against the free memory: the device loop gives
  way to the host loop, and the host loop's GPU J/K to the CPU's, each with its reason (`device
  call: ...`, naming the closed-shell probe's builds when they were counted) in
  `resolved.backend.fallbacks`. A build that still runs out of device memory is error kind
  `resource` with a message naming `LINUS_GPU_MEMORY_GB` and backend `"cpu"`. A stability probe
  whose Fock builds failed ended in a panic (error kind `internal`, "symmetric eigensolver
  failed"); it now returns the builds' error. Uncapped runs are unchanged.
- Internal: the GPU XC product plans hold their arithmetic (FP64, FP64 grouped, FP32), so a plan
  built for FP32 operands cannot be run as FP64 (the precision was passed separately to the run).
  Results are unchanged.
- Internal: the GPU's VV10 response engine refuses FP32 as its arithmetic (it ran double-float
  without a word); FP32 stays the per-call choice of the CPKS kernel's pair sums. Results are
  unchanged.
- **Records after a probe's rotated Fock builds are the converged state's.** The zero-mode check
  of open-shell Hessians (`response.mode_check`, on by default, both loops) and the open-shell
  stability probe on the GPU's device loop left the terms that keep their last state at the
  density of their last rotated build. With the check, an implicit solvent's `solvation` summary
  and `components.solvation_electrostatic` were that build's: triplet CH2 HF/def2-SVP in C-PCM,
  -0.0016460 instead of -0.0016437 Eh, so the components no longer added up to `energy` (2.3e-6
  Eh off). A -V functional's `vv10` energy and statistics were affected the same way, after the
  check and after the device loop's probe. Now a build at the converged state follows them, as it
  already did after the host loop's probe. The energy, gradient and Hessian were not affected.
- **The stability probe's restoring build runs only where a term keeps state** (a solvent, VV10).
  On the CPU it ran after every probe that found no negative mode, in the gas phase too. The
  device loop already skipped it there. Results are bitwise unchanged; where the build runs, it
  is counted in the probe's `fock_builds`.
- **GPU setups under memory pressure fall back instead of failing.** An implicit solvent in the
  predicted mode takes the device J build's fit through the metric's Cholesky factor on the
  device; when that factorization did not fit (cuSOLVER failed on a full card), the calculation
  failed with "solvent.scf = predicted: the J builder is not density-fitted". The hand-over now
  solves on the host in that case (results within round-off), and should a J builder still
  refuse it, the surface terms run on the CPU with the reason in `resolved.backend.fallbacks`
  (`pcm`). cuSOLVER's allocation failure, its internal error while the device is short of
  memory, cuBLAS's allocation failure and the driver's out-of-memory are now out-of-memory
  errors, so the setups fall back as for any other. A device handle returns the memory its pool
  kept to the driver when dropped, so a later calculator in the same process sees it free.
  Results are unchanged.
- A UHF/UKS stability analysis that found a negative orbital-Hessian mode but whose trial
  rotations (0.25, 0.5, 1) all raised the energy took no step, and the SCF reported convergence
  on the saddle point. It now tries 0.125 halved down to 1/64. HF+ r2SCAN/def2-SVP (eigenvalue
  -1.7e-3) ends 1.1e-5 Eh lower, on a minimum; tetrahedral CuCl4 2- HF reaches the solution 1.0e-4
  Eh below the old saddle. Results change only for SCFs that stopped on such a saddle.
- An implicit solvent whose dense surface matrices do not fit in `memory_gb` (or the machine's or
  cgroup's memory when unset) now stops before allocating them, with error kind `resource` and a
  message naming the points, the GiB needed and the ways out (fewer `solvent.lebedev` points per
  sphere, more memory). It used to be killed by the out-of-memory killer (e.g. ethanol with 5810
  points per sphere, 18,817 surface points, in 1 GB). Peak while building: 2 n² doubles for
  C-PCM/COSMO, 6 n² for IEF-PCM, SS(V)PE and SMD. Results are unchanged.
- The GPU kernel cache named its temporary files after the process id only, which repeats
  across containers sharing a cache volume (`LINUS_GPU_CACHE`): two containers could write one
  temporary file. The name now carries 64 random bits too; the rename stays atomic.
- Clean checkouts could not build: linus-disp pinned axtb-d4 to a git commit that exists only in
  a private repository. It now uses the crates.io release `axtb-d4 = "=0.1.1"`, which gives
  bitwise-identical D4 energies, gradients and Hessians.
- Hessians, dipole derivatives and polarizabilities depended on the thread count in their last
  digits (a u1an39 ωB97X-D3BJ density-fitted Hessian differed by 2.7e-7 between 1 and 8
  threads): products with one row or column, and solves with one right-hand side, now run
  sequentially, so results are bitwise identical at any thread count again
  (`crates/linus/tests/faer_parallelism.rs` keeps it so).
- D3/D4 dispersion runs in the calculator's own thread pool (it used the global pool), and
  Python's `vib.harmonic_analysis` in a pool of its own, with panics contained.
- After a failed SCF the next one starts from the guess the failed one used (the projected
  orbitals after a move), not from the last converged density at another geometry.
- **No parallel work inside lazily built shared data.** The XC integrator's batch shell lists
  and the PCM's exact surface integrals were built in parallel inside a lazily initialized cell,
  and the XC Hessian terms' batch loops could build the shell lists while holding a per-thread
  lock: a worker could then wait on itself and deadlock the thread pool (calculations primed the
  lists before those loops; integrators made directly, in tests and studies, did not). Numerical
  results are unchanged.
- The device SCF loop's stability probe starts towards the lowest-energy iterate only when the
  saddle check is on (`scf.saddle_check` > 0), as the host loop's does; with `scf.saddle_check: 0`
  it started there whenever the escalation had kept that iterate's orbitals. The two loops now
  take this decision, like every other SCF decision, from the same code.
- Internal: the SCF is one algorithm (evaluate, observe, decide, step) over named units: the
  Fock evaluator with accepted and trial evaluations, the control policy as state machines on each
  iteration's scalars, the accelerators, and the analyses of converged points. The device loop
  asks the host loop's controls through a hook instead of keeping its own copy of the settings
  and rules, and the solvent is an environment term that the driver runs the SCF with, no longer
  the SCF's entry point. Both loops share one stability search, mode check and closed-shell probe
  (over their rotated gradients) and one assembly of the SCF result. Results are unchanged (SCF
  traces, CPU bitwise suite, GPU A/B).
- Internal: the CPU/GPU boundary is one provider per backend (`crate::backend`). For each method
  family and derivative order (the SCF's J/K, XC term, loop and solvent term; the gradient's
  two-electron, XC and solvent terms; the Hessian's two-electron and XC terms; the response's
  J/K, CPKS kernel and VV10 sweeps), the plan's placement picks the device's provider or the
  CPU's. The provider answers with the family's engine as a trait object or declines with the
  reason, and one resolver records the decision. The per-site "try the device, else the CPU"
  code and the closed enums over the backends (the SCF's J/K and XC term, the CPKS kernel, the
  solvent's SCF term) are gone. The gradient's device schedule (the device's XC and two-electron
  terms beside the CPU's one-electron pass) is the gradient terms' own scheduling, one driver for
  every backend. The GPU's provider is one module with its adapters and the device session
  (`backend::gpu`), the boundary's only conditional compilation: the rest of the crate compiles
  the same with and without the `gpu` feature, and its traits name no GPU type. Results are unchanged (CPU
  bitwise suite, golden JSON, SCF traces, GPU A/B with identical kernels). One timing change in a
  rare case: when the device's XC gradient engine does not fit (its setup, or a call's
  workspaces under a memory cap), the integrator's XC gradient now runs after the overlap with
  the one-electron pass instead of inside it (the same bits).

### Known limitations

- **ROHF/ROKS** is not implemented: open shells run UHF/UKS (`restricted: true` on an open
  shell is refused as `unsupported`).
- **Effective core potentials:** def2-type scalar ECPs only; LANL2-type ECPs (r^-2 local terms)
  and spin-orbit potentials are refused as `unsupported`.
- **RIJ + COSX (`jk: "rij_snk"`)** gives energies and gradients only: no seminumerical exchange
  in the CPHF/CPKS response and no COSX Hessians (D19).
- **VV10 Hessians** spend most of their response time in VV10's kernel; on the GPU its pair sums
  run on the device, its NLC grid's passes on the CPU.
- **Closed-shell stability is a probe** (D36): RKS solutions are checked for a lower
  broken-symmetry UKS solution only below the gap gate and only by default for KS-DFT (HF is
  opt-in, `scf.closed_shell_probe: 2`); it warns and never switches. Its 2 products miss some
  instabilities, and internal ones (a lower RHF/RKS solution) are not looked for: run
  `restricted: false` with `scf.stability: true` for the full analysis and descent.
- **The stability probe is a probe.** Its two products per irrep can miss a negative mode that
  the full analysis finds (on the open set at the default settings: FeH, 7.5 mEh lower, and a Ti
  complex, 5.1 mEh lower; a Hessian of either probes with the zero-mode check's six and steps),
  irreps whose lowest pair lies more than 0.1 Eh above the lowest are not searched, and in a
  degenerate set whose orbitals share an abelian irrep (cubic and icosahedral groups, an atom's d
  shell) the start's direction within the set follows the eigensolver's basis.
- **The full analysis is not exhaustive either.** Starting from an irrep's three lowest pairs, it
  can still miss a mode that grows from a pair further up (on the open set, against the lowest
  state any variant reached: a Ta complex's anion 0.36 mEh, FeO 6.6e-5 Eh, the Co atom, Ni2,
  cobaltocene below 4e-5). Which minimum a step reaches depends on the mode's vector: the
  glutamine radical cation's saddle check steps along the same -1.3e-2 mode, converged further than
  in epoch 7, and the SCF ends in a minimum 1.8 mEh above epoch 7's; the L-lactide cation (UHF,
  default settings) steps along its -0.11 mode into the minimum 1.85e-5 Eh above epoch 7's, the one
  epoch 6 reached, while at tighter settings (openshell_scf's) it reaches the lower one. A negative mode no step lowers,
  and a plateau found with the escalation's budget spent, stay as they are, with the warning.
- **Which state an SCF converges to in a landscape of several minima can depend on the last bits
  of its path** (transition-metal complexes, a linear radical's hole orientation about its axis):
  the per-irrep searches remove the probe's dependence on rounding, not the SCF path's. On the
  open set a one-ulp change of the starting guess moves 9 converged energies by 1e-6 to 5.4e-5 Eh
  (the Fe²⁺ ion, the V and Co atoms, Ni2, cobaltocene, a Re complex's cation, the O atom, two
  ALKBDE10 radicals) and leaves the Sc and Ti atoms unconverged after the escalation (the Ti atom
  fails on epoch 6 without the change). This epoch's guess moved Fe(O)Cl4²⁻ to a state 48 mEh
  higher, FeH onto its saddle (above), Fe(CN)6³⁻ and a Sc complex's cation to states 1.3 and 0.8
  mEh lower, and the Fe²⁺ ion, FeO, the V, Co and Pt atoms and the CH radical by 2e-6 to 9e-5 Eh;
  the Ti atom now converges.
- **The GPU backend** needs NVIDIA GPUs with CUDA 13, runs one device per calculation, and has
  J/K kernels for density fitting only.
- **Non-x86-64 builds** (macOS arm64) agree with x86-64 to round-off, not bit for bit (portable
  kernels; libxc compiled with the platform's floating-point contraction).
- **IR intensities and polarizabilities** are validated against finite differences and finite
  fields, not against another program.

## Internal baseline - 2026-11-08

The feature-complete internal baseline (built as 0.1.0 before the first public release; never
published).

### Methods

- RHF and UHF; RKS and UKS with LDA, GGA and meta-GGA functionals, global hybrids,
  range-separated hybrids (ωB97X, ωB97X-D3BJ, ωB97M-D3BJ, CAM-B3LYP, ωPBE), VV10 functionals
  (ωB97M-V, ωB97X-V, B97M-V), any libxc 7.0.0 functional and mixtures of them.
- Energies, analytic gradients and analytic Hessians; harmonic frequencies, IR intensities,
  dipole derivatives and polarizabilities; point groups and RRHO/quasi-RRHO thermochemistry,
  also for any program's Hessian (`linus.vib`). VV10 Hessians are not implemented (refused
  with a message).
- Coulomb and exchange: integral-direct (`exact`), stored integrals (`incore`), density
  fitting (`df`, RI-JK, with Psi4's wK scheme for range-separated exchange) and RI-J with
  seminumerical exchange (`rij_snk`, COSX) for energies and gradients; `exact`, `incore` and
  `df` for Hessians.
- Effective core potentials (def2, Rb-Rn), Cartesian or spherical basis functions.
- Implicit solvent: C-PCM, COSMO, IEF-PCM and SS(V)PE with Gaussian surface charges, and SMD,
  with energies, gradients and Hessians.
- Dispersion: D3(BJ), D3(BJ)+ATM and D4 with analytic gradients and Hessians.
- DFT grids that reproduce PySCF (`pyscf:N`) and Psi4 (`psi4`) point for point, and linus's own
  (`linus:N`) tuned for speed and for second derivatives.

### Interfaces

- `linus` command line: XYZ files or a JSON request on stdin, a summary or the result as JSON,
  every setting by name (`--set`, `--options`), typed exit codes.
- Python package `linus`: `Calculator`, typed exceptions, `ScfResult`, orbital and basis set
  models with PySCF and Molden converters, and qcdata integration (`linus.compute` and a
  qccodec codec for qccompute).
- Every result reports what the calculation used (`resolved`), its timings and its resources.
- Results are bitwise identical at any thread count.

### Changes during development

For users of pre-release builds: `timings.total` became the wall time of the whole call (it
was the SCF's; the SCF stage is `timings.energy`, and `timings.calculation` is gone);
`df_store_gib`, `df_bytes_per_elem`, `df_one_sided`, `mem_hwm_gb_*` and the block counts moved
from `timings` to `resources`; the result gained `success`, `molecule` and `resolved`.

[0.1.3]: https://github.com/atomsforhumanity/linus/releases/tag/v0.1.3
[0.1.2]: https://github.com/atomsforhumanity/linus/releases/tag/v0.1.2
[0.1.1]: https://github.com/atomsforhumanity/linus/releases/tag/v0.1.1
[0.1.0]: https://github.com/atomsforhumanity/linus/releases/tag/v0.1.0
