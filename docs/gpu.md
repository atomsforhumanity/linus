# The GPU backend

linus can run its density-fitted SCF, gradients and the response equations of Hessians and
polarizabilities on an NVIDIA GPU, an implicit solvent's (PCM, SMD) surface terms included.
Everything else runs on the CPU as before, and every part that has no device kernels falls back
to the CPU on its own, with the reason in the result. The
GPU results agree with the CPU's to round-off (energies within 4e-12 Eh, gradients within
5e-12 Eh/bohr, Hessians within 5e-11 Eh/bohr² in the validation below) and are bitwise
reproducible run to run on one device.

## Building

The GPU backend is the `gpu` cargo feature, off by default. The Python package with it is on
PyPI as `afh-linus-cuda13` (`pip install afh-linus-cuda13`, instead of the CPU build `afh-linus`;
[python.md](python.md#installing)); `linus.GPU_BUILD` is True in it. From source:

```bash
cargo build --release -p linus-cli --features gpu                         # the linus command
cd crates/linus-py && maturin build --release --features gpu               # a Python wheel
```

A wheel built this way is named `afh-linus` like the CPU build; release builds of the GPU
wheel first rename the distribution in their copy of the tree with
`python3 tools/cuda13_distribution.py`, which makes it `afh-linus-cuda13`, with the same version.
Nothing CUDA is needed to build. At run time linus loads, when a calculation first asks for the
GPU:

- an NVIDIA driver with CUDA 13 support (`libcuda`);
- the CUDA 13 libraries NVRTC, cuBLAS and cuSOLVER, where the dynamic loader finds them (a
  CUDA 13 toolkit's `lib64` on the loader path, or `LD_LIBRARY_PATH`).

A machine without them still runs every calculation with `backend: "cpu"` (the default) or
`"auto"`; `backend: "gpu"` then fails with a `backend` error that says what is missing.

The device kernels are compiled at run time (NVRTC) for the card at hand and cached on disk in
the directory of `backend.kernel_cache` (`off`: memory only), by default (`auto`)
`$LINUS_GPU_CACHE` if set (`off` disables the disk cache), else
`$XDG_CACHE_HOME/linus/gpu-kernels` or `~/.cache/linus/gpu-kernels`. A cached kernel is keyed by
its source, the compile options, the GPU architecture and the NVRTC version, so the first
calculation on a new GPU architecture or CUDA toolkit, or with a linus version that changed its
kernels, compiles them (seconds to a minute); later ones, in any process, load them. A calculation
compiles only the kernels it needs: the integral classes of its basis sets, plus the derivative
kernels for gradients and Hessians. For debugging, `LINUS_GPU_KERNEL_DUMP=<directory>` also writes
each module's full source as the compiler sees it (`<key>.cu`) and its name and compile options
(`<key>.json`) there, `<key>` being the cache key.

One compile option depends on the card: on H100/H200 (compute capability 9.x) the three-center
second-derivative kernels of Hessians are built with ptxas `-O1`, because ptxas's default `-O3`
(CUDA 13.0 and 13.1) builds four of their classes wrongly for sm_90 (wrong Hessians with
f and g shells, such as def2-TZVP's C-F pairs with the def2-universal-jkfit g functions: 4.9e-4
Eh/bohr^2 on a 39-atom molecule). It costs these kernels about a third of their time there
(0.79 to 1.04 s of a 44 s def2-TZVP Hessian). linus-gpu's `compile_options` names the evidence.

**Containers and cluster jobs: put the cache on persistent storage.** The default directory is
inside the container's home, which is empty in every new container, so every pod of a cluster job
compiles the kernels again while the GPU waits. Measured on an NVIDIA H200 (sm_90) with a
linus 0.1.0 container image and CUDA 13, a B3LYP-D3(BJ)/def2-TZVP gradient with `jk: "df"` of
C10H22 (442 functions): 58.5 s in a fresh container, 52.4 s of it in `timings.grad_2e_setup`
compiling, and 5.5 s (setup 0.7 s) with a warm cache, in the same or another container. Set
`backend.kernel_cache` (or `LINUS_GPU_CACHE`, its default) to a directory the pods share (a
shared filesystem works: files are written to a
temporary name unique to the process, also across containers, and renamed, and one that fails to
load is recompiled). Before starting many pods at
once, warm it by running one calculation of each kind the jobs will run (each basis set; energy,
gradient or Hessian), so they don't all compile the same kernels. The cache's subdirectories are
created mode 0700, so they are shared by processes of the same user only.

## Using it

The `backend` setting picks the hardware and the `precision` settings the GPU's arithmetic,
the same in every front end:

| front end | GPU | GPU with FP64-emulated exchange |
|---|---|---|
| CLI | `--backend gpu` | `--backend gpu --precision emulated` |
| Python (qcdata `ProgramInput`) | `keywords={"backend": "gpu"}` | `keywords={"backend": "gpu", "precision": "emulated"}` |

`backend`:

- `kind`: `cpu` (the default), `gpu`, or `auto` (the GPU when the build has the feature and a
  device opens, else the CPU, with the reason in `resolved.record.backend.no_device`).
- `devices`: `[n]` for device n (default 0); one device per calculation.
- `scf_loop`: where the SCF's own work runs (orthogonalization, Fock assembly, DIIS,
  diagonalization). `host`: the CPU drives the device's Fock builds, and the matrices cross
  PCIe once per iteration. `device`: the whole loop on the GPU, with orbitals, densities, Fock
  matrices and the DIIS subspace on the device (CLI: `--set backend.scf_loop=device`). `auto`
  (the default): `device`, closed and open shells, on every card (decisions D27 and D31).
- `kernel_cache`: where the compiled kernels are kept (above): a directory, `off` (memory only)
  or `auto` (the default: `$LINUS_GPU_CACHE`, else `$XDG_CACHE_HOME/linus/gpu-kernels`, else
  `~/.cache/linus/gpu-kernels`).

`precision` (Precision below has the details and measurements):

- `fp64`: how the GPU reaches FP64 accuracy in the density-fitted exchange GEMMs. `native`
  (synonyms `double`, `fp64`): the FP64 units. `emulated`: the GPU's density-fitted exchange
  GEMMs (SCF and response) in FP64 emulation on INT8 tensor cores, FP64 accuracy on this data and
  much faster on cards with little native FP64 throughput. `auto` (the default): the card's
  faster of the two, by its compute capability: `native` on FP64-rich cards (A100, H100/H200,
  V100, P100) and on the CPU, `emulated` on the others (RTX and RTX PRO, L40/L4, A40/A10, T4,
  B200, B300); `resolved.settings.precision.fp64` says which and
  `resolved.reasons["precision.fp64"]` why.
- `fp32`: where FP32 may run outside the SCF loop. `off`: nowhere, FP64 throughout (VV10's pair
  sums in FP64 too, and `precision.response: auto` FP64). `auto` (the default): linus's validated
  uses: the single-precision response exchange of large RI Hessians (CPU and GPU), the GPU's CPKS
  and VV10 kernels in FP32 with it, and VV10's pair sums in double-float on FP64-poor cards (FP64
  on FP64-rich ones). `on` (opt-in): also the GPU's XC gradient, CPKS XC kernel, Hessian XC and
  VV10 gradient products in FP32, and the stored FP32 response exchange operator where it pays.
- `response`: the arithmetic of the RI response exchange of Hessians: `fp32` (synonym
  `single`), `fp64` (synonym `double`) or `auto` (the default: as `fp32` says; with `fp32: off`
  FP64, else the CPU's single-precision rule, and on the GPU the card's rule, below).
- A string is a shorthand: `precision: "fp64"` (synonyms `native`, `double`) is `{"fp64":
  "native", "fp32": "off"}`, FP64 throughout; `"emulated"` is `{"fp64": "emulated"}`; `"mixed"`
  is `{"fp32": "on"}`, which leaves the exchange to `precision.fp64` (on FP64-poor cards its
  `auto` emulates it; `{"fp64": "native", "fp32": "on"}` keeps it in native FP64); `"auto"` keeps
  the defaults. A dotted key refines a shorthand (`--precision mixed --set
  precision.fp64=native`), and an object sets single keys (Python `precision={"response":
  "fp32"}`, CLI `--set precision.response=fp32`).

The CPU computes in FP64 except the response exchange (`precision.response`): with `backend:
"cpu"`, `precision.fp64: "emulated"` and `precision.fp32: "on"` are refused. `device_memory_gib`
(top level) caps the device memory a process uses (Sharing a GPU, below).

The GPU backend's J/K kernels are density fitting's (`jk: "df"`, the default). With `jk:
"exact"` or `"rij_snk"` the J/K terms stay on the CPU and only the XC terms move.

A session opens its device once and keeps it for all its calculations (geometry steps,
frequencies). Each calculation sets its engines up again (the density-fitting tensor, the
grid's blocks); that setup is part of the timings below. The density-fitting setup runs on the
device: the three-center integrals from linus's own kernels (on the CPU only for basis sets they
do not cover, e.g. Cartesian d shells: `resolved.record.df.integrals` says `cpu`, with the reason) and
the fit with the metric's factor.

## What runs where

| part | on the GPU | on the CPU |
|---|---|---|
| SCF J/K (`jk`) | `jk: "df"`: HF, pure, global and range-separated hybrids (the long-range exchange in Psi4's wK scheme, as the CPU) | `jk` exact or rij_snk; `solvent.df: true` (the density-fitted PCM's terms ride on the CPU's density-fitted J) |
| SCF XC (`xc`) | LDA, GGA, meta-GGA, closed and open shells | - |
| VV10 nonlocal correlation of -V functionals (`vv10`): its energy, SCF potential and gradient (grid response included), on the NLC grid | always (ωB97M-V, ωB97X-V, B97M-V) | - |
| gradient, two-electron term (`gradient_jk`) | `jk: "df"`, range-separated hybrids included | `jk` exact or rij_snk |
| gradient, XC term (`gradient_xc`) | always (grid response included) | - |
| Hessian, explicit two-electron terms (`hessian_jk`): the second derivatives at fixed orbitals (three- and two-center integrals, the RI cross terms) and the derivative Fock matrices of the CPHF right-hand sides | `jk: "df"`, range-separated hybrids included (the two-center second derivatives' contraction on the host) | `jk: "exact"` |
| Hessian, explicit XC terms (`hessian_xc`): the XC potential's derivatives, the XC energy's second derivative, the grid response | LDA, GGA and meta-GGA functionals (hybrids and range-separated hybrids included), closed and open shells; the grid response on any partition (the weight derivatives from linus-grid, as the CPU) | - |
| CPHF/CPKS J/K of Hessians and polarizabilities (`response_jk`) | `jk: "df"`, range-separated hybrids included | `jk: "exact"` |
| CPKS XC kernel (`response_xc`) | always | - |
| VV10 in Hessians and polarizabilities (-V functionals, `vv10`): the CPKS kernel and the explicit terms, on the NLC grid | the pair sums (the O(n^2) part per perturbation; in the SCF VV10's arithmetic: by default FP64 on FP64-rich cards and double-float on the others, FP64 with `precision.fp32: "off"`; the CPKS kernel's in FP32 when the device's semilocal CPKS kernel runs FP32, `response` `xc: "fp32"`) | the NLC grid's passes (first-order densities, contractions) and the potential's own terms |
| implicit solvent: the surface terms of the SCF and of gradients (`pcm`) | `solvent.scf_terms` exact or predicted (the default with density fitting): the surface charges' integrals, the charge solve, the reaction field's Fock matrix and the gradient's integral derivatives | `solvent.df: true` |
| the SCF's nuclear attraction V (`nuclear_attraction`) | large molecules beside a device J/K: kept shell pairs x point charges at least 2e8 (ubiquitin, 1231 atoms; the device PCM engine's pair x point integrals; within 1e-15 of the CPU's, relative to its largest element) | below that, and on any device failure |
| one-electron integrals (overlap, kinetic energy; the nuclear attraction of smaller molecules) and their derivatives, ECPs, the cavity and SMD's CDS term (`solvent`), the PCM's Hessian terms (`solvent`), dispersion | - | always |

The SCF loop itself (DIIS, diagonalization) runs on the GPU by default, closed and open shells
(`backend.scf_loop: "auto"`, decisions D27 and D31); `"host"` keeps it on the CPU, the matrices
going to the device and back once per iteration. The device loop runs when its Fock terms do
(`jk: "df"`; VV10 included) and the algorithm is the default one (no `scf.adiis`,
`scf.soscf` or `scf.damping`); otherwise the host loop runs, with the reason (`scf_loop` in
the fallbacks). An implicit solvent's reaction field runs on the device's own
buffers in the device loop (with `pcm` on the CPU, its terms come from the CPU each iteration).
Open shells (UHF/UKS) on the device loop take the host loop's decisions. The default stability
probe (`scf.stability_probe`) and the zero-mode check of open-shell Hessians
(`scf.stability_hessian_check`) run on the device: their Hessian-vector products rotate the orbitals,
build the Fock matrices on the loop's own engines and form the orbital gradient there, and the
same Davidson iterations as the host loop's decide. When the probe finds no negative mode (the
usual case) the SCF ends on the device. The closed-shell probe of RKS solutions
(`scf.stability_closed_probe`) runs on the device the same way, with UKS-type Fock builds on the
loop's engines; its eigenvalue agrees with the host loop's and the CPU's. The soft-mode plateau rule (`scf.plateau_window`)
applies to the device's iterations. What the host loop does beyond plain DIIS, all rare (a
negative probe and the analysis and steps that follow, the saddle check, `scf.stability: "always"`,
a two-cycle), continues on the host loop: the device loop stops at that iteration and the host
loop redoes it from the device's state (its orbitals, the lowest-energy iterate, the history)
and goes on as it would have from the start. When the escalation comes due on the device (DIIS
stagnated or blew up, or 2/3 of `max_iter` spent), the device loop hands the state after that step to the
host loop, which escalates (`scf.escalate`) from the lowest-energy iterate the device kept, for
the rest of the same `max_iter`. The stability decisions, the records in
`scf.stability` and the iteration counts are then those of the host loop.

The implicit solvent on the GPU keeps the CPU's SCF modes (docs/solvation.md): `predicted`
(the default with density fitting: one pass over shell pairs x surface points per iteration,
the charges predicted from the J build's fit) and `exact` (two passes). The device always does
full passes: the CPU's `solvent.incremental` (passes on the change of the density) does not
apply. The cavity is built on the CPU; the PCM's matrices are factored on the GPU (Cholesky for
C-PCM and COSMO, LU for IEF-PCM and SS(V)PE). The density-fitted PCM (`solvent.df: true`) runs
on the CPU (`pcm` in the fallbacks), and so do the PCM's terms of Hessians (`solvent: "cpu"`).

Open shells with several stable UHF/UKS solutions can, in knife-edge cases, converge to a
different one on the GPU than on the CPU: a soft mode amplifies the backends' round-off, and the
saddle check or the stability step then fires at another iteration (HF+ r2SCAN at the extreme
preset: the CPU's and the device loop's end on the same minimum by different paths, the GPU's
host loop on another 1.8e-7 Eh above it). It is deterministic (the same solution every run on
one device). Compare such energies against the solution's own stability analysis
(`scf.stability: "always"`), not against another backend's run. (Before the device loop resumed the
host loop from its state, its hand-off restarted from the densities and reached other minima
more often: the L-lactide cation, def2-SVP, UHF 1.8e-5 Eh and ωB97X UKS 3.5e-6 Eh; these now
agree with the CPU within 3e-12 Eh.)

With `backend: "gpu"` (not `auto`), linus also prints one warning line on stderr when the SCF's
`jk` or `xc` runs on the CPU after all, by rule or at setup (`linus: warning: backend gpu: jk
runs on the CPU: <reason>`): a GPU job that quietly becomes a CPU job is easy to miss.

The density-fitting tensor sets up on the cheapest rung that fits the device memory the engine
may hold, each counting the tensor, a build's workspace and the exchange arithmetic's (the Ozaki
scheme's slices where it is the arithmetic). First on the device with the automatic K block (the
auxiliary functions per exchange block). Then, by their costs per J/K build (decision D11): on the
device with the K block shrunk to fit (a multiple of 8, at least 32), and streamed from
page-locked host memory with the automatic block; the device-resident tensor goes first unless
even its smallest block would cost more than streaming. Then streamed with the block shrunk; then
the SCF's host loop and the CPU's J/K. The costs are measured: an extra K block about 0.2 ms
per build, streaming the tensor about its size over 68 GB/s per build. (u1an39 B3LYP/def2-TZVP on
an RTX PRO 6000, the 5.49 GiB tensor, 16 builds: on the device 0.154 s per build with the
automatic block of 496, 0.155 s with 136 and 0.166 s with 40; streamed 0.240 s. At 10 and 12 GB
caps the SCF's J/K therefore runs on the device with blocks of 40 and 136, 2.5-2.7 s instead of
3.8 s streamed.) `resolved.record.df` names the rung (`rung`, with why and, for a shrunk block,
the two costs) and the K block (`k_block`). The K block orders the exchange's sums over the
auxiliary functions, so a capped device (`device_memory_gib`, or a busy card) can change the
result at round-off, as the storage choice can (u1an39/def2-TZVP: the energies of every rung agree
to 1e-12 Eh). (At a 6 GB cap (5.59 GiB) the tensor fits the device on no rung: streamed with a K
block of 216, 12.3 s against 36.1 s with the CPU's J/K it fell back to before.)

Two cases fall back at setup time instead of by rule: a density-fitting tensor that fits
neither the device on any rung nor, streamed from page-locked host memory, 75 % of `memory_gib`
(the CPU's own storage plans take over), and a range-separated response whose tensor does not
fit the device. Any other device error, and every error after a setup, stops the calculation: out
of device memory with error kind `resource` and code `device_memory` (page-locked host memory that
cannot be pinned: `host_memory`; Python `linus.ResourceError`), a driver fault with kind `backend`
(Python `linus.BackendError`): a calculation never continues on the CPU midway, so its result does
not depend on the device's load. A setup falls back only out of device memory or in a case its engine does not cover; a
device factorization that fails numerically (a density-fitting metric that is not positive
definite, a singular matrix) stops the calculation with error kind `numerical`, as on the CPU.
The XC gradient's call counts as its setup (its workspaces are sized only then): when they do
not fit, the XC gradient runs on the CPU (`gradient_xc` in the fallbacks). The SCF's J/K builds
are sized before its first iteration against the memory the other engines left: when they do
not fit, the device loop gives way to the host loop, and then the J/K to the CPU (`scf_loop`
and `jk` in the fallbacks); the SCF never changes engines between iterations. A build that
still runs out of device memory (one not known in advance) stops the calculation with error
kind `resource`, code `device_memory`.

`resolved.record.backend` reports where each part ran and why:

```json
{"kind": "gpu", "device": {"id": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
  "memory_gib": 95.0, "compute_capability": "12.0"}, "precision": "double",
 "jk": "cpu", "xc": "gpu", "vv10": null, "gradient_jk": "cpu", "gradient_xc": "gpu",
 "hessian_jk": "cpu", "hessian_xc": "gpu", "response_jk": "cpu", "response_xc": "gpu", "scf_loop": "host",
 "one_electron": "cpu", "pcm": null, "solvent": null, "dispersion": null,
 "fallbacks": [
  {"kernel": "jk", "reason": "jk exact: the four-center integrals have no device kernels (jk df has them)"},
  {"kernel": "gradient_jk", "reason": "jk exact: the four-center integrals have no device kernels (jk df has them)"},
  {"kernel": "hessian_jk", "reason": "jk exact: the four-center integrals have no device kernels (jk df has them)"},
  {"kernel": "response_jk", "reason": "jk exact: the four-center integrals have no device kernels (jk df has them)"}]}
```

(r2SCAN frequencies with `jk: "exact"`, `backend: "gpu"` and `precision: "fp64"`: the XC
families on the device, the exact integrals on the CPU; with `jk: "df"` every family runs on the
device and `fallbacks` is empty; a -V functional's energy or gradient reports `vv10: "gpu"`;
`precision` is the device's arithmetic: `emulated` where the exchange runs in FP64 emulation,
else `mixed` with `precision.fp32: "on"`, else `double`. At the default `precision.fp64:
"auto"`, `resolved.settings.precision.fp64` reports the choice for the card and
`resolved.reasons["precision.fp64"]` why.)
`resolved.record.df` describes the SCF's density-fitting tensor:
`storage` is `device` or `host_streamed` on the GPU. The timings keep their names (`jk`, `xc`,
`grad_2e`, `grad_xc`, `hess_cphf`, `vv10`, `grad_vv10`, ...); the GPU adds its setups (`jk_setup`,
`xc_setup`, `vv10_setup`, `grad_2e_setup`, `grad_xc_setup`, `cphf_xc_setup`). A Hessian's explicit two-electron terms are
`hess_2e` on the device (with `hess_2e_setup` and its phases `hess_2e_*`) in place of the CPU's
`hess_df_*`. With the grid's partition weights on the device (PySCF's Becke grids), `grid` has
the parts `grid_partition` (the device's weights) and `grid_batches` (the batching).

## Precision

Three settings decide the arithmetic (the `precision` group, Using it above): `precision.fp64`,
how the GPU reaches FP64 accuracy in the exchange GEMMs (native or emulated); `precision.fp32`,
where FP32 may run outside the SCF loop (`off`, linus's validated uses at `auto`, or the opt-in
`on`); and `precision.response`, the arithmetic of the response exchange.

- FP64 accuracy by default, everywhere (VV10's pair sums in double-float on FP64-poor cards,
  below).
  `precision.fp64: "auto"` (the default) picks the card's faster FP64-quality arithmetic from its
  compute capability (decision D27), a fixed rule, so a card's results do not depend on a
  timing:
  - `native` where native FP64 is faster: cards whose INT8 tensor cores run at most about 32x
    their FP64 rate (A100: 8.0, H100/H200: 9.0) or that have none (V100, P100). On an H200
    the exchange per call is 1.2-1.9x faster in FP64 than emulated (ts61, ts113 TZVP);
  - `emulated` (below) where emulation is faster: the cards with FP64 at 1/32-1/64 of FP32
    (RTX and RTX PRO, L40/L4, A40/A10, T4: INT8 about 500x FP64; on the RTX PRO 6000 the
    exchange per call is 5.6-7.6x faster and a TZVP SCF about 2x), and B200 (sm_100: INT8 about
    112x FP64; extrapolated from the two measured cards, not measured), B300, Thor, GB10.
  - `resolved.settings.precision.fp64` reports the choice and `resolved.reasons["precision.fp64"]`
    why, e.g. `"auto: emulated (compute capability 12.0: INT8 tensor cores about 500x FP64; FP64
    emulation on INT8 tensor cores is faster than native FP64 here, at FP64 accuracy)"`.
    `native` and `emulated` set explicitly are used as set on every card. Energies of
    `emulated` stay within about 3e-12 Eh of `native` with the same iterations, so the defaults
    of two cards agree to round-off.
- **VV10's pair sums** (-V functionals) follow the card class, as the exchange does
  (`precision.fp32: "auto"`, the default): native FP64 on FP64-rich cards (H100, H200, A100;
  on an H200 3.2x faster than double-float: the u1an39/def2-TZVP ωB97M-V pair kernel 27 against
  86 ms per SCF call), and on cards with little FP64 throughput double-float arithmetic: each
  value the unevaluated sum of two FP32 numbers (about 46 bits), each point's sums added in FP64
  in a fixed order. A pair costs one division and about 25 operations, so there the FP32 units
  do it several times faster (on the card below 6.5x native FP64, which there runs slower than 8
  CPU cores). A precision study found double-float FP64-equal: on 55 -V cases (water,
  L-lactide, the 39-atom anion; ωB97M-V, ωB97X-V, B97M-V; closed and open shells; energies,
  gradients, the device loop, C-PCM) the energies are within 2.7e-12 Eh of the CPU backend with
  the same SCF iterations and within 4.3e-12 Eh of native FP64 pair sums (VV10's energy within
  2.3e-14 Eh), the gradients as close to the CPU's as with FP64. `precision.fp32: "off"` (as in
  `precision: "fp64"`) runs them in native FP64 on every card, `"on"` in double-float (the
  gradient's in FP32); `resolved.record.backend.vv10_arithmetic` reports which (e.g.
  `{"scf": "fp64", "gradient": "fp64"}` on an H200, `"df64"` on an RTX PRO 6000) and
  `resolved.record.backend.arithmetic` why. Each arithmetic computes the same bits on every card
  of its class (IEEE-rounded operations only, no atomics, a fixed order); see
  [Reproducibility](#reproducibility).
- `precision.fp64: "emulated"`: the density-fitted exchange GEMMs of the SCF, of the
  CPHF/CPKS responses and of a Hessian's explicit terms (the exchange part of the derivative
  Fock matrices) run in FP64 emulation on the card's INT8 tensor cores (the Ozaki scheme:
  each value cut into 7 signed 8-bit slices, 55 bits relative to the largest value it is
  multiplied with, the slice products summed exactly in integers). The density-fitting tensor
  is sliced once per calculation when its sliced copy fits the device beside the other
  engines; otherwise each build slices the parts it streams. The exchange matrices stay within
  2e-14 (relative) of FP64 and the energies within 2e-12 Eh, with the same SCF iterations and
  convergence history even at tight thresholds: the emulation has a fixed precision, so it is
  smooth in the density, and it is bitwise reproducible run to run.
  - The density-fitting setup's fit (the triangular solve with the metric's Cholesky factor,
    most of the setup) runs on the INT8 tensor cores too: blocked, every product an Ozaki GEMM,
    the diagonal blocks refined once, so it keeps the triangular solve's accuracy (the SCF's
    Coulomb and exchange energies equal the FP64 fit's to 1e-13 Eh). It is FP64 when its
    workspace (about 1 GiB at 2155 functions) does not fit the device, with the reason.
  - When the emulation cannot be set up (device memory, a card or library without INT8 GEMMs)
    it falls back at setup to cuBLAS's own FP64 emulation (55-bit fixed mantissa), and then to
    FP64 (the Hessian's terms directly to FP64). `resolved.record.backend.emulation` reports, per
    family (`jk`, `response_jk`, `hessian_jk`), what ran and why, and for `jk` the fit's tier:
    `[{"kernel": "jk", "scheme": "ozaki", "bits": 55, "presliced": true, "fit": {"scheme":
    "ozaki", "bits": 55}}]`.
  - It pays for TZVP-size molecules and up; on small ones it is about as fast as FP64.
  - A response whose exchange runs on the stored operator (below) emulates the operator's
    setup with cuBLAS's 55-bit emulation instead (`"scheme": "cublas"` for `response_jk`, with
    the reason); its large, square GEMMs gain 2.5x there, and the Ozaki planes would sit unused.
  - The XC terms stay in FP64 under `emulated`: their products are small (one per grid
    batch), where emulation costs time. The CPKS kernel follows `precision.response` (below);
    where it runs in FP64, its two large GEMMs per 512-point block (P = F A, dV = F^T Z) run in
    the Ozaki scheme (`resolved.record.backend.response`'s `xc: "emulated"`): 1.5-1.6x per product
    (u1an39/def2-TZVP 4.50 -> 2.79 s), within 5e-14 of FP64.
- `precision.fp32: "on"` (opt-in, never the default; the shorthand `precision: "mixed"`): the
  XC gradient's, the CPKS XC kernel's and the Hessian XC terms' potential-derivative products in
  FP32 (each grid batch, at most 256 points or its functions, summed in FP32; the batches in
  FP64), where a precision study found the errors at least 10x below the default grids' own:
  gradients within 3e-8 Eh/bohr of FP64 (over 500x below the grid's error), frequencies within
  2e-4 cm⁻¹ (u1an39 and ts61 B3LYP/def2-SVP Hessians: over 1000x below the grid's error, linus:3
  against linus:4). On this card the FP32 CPKS kernel is about 9x faster than in FP64, the XC
  gradient about 1.8x and the Hessian's XC terms about 2.3x. The VV10 gradient's pair sums run in
  FP32 too (each tile of 128 pairs summed in FP32, the tiles in FP64; the SCF's stay
  double-float): on -V gradients it is within 4e-8 Eh/bohr of FP64 pair sums (VV10's own part
  about 1e-10), at least 130x below the NLC grid's own error (linus:1 against PySCF level 4), and
  its pair kernel is about 5x faster than double-float. SCF energies and iterations are those of
  the default `precision.fp32: "auto"` (the SCF loop is untouched), and `precision.response`
  decides the response J/K as usual, except below.
  - **The exchange GEMMs follow `precision.fp64`**: `precision.fp32: "on"` takes the card's
    choice there, so on FP64-poor cards at the default `precision.fp64: "auto"` it runs with the
    emulated exchange. Before the precision settings were split, `mixed` kept the exchange in
    native FP64; that is `precision: {"fp64": "native", "fp32": "on"}` now, and the `mixed`
    measurements on this page were taken with it.
  - Where the response runs single precision (`precision.response: fp32`, or `auto` where the
    CPU's rule runs single precision), `precision.fp32: "on"` runs its exchange on a **stored
    operator in FP32** (setup in FP32 sums over blocks of 256 auxiliary functions added in
    FP64, half the FP64 operator's bytes, streamed from host memory as far as needed; else the
    FP32 exchange through the tensor). Against FP64 its Hessians lie within 2.7e-7 Eh/bohr² and
    4.4e-4 cm⁻¹ (la18 and u1an39, B3LYP and ωB97X-D3BJ, RKS and UKS; 2-3.5x the FP32 tensor
    exchange's, still far below the grid's error); a ts61/def2-TZVP product takes 1.25 s instead
    of 13.6 s (setup 13 s).
  - Measured and not offered: an FP32 potential (densities and E_xc kept in FP64) changed SCF
    energies by at most 1.4e-11 Eh in 40 of 41 test cases (a slow open-shell SCF stopped
    3.8e-6 Eh away at the default thresholds) but made the SCF take 1-6 more iterations on
    TZVP-size molecules, which cost more than it saved; full FP32 XC (FP32 densities too)
    changes E_xc at first order, by as much as the grid error itself on small molecules.
- **A Hessian's response exchange on a stored operator.** K is linear in the response's
  first-order orbitals, so per spin channel it is one (n o) x (n o) matrix (n functions, o
  occupied orbitals), built once and then applied to all perturbations as GEMMs. Each CPKS
  iteration's exchange then costs 35-75x less than through the density-fitting tensor
  (range-separated hybrids no more than global ones), for a one-time setup of a third to half
  an iteration's former cost and (n o)²/2 doubles of device memory: 13 GiB for a 39-atom anion
  with def2-SVP (483 functions), 41 GiB for a 61-atom molecule (666 functions), 61 GiB for the
  anion with def2-TZVP. linus builds it only when the products pay for it (a Hessian's 3N
  perturbations; a polarizability's three only for small molecules). Its device memory is
  granted at the response's setup, where the decision is made, and the operator is built at
  the first product within it, so the Hessian's other device work in between cannot change
  the decision (`operator_granted_gib` in `resolved.record.backend.response`; the DF gradient's
  and Hessian's, the XC gradient's and Hessian's workspaces and the PCM's surface matrices are
  granted the same way, an engine that holds more at once than its grant recorded in
  `resolved.record.backend.plan_violations`). An IR run's
  electric-field response, which needs the same exchange of the same orbitals, runs on the
  Hessian's operator, kept alive past the Hessian (when it is FP64; not after `auto`'s FP32
  fallback): 750 -> 145 ms per field iteration's exchange for the 39-atom anion. It is stored in chunks
  of whole orbitals (1 GiB): as many as fit beside the other engines stay on the device, and
  with `precision.response: "fp64"` (or where `auto` finds the streamed operator the cheaper
  path, below) the rest go to page-locked host memory (at most 60% of
  `memory_gib`, else of the machine's or the container's memory) and are copied back to the
  device in every product on a second stream, behind the products of the chunks before them.
  A 61-atom molecule with def2-TZVP (1343 functions) needs 167 GiB, 115 GiB of them in host
  memory on a 96 GB card: its products take 12.9 s (all 183 perturbations) whether 113 or all
  167 GiB stream over PCIe (57 GB/s each way on this card), against ~357 s through the tensor,
  and its FP64 Hessian takes 628 s instead of 2548 s. The products are bitwise the same with
  any split between device and host. Where the operator does not fit (not even with the host
  part), the exchange runs through the tensor, with the reason. Its results agree with the
  tensor path to round-off (the products within 1e-14, relative). Its products always run in
  FP64; its setup in FP64, or in cuBLAS's FP64 emulation (55 bits) with `emulated`.
- `precision.response` on the GPU (Hessians, frequencies, IR spectra; polarizabilities always
  run in double precision):
  - `fp64`: FP64 throughout (the exchange on the operator, on the device and in host memory
    as far as needed, else the tensor; the CPKS XC kernel in FP64).
  - `fp32`: the exchange GEMMs in FP32 through the tensor (J in FP64) and the CPKS XC
    kernel's products in FP32, as `precision.fp32: "on"` runs them. The GPU's single-precision
    Hessians stay within the CPU single-precision mode's errors (about 1e-8 Eh/bohr²). (The FP32
    stored operator, 2-3.5x further from FP64, runs only with `precision.fp32: "on"`.)
  - `auto` (the default): with `precision.fp32: "off"` FP64 throughout, as `fp64`. Otherwise,
    where the CPU's rule runs single precision (a large exchange, at the default `response.tol`;
    la18-size molecules and up), the GPU runs the CPKS XC kernel in FP32 (its frequency errors,
    at most 5.8e-5 cm⁻¹ in the precision study, are far below the grid's).
    Its exchange takes, on cards with little FP64 throughput, **the path a cost model estimates
    cheaper** (decision D37): the stored operator in FP64 (its setup in cuBLAS's emulation under
    the card's `precision.fp64` default, emulated; on the device, or streamed from host memory as
    far as it does not fit) or FP32 through the tensor in every product. Both are far more
    accurate than the response needs: against the operator, the FP32 exchange moves Hessians by
    1e-8 to 5e-8 Eh/bohr² and frequencies by 2e-5 to 1.1e-4 cm⁻¹ with def2-SVP (la18, the
    39-atom anion, ts61), 1.8e-7 Eh/bohr² and 6.9e-4 cm⁻¹ for the anion with def2-TZVP, inside
    the CPU default's own error. Where the CPU runs double precision (a small exchange, or `response.tol` set below what
    single precision reaches, 1e-6), FP64 throughout. On devices with full FP64 throughput (A100,
    H100/H200: compute capability 8.0, 9.0) `auto` stays FP64 throughout, with the FP64 operator
    streamed beyond the device as for `fp64` (on an H200 a ts61/def2-TZVP product takes 1.4 s
    with 70 of its 167 GiB in host memory). The CPU keeps its own rule and FP64 kernel.
  - **The cost model** estimates each path's seconds from quantities known before the solve:
    the operator's setup and product flops and the tensor path's product flops (from the
    functions, auxiliary functions and occupied orbitals per spin, and the range-separated
    exchange's second operator), the operator's bytes on the device and in host memory, six
    products of the 3N perturbations over the CPKS (5.7-6.4 measured), and with IR intensities
    (`--ir`) the electric-field solve's 12-13 products of three perturbations, which the
    operator serves too while after the FP32 exchange the field solve takes its own FP64 J/K
    through the tensor. The rates (FP64 GEMMs 1.46 TFLOP/s, the emulated setup 3.8, the FP32
    tensor products 34.3, a three-column product on the operator 100 GB/s, PCIe 50 GB/s) were
    measured on the RTX PRO 6000 and are scaled to another card by its multiprocessors, clock and
    class (FP32 and FP64 units per multiprocessor): a fixed function of the device, so a card's
    choice never depends on a timing, and a B200 (FP32 only twice its FP64) always keeps the
    operator. On the RTX PRO 6000 (B3LYP, the defaults otherwise):

    | Hessian | exchange: operator / FP32 tensor (s) | whole run: before / after (s) | auto now |
    |---|---|---|---|
    | la18/def2-SVP | 0.17 / 0.12 | 2.94 / 2.83 | FP32 tensor |
    | u1an39/def2-SVP | 7.8 / 6.2 | 28.4 / 27.0 | FP32 tensor |
    | u1an39/def2-SVP with IR intensities | 9.8 / 13.3 (with the field solve's) | 32.9 / 33.2 (the same path, bitwise) | operator |
    | ts61/def2-SVP | 32.6 / 31.9 | 98.0 / 98.9 (a tie: within the runs' spread) | FP32 tensor |
    | u1an39/def2-TZVP | 29.3 / 20.3 | 87.2 / 79.1 | FP32 tensor |
    | ts61/def2-TZVP (operator streamed) | 138 / 95 | 369.5 (FP32 tensor before too) | FP32 tensor |

    `resolved.record.backend.response` records the choice (`exchange`), the two estimates
    (`estimate_s`) and the reason, e.g. `"auto: the FP32 exchange through the tensor (estimated
    5.5 s against 6.6 s on the stored exchange operator in FP64), the CPKS kernel in FP32"`. An
    IR run's field solve reuses the Hessian's J/K only when it ran on the operator; after the
    FP32 exchange it builds its own in double precision, as before.
  - The GPU solves the response in one level (no cheaper iteration J/K), so
    `resolved.record.response.two_level` is false.
- **The electric-field response of polarizabilities and IR spectra** (three perturbations,
  10-15 iterations; FP64 J/K and kernel at any `precision.response`) runs its XC kernel on
  2048-point blocks with P = F A of every component and keeps the kernel's
  perturbation-free part (collocation, Phi = F C) on the device across the iterations where it
  fits; the Hessian's CPKS kernel keeps its 512-point blocks. On an RTX PRO 6000
  (B3LYP/def2-SVP, FP64) a field iteration's kernel takes 55 ms for L-lactide (205 ms before)
  and 494 ms for the 39-atom anion (800 ms), and an IR run's field response 0.68 s (2.65 s) and
  8.3 s (21.0 s) with the Hessian's operator; at the card's `auto` (FP64 emulation, the large
  GEMMs in the Ozaki scheme) L-lactide's takes 0.42 s (2.60 s). VV10's kernel (-V functionals)
  adds to it.
- `resolved.record.backend.response` reports each response's device arithmetic and why: e.g.
  `[{"purpose": "hessian", "jk": "emulated", "exchange": "operator", "operator_gib": 13.3,
  "estimate_s": {"operator": 8.3, "tensor": 11.8}, "xc": "fp32", "reason": "auto: the stored
  exchange operator in FP64 (estimated 8.3 s against 12 s for the FP32 exchange through the
  tensor), the CPKS kernel in FP32"}, {"purpose": "polarizability", "jk": "emulated",
  "exchange": "operator", "operator_gib": 13.3, "xc": "emulated", "reason": "the hessian's
  response J/K in double precision, kept for these products"}]` for an IR run on an RTX PRO 6000
  (a standalone polarizability: its own J/K, `"reason": "polarizability: double precision"`).
  `jk` is `fp64`, `fp32` or `emulated` (the tier in `emulation`), `exchange` is `operator` (with
  `operator_gib`, and `operator_host_gib` for its part in host memory) or `tensor` (with
  `operator_note` when the operator would have paid but did not fit), `estimate_s` the cost
  model's seconds where `auto` weighed the two paths, `xc` is `fp64`, `emulated` or `fp32`.
  A Hessian's `solver` says where the CPHF/CPKS solver's own algebra ran: `gpu` (the subspace,
  its products and the residuals on the device, beside the products) or `cpu` with the reason in
  `solver_note` (its subspace would take more than a quarter of the device memory free at the
  response's setup, or outgrew the device and was solved again on the host).

## Sharing a GPU

Several linus processes can run on one card at once (validation shards, job-runner tasks):
small calculations use a few GB each and leave most of the card idle, so 4-8 of them per card
give 2-3x the throughput. Each process plans its engines'
storage against 95 % of the device memory free when it opens the device, so set
`device_memory_gib` to cap it per process (its default comes from the environment,
`LINUS_GPU_MEMORY_GIB`, in GiB), e.g. 7.45 GiB (8 GB) for 8-16 small processes per card;
`resolved.record.backend.device` reports `memory_limit_gib` (what the process plans against) and
`memory_cap_gib`. A setup that does not fit the cap falls back to the CPU with its reason in
`resolved.record.backend.fallbacks`, as any out-of-memory setup does (an implicit solvent in the
predicted mode then follows the J/K to the CPU); results do not change. The explicit XC terms of
Hessians size their accumulators to the memory left instead (more passes over the grid, the same
bits). The engines' device
reserves take at most a quarter of the limit each (the response's exchange workspace, the memory
its stored operator and the Ozaki exchange's pre-sliced planes leave free; else 8 GiB, 6 GiB and
the larger of 8 GiB and a tenth of the limit), so a calculation that fits the cap takes the same
path as without it. Hessians need the most: for the 39-atom anion of the validation (def2-SVP,
483 functions, 2738 auxiliary) the stored FP64 exchange operator takes 13.3 GiB (26.5 GiB for
its open-shell cation's two spins) and the explicit terms' exchange 12.6 GB, so a cap of 32 GB
(29.8 GiB; 44 GB, 41.0 GiB, for the open shell) keeps both on the device as without a cap. Under
less the response's exchange runs on the operator streamed from host memory or through the
tensor in FP32 (`precision.response: auto`, by its cost model; with `fp64` the operator streams)
and, under 8 GB (7.45 GiB), the explicit terms on the CPU (`hessian_jk` in the fallbacks), 2x
slower, the Hessian within 3e-8 au.

## Reproducibility

GPU results are bitwise identical run to run on one device with the same driver and libraries,
and do not depend on the number of host threads: every device kernel has one writer per
output or sums in fixed point, cuBLAS runs in a fixed configuration, and the host parts are
order-independent. They are not bitwise equal to the CPU's: they agree to round-off.

The grid is the CPU's: the device computes the partition weights of PySCF's (Becke) grids with
the CPU's operations, but per point only for the atoms whose cell function can matter. The
others, each below 2^-100 (8e-31) of the sum of the cell functions by a rigorous bound, are left
out of that sum. A weight can therefore differ from the CPU's only where those terms change the
sum's last bit (none did in the tests, up to 376 atoms); far inside every tolerance below.

Across cards, the defaults (`precision: "auto"`) are chosen per card class, so bits are shared
only within a class: FP64-rich cards (H100, H200, A100) run the exchange in native FP64 and
VV10's pair sums in FP64, FP64-poor cards (RTX, RTX PRO, L40) the exchange in FP64 emulation
and VV10 in double-float. Two cards of different classes agree to round-off (energies within
about 3e-12 Eh with the same iterations), not bitwise. Within a class, two cards give the same
bits only with the same architecture, driver and libraries (cuBLAS chooses its kernels per
architecture; the generated kernels are compiled per architecture). `precision: "fp64"`
(native FP64 everywhere) removes the class difference but not the architecture's.

## Validating

```bash
cargo build --release -p linus-cli --features gpu
python3 reference/validate_gpu.py --linus-bin target/release/linus            # the whole matrix (291 cases)
python3 reference/validate_gpu.py --cases 'la18.*' --prefix 'taskset -c 0-7'    # a subset, pinned
LINUS_GPU_REQUIRED=1 cargo test --profile test-fast -p linus --features gpu --test gpu   # the Rust tests
LINUS_GPU_REQUIRED=1 cargo test --profile test-fast -p linus-gpu --features gpu          # the engines
```

`reference/validate_gpu.py` runs each case with `--backend cpu` and `--backend gpu` of the same
binary and checks energies (1e-10 Eh, same SCF iterations), gradients (1e-9 Eh/bohr), Hessians
(1e-10 Eh/bohr², same CPHF iterations; one slow UHF case, la18 HF UKS, 2e-10), frequencies (1e-6
cm⁻¹), polarizabilities (1e-8 au), dipole derivatives (1e-8 au) and IR intensities (1e-8 of the
largest), `resolved.record.backend` (placement and fallbacks) and, for marked cases, bitwise
reproducibility of a second GPU run. Without a device the Rust tests skip unless
`LINUS_GPU_REQUIRED=1`. The GPU side pins `precision: "fp64"` and `scf_loop: "host"` unless a
case names another (the `*.emu.*` cases `precision.fp64: "emulated"`, the `*.dloop.*` cases the
device loop; the `*.default.freq` cases pin only `precision.fp64: "native"`, so that their
response keeps its default), so every card checks the same arithmetic; the `*.auto.*` cases keep
the defaults and check what `auto` chose for the card.

The `*.default.freq` cases keep the default `precision.response` (`auto`) on both backends and are
held to the default-precision bars: frequencies within 3e-3 cm⁻¹, IR intensities within 1e-3 of
the largest and Hessians within 1e-6 Eh/bohr² of the CPU's default, and, the device's own error,
frequencies within 3e-4 cm⁻¹ and Hessians within 1e-7 Eh/bohr² of the CPU's FP64 response. The
first bar covers the CPU default's two-level single-precision response, the larger error by far
(L-lactide cation ωB97X-D3BJ: 2.0e-3 cm⁻¹ from FP64 on its 629i mode; the C library's `erf`
rounding moves it by ~4e-4 cm⁻¹); the GPU's default is within 8.9e-5 cm⁻¹ of FP64 there on an
RTX PRO 6000 (its FP32 CPKS kernel) and within 1e-7 cm⁻¹ on an H200.

The dipole derivatives (and IR intensities) are stationary in the CPHF/CPKS response, as the
Hessian is: their error is quadratic in the residual, so the backends' different round-off in
the solve no longer shows up there (water, L-lactide and the 39-atom anion, B3LYP and
ωB97X-D3BJ, FP64 and emulated: dipole derivatives within 4.7e-10 au, IR intensities within
7.7e-10 of the largest; when they were linear in the residual, up to 2.9e-7 au). One check
follows from the solve rather than from the device's arithmetic: the
shared-subspace solver's final residual is round-off sensitive (it drops near-dependent
directions and converged right-hand sides: GPU and CPU residuals differ by 1-25 % at the same
iteration count), so a solve that stops right at its tolerance can take one iteration more on
either backend. The script allows that one iteration only when the side with fewer iterations
stopped within a factor 2 of the tolerance, shows each such flip in the case's line and counts
them; it never excuses the accuracy checks, and the SCF's iteration counts must match.

On the card below the whole matrix passes (75 cases, def2-SVP, `jk: "df"`): energies of water,
L-lactide (la18), a 39-atom anion (u1an39) and a 61-atom transition state (ts61) with HF,
B3LYP, ωB97X-D3BJ and CAM-B3LYP, closed and open shells, within 3.6e-12 Eh of the CPU backend
with the same SCF iterations; gradients within 4.3e-12 Eh/bohr; Hessians (B3LYP, HF,
ωB97X-D3BJ, CAM-B3LYP; response in FP64) within 4.5e-11 Eh/bohr², frequencies within 1.1e-7
cm⁻¹, with the same CPHF iterations; a polarizability within 1.1e-11 au; the device SCF loop,
an implicit solvent included; every repeated run bitwise identical.

The implicit-solvent cases (67, `--cases '*pcm*'`, THF): C-PCM, IEF-PCM and SMD at the `pyscf`
and `orca` presets, the `predicted` and `exact` SCF modes, the host and device loops, water,
L-lactide and the 39-atom anion with B3LYP and ωB97X-D3BJ, closed and open shells: energies
within 3.6e-12 Eh with the same SCF iterations, gradients within 4.4e-12 Eh/bohr, a C-PCM
Hessian (its PCM terms on the CPU) within 2.0e-11 Eh/bohr², repeated runs bitwise identical. The
device's surface integrals match linus-ints' reference to a few 1e-15 (relative), Gaussian and
point charges alike.

The -V functionals' cases (55, `--cases '*-v.*'`: water, L-lactide and the 39-atom anion with
ωB97M-V, ωB97X-V and B97M-V, closed and open shells, energies and gradients, the device loop,
C-PCM) are held to a tighter bar, 1e-11: energies within 2.7e-12 Eh with the same SCF
iterations, gradients within 5.6e-12 Eh/bohr except two open-shell L-lactide-cation gradients
whose SCF takes ~60 iterations (4e-11 Eh/bohr, as far as the GPU backend was from the CPU there
before VV10 moved to the device; bound 1e-10). The stored PySCF VV10 comparisons
(`reference/checks/pyscf_vv10_gpu.py`) pass on the GPU backend as on the CPU.

The meta-GGA frequencies (8 cases, `--cases '*.r2scan.*,*.wb97m-d3bj.*'`: water and L-lactide
with r2SCAN and ωB97M-D3BJ, closed and open shells, the Hessians' explicit XC terms on the
device) pass at the global tolerances on an H200 and on the card below: Hessians within 3.3e-11
Eh/bohr², frequencies within 1.9e-7 cm⁻¹, the same CPHF iterations, a repeated run bitwise
identical. r2SCAN's cases converge the response to 1e-9 on both backends: at the default 1e-6
its Hessians follow where the CPHF solver stops, which round-off moves (water: the same 6
iterations end at a max residual of 7.9e-7 on the CPU and 1.3e-7 on the GPU, and the Hessians
differ by 3.4e-9 Eh/bohr² whichever backend computes the explicit XC terms; at 1e-9, 3.2e-12).
Its frequencies stay within 1e-6 cm⁻¹ at the default either way (L-lactide 8.6e-7 cm⁻¹).

## Performance

Wall-clock seconds of whole calculations through the CLI (setup, SCF, derivatives) on one NVIDIA
RTX PRO 6000 Blackwell Max-Q (a workstation card with modest FP64 throughput, about 1.7
TFLOP/s) against 8 Zen 5 cores (8 threads) of the same machine; `jk: "df"`, default grids and
settings; one run each on an otherwise idle GPU (rows marked * were measured before the
per-batch XC engines: the better of two runs, ts113 one run; the emulated column with the host
SCF loop; the Hessians one run each; the `mixed` column with the exchange in native FP64, now
`precision: {"fp64": "native", "fp32": "on"}`):

| calculation | CPU | GPU | GPU, device loop | GPU, emulated exchange | GPU, `mixed` |
|---|---|---|---|---|---|
| u1an39 B3LYP/def2-TZVP energy (1034 functions, 14 iterations) | 34.8 | 21.0 (1.7x) | 18.3 (1.9x) | 10.4 (3.4x) | (as GPU) |
| u1an39 ωB97X-D3BJ/def2-TZVP energy * | 54.7 | 34.6 (1.6x) | 31.8 (1.7x) | 15.8 (3.5x) | |
| ts61 B3LYP/def2-TZVP energy (1343 functions, 13 iterations) | 87.6 | 51.7 (1.7x) | 47.3 (1.9x) | 21.7 (4.0x) | (as GPU) |
| ts113 B3LYP/def2-TZVP energy (2155 functions, 14 iterations) * | 487.6 | 295.8 (1.65x) | | 125.6 (3.9x) | |
| u1an39 B3LYP/def2-TZVP gradient | 45.3 | 27.4 (1.65x) | | | 27.0 (1.7x) |
| ts61 B3LYP/def2-TZVP gradient | 108.2 | 63.4 (1.7x) | | | 62.2 (1.7x) |
| la18 B3LYP/def2-SVP Hessian * | 8.1 | 7.1 (1.1x) | | | |
| u1an39 B3LYP/def2-SVP Hessian (default response precision) | 196.2 | 47.3 (4.1x) | | | 44.7 (4.4x) |
| u1an39 B3LYP/def2-SVP Hessian, `precision.response: "fp64"` | 321.4 | 59.4 (5.4x) | | | |
| ts61 B3LYP/def2-SVP Hessian (666 functions; default response precision) | 850.4 | 168.6 (5.0x) | | | 159.8 (5.3x) |
| ts61 B3LYP/def2-SVP Hessian, `precision.response: "fp64"` | 1584.1 | 213.8 (7.4x) | | | |
| ts61 B3LYP/def2-TZVP Hessian (1343 functions; default response precision) | | 369.5 | | | 263.1 |
| ts61 B3LYP/def2-TZVP Hessian, `precision.response: "fp64"` | | 602.4 | | 385.3 | |
| u1an39 B3LYP/def2-TZVP Hessian (default response precision) | | 147.7 | | | |
| u1an39 B3LYP/def2-TZVP Hessian, `precision.response: "fp64"` | | 173.1 | | 114.8 | |

- On this card the SCF's J/K and XC run at the speed of roughly 12-16 Zen 5 cores in FP64; the
  XC terms alone (energy, potential, gradient) at about 3-3.4x 8 cores, and with `mixed` the
  CPKS XC kernel at about 20x. The exchange GEMMs are FP64-bound, so `precision: "emulated"`
  (FP64 emulation on the INT8 tensor cores) takes most of the J/K time away: a ts61
  B3LYP/def2-TZVP iteration's J/K drops from 2.6 s to 0.35 s, a ts113 one from 15 s to 2.8 s,
  and XC becomes the larger part. Before this release the emulated column used cuBLAS's
  emulation (u1an39 19.3 s, ts61 40.2 s, ts113 162.1 s). Cards with full FP64 rates
  (data-center GPUs) shift all of this further towards the device.
- A Hessian on the GPU (u1an39 B3LYP/def2-SVP at the default precision, 47.3 s): the CPHF/CPKS
  solve 21.3 s (its exchange on the stored operator in FP64, the CPKS kernel in FP32 where the
  CPU runs single precision), the explicit two-electron terms 10.3 s (against 52.3 s on 8
  cores), the SCF 7.1 s, the explicit XC terms 4.2 s (against 16.1 s). At ts61: the response
  85 s, the two-electron terms 35.7 s (218.8 s), the SCF 19.8 s, the XC terms 16.5 s (60.1 s).
  Both explicit parts are FP64-bound (DGEMMs; the XC terms' fused kernel at 96 % of the FP64
  pipe): with `precision: "emulated"` the two-electron terms' largest GEMM runs on the tensor
  cores, with `mixed` the XC terms' products in FP32 (u1an39 4.2 -> 2.0 s, ts61 16.5 -> 7.1 s).
  With `precision.response: "fp64"` the response takes 33.5 s at u1an39, 130 s at ts61 and
  349 s at ts61/TZVP (2304 s before its stored operator streamed beyond the card; the TZVP
  rows were measured on an identical card, the response's solver already in
  BLAS-3).
- Meta-GGA Hessians (since this release their explicit XC terms run on the device too; before,
  on 8 CPU cores beside the GPU; u1an39/def2-SVP frequencies at the default precision, one run
  each): r2SCAN 51.8 -> 30.5 s, the explicit XC terms 34.2 -> 12.7 s; the radical's r2SCAN UKS
  68.8 s (its XC terms 24.8 s); ωB97M-D3BJ 104.4 -> 82.8 s. The tau term triples the fused
  potential kernel's work (three more products per point; at 97 % of the FP64 pipe, as the
  GGA's), so a meta-GGA's XC terms take about 3x a GGA's (B3LYP 4.2 s). With `mixed`: r2SCAN
  21.3 s (XC terms 3.7 s), UKS 49.6 s (6.8 s), ωB97M-D3BJ 53.7 s; frequencies within 5.7e-4
  cm⁻¹ (r2SCAN, closed and open shells) and 1.6e-3 cm⁻¹ (ωB97M-D3BJ) of FP64, IR intensities
  within 3.3e-6 of the largest, the CPHF's own stopping noise included (L-lactide r2SCAN with
  the response converged further: 1.7e-4 cm⁻¹, IR 2.8e-6); r2SCAN's frequencies move by up to
  several cm⁻¹ between the default grid and a converged one.
- Small molecules (water, la18 energies) do not gain: kernel launches and transfers dominate.

**-V functionals** (ωB97M-V; VV10 on the device since this release, before on 8 CPU cores beside
the GPU; the same settings, medians of 2-3 runs after a warm-up, the two builds interleaved):

| calculation | GPU, VV10 on the CPU | GPU | of which VV10 (before -> now) |
|---|---|---|---|
| u1an39 ωB97M-V/def2-SVP energy (483 functions, 14 iterations) | 15.5 | 10.7 (1.44x) | 5.96 -> 1.13 |
| u1an39 ωB97M-V/def2-SVP gradient | 19.6 | 13.9 (1.41x) | 7.31 -> 1.58 |
| ts61 ωB97M-V/def2-TZVP energy (1343 functions, 14 iterations) | 120.1 | 106.6 (1.13x) | 17.4 -> 3.6 |
| ts61 ωB97M-V/def2-TZVP gradient | 144.2 | 128.1 (1.13x) | 21.6 -> 5.1 |

- A VV10 evaluation (the NLC grid's densities, the pair sums of ~113k active points, the
  potential matrix) takes 0.075 s at u1an39 against 0.40 s on 8 cores; the pair kernel in
  double-float 66 ms (FP64 430 ms, FP32 13 ms). At ts61/TZVP the FP64 exchange is most of the
  remaining run; `precision: "emulated"` shortens it.

**AFH's production job**, single-point energy (no gradient): a 113-atom transition state
(ts113), ωB97X-D3BJ/def2-TZVP (2155 functions), C-PCM(THF) on 11216 surface points, the default
grid, `predicted` surface terms; linus's own total in seconds, one run each (the GPU runs alone
on the card; CPU threads on one CCD, 64 threads on all eight):

| | total | SCF iterations | of which the PCM |
|---|---|---|---|
| CPU, 8 cores, RI-JK (`jk: "df"`) | 936.7 | 13 | 44.1 |
| CPU, 8 cores, RIJCOSX (`jk: "rij_snk"`, the CPU's production setting) | 394.7 | 13 | 43.7 |
| CPU, 64 cores, RI-JK | 379.3 | 13 | 12.0 |
| GPU, host loop | 502.8 | 13 | 9.4 |
| GPU, device loop | 487.3 | 13 | 9.2 |
| GPU, device loop, `precision: "emulated"` | **118.0** | 13 | 8.4 |
| GPU, device loop, `precision: "emulated"`, before the device's density-fitting setup and XC engines | 188.6 | 13 | 8.4 |
| GPU, device loop, gas phase (no solvent) | 508.7 | 14 | - |

- The implicit solvent costs the GPU 0.64 s per SCF iteration on 11216 points (3.2 s on 8
  cores, 0.8 s on 64); the solvated SCF needs one iteration fewer than the gas phase here, as
  on the CPU.
- On this card the run is 1.9x 8 cores with the same method (RI-JK), but the CPU's RIJCOSX
  (whose exchange approximation is about 1 mEh off RI-JK here) and 64 cores are faster: the
  exchange is FP64-bound on a workstation card.
- On the GPU the run is the exchange: J/K 369.7 s (13 builds of a range-separated hybrid's two
  tensors, B^ω streamed from page-locked host memory at this size), the density-fitting setup
  70.3 s, XC 32.6 s, the PCM 9.4 s, the SCF loop's own work 11.1 s (host loop; 2.3 s on the
  device loop), the rest 9.7 s.
- With `precision: "emulated"` the exchange runs on the INT8 tensor cores: J/K 70.9 s instead
  of 368.5 s, the whole job 118.0 s, 4.1x the FP64 GPU run, 3.3x the CPU's RIJCOSX on 8 cores
  and 3.2x RI-JK on 64 cores, with the same energy (1.8e-12 Eh from the FP64 GPU run) and
  iterations. The tensors stay where FP64 keeps them (B on the device, B^ω streamed); the
  sliced copy of the tensor (168 GiB) does not fit, so each build slices the blocks it reads
  (`resolved.record.backend.emulation` says so). The density-fitting setup takes 16.8 s: its
  integrals run on the device and its fit (the triangular solve with the metric's factor,
  most of the setup) on the INT8 tensor cores at FP64 quality (70.2 s before, when the fit ran
  in FP64 and B^ω's page-locked memory took 8 s to set up); XC 14.7 s (31.5 s before the
  per-batch XC engines). The exchange's builds are now most of the run.
