# Session state vs a new calculation per gradient (SESSBENCH, 2026-11-12)

**Question (Colton):** how much does computing gradients (or similar repeated calculations) as new
calculations from start to finish cost against sharing state across calculations within a session? Does it
matter, or is it only conceptually interesting?

**Short answer:** it matters, and one piece of state does most of the work. Replaying real optimization
trajectories, a new calculation per gradient takes 1.1-1.4× as long as today's session on 8 CPU cores
(water 1.9×) and 1.6-2.1× on the GPU (water 3.8×). Three kinds of state account for it, measured separately:

1. **The previous geometry's orbitals (the warm start):** 14-16 SCF iterations become 10-11, which saves
   15-25 % of every step (C → D below). This is the largest term everywhere except water.
2. **The probes a calculator runs only at its first geometry:** molecules with a small HOMO-LUMO gap (the
   anion u1an39) or open shells pay 2 orbital-Hessian products on every new calculator: +14-16 % per step for
   u1an39 (B → C).
3. **The process:** on the CPU a new process costs only 15-40 ms per gradient (0.3 s on u1an39, mostly page
   faults of a fresh heap), but on the GPU 0.3-0.8 s (CUDA context, libraries, kernel modules, first-call
   setup), which doubles an 18-atom gradient (A → B).

Re-creating the calculator object itself in a running process costs milliseconds, and Python adds 0.2 ms per
call. What a warm session still repeats every step is geometry-dependent setup (DF tensor and metric, grid, pair
data): 5-19 % of a step, of which a deeper session could save only 1-5 % of the step.

What follows for workflows: keep one session per molecule and model, with a lineage per chain of geometries
(linus.Session; qcopt 0.3 opens them itself), and avoid a new process per gradient on the GPU. A worker process
costs nothing measurable on top of the session ([the rerun below](#rerun-through-the-session-api-2026-11-13)).

## Rerun through the session API (2026-11-13)

The same trajectories replayed through `linus.Session` (SESSION MR3b, 552fb995, built with the gpu features;
qcdata data-only 370a441, qccompute execution-protocol eca1da2), with the same conditions as below (CCD0's 8
physical cores under the bench lock, the exclusive GPU lock, memory caps per system), into
[results-session-api/](results-session-api/) (`summary.json`, `tables.md`, raw steps). The modes:

- **A**: a new process per gradient (the CLI), as before;
- **B**: a new `linus.Session` per gradient in one Python process (opened, `compute(x)`, closed);
- **D**: one `Session`, `compute(x)` in its default lineage (the warm start);
- **W** (new): one session in a **worker process**, qccompute's serve loop (the draft of SESSION MR4, `python
  serve.py linus`; it landed in qccompute as `python -m qccompute.serve linus`, which mode W runs now): the
  client writes `{"op": "compute", "session", "positions"}` per step on a pipe and reads the frame back (JSON
  lines); a step runs from before the write to after the reply's parse.

Two repeats of everything (u1an39-svp on the CPU and u1an39-tzvp: one), modes in rotating order:

| system | platform | A (s) | B (s) | D (s) | W (s) | A / D | B / D | W / D | (A − W) per step (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| water-svp | cpu | 0.3 | 0.2 | 0.2 | 0.2 | 1.83 | 1.22 | 1.00 | 0.026 |
| water-svp | gpu | 2.5 | 0.7 | 0.7 | 0.7 | 3.79 | 1.07 | 1.00 | 0.364 |
| la18-svp | cpu | 25.6 | 25.0 | 20.0 | 19.9 | 1.28 | 1.25 | 1.00 | 0.281 |
| la18-svp | gpu | 21.1 | 13.0 | 10.6 | 10.6 | 1.99 | 1.23 | 1.00 | 0.523 |
| u1an39-svp | cpu | 353.5 | 358.7 | 247.3 | – | 1.43 | 1.45 | – | – |
| u1an39-svp | gpu | 118.3 | 103.8 | 73.8 | 74.6 | 1.60 | 1.41 | 1.01 | 1.458 |
| u1an39-tzvp | gpu | 317.4 | 301.3 | 209.9 | 210.1 | 1.51 | 1.44 | 1.00 | 3.579 |
| la18cat-svp | cpu | 66.9 | 66.3 | 62.7 | 62.6 | 1.07 | 1.06 | 1.00 | 0.181 |

- **The session API keeps the study's conclusions.** A new session per gradient (B) costs 1.06-1.45× the
  session (D), as a new calculator did (1.04-1.43× then); the warm start still turns 14-16 SCF iterations into
  10-11; mode D's totals are 0-9 % below 2026-11-12's (the code moved since: numerics epoch 7, the
  lineages). u1an39 on the CPU ran once; its A < B by 0.17 s per step is within that case's noise.
- **The worker costs nothing measurable.** W / D is 1.00 everywhere (1.01 for u1an39-svp on the GPU, within the
  repeats' spread: 74.9 and 74.2 s against 73.2 and 74.3 s), and W's energies, gradients and SCF iterations are
  bitwise D's at every step (positions and results cross the pipe as shortest round-trip decimals).
- **Against a process per gradient on the GPU (A), the worker is 1.5-3.8× faster**: 0.36 s per step on water,
  0.52 s on la18, 1.5 s on u1an39-svp and 3.6 s on u1an39-tzvp, of which 0.36-0.54 s is the new process's CUDA
  context, libraries and module loads and the rest the warm start and the probes the lineage runs only once.

### In process against the worker, per call ([pipe.py](pipe.py), `results-session-api/pipe_water_cpu.json`)

SESSION2 measured 1.7 ms per call more through the worker than in process (water B3LYP/STO-3G gradients, 32.5
against 34.2 ms) and asked where it goes. `pipe.py` stamps both sides of the pipe with one clock
(CLOCK_MONOTONIC) over 300 calls, water B3LYP/STO-3G gradients 0.002 bohr apart, 2 threads, CCD0 under the bench
lock, client and worker pinned to the same cores (medians, microseconds):

| in process | µs | through the worker | µs |
|---|---:|---|---:|
| `Session.compute` | 36228 | round trip | 36475 |
| of it: Python before the binding (request, peak-memory probe) | 53 | client write + flush | 3 |
| the binding's `compute` | 36024 | pipe to the worker (wake-up included) | 5 |
| of it: beyond linus's own `timings.total` (report to dict) | 174 | worker `json.loads` + Request validation | 21 |
| Python after the binding (frame) | 144 | `Session.compute` in the worker | 36154 |
| linus's `timings.total` | 35845 | frame `model_dump` | 67 |
| | | `json.dumps` (7.8 kB reply) | 91 |
| | | worker write + flush, pipe to the client | 12 |
| | | client `json.loads` | 71 |

The worker adds **0.25 ms per call**, and 0.23 ms of it is the frame's JSON round trip (dump, encode, parse);
the pipe and the process wake-ups are 20 µs, and the computation itself takes as long in the worker as in
process (a sleep between in-process calls as long as the worker's gap between computes, 0.31 ms, changes
nothing either: 36.30 against 36.23 ms). The 1.7 ms does not reproduce on a clean CCD; it was most likely
contention from other jobs then. The SCF history is about 13 % of the frame, so making it opt-in would save
some 30 µs per call (SESSION2 measured 1 kB of history in a 6.6 kB frame on this case): not worth changing the
default. In process, the package adds 0.4 ms per call to linus's
total on a water def2-SVP gradient (`overhead.py`: the binding's report conversion 0.13 ms, the frame 0.27
ms), and opening a session costs 3.8 ms on the CPU (closing it 0.1 ms).

## What a `linus.Calculator` kept (code reading, main a6a9de8; the 2026-11-12 study)

| Kept for the calculator's life (built once) | Rebuilt for every SCF (every geometry) |
|---|---|
| settings, normalized and checked; method info and dispersion parameters | shell-pair data (`pairs`), one-electron integrals and ECP matrices (`one_electron`, `ecp`) |
| the basis set definitions (JSON read once) and the basis set placed on the atoms (`set_positions` moves its centers) | the auxiliary basis placed on the atoms, the DF tensor and Coulomb metric (`jk_setup`; GPU: device integrals, metric, emulation planes, uploads) |
| the rayon thread pool (`threads`) | the molecular grid and its weights (`grid`); GPU: the device XC setup (`xc_setup`) |
| the GPU device: context, stream-ordered memory pool, cuBLAS/cuSOLVER handles, the device policy and memory plan (`BackendState::open`) | the DF memory budget and reservations (read from the cgroup and `MemAvailable`) |
| the in-memory kernel-module map of that device (`KernelCache`) | the SAD guess (unless the warm start applies), the device SCF loop's buffers (`scf_setup`) |
| the last converged orbitals: the next SCF starts from them, Löwdin-projected at the new positions (`set_positions`); `reset_guess()` forgets them | the gradient's setups: derivative pair data (`grad_pairs`), GPU derivative records and metric (`grad_2e_setup`), GPU XC-gradient layout (`grad_xc_setup`); on the CPU the gradient builds its own grid again |
| probe state: the stability probes (UKS, and the closed-shell probe below a 0.15 Eh gap) run at a calculator's first geometry only | |

Within one call the GPU hands the SCF's DF data and XC integrator to the gradient at the same geometry
(crates/linus/src/driver/handoff.rs). Per **process**, not per calculator: integral tables (Boys, Hermite,
3c recursion tables; `OnceLock`s in linus-ints), the SAD atomic densities (`scf/guess.rs` cache), libxc's
initialization, the CUDA driver and libraries. A new process (the CLI) pays all of these on every call; a new
`Calculator` in a warm process pays only the per-calculator column.

## The experiment

**Trajectories** (`trajectory.py`, `results/trajectories/`). One real geometry optimization per system with a
persistent Calculator (warm start), generated on the GPU: scipy 1.18 BFGS on linus energies and gradients, initial
Hessian 0.72 Eh/bohr² × identity (ASE BFGS's 70 eV/Å²; scipy's default unit step threw water 3 Eh uphill on its
first trial), from the input structure displaced by Gaussian noise of 0.1 bohr RMS (fixed seed, no net
translation), until the gradient RMS is below 3e-4 Eh/bohr or after 30 gradients. Every gradient evaluated
(line-search trials included) is a step. Every mode replays exactly these geometries.

| system | molecule, method / basis | atoms | basis functions | charge, mult. | steps | stop | E(first) - E(last) (Eh) |
|---|---|---:|---:|---|---:|---|---:|
| water-svp | water, B3LYP/def2-SVP | 3 | 24 | 0, 1 | 5 | converged, final gradient RMS 2.8e-04 | 0.0096 |
| la18-svp | la18 (L-lactide), B3LYP/def2-SVP | 18 | 180 | 0, 1 | 20 | converged, final gradient RMS 3.0e-04 | 0.1142 |
| u1an39-svp | u1an39 (anion), B3LYP/def2-SVP | 39 | 483 | -1, 1 | 30 | max steps, final gradient RMS 4.1e-04 | 0.2039 |
| u1an39-tzvp | u1an39 (anion), B3LYP/def2-TZVP (GPU only) | 39 | 1034 | -1, 1 | 30 | max steps, final gradient RMS 3.6e-04 | 0.2424 |
| la18cat-svp | la18⁺ (radical cation, UKS), B3LYP/def2-SVP (CPU only) | 18 | 180 | 1, 2 | 24 | converged, final gradient RMS 2.7e-04 | 0.1129 |

The input structures are not B3LYP minima (la18 and u1an39 relax by 0.11 and 0.20 Eh), so these are early,
large-step optimizations: the warm start's least favourable regime.

**Modes** (`replay.py`), each replaying the same trajectory:

- **A** – a new process per gradient: `linus step.xyz --method b3lyp --basis B --jk df --grad --threads 8
  --backend cpu|auto --charge Q --mult M --json` (the xyz files written before timing). Each step pays process
  start, basis-file parsing, the per-process tables and caches, and on the GPU the CUDA context, cuBLAS/cuSOLVER
  handles and the kernel modules' load from the on-disk kernel cache (warm: the trajectories filled it).
- **B** – a new `linus.Calculator` per gradient in one Python process: construct, `gradient()`, drop. The process,
  its tables and the CUDA driver are warm; every per-calculator setup is paid again.
- **C** – one Calculator; before each step `reset_guess()` then `set_positions(x)`: the session without the warm
  start (SAD every step).
- **D** – one Calculator; `set_positions(x)` before each step: today's session (warm start from the last orbitals,
  Löwdin-projected).

A step's wall time runs from before the CLI's start / the construction / `set_positions` to the CLI's exit / the
`CalcResult` returned (B: and the Calculator dropped). The first step of B, C and D includes the first
construction in the process (on the GPU: the CUDA context and the first module loads). Python's own start and
`import linus` (~0.3 s) are not in any mode's total.

**Settings:** B3LYP (VWN-RPA), `jk df` (def2-universal-jkfit), grid `auto` (PySCF level 3), the gradient's
default SCF convergence (|ΔE| < 1e-9 Eh, max |F_ai| < 1e-7), 8 threads; CPU: `backend cpu`; GPU: `backend auto`
(the device SCF loop, `precision.fp64` auto = FP64 emulation on this card).

**Conditions:** linus at main a6a9de8 (`cargo build --release --workspace` with the gpu features; the CPU runs
use the same build with `backend cpu`), AMD Threadripper PRO 9985WX, 8 threads pinned to one CCD's 8 physical
cores (SMT siblings idle) under the shared bench lock; NVIDIA RTX PRO 6000 Blackwell Max-Q (driver 610.57) under
the node's exclusive GPU lock (turnstile + lock, one replay at a time), its host side on the same 8 cores under
the bench lock; a GPU system's four modes of one repeat run back to back under one hold of the locks (a few
minutes; the TZVP case one mode per hold, about 4 min each), so that a gate's queue of short GPU runs is drained
once per system rather than once per mode. Every job in its own systemd scope with a memory cap per system
(water 16G, la18 24G, u1an39 SVP 32G, TZVP 48G); with `memory_gib` unset linus sizes its DF budget from that cap
(0.75 of it), the same for every mode. Repeats: two of every (system, platform, mode) (TZVP: one, see the
caveats), modes in rotating order within a repeat (repeat r starts at mode r mod 4), repeat 1 run after repeat
0. Each step's wall is the median over the repeats; totals are sums of those medians (per-repeat totals in the
table). The node ran other packages' gates throughout (CCDs 1-7, the GPU between our jobs); `results/jobs.jsonl`
records the busy threads per CCD before each job.

## Results

All numbers are measured (medians of the repeats; `results/summary.json`, `results/tables.md` from
`summarize.py`), except where a row says "estimate".

### Whole trajectories

| system | platform | mode | steps | first step (s) | median later step (s) | total (s) | totals per repeat (s) | median SCF it. | total / D |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| water-svp | cpu | A | 5 | 0.07 | 0.066 | 0.3 | 0.3, 0.3 | 10.0 | 1.87 |
| water-svp | cpu | B | 5 | 0.06 | 0.037 | 0.2 | 0.2, 0.2 | 10.0 | 1.18 |
| water-svp | cpu | C | 5 | 0.06 | 0.033 | 0.2 | 0.2, 0.2 | 10.0 | 1.10 |
| water-svp | cpu | D | 5 | 0.06 | 0.028 | 0.2 | 0.2, 0.2 | 8.0 | 1.00 |
| water-svp | gpu | A | 5 | 0.52 | 0.508 | 2.5 | 2.5, 2.5 | 10.0 | 3.81 |
| water-svp | gpu | B | 5 | 0.45 | 0.090 | 0.8 | 0.8, 0.8 | 10.0 | 1.22 |
| water-svp | gpu | C | 5 | 0.34 | 0.080 | 0.7 | 0.7, 0.7 | 10.0 | 1.00 |
| water-svp | gpu | D | 5 | 0.35 | 0.079 | 0.7 | 0.7, 0.7 | 8.0 | 1.00 |
| la18-svp | cpu | A | 20 | 1.43 | 1.318 | 26.7 | 26.5, 26.9 | 14.0 | 1.24 |
| la18-svp | cpu | B | 20 | 1.38 | 1.284 | 26.0 | 25.8, 26.1 | 14.0 | 1.21 |
| la18-svp | cpu | C | 20 | 1.40 | 1.292 | 26.2 | 25.9, 26.5 | 14.0 | 1.22 |
| la18-svp | cpu | D | 20 | 1.38 | 1.030 | 21.5 | 21.9, 21.1 | 10.0 | 1.00 |
| la18-svp | gpu | A | 20 | 1.20 | 1.133 | 22.8 | 22.9, 22.8 | 14.0 | 2.12 |
| la18-svp | gpu | B | 20 | 1.02 | 0.639 | 13.2 | 13.2, 13.3 | 14.0 | 1.23 |
| la18-svp | gpu | C | 20 | 0.95 | 0.620 | 12.8 | 12.8, 12.8 | 14.0 | 1.19 |
| la18-svp | gpu | D | 20 | 0.96 | 0.500 | 10.8 | 10.7, 10.8 | 10.0 | 1.00 |
| u1an39-svp | cpu | A | 30 | 13.04 | 12.943 | 388.8 | 383.3, 394.4 | 15.0 | 1.43 |
| u1an39-svp | cpu | B | 30 | 12.84 | 12.656 | 380.0 | 377.9, 382.2 | 15.0 | 1.40 |
| u1an39-svp | cpu | C | 30 | 13.26 | 11.363 | 343.5 | 335.2, 351.8 | 15.0 | 1.27 |
| u1an39-svp | cpu | D | 30 | 12.98 | 8.892 | 271.5 | 268.3, 274.6 | 11.0 | 1.00 |
| u1an39-svp | gpu | A | 30 | 4.11 | 4.023 | 120.9 | 119.7, 122.1 | 15.0 | 1.60 |
| u1an39-svp | gpu | B | 30 | 3.89 | 3.531 | 108.0 | 104.7, 111.3 | 15.0 | 1.43 |
| u1an39-svp | gpu | C | 30 | 3.83 | 2.974 | 90.1 | 89.7, 90.6 | 15.0 | 1.19 |
| u1an39-svp | gpu | D | 30 | 3.83 | 2.487 | 75.5 | 75.2, 75.8 | 11.0 | 1.00 |
| u1an39-tzvp | gpu | A | 30 | 10.97 | 10.954 | 325.6 | 325.6 | 16.0 | 1.55 |
| u1an39-tzvp | gpu | B | 30 | 10.89 | 10.169 | 302.0 | 302.0 | 16.0 | 1.43 |
| u1an39-tzvp | gpu | C | 30 | 10.65 | 8.617 | 257.7 | 257.7 | 16.0 | 1.22 |
| u1an39-tzvp | gpu | D | 30 | 10.74 | 6.652 | 210.7 | 210.7 | 10.0 | 1.00 |
| la18cat-svp | cpu | A | 24 | 3.57 | 2.806 | 71.1 | 69.1, 73.1 | 20.0 | 1.09 |
| la18cat-svp | cpu | B | 24 | 3.35 | 2.735 | 68.1 | 68.6, 67.6 | 20.0 | 1.04 |
| la18cat-svp | cpu | C | 24 | 3.48 | 2.698 | 67.2 | 66.2, 68.3 | 20.0 | 1.03 |
| la18cat-svp | cpu | D | 24 | 3.50 | 1.883 | 65.3 | 63.2, 67.4 | 14.0 | 1.00 |

### Where the differences come from

| system | platform | A / D | B / D | C / D | (A - B) per step (s) | (B - C) per step (s) | (C - D) per step (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| water-svp | cpu | 1.87 | 1.18 | 1.10 | 0.024 | 0.003 | 0.004 |
| water-svp | gpu | 3.81 | 1.22 | 1.00 | 0.345 | 0.029 | -0.000 |
| la18-svp | cpu | 1.24 | 1.21 | 1.22 | 0.036 | -0.011 | 0.235 |
| la18-svp | gpu | 2.12 | 1.23 | 1.19 | 0.480 | 0.021 | 0.103 |
| u1an39-svp | cpu | 1.43 | 1.40 | 1.27 | 0.293 | 1.218 | 2.401 |
| u1an39-svp | gpu | 1.60 | 1.43 | 1.19 | 0.430 | 0.596 | 0.489 |
| u1an39-tzvp | gpu | 1.55 | 1.43 | 1.22 | 0.788 | 1.476 | 1.566 |
| la18cat-svp | cpu | 1.09 | 1.04 | 1.03 | 0.125 | 0.035 | 0.082 |

The three differences between neighbouring modes isolate one kind of state each:

- **C → D, the warm start** (the last geometry's orbitals, Löwdin-projected): C/D = 1.19-1.27 on la18 and
  u1an39, CPU and GPU (0.10-0.24 s per la18 step, 0.5-2.4 s per u1an39 step), 1.03 over the la18⁺ trajectory
  (the warm start followed another SCF solution for two steps, below), 1.01-1.10 on water. It is the largest of
  the three on every system above water: 14-16 SCF iterations from SAD become 10-11 (open shell: 20 → 14), and
  the iterations are 50-75 % of a step.
- **B → C, a new calculator in a warm process** (settings, basis set, thread pool, device handles, 72 kernel
  modules, and the probes a calculator runs only at its first geometry): the construction costs 4-5 ms and the
  drop 0.2 ms on the CPU, 7-106 ms on the GPU (freeing and trimming the device pool; 106 ms for TZVP), so on
  la18 and water B and C agree to 1-3 %. The difference is the **probes**: u1an39's HOMO-LUMO gap (0.124 Eh) is
  below the closed-shell probe's 0.15 Eh gate, so every new calculator runs 2 orbital-Hessian products, 1.5 s on
  the CPU, 0.47 s on the GPU (SVP) and 1.36 s (TZVP) per step (+14-16 %); la18⁺'s UKS stability probe costs 0.19
  s (CPU, 7 % of its step) per new calculator, within that case's repeat-to-repeat noise. A session (or a
  calculator told `set_guess(..., continuation=True)`) runs them at the first geometry only, unless the gap
  drops.
- **A → B, a new process**: on the CPU 25-40 ms per gradient on water and la18, almost all inside linus's own
  first-call costs (integral tables, the SAD atomic densities, first-touch page faults; the process start and
  JSON output are 5-15 ms): +3 % on la18, +90 % on water; 0.3 s (+2 %) on u1an39, where a fresh heap takes 0.38
  M more page faults per step (2.1 M against 1.7 M). On the GPU **0.3-0.8 s per gradient**: the CUDA context,
  cuBLAS/cuSOLVER and their libraries, the kernel modules' first load in the process, and first-call device
  setup (water's `jk_setup` is 0.10 s in a new process against 0.03 s in a warm one). That doubles a la18
  gradient on the GPU and is still +12 % for u1an39 SVP and +8 % for TZVP.

### Per-step breakdown

Per later step (median of steps 2..N), seconds: wall = outside linus's timers + linus total; total = SCF setup +
SCF loop + gradient.

| system | platform | mode | wall | outside linus | SCF setup | SCF iterations | gradient | setup (SCF + gradient) | of it: ints + pairs | DF tensor + metric | grid + XC setup | SAD guess | device loop setup | probes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| water-svp | cpu | A | 0.066 | 0.006 | 0.032 | 0.019 | 0.009 | 0.033 | 0.001 | 0.019 | 0.002 | 0.007 | 0.000 | 0.000 |
| water-svp | cpu | B | 0.037 | 0.004 | 0.005 | 0.019 | 0.009 | 0.006 | 0.000 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 |
| water-svp | cpu | C | 0.033 | 0.000 | 0.005 | 0.019 | 0.009 | 0.006 | 0.000 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 |
| water-svp | cpu | D | 0.028 | 0.000 | 0.005 | 0.014 | 0.009 | 0.006 | 0.000 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 |
| water-svp | gpu | A | 0.508 | 0.309 | 0.128 | 0.042 | 0.029 | 0.137 | 0.001 | 0.100 | 0.007 | 0.007 | 0.018 | 0.000 |
| water-svp | gpu | B | 0.090 | 0.007 | 0.022 | 0.030 | 0.030 | 0.037 | 0.000 | 0.027 | 0.005 | 0.000 | 0.001 | 0.000 |
| water-svp | gpu | C | 0.080 | 0.000 | 0.018 | 0.030 | 0.031 | 0.034 | 0.000 | 0.025 | 0.004 | 0.000 | 0.001 | 0.000 |
| water-svp | gpu | D | 0.079 | 0.000 | 0.018 | 0.023 | 0.037 | 0.040 | 0.000 | 0.031 | 0.004 | 0.000 | 0.001 | 0.000 |
| la18-svp | cpu | A | 1.318 | 0.014 | 0.103 | 0.933 | 0.267 | 0.131 | 0.007 | 0.063 | 0.047 | 0.012 | 0.000 | 0.000 |
| la18-svp | cpu | B | 1.284 | 0.005 | 0.068 | 0.941 | 0.266 | 0.096 | 0.006 | 0.043 | 0.047 | 0.000 | 0.000 | 0.000 |
| la18-svp | cpu | C | 1.292 | 0.000 | 0.069 | 0.949 | 0.266 | 0.096 | 0.006 | 0.043 | 0.046 | 0.000 | 0.000 | 0.000 |
| la18-svp | cpu | D | 1.030 | 0.001 | 0.071 | 0.679 | 0.272 | 0.098 | 0.007 | 0.044 | 0.046 | 0.000 | 0.000 | 0.000 |
| la18-svp | gpu | A | 1.133 | 0.348 | 0.196 | 0.443 | 0.141 | 0.221 | 0.009 | 0.136 | 0.031 | 0.013 | 0.027 | 0.000 |
| la18-svp | gpu | B | 0.639 | 0.013 | 0.073 | 0.433 | 0.120 | 0.097 | 0.007 | 0.048 | 0.032 | 0.000 | 0.005 | 0.000 |
| la18-svp | gpu | C | 0.620 | 0.000 | 0.069 | 0.432 | 0.120 | 0.092 | 0.007 | 0.045 | 0.030 | 0.000 | 0.005 | 0.000 |
| la18-svp | gpu | D | 0.500 | 0.002 | 0.070 | 0.308 | 0.120 | 0.094 | 0.008 | 0.047 | 0.030 | 0.000 | 0.005 | 0.000 |
| u1an39-svp | cpu | A | 12.943 | 0.038 | 1.074 | 9.195 | 2.640 | 1.293 | 0.035 | 0.844 | 0.400 | 0.026 | 0.000 | 1.625 |
| u1an39-svp | cpu | B | 12.656 | 0.007 | 1.014 | 9.021 | 2.605 | 1.229 | 0.030 | 0.817 | 0.400 | 0.000 | 0.000 | 1.543 |
| u1an39-svp | cpu | C | 11.363 | 0.001 | 1.040 | 7.672 | 2.660 | 1.261 | 0.030 | 0.840 | 0.405 | 0.000 | 0.000 | 0.000 |
| u1an39-svp | cpu | D | 8.892 | 0.006 | 1.035 | 5.287 | 2.638 | 1.251 | 0.035 | 0.832 | 0.401 | 0.000 | 0.000 | 0.000 |
| u1an39-svp | gpu | A | 4.023 | 0.363 | 0.495 | 2.340 | 0.822 | 0.561 | 0.043 | 0.320 | 0.135 | 0.027 | 0.029 | 0.500 |
| u1an39-svp | gpu | B | 3.531 | 0.038 | 0.365 | 2.311 | 0.816 | 0.428 | 0.033 | 0.230 | 0.141 | 0.000 | 0.013 | 0.482 |
| u1an39-svp | gpu | C | 2.974 | 0.001 | 0.356 | 1.800 | 0.817 | 0.417 | 0.033 | 0.223 | 0.139 | 0.000 | 0.012 | 0.000 |
| u1an39-svp | gpu | D | 2.487 | 0.006 | 0.363 | 1.310 | 0.815 | 0.423 | 0.038 | 0.227 | 0.138 | 0.000 | 0.012 | 0.000 |
| u1an39-tzvp | gpu | A | 10.954 | 0.501 | 1.391 | 7.004 | 2.062 | 1.642 | 0.238 | 0.894 | 0.174 | 0.268 | 0.047 | 1.368 |
| u1an39-tzvp | gpu | B | 10.169 | 0.115 | 0.990 | 6.992 | 2.066 | 1.198 | 0.175 | 0.798 | 0.173 | 0.001 | 0.032 | 1.361 |
| u1an39-tzvp | gpu | C | 8.617 | 0.003 | 0.971 | 5.580 | 2.065 | 1.180 | 0.177 | 0.778 | 0.171 | 0.001 | 0.032 | 0.000 |
| u1an39-tzvp | gpu | D | 6.652 | 0.020 | 1.017 | 3.510 | 2.065 | 1.228 | 0.198 | 0.806 | 0.172 | 0.000 | 0.032 | 0.000 |
| la18cat-svp | cpu | A | 2.806 | 0.017 | 0.105 | 2.364 | 0.326 | 0.133 | 0.007 | 0.064 | 0.048 | 0.012 | 0.000 | 0.189 |
| la18cat-svp | cpu | B | 2.735 | 0.005 | 0.068 | 2.345 | 0.315 | 0.095 | 0.006 | 0.042 | 0.046 | 0.000 | 0.000 | 0.185 |
| la18cat-svp | cpu | C | 2.698 | 0.001 | 0.070 | 2.297 | 0.330 | 0.098 | 0.006 | 0.044 | 0.047 | 0.000 | 0.000 | 0.000 |
| la18cat-svp | cpu | D | 1.883 | 0.002 | 0.071 | 1.490 | 0.321 | 0.098 | 0.007 | 0.045 | 0.047 | 0.000 | 0.000 | 0.000 |

### What mode D still pays per step

| system | platform | step wall | SCF iterations + gradient compute | setup per step | of it: ints + pairs | DF tensor + metric | grid + XC setup | device loop setup | minor faults per step | kernel CPU time per step |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| water-svp | cpu | 0.028 | 0.022 | 0.006 (20 %) | 0.000 | 0.001 | 0.002 | 0.000 | 4852 | 0.011 |
| water-svp | gpu | 0.079 | 0.039 | 0.040 (51 %) | 0.000 | 0.031 | 0.004 | 0.001 | 1210 | 0.007 |
| la18-svp | cpu | 1.030 | 0.931 | 0.098 (10 %) | 0.007 | 0.044 | 0.046 | 0.000 | 116641 | 0.155 |
| la18-svp | gpu | 0.500 | 0.405 | 0.094 (19 %) | 0.008 | 0.047 | 0.030 | 0.005 | 8652 | 0.019 |
| u1an39-svp | cpu | 8.892 | 7.635 | 1.251 (14 %) | 0.035 | 0.832 | 0.401 | 0.000 | 1561626 | 1.725 |
| u1an39-svp | gpu | 2.487 | 2.058 | 0.423 (17 %) | 0.038 | 0.227 | 0.138 | 0.012 | 68308 | 0.151 |
| u1an39-tzvp | gpu | 6.652 | 5.403 | 1.228 (18 %) | 0.198 | 0.806 | 0.172 | 0.032 | 130093 | 1.092 |
| la18cat-svp | cpu | 1.883 | 1.784 | 0.098 (5 %) | 0.007 | 0.045 | 0.047 | 0.000 | 118110 | 0.162 |

### Energies

| system / platform | runs | max ΔE, all modes (Eh) | max ΔE, SAD modes A-C (Eh) | max Δgradient, all modes (Eh/bohr) | max Δgradient, A-C (Eh/bohr) |
|---|---:|---:|---:|---:|---:|
| la18-svp/cpu | 8 | 3.1e-12 | 1.3e-12 | 1.1e-06 | 1.8e-12 |
| la18-svp/gpu | 8 | 2.6e-12 | 8.0e-13 | 1.1e-06 | 1.8e-12 |
| la18cat-svp/cpu | 8 | 9.2e-04 | 1.0e-12 | 1.8e-02 | 1.6e-12 |
| u1an39-svp/cpu | 8 | 8.2e-12 | 7.3e-12 | 6.9e-07 | 2.8e-12 |
| u1an39-svp/gpu | 8 | 5.0e-12 | 4.5e-12 | 6.9e-07 | 2.4e-12 |
| u1an39-tzvp/gpu | 4 | 8.2e-12 | 8.2e-12 | 1.2e-06 | 2.1e-12 |
| water-svp/cpu | 8 | 1.4e-13 | 8.5e-14 | 1.0e-08 | 5.1e-13 |
| water-svp/gpu | 8 | 9.9e-14 | 9.9e-14 | 1.0e-08 | 3.4e-13 |
| water-svp/cpu-vs-gpu (mode D, repeat 0) | 2 | 9.9e-14 | | | |
| la18-svp/cpu-vs-gpu (mode D, repeat 0) | 2 | 9.1e-13 | | | |
| u1an39-svp/cpu-vs-gpu (mode D, repeat 0) | 2 | 4.5e-12 | | | |

The SAD modes (A, B, C) start from the same guess and agree to ~1e-12 in energy and gradient (the CLI reads
positions from an xyz file in Å: ~1e-12 bohr round trip). The warm start converges to a slightly different point
within the gradient's SCF thresholds (|ΔE| < 1e-9 Eh, max |F_ai| < 1e-7): energies within 1e-11 Eh, gradients
within ~1e-6 Eh/bohr, for every closed shell. CPU and GPU (FP64 emulation on this card) agree to 5e-12 Eh. The
one exception is the open shell below, where the warm start and SAD found different SCF solutions at one step.

### The open-shell case (la18⁺, UKS)

The radical cation has two UKS solutions close together near the start of its trajectory. At step 2 the warm
start (mode D, and the optimization that made the trajectory) converged to one 0.92 mEh lower than the one SAD
finds (<S²> 0.7664 against 0.7633; gradients differ by up to 0.018 Eh/bohr); at step 3, the first large BFGS
step, the projected orbitals of that solution needed 152 SCF iterations (the escalation path) where SAD needed
24, and from there on both starts give the same state (energies agree to 1e-12 Eh). So the warm start is not
only faster: it can carry the SCF along a different solution than a fresh start would pick, which for an
optimization is usually what one wants (continuity), but it changes results, and SAD-vs-warm agreement is not
guaranteed for open shells. Over the trajectory the warm start still wins (later steps 12-14 iterations against
20 from SAD), but by 3 % in total rather than the 30 % of its median step.

### Python and pyo3 overhead (water, CPU, `overhead.py`)

| per call | ms |
|---|---:|
| the binding's `gradient()` outside linus's own `timings.total` (pyo3, report → Python objects) | 0.13 |
| the package's `gradient()` (adds the `CalcResult`) outside `timings.total` | 0.21 |
| `set_positions` (checks, overlap, Löwdin projection of the last orbitals) | 0.08 |
| `Calculator(...)` construction / drop (CPU backend) | 3.7 / 0.07 |
| for scale: water's warm B3LYP/def2-SVP gradient | 19.7 |

### CUDA API time per step under Nsight Systems (u1an39 SVP, GPU)

| replay under nsys | steps | synchronization (waits for the device) | copies and memsets (D2H copies include waits) | launches | device memory alloc/free (pool, runtime) | page-locked host memory (register/alloc/free) | modules and kernels (load/unload/get) | context, events, streams (create/destroy/set) | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| u1an39-svp_gpu_B | 30 | 4.6252 | 1.6257 | 0.2890 | 0.0580 | 0.0196 | 0.2109 | 0.0029 | 0.0220 |
| u1an39-svp_gpu_D | 30 | 1.4542 | 0.7173 | 0.1355 | 0.0094 | 0.0076 | 0.0035 | 0.0017 | 0.0122 |

Under the tracer mode B's walls double (it instruments every module load), so its durations are not
representative; its call counts are: every new calculator loads 72 kernel modules (cuModuleLoadData) and, on
drop, trims the device memory pool once (cuMemPoolTrimTo). Mode D's walls are within 4 % of the untraced run.

## What a deeper session would add

Mode D's later steps still pay a setup per geometry (table "mode D's later steps" above): **5-14 % of a warm
step on the CPU, 17-19 % on the GPU (SVP and TZVP), more on water.** Almost all of it depends on the geometry
and no session can keep it: the three-center integrals, Coulomb metric and its factorization (`jk_setup`, the
largest part), the grid points and Becke weights (`grid`), shell-pair screening data (`pairs`, `grad_pairs`),
the one-electron integrals, the device XC screening and the gradient's XC layout (`xc_setup`, `grad_xc_setup`).
What a deeper session (one that keeps workspaces, plans and buffers between geometries) could avoid, per warm
step:

| item | CPU la18 / u1an39 SVP | GPU la18 / u1an39 SVP | measured or estimated |
|---|---:|---:|---|
| the gradient's second grid build at the SCF's geometry (the CPU gradient builds its own integrator; the GPU already hands the SCF's over) | 23 / 200 ms (2.2 / 2.3 %) | 0 | measured `grid` (both builds), split half and half: estimate |
| per-geometry kernel lookups and plan decisions (`grad_2e_setup_kernels`, `_plan`, `grad_xc_setup_layout_plans`, `_lists`, `_blocks`, `_aux_pairs`) | – | 7 / 11 ms (1.5 / 0.4 %) | measured sub-timers; that they could be cached is an estimate |
| the device SCF loop's buffers (`scf_setup`: allocations and the uploads of S and H) | – | 5 / 12 ms (1.0 / 0.5 %) | measured; upper bound (the uploads stay) |
| CUDA allocation, page-locked registration, event creation (driver API time inside those calls, Nsight Systems) | – | – / 19 ms (0.8 %) | measured under nsys (overlaps the two rows above) |
| host page faults (memory freed and touched again; kernel CPU time per step) | 0.16 / 1.7 CPU-s (2.3 / 2.7 % of the step's CPU time) | 0.02 / 0.15 CPU-s | measured `ru_stime`, `ru_minflt` (117 k / 1.56 M minor faults per step, ~1.1 µs each); the wall share is an estimate (≈ the CPU-time share) |
| of the faults: the DF tensor's first touch (0.1 / 1.25 GiB) | 0.03 / 0.35 CPU-s | on the device | estimate (pages × 1.1 µs) |
| **total a deeper session could save** | **≈ 4-5 %** | **≈ 1-2 %** | estimate |

Most of the page faults come from temporaries inside the SCF iterations (≈ 140 k faults, about 0.55 GiB of
fresh pages, per u1an39 Fock build), which persistent workspaces would remove within one SCF too: it is not
state *between* calculations. On the GPU the device memory already comes from a stream-ordered pool that keeps
freed blocks (crates/linus-gpu/src/cuda/pool.rs), so allocations are cheap; the per-step CUDA API time outside
synchronization and launches is under 1 % of the step.

The larger session lever is the one D already pulls: fewer SCF iterations. Each iteration saved is worth
≈ 0.56 s on u1an39 SVP on the CPU and ≈ 0.12 s on the GPU ((C - D) / Δiterations). A session that keeps the
last two or three geometries' orbitals could extrapolate them (as MD codes do) instead of projecting the last
one; not measured here, and the open-shell example below shows that a warm start can also go wrong.

## Caveats

- **Node noise.** Other packages' gates ran on CCDs 1-7 and on the GPU (between our jobs) the whole time; CCD0
  and the GPU were ours only for the duration of each job. Shared L3 is per CCD, but memory bandwidth and the
  package power budget are shared, so absolute walls carry a few percent of noise (compare the per-repeat
  totals). The ratios between modes are more robust than the absolute times: modes were interleaved in rotating
  order and every step is a median over repeats.
- **Two repeats; TZVP one.** Two repeats are enough to see effects of 5 % or more; differences of 1-2 % between
  A, B and C on the larger systems are within noise. The u1an39 def2-TZVP case (GPU) has one repeat: its second
  was cut for time (the brief's first cut), because gates' GPU test suites held the shared GPU lock for 10-15
  minutes at a time and our timed GPU jobs queued behind them (lock waits in `results/jobs.jsonl`); for the
  same reason the open-shell case ran on the CPU only. GPU walls repeat to 1-2 % between our repeats of the SVP
  cases.
- **How the jobs were scheduled changed during the run** (two drivers sharing the bench lock, then one serial
  driver, then CPU-first): every job still ran alone on CCD0 under the bench lock, and every GPU job under the
  exclusive GPU lock, so the measurements are unaffected; only the interleaving order differs from a strict
  rotation in places.
- **Real, but one kind of, trajectory.** Each trajectory is one real BFGS optimization in Cartesian coordinates
  (line-search trials included), from a 0.1 bohr RMS perturbation, capped at 30 gradients. The warm start's gain
  depends on the step size: smaller steps (late optimization, IRC, MD, finite differences) gain more, larger
  steps (early optimization, conformer jumps) less; the first steps here are the largest. A different optimizer
  (internal coordinates, fewer and larger steps) changes the step count, not the per-step picture.
- **The warm start converges to a slightly different point** (within the SCF thresholds): for the closed
  shells energies agree to ~1e-11 Eh between all modes, gradients to ~1e-6 Eh/bohr between the warm and the SAD
  starts (SAD modes A-C agree to ~1e-12). An optimizer sees the warm-start gradient as noise of that size. For
  the open shell the two starts found different SCF solutions at one step (0.9 mEh apart).
- **Mode A's CLI writes the whole JSON result** (orbital energies, SCF history; tens of kB) to a pipe; its parse
  in the driver is not timed.
- **The CPU runs use the GPU-feature build** with `backend cpu` (no device opened); the CUDA libraries are
  loaded only when a device is asked for.
- **Open shell:** la18⁺ (UKS doublet) is a hard SCF case (one early step needs 150 iterations from the warm
  start); read its rows as a probe-and-convergence example rather than a typical timing.

## Files

| File | What |
|---|---|
| `common.py` | systems, paths, locks, the job runner, the timer classification (`breakdown`) |
| `trajectory.py` | one geometry optimization per system (the replayed trajectory) |
| `replay.py` | one trajectory replayed in one mode, every step timed |
| `run_all.py` | the driver: `traj`, `replay` (rotating mode order per repeat), `overhead`, `nsys` |
| `overhead.py` | Python/pyo3 overhead per call on water |
| `summarize.py` | `results/summary.json` and `results/tables.md` from `results/raw/` |
| `results/trajectories/` | the visited geometries (bohr), energies, gradient RMS, SCF iterations of the optimization |
| `results/raw/<system>_<platform>_<mode>_r<repeat>.json` | per step: wall, linus's `timings`, iterations, energy, gradient, rusage, probes, DF storage, backend |
| `results/jobs.jsonl` | every job: lock waits, wall, busy threads per CCD before it |
| `results/nsys/` | CUDA API summaries of a GPU replay under Nsight Systems (modes B and D) |

Reproduce (on the bench node, from this directory, with the GPU-feature build in `target/release` and its
`lib_linus.so` copied to `crates/linus-py/python/linus/_linus.abi3.so`):

```bash
python run_all.py traj water-svp la18-svp u1an39-svp la18cat-svp u1an39-tzvp
python run_all.py overhead
python run_all.py replay --repeats 0 1          # one driver, jobs one after the other; rerun to resume
python run_all.py nsys u1an39-svp D B --cpus 48-55   # untimed, shared GPU lock, another CCD's cores
python summarize.py
```
