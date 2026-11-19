# MEMFIT: peak memory within `memory_gib` (2026-11-17)

`memory_gib` is a limit on the whole process: a calculation's peak resident memory stays within
it, and the plan states the smallest `memory_gib` that runs it (`memory.required_gib`). These are
the measurements behind that rule (`crates/linus/src/driver/needs.rs`) and its proof. All runs:
AMD Threadripper PRO 9985WX, the release CLI, ωB97X-D3BJ/def2-TZVP with def2-universal-jkfit and
grid `linus:2` unless named otherwise.

## Tools

- `rss_trace.py OUT.json [--smaps] --stdin REQ -- CMD...`: resident memory over time (5 ms) with the
  stderr lines timestamped (`output.verbose`: the SCF iterations mark the phases); `--smaps` keeps a
  snapshot of /proc/PID/smaps nearest the peak.
- `attrib.py RUN.json RUN.strace BIN`: the resident mappings of that snapshot attributed to the mmap
  call that created each (`strace -f -k -e trace=mmap,munmap,mremap`, frames symbolized with
  addr2line: file offset + 0x1000 - 1). Adjacent anonymous mappings merge, so a mapping can hold
  several allocations; the largest are unambiguous.
- `spread.py CASES.json OUT BIN [--front cli|py|qcc|qccw] [--cg]`: each case at an explicit
  `memory_gib` (`"plan"`: the plan's `memory.required_gib` rounded up to 0.1 GiB, what a scheduler
  sets; `"min"`: a failed probe's `required_gib`), peak RSS from `/usr/bin/time -v`; `--cg` runs it
  in a systemd scope with `MemoryMax` equal to `memory_gib` and no swap, so an overshoot is an
  out-of-memory kill. `py_runner.py` is the Python side (`linus.compute`, qccompute in process,
  qccompute's worker process). `split_peak.py` splits a trace's peak at the last SCF iteration;
  `phases.py` prints a plan's phases.

## Before: where the overshoot came from (main 12f4916, one thread)

| case | memory_gib | peak (GiB) | over |
|---|---|---|---|
| caffeine energy | 1.5 | 1.60 | 0.10 |
| caffeine energy, 8 threads | 1.4 | 1.78 | 0.38 |
| lidocaine energy | 3.0 | 3.15 | 0.15 |
| caffeine B3LYP gradient | 1.4 | 2.11 | 0.71 |
| caffeine gradient | 1.4 | 2.14 | 0.74 |
| ts61 ωB97X-D3BJ/def2-SVP gradient, 4 threads | 4.7 | 5.43 | 0.73 |
| caffeine HF/def2-SVP Hessian | 0.4 | 1.24 | 0.84 |
| caffeine B3LYP/def2-SVP Hessian | 0.4 | 1.20 | 0.80 |
| lidocaine B3LYP/def2-SVP Hessian, 4 threads | 1.0 | 2.27 | 1.27 |

Only the SCF's DF tensors (75% of the limit) and a Hessian's fixed arrays were counted. The rest:

- caffeine energy at 1.5: DF coefficients 1.03 GiB; the one-sided exchange's block scratch
  (`DfJk::k_blockseq`: Z panels and the raw integrals of a block, sized to ~1 GiB from the problem,
  here 0.38) from the first Fock build on; heap and arenas 0.11 (XC, Fock and DIIS matrices, the
  allocator's free blocks); the metric 0.012; the program 0.014. Lidocaine at 3.0: 2.18 + 0.50 +
  0.37 + 0.02.
- the process: the CLI 0.02 GiB; Python with numpy, qcdata, linus and qccompute 0.056 after the
  imports, 0.065 after a calculation; a qccompute worker 0.066. Small: not the cause.
- per thread: about 25 MB (caffeine energy, 1 against 8 threads).
- the allocator: glibc raises its mmap threshold to up to 32 MiB after freeing large blocks and keeps
  freed blocks below it in its arenas. A fixed 256 KiB threshold (`GLIBC_TUNABLES`) lowers the
  peaks by 0.056 (caffeine), 0.12 (lidocaine), 0.144 (caffeine, 8 threads) and 0.02 GiB (gradient)
  for 1-6% more wall time: not taken, counted instead.
- gradients: `df_gradient`'s pass-2 G block (`G_BLOCK_BYTES`, 3 GiB: all of caffeine's 1242
  auxiliary functions in one block, its lower triangle 1.13 GiB resident) and the half-transformed
  factors; with several blocks of different sizes the whole buffer is touched (ts61: 2.55 of
  3 GiB). The 3-center derivative pair layout adds ~0.3 GiB with f functions.
- Hessians: the response exchange's term-1 dense blocks (`T1_DENSE_BYTES`; 0.63 GiB for caffeine
  HF/def2-SVP), the response tensor, the CPKS subspace, the explicit terms' blocks.

## After: the rule and its proof

A phase's need is the process's baseline, the engines' sizes from their own functions (the DF
storage and its exchange workspace: `DfJk::scf_storage_needs`; the DF gradient's blocks:
`df_gradient_workspace`; a response's arrays), the shares of the limit (H-bar 1/4, response slots
and XC kernel 1/8) and a measured allowance (`RestModel`: per basis matrix, per thread, fixed). The
SCF's tensors also keep their 75%. Peak RSS at `memory_gib` = the plan's `required_gib` rounded up
to 0.1, in a scope with `MemoryMax` = `memory_gib` (no kill anywhere):

| case (mepd: 1 thread; packed: qccompute lists) | memory_gib | peak (GiB) | margin |
|---|---|---|---|
| mepd water energy | 0.2 | 0.024 | 0.176 |
| mepd caffeine energy | 1.8 | 1.678 | 0.122 |
| mepd lidocaine energy | 3.3 | 3.203 | 0.097 |
| mepd caffeine gradient | 2.4 | 2.190 | 0.210 |
| mepd p-benzyne triplet UKS gradient | 0.6 | 0.293 | 0.307 |
| mepd p-benzyne triplet UKS Hessian | 1.1 | 0.922 | 0.178 |
| packed caffeine energy, 8 threads | 2.0 | 1.789 | 0.211 |
| packed caffeine B3LYP/def2-SVP gradient, 8 threads | 1.1 | 0.837 | 0.263 |
| packed caffeine B3LYP/def2-SVP Hessian, 8 threads | 2.0 | 1.472 | 0.528 |
| packed caffeine HF/def2-SVP exact J/K gradient, 8 threads | 0.7 | 0.160 | 0.540 |
| packed ts61 ωB97X-D3BJ/def2-SVP gradient, 4 threads | 6.1 | 5.496 | 0.604 |
| Python `linus.compute`: caffeine energy | 1.9 | 1.715 | 0.185 |
| Python `linus.compute`: caffeine gradient | 2.4 | 2.238 | 0.162 |
| Python `linus.compute`: caffeine B3LYP/def2-SVP Hessian, 8 threads | 2.0 | 1.566 | 0.434 |
| qccompute in process: caffeine energy | 1.9 | 1.699 | 0.201 |
| qccompute worker process: caffeine energy (the worker's peak) | 1.9 | 1.726 | 0.174 |

The smallest `memory_gib` (plan, command line, one thread): caffeine energy 1.74 (was 1.4),
gradient 2.30, Hessian 2.24; lidocaine energy 3.27 (was 3.0), gradient 4.76; caffeine
def2-SVP Hessian 1.54 (B3LYP and ωB97X-D3BJ). A Python process needs 0.06 GiB more.

A process with a GPU: the CUDA context and libraries take 0.61 GiB of host memory (water
B3LYP/def2-SVP on the RTX PRO 6000: 0.638 GiB peak against 0.028 on the CPU; caffeine 0.697),
counted as 0.625 GiB (`DEVICE_HOST_BYTES`); the plans ask 0.78 and 0.82 GiB.

What moved (gate v2 on 32875c54): golden CLI JSON 31 of 31 identical, SCF traces 17 of 17, the
bitwise suite all bitwise, validation 0 unexplained; no DF storage rung changed anywhere. GPU A/B
59 of 61 bitwise: la18 `maxiter14` fails on both sides as designed, and
`u1an39.b3lyp.rks.cap1.5g.emu.energy` (a 1.5 GB device cap that sends the SCF's J/K to the CPU,
`--memory 2`) now fails before any work: its CPU fallback holds a 1.25 GiB B tensor beside the
CUDA process, 2.65 GiB in all, and main's run peaked at 2.33 GiB in its 2 GiB. At `--memory 2.7`
or 3 it runs bitwise main's result (B storage, peak 2.33 GiB).

Without `memory_gib` nothing changes: caffeine gradient and lidocaine energy at `auto` are bitwise
main's with the same DF storage (B and B^ω). The up-front check then runs against the machine's
limit within 90% of the free memory (a check only).

## What would lower the minimums (EPOCH-list candidates, not done: each changes the rounding)

The workspaces are sized from the problem, never from the limit (D43); sizing them from the limit
needs sums that do not depend on the block partition first.

| workspace | measured | what a budget-sized block would save |
|---|---|---|
| DF gradient pass-2 G block and its half-transform (`G_BLOCK_BYTES` 3 GiB, per-block gradient sums) | caffeine 1.13 + 0.23 GiB; ts61 2.55 + 0.7 | caffeine gradient minimum 2.3 to ~1.8 (the energy's); ts61 6.1 to ~5.2 (the SCF's) |
| SCF one-sided exchange blocks (`k_blockseq`, ~1 GiB target, lane sums per block) | caffeine 0.38-0.44, lidocaine 0.5, ts61 1.09 GiB | caffeine energy 1.74 to ~1.45, lidocaine 3.27 to ~2.9 |
| SCF in-memory exchange (`k_groups`: up to 64 partial K, 256 MB panels per running group) | caffeine B3LYP/def2-SVP 8 threads ~0.1 GiB | small for these sizes; grows with threads |
| response term-1 dense blocks (`T1_DENSE_BYTES` 256 MiB per operator) and the DF Hessian's derivative blocks (`MO_BLOCK_BYTES` 1 GiB, `DBLOCK_BYTES`, `RZ_BYTES`) | caffeine HF/def2-SVP 0.63 + 0.12 GiB | Hessian minimums ~0.5 GiB lower |
| a sizing function for the 3-center derivative pair layout | ~0.3 GiB with f functions | the gradient allowance could follow the basis: exact and def2-SVP gradients are over-allotted by up to 0.5 GiB |
