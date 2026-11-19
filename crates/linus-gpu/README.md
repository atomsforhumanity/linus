# linus-gpu

GPU kernels and engines for linus. The `linus` crate's `gpu` feature wraps them behind its
builder traits (`backend: "gpu"`; [docs/gpu.md](../../docs/gpu.md)); the API of this crate is
unstable (`publish = false`).

The engines are the device counterparts of linus's CPU builders, with plain host matrices at
their boundary; linus wraps them behind its own traits (`JkBuilder`, `XcPotential`,
`ResponseJk`, `ResponseKernel`, and its gradient dispatch):

- `df::GpuDfJk`: density-fitted J and K, the same screened tensor and results as linus's CPU
  `DfJk` (to round-off), the tensor on the device or streamed from pinned host memory, K
  optionally in FP64 emulation on INT8 tensor cores: our Ozaki scheme (`emu`, the tensor
  sliced once) or cuBLAS's fixed-mantissa emulation. Range-separated hybrids
  (`set_exchange_operator`): the CPU's wK exchange with a second tensor B^omega of the erf
  operator. Both tensors are set up on the device (`eri3c` integrals, the fit in FP64 or,
  `DfFit::Ozaki`, on INT8 tensor cores at FP64 quality).
- `xc::GpuXc`: the XC energy and potential on linus's grid, closed and open shells, LDA, GGA
  and meta-GGA (functionals through linus-xc on the host).
- `grad::df::GpuDfGrad`: the density-fitted two-electron gradient (J, a global hybrid's K and
  a range-separated hybrid's wK exchange, closed and open shells) of linus's CPU
  `df_gradient`, from the device three-center integrals and their derivatives (`eri3c`,
  Coulomb and erf, Gamma contracted inside the derivative kernels); the two-center metric term
  on the host.
- `grad::xc::GpuXcGrad`: the XC gradient of linus's `XcIntegrator::gradient` (LDA, GGA,
  meta-GGA), with the grid response; Becke weight derivatives on the device.
- `hess::df::GpuDfHess`: the density-fitted two-electron terms of an analytic Hessian, linus's
  CPU `df_hessian_2e` (global and range-separated exchange, closed and open shells): the
  derivative Fock matrices F^x C of the CPHF right-hand sides for every perturbation, the RI
  cross terms of the second derivative, and the second-derivative integrals (three-center from
  the generated second-derivative kernels with a Gamma epilogue, two-center contracted on the
  host); derivative integrals per auxiliary atom block, H stored on the device, the exchange's
  cross term as the Gram of L^-1 R.
- `hess::xc::GpuXcHess`: the explicit XC terms of an analytic Hessian, linus's CPU
  `XcIntegrator::hessian_terms` and `grid_terms` (LDA, GGA and meta-GGA, closed and open
  shells): the XC potential's derivatives times the occupied orbitals, the XC energy's second
  derivative and the grid response, in one pass over the per-batch layout of the XC engines.
- `pcm::GpuPcmInts` and `pcm::solver::GpuPcmSolver`: an implicit solvent's surface terms, the
  device counterparts of linus-ints' `SurfacePlan` and linus's `PcmOperator`: the potential of a
  density at the cavity's points and the charges' potential matrix (one pass each, or fused),
  the gradient's integral derivatives (shell centres and points), from generated kernels
  (Gaussian or point charges); the dense charge solve with K factored by cuSOLVER (Cholesky for
  C-PCM/COSMO, LU for IEF-PCM/SS(V)PE).
- `response::df::GpuDfResponse` and `response::xc::GpuXcKernel`: the CPHF/CPKS products of
  Hessians and polarizabilities (linus's `ResponseJk` and `XcIntegrator::kernel_contract_occ`)
  on those engines' tensor and grid: the projected J/K of factored first-order densities for
  all perturbations at once (global and range-separated exchange; FP64, Ozaki- or
  cuBLAS-emulated FP64 or single-precision exchange GEMMs; the exchange on a stored
  (n o) x (n o) operator when the products pay for it and it fits: `KOperator`), and the XC
  kernel contraction (closed and open shells, LDA, GGA and meta-GGA; FP64 or FP32 per batch).

All are deterministic run to run on a device.

## Building

Everything that touches a device is behind the `gpu` feature, which is off by default. A
CPU-only build never needs CUDA, and it still compiles the kernel source generators, the test
harness and the deterministic-sum format.

With the feature on, CUDA is still not needed at build time. The device layer (cudarc) loads
the CUDA driver, NVRTC, cuBLAS and cuSOLVER from a CUDA 13 toolkit when a GPU is first opened.
A machine without them gets `GpuError::Unavailable` from `Gpu::new`, not a link error.

```bash
cargo test -p linus-gpu                       # generators and harness, no GPU needed
cargo test --profile test-fast -p linus-gpu --features gpu   # also every kernel against the CPU code
cargo test --profile test-fast -p linus-gpu --features gpu --test nvrtc   # every kernel module compiled by NVRTC, no GPU needed
LINUS_GPU_REQUIRED=1 cargo test --profile test-fast -p linus-gpu --features gpu   # fail instead of skipping without a GPU
```

## Layout

| Path | Content |
|---|---|
| `src/cuda/` | **The device layer**, the only code that calls the vendor runtime: `Gpu` (device, stream, memory, copies, run-time compilation, launch, event timing and marks), `DeviceBuf` and its element-typed addresses `DevPtr` (`buf.at(i)`: no call site spells an element size), `HostBuf` (page-locked host memory: an anonymous mapping registered with the driver; asynchronous copies), `StreamedStore` (host memory streamed through a ring of device staging buffers, in order on the compute stream or overlapped on a copy stream, every staging buffer's cross-stream uses ordered by fences inside the store; `stream.rs`), `Kernel`, `Launch`, `Arg`, `blas` (one cuBLAS handle in one FP64 arithmetic: GEMMs on typed operands `MatRef`/`MatMut` (one bounds check), generic over f64 and f32 (`GemmScalar`), single, strided-batched, raw-address and grouped, with today's entry point fixed per call shape; SYRK, TRSM; INT8 GEMMs with INT32 accumulation in `ozaki.rs`), `solver` (symmetric eigensolver, repeated eigendecompositions by plan: syevd, Xsyevd, syevj; Cholesky). A HIP backend would be a sibling module exporting the same items. A test fails if cudarc is used anywhere else. |
| `src/df/` | **DF-J/K engine** (`GpuDfJk`): the screened pair-major tensor (`DfLayout`), set up on the device: the integrals from `eri3c` (linus-ints' on the CPU where the kernels do not cover the basis), the fit a triangular solve with the metric's factor in FP64 or (`DfFit::Ozaki`) blocked with Ozaki-GEMM products and refined diagonal blocks; J by two passes over it, K per auxiliary block (unpack, two GEMMs), device or host-streamed storage within a memory budget; range-separated exchange (wK) with B^omega from the device erf kernels. `KMath::Ozaki`: the tensor's scales computed once and, where they fit, its dense GEMM1 planes sliced once (else each block's digits gathered from the stored rows per build), K by Ozaki GEMMs. |
| `src/linalg/` | **The GEMM service** (`Gemm`): the engines' GEMMs, SYRKs and TRSMs, each call naming its arithmetic (linus-core's `Arith`: FP64, FP64 emulated with fixed bits, FP32); one cuBLAS handle per arithmetic prepared, configured once (no handle whose mode is hidden state). **The vector service** (`Blas1`, `device/blas1.cu`): scale (by a scalar or a diagonal) and axpby of up to three terms over the first n elements, in place or not, one definition for every engine; each operation's rounding is part of its definition (axpby: the first product rounded, each later term fused, so a caller fixes its rounding by the order of its terms). |
| `src/emu/` | **FP64 GEMMs emulated on INT8 tensor cores** (Ozaki scheme I, portable kernels): operands sliced into signed 8-bit digits with one power-of-two scale per vector (`Sliced`, `SlicedRef` views of a stored operand), each diagonal of the digit products summed exactly in INT32 (concatenated along k), one FP64 recombination in a fixed order; `gemm`, `syrk_lower`, `dgemm_at`. The INT8 GEMM itself is `Blas::gemm_i8` in `src/cuda/ozaki.rs`. |
| `src/xc/` | **XC engine** (`GpuXc`): per grid batch the functions the CPU integrator keeps (its value test, taken on the device once per geometry: `batched`), collocation per block of batches compressed to each batch's functions, densities and the potential by per-batch products in one batched launch per chunk (`kernels::vgemm`), the functional on the host (linus-xc), V_xc and E_xc in fixed point. Block screening (one product per block, both density routes) remains as an option. FP32 variants (`XcMath`) for the precision study and linus's `precision.fp32: on`. `vv10`: VV10 nonlocal correlation (`GpuVv10`, `GpuVv10Grad`): the engine (and the XC gradient's) on the NLC grid as a GGA of the total density, the pair sums in place of libxc (active points compacted on the device, one thread per point over all points in a fixed order, FP64, double-float or FP32 pair arithmetic, `Vv10Math`), the explicit point derivatives joining the gradient's point-motion term; `GpuVv10Resp`: the per-perturbation pair sums of VV10's CPKS kernel and Hessian terms (linus's `xc::vv10_resp`), FP64 or double-float. |
| `src/grad/` | **Gradients**: `df` (`GpuDfGrad`: MO integrals by GEMMs over dense blocks of whole first shells, the fit by triangular solves, the metric term's GEMM on the device and its two-center derivative integrals on the host, Gamma contracted inside the eri3c derivative kernels (or the six derivative planes stored and contracted here), fixed-point atom sums), `xc` (`GpuXcGrad`: per-batch screening as the XC engine's, the density pass to first order with X = phi D kept for the contraction, collocation to second derivatives, the contraction per (batch, function) and (batch, point), the Becke weight derivatives with each atom pair once per point and negligible pairs left out). |
| `src/hess/` | **Hessian terms**: `df` (`GpuDfHess`): the MO pass over blocks of whole first shells (W = (mn\|P) C by strided-batched GEMMs, Z = V^-1 W, X), pass 1 over auxiliary atom blocks (derivative planes unpacked to the dense I^k[a][q][b], H = I C, gamma^x, J1, the K terms of F^x C), E^k and H~ = L^-1 H, pass 2 over sub-chunks (R~^x = L^-1 R^x for all perturbations, the Gram GEMM, sum_P W~_P R~^x_P, optionally in the Ozaki scheme), and the second derivatives (Gamma in the pair-row layout, `Eri3c::hessian` into a fixed-point (3N)^2 accumulator)); `xc` (`GpuXcHess`): the explicit XC terms of the CPU's `hessian_terms` and `grid_terms` in one pass over the XC engines' per-batch layout (third-order collocation, X for ten components, u, t and the E1 parts per point and local atom, the E terms by GEMMs and per-atom kernels, the G matrices per point owner, and one fused kernel for every potential slot of a batch: phi^T Y[t] with Y formed on the fly and each element added to its packed fixed-point V; the grid response with linus-grid's weight derivatives, any partition; V^x C and dV^y C by GEMMs at the end). |
| `src/pcm/` | **PCM engine**: `PcmIntsHost` (pair records in each class's E patterns, per-degree primitive lists, the tables; no GPU) and `GpuPcmInts` (Hermite densities, the generated kernels over primitives x point chunks with fixed-order reductions, the matrix assembly, the gradient's atom sums in detsum); `solver` (`GpuPcmSolver`: S and D by element kernels, D A S by cuBLAS, K factored and solved by cuSOLVER). |
| `src/kernels/pcm_gen/` | The PCM kernels, two per Hermite degree L = la + lb (potentials and charge accumulators, first derivatives), generated by `tools/intgen/intgen.py pcm` (`cuda_pcm.py`). |
| `src/response/` | **Response engines** for CPHF/CPKS. `df` (`GpuDfResponse`): Y = B C stored at setup, J by two GEMMs over it, K per auxiliary block of `GpuDfJk`'s tensor (one GEMM for all perturbations, then strided-batched contractions), wK exchange with a second tensor of the operator. `xc` (`GpuXcKernel`): the kernel's weighted second derivatives evaluated once at setup (linus-xc), then per block (FP64) or per batch (FP32) the perturbed densities and the contraction by GEMMs on `GpuXc`'s blocks or batches. |
| `src/scf/` | **Device-resident SCF loop** (`run_scf`): linus's CPU `scf::run_scf` on the device (orthogonalizer, Fock assembly, energy, orbital gradient, DIIS, level shift, diagonalization, densities; RHF/UHF/RKS/UKS), only scalars crossing PCIe per iteration. The Fock terms sit behind `FockTerms`: `DeviceTerms` (the DF-J/K and XC engines) or `OnHost` (the CPU fallback). Its kernels: element-wise updates with the CPU's roundings and deterministic compensated reductions. |
| `src/kernels/` | One module per kernel family: a source generator (plain Rust that emits CUDA C++ specialized for one class of work, testable without a GPU) and the launcher. `colloc`: basis functions and derivatives on grid points, for a list of shells with output strides as arguments. `eri3c`: three-center Coulomb integrals of one shell class, McMurchie-Davidson as linus-ints. `vgemm`: variable-size batched FP64 GEMMs (a tile list over many small products of different shapes, one launch). |
| `src/eri3c.rs` | **Three-center integrals (mn\|P) and their first and second derivatives on the device**, for 1/r and erf(omega r)/r, for a basis, an auxiliary basis and a list of shell pairs with output rows (the row layout of a density-fitting tensor): host packing (`Eri3cHost`) and the builder (`Eri3c`, feature `gpu`) that fills any pair range, against every auxiliary shell or one block of them (`compute_block_op`, block-relative columns), contracts the first derivatives with a Gamma into a gradient (`gradient`) and the second ones into a Hessian (`hessian`). Every class with orbital l <= 4 and auxiliary l <= 6. |
| `src/kernels/eri3c_gen/` | The fused three-center kernels, one CUDA source per class and kind (integrals, first derivatives, second derivatives), generated by `tools/intgen/intgen.py cuda` with the schedule the tuner recorded (or the default), `operator.cuh`, the operator step of their Boys stage (1/r, or erf(omega r)/r with `ERI3C_ERF`), and `hessian.cuh`, the second derivatives' epilogue (the Hessian's centre blocks in fixed point); `src/kernels/eri3c_class.rs` derives the record shapes and tables from linus-ints and checks them against the generator's hash. |
| `src/df_deriv.rs`, `device/df/deriv.cu` | What the DF derivative engines share: one setup (`DerivSource`: builders, metric factor, row tables) and their common kernels (`DerivKernels`: a fitted block unpacked to its dense form, the gradient's way or the Hessian's, and Gamma). |
| `src/detsum.rs`, `device/detsum.h` | Fixed-point accumulation on 64-bit integer atomics. Bit-identical to the CPU accumulator (linus-num's `det::FixedAcc`) in any order. |
| `device/portable.h` | The few names that differ between CUDA and HIP. The host pastes it, with the device's warp size, in front of every kernel source. |
| `device/common.cuh` | Device functions several modules share (the packed lower-triangle index), pasted in front of the sources that use them (`kernels::registry::COMMON_H`). Kernels several engines launch are modules of their own instead (`blas1.cu`, `df/deriv.cu`), compiled once. |
| `device/<area>/*.cu`, `*.cuh` | **The hand-written kernel sources**, one file per module part (`df/jk.cu`, `xc/vv10.cu`, `emu/digits.cuh`, ...; `hess/xc_vslots.cu.in` is a template the XC Hessian instantiates four times), pulled in with `include_str!` by the engine that owns them. Each engine declares its modules once (`KernelModule`: name, assembled source, entry points) and `src/kernels/registry.rs` lists them all. Run-time generators (collocation, vgemm) and the offline-generated families (`eri3c_gen`, `pcm_gen`) keep their own sources. |
| `tests/nvrtc.rs` | Every module (the registry, every generated three-center and PCM class, the collocation kernels of every (l, pure) and order, vgemm) compiled by NVRTC for sm_90 and sm_120 without a GPU, each declared entry point checked in the image; CI runs it with NVRTC from the pip wheel. `LINUS_KERNEL_MANIFEST=<file>` writes every module source's SHA-256. |
| `src/harness.rs` | Benchmark molecules, basis sets, grids, error statistics for GPU-vs-CPU checks (feature `testkit`: the crate's tests and examples only, never the library linus links). |
| `src/cpu.rs` | Everything taken from linus-ints, linus-grid and linus-xc, in one place. |
| `examples/spike_bench.rs` | Roofline numbers of the card (FP64 FMA, DGEMM, bandwidth) and the kernels against the CPU. |
| `examples/gemm_emulation.rs` | Native vs emulated DGEMM on density-fitted-exchange GEMMs (data from `tools/dfk_data.py`). |
| `examples/grouped_gemm_bench.rs` | The XC engines' per-batch product shapes: one DGEMM per product, cuBLAS's grouped GEMM (FP64, FP32, and in an emulated mode), per-product emulation and `vgemm` tile shapes. |
| `examples/ozaki_bench.rs` | The Ozaki emulation: which INT8 GEMM shapes cuBLAS takes (`probe`), INT8 throughput (`int8`), and native, cuBLAS-emulated and Ozaki GEMMs on the DF-K shapes with their errors (`dfk`). |
| `examples/eigh_bench.rs` | cuSOLVER's eigensolvers (syevd, Xsyevd, syevj) at the SCF's matrix orders: time, residual, bitwise repeat. |
| `examples/hess_bench.rs` | `GpuDfHess` of one molecule (random orthonormal orbitals, no SCF): setup and per-call phase times; the target for Nsight Compute. |
| `examples/eri3c_bench.rs`, `examples/eri3c_tune.rs` | The generated three-center kernels against the card's FP64 peak, the CPU and the spike's (dd\|d); the auto-tuner that times each class's candidate schedules (`intgen.py cuda --candidates`) and records the fastest (`intgen.py cuda --record`). |
| `examples/dmma_bench.rs` | NVIDIA-only (inline PTX): FP64 tensor cores (DMMA) against FP64 FMA on SHARK-style T/I-step and three-center contraction shapes; sm_80/sm_90, skips elsewhere. Decides whether a data-center kernel variant pays; the portable kernels stay the reference. |

## Kernel rules

- **FP64** unless a kernel says otherwise. Reduced or emulated precision only as a per-kernel
  policy, after an accuracy study.
- **Deterministic** run to run on a device. Each output has one writing thread, or it is summed
  with `detsum`. No floating-point atomics.
- **Portable source.** C-style C++ in the subset that CUDA and HIP both compile:
  - no inline PTX;
  - vendor-specific paths (tensor cores, emulated GEMMs) only as optional variants beside a
    portable kernel;
  - the warp size is never a literal (`LINUS_WARP_SIZE`).
- **Large GEMMs go through the vendor BLAS.**

## Kernel cache

Compiled kernels are cached in memory and on disk. The key is the SHA-256 of the full source,
the compiler options, the target architecture and the NVRTC version. A CUBIN is built for the
device's `sm_XY`, or PTX when NVRTC is older than the device.

The cache lives in the directory the device opens with (linus's `backend.kernel_cache`), by
default in `$LINUS_GPU_CACHE` if set (`off` disables the disk cache), else in
`$XDG_CACHE_HOME/linus/gpu-kernels` or `~/.cache/linus/gpu-kernels`. Files are written
atomically. A file that fails to load is recompiled.
