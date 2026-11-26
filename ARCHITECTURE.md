# linus architecture

This is a map of the code for people who change it: which crate and module owns each concept
of the electronic-structure problem, which way the dependencies point, where a new method or
backend plugs in, and which files are generated. Usage is in README.md and docs/; the names
of things (one per concept, also in the code) are in docs/conventions.md.

## Crates

```
 5   linus-cli        linus-py        linus-dev (not published)
          \               |              /
 4   ---------------- linus (the facade) ----------------
       /         /          |          \           \
 3  linus-jk  linus-dft  linus-scf  linus-solvent  linus-vib      (no edges between them)
       \         \          |          /           /
 2   ------------------ linus-base ---------------          linus-gpu (feature gpu)
                            |                                   |
 1   linus-ints   linus-grid   linus-xc   linus-disp (feature dispersion, LGPL)
 0   linus-core   linus-num
```

Dependencies point to lower layers only, and the method crates of layer 3 have no edges between
them: the SCF and the response (linus-scf) reach the J/K, XC and solvent families only through
linus-base's terms, and the families do not know each other or the algorithms. The compiler
enforces the graph; `tools/check_layering.py` checks every manifest against the table below
(exactly these edges between linus crates, each to a lower layer; dev-dependencies from its own
list) and that the table and the script agree. CI's lint job and the gate's chain L run it.

<!-- layering:begin -->
| crate | layer | its linus dependencies |
|---|---|---|
| `linus-num` | 0 | none |
| `linus-core` | 0 | none |
| `linus-ints` | 1 | `linus-num`, `linus-core` |
| `linus-grid` | 1 | `linus-num`, `linus-core` |
| `linus-xc` | 1 | none |
| `linus-disp` | 1 | none |
| `linus-base` | 2 | `linus-core`, `linus-ints`, `linus-grid`, `linus-xc` |
| `linus-gpu` | 2 | `linus-num`, `linus-core`, `linus-ints`, `linus-grid`, `linus-xc` |
| `linus-jk` | 3 | `linus-num`, `linus-core`, `linus-ints`, `linus-grid`, `linus-base` |
| `linus-dft` | 3 | `linus-num`, `linus-core`, `linus-ints`, `linus-grid`, `linus-xc`, `linus-base` |
| `linus-scf` | 3 | `linus-num`, `linus-core`, `linus-ints`, `linus-base` |
| `linus-solvent` | 3 | `linus-num`, `linus-core`, `linus-ints`, `linus-grid`, `linus-base` |
| `linus-vib` | 3 | `linus-core`, `linus-base` |
| `linus` | 4 | `linus-num`, `linus-core`, `linus-ints`, `linus-grid`, `linus-xc`, `linus-disp`, `linus-base`, `linus-gpu`, `linus-jk`, `linus-dft`, `linus-scf`, `linus-solvent`, `linus-vib` |
| `linus-cli` | 5 | `linus` |
| `linus-py` | 5 | `linus` |
| `linus-dev` | 5 | `linus-core`, `linus-ints`, `linus-grid`, `linus-xc`, `linus-base`, `linus-jk`, `linus-dft`, `linus-scf`, `linus-solvent`, `linus-vib`, `linus` |
<!-- layering:end -->

| crate | owns |
|---|---|
| `linus-core` | the vocabulary every layer shares, so no crate keeps a copy of another's type: units and constants (one CODATA table; the Bohr radius), elements, `Structure` (the atoms, charge and spin of a calculation) and its XYZ reader, spin channels (`Channels`), nuclear gradients as sums of parts (`Gradient`), the error kinds (`ErrorKind`, `DeviceErrorKind`), memory plans and budgets, the arithmetic of a kernel family (`Arith`), timings (`Timers`, `Span`) and the run record; types only, the policy stays in the driver |
| `linus-num` | the numerical primitives: deterministic accumulation (`det`: the 2^-82 `FixedAcc` and 2^-64 `Fixed128` fixed-point formats, ordered chunks `par_fold_det`, shape-blocked `det_matmul`, `ThreadAccs`) and the SIMD layer (`simd`: `Lanes`, the run-time `Isa` with `LINUS_ISA`, the `multiversion!` and `multiversion_lanes!` macros) |
| `linus-ints` | Gaussian integrals: the basis set library, the McMurchie-Davidson engine, one-electron, ECP, four- and three-center integrals, integrals on points, and all their nuclear derivatives |
| `linus-grid` | molecular integration grids: radial and angular quadratures, pruning, partitioning with weight derivatives, batching, presets |
| `linus-xc` | exchange-correlation functionals behind a backend-agnostic trait (`XcKernel`), with libxc as the backend |
| `linus-disp` | DFT-D3(BJ) and D4 dispersion with analytic derivatives; LGPL, linked only through `linus`'s `dispersion` feature (the facade converts its errors, `dispersion.rs`) |
| `linus-base` | what the method crates share and none of them owns: the errors of a calculation (`error`: `LinusError`, the SCF's and the responses' failures, the did-you-mean hints) and the SCF's convergence thresholds they report (`convergence`), dense linear algebra (`linalg`, the only route to faer's products, solves and factorizations: thread-count determinism, clippy.toml; products with less work per thread than their threads' synchronization run sequentially, the same bits; faer's kernels add differently by shape, so a product cut into parts by a pass, a memory budget or the thread count goes through `matmul_part` with its `Whole`, and solves cut into column blocks through `column_blocks`: D43's bits on every CPU), memory limits and probes, the process kinds' baselines and a phase's need (`memory`: `MemoryLimit`, `Process`, `PhaseNeed`; `memory_gib` limits the whole process), a calculation's progress (`progress`: `Progress`, its lines to stderr for the command line or collected for an in-process caller's `logs`, its `Event`s, the stages and the SCF's and responses' iterations, to a listener: the Python package's `ctx.progress`, and the caller's stop request, `StopCheck`: the Python package's `ctx.stop_requested()`, asked before each stage and between the SCF's and the responses' iterations, which ends a calculation with `LinusError::Stopped`), molecular point-group symmetry (`symmetry`: the point group and rotational symmetry number of a structure, the largest abelian subgroup in the D2h family with a lab-aligned frame and Mulliken's axis conventions, and how the group's operations act on a basis set's functions), and the method families' terms (`terms`): the traits through which the SCF and the response call the families (`JkBuilder`, `ResponseJk`, `CoulombHook`, `XcPotential`, `ResponseKernel`, `EnvironmentTerm`, `EnvironmentResponse`) and the data they exchange |
| `linus-jk` | the J/K family: Coulomb and exchange builders and their derivatives, one module per approximation (`exact`, `df`, `cosx`) and the composed builders (`CompositeJk`, `DenseResponseJk`); `df::reference` (feature `reference`) holds the dense test oracles |
| `linus-dft` | the XC family: the quadrature (`XcIntegrator`: energy, potential, gradient; `hessian`; the CPKS `kernel`; the batch engine every pass runs through) and VV10 at every derivative order |
| `linus-scf` | the SCF as one algorithm over the families' terms, its control policy, accelerators and analyses; the linear response (`response`: perturbations, the response operator, the CPHF/CPKS problem and solvers, the second-order assembly and its stationary form); the reference determinant derivatives and responses expand around (`reference`) |
| `linus-solvent` | the solvent family: polarizable continuum models (C-PCM, COSMO, IEF-PCM, SS(V)PE) on Gaussian surface charges, SMD and its surface area, the solvent database (`db`), and the models' resolved parameters (`model`: `ResolvedPcm`) |
| `linus-vib` | harmonic analysis and thermochemistry: pure functions of any program's Hessian (the point group and symmetry number from linus-base's `symmetry`) |
| `linus-gpu` | the GPU engines (CUDA sources under `crates/linus-gpu/device/` compiled by NVRTC at run time and in CI without a GPU; cuBLAS, cuSOLVER loaded at run time): density-fitted J/K, XC, response products, gradients, the Hessian's explicit two-electron and XC terms and the PCM's surface terms with plain matrices at their boundary, and FP64 GEMMs emulated on INT8 tensor cores (Ozaki scheme). Below them the device layer (the only vendor code): typed device addresses (`DevPtr`), host memory streamed through device staging buffers (`StreamedStore`), errors typed by kind (`GpuError::kind`, linus-core's `DeviceErrorKind`, which the fallback policy reads), one GEMM service (`linalg::Gemm`) whose every call names its arithmetic (linus-core's `Arith`) and one vector service (`linalg::Blas1`: scale and axpby, each with its rounding fixed by its definition). The engines come in families on shared parts: the XC engines (energy and potential, gradient, Hessian terms, CPKS kernel, VV10) on one per-batch layout (`xc::layout::XcLayout`), one density pass (`xc::density`) and one host functional evaluator (`xc::functional`); the DF derivative engines (gradient, Hessian terms) on one setup and one module of common kernels (`df_deriv`: the dense blocks' unpack, Gamma) and as phase methods; every engine reports its record as named phases in seconds (`engine::Phases`), its device phases timed by events (`engine::Timeline`). The precision studies' variants (the XC block paths, density routes, emulated and FP32-potential XC products) are behind the `study` feature. Linked only through `linus`'s `gpu` feature, which wraps the engines behind its traits |
| `linus` | the facade: the driver (`Session`, the settings and their schema, the plan, the report, the run record), the backend providers (`backend`: the CPU's and the GPU's, its adapters over linus-gpu's engines and the device loop), the integral context of a geometry, the assembly of gradients, Hessians and properties from the families' terms, the families' dispatch (`two_electron.rs`, `solvent/`: which engine a calculation's settings choose, at which order), dispersion, and the public API |
| `linus-cli` | the `linus` command-line program (JSON in and out) |
| `linus-py` | the Python package `linus` on qcdata objects: `Session` (one molecule and model at many geometries, its lineages in the facade) and `compute` (one calculation), qccompute's program protocol (`linus.qc.program`, the `qccompute.programs` entry point; a runner's grant reaches `compute` and `Session` as `ctx`), the report as a qcdata `Frame` (`_frames`), the converters on qcdata's models (`vib`: any Hessian's analysis and thermochemistry; `convert`: PySCF; Molden files are qcinf's), over the pyo3 binding (`_linus.Session`) |
| `linus-dev` | study, check and benchmark programs (examples/), on the internals of the crates |

Every crate is `publish = false`. The stable interfaces are the CLI, the JSON request and result,
and the Python package. The Rust API is unstable (0.x): `linus`'s documented facade is what
lib.rs re-exports (`Session`, `CalcOptions`, `Structure`, the result types, `LinusError`,
`vib`, and the `JkBuilder`/`ResponseJk` traits for backend authors). Its other modules are
`pub(crate)`, or `#[doc(hidden)] pub` where the front ends or linus-dev use them. The concept
crates are public to the facade and linus-dev: what the facade's device loop, dispatch or
whole-calculation tests use of a crate is public there, the device loop's SCF units hidden from
the documentation.

## linus-ints

```
basis.rs     basis sets from Basis Set Exchange JSON; the library: load by name, composites a|b,
             listing, did-you-mean, file names
contract.rs  the typed inputs and outputs of the integral contractions, with no spin, hybrid or
             solvent semantics: DensityTerms (the density factor of the four-center
             derivatives), AoMatrix/AoMatrixMut (strided nbf x nbf views: no linear-algebra
             crate at the boundary, no copy)
engine/      the inside of every plan: batch engines walk a pair set's batches in a fixed order
             and hand them to monomorphized digesters (block: the batched four-center walk and
             BlockDigester); the determinism contract in code (schedule: a digester's
             Reduction, Exact or Ordered, picks the schedule; one dispatcher runs the Ordered
             schedule on per-thread scratch with per-chunk partials: strided chunks
             (ordered_strided; ordered_pool, its partials carved from one workspace pool) or
             contiguous ones (ordered_ranges)); WorkspaceSize
error.rs     IntsError, what a plan's constructor refuses, with its cross-package ErrorKind
operator.rs  the Operator enum (Coulomb, LongRange, Attenuated: 1/r, erf(omega r)/r and their
             combinations), the one input of every two-electron engine; how the kernels
             evaluate them (Operator::kernel, the one selection rule) and the screening
             threshold scaled for them
pairs.rs     PairSet, the one owner of a basis's shell-pair data: E and, up to its DerivOrder,
             E^A and E^AA of every pair; the Schwarz factors, derivative-integral bounds and
             primitive Schwarz factors, computed on first use; the (P, 1) pairs of an
             auxiliary basis (PairSet::aux; PairKind tells them apart, and the plans check it). One
             orientation (pair (i, j), i >= j) serves every engine: the batched four-center
             layout orients its own copy (higher l first, E transposed)
core/        the shared engine: Boys function and Rys quadrature, Hermite machinery (hermite;
             pair_data: the one primitive loop of E, E^A and E^AA), derivative expansions of
             shell pairs (deriv), solid harmonics, lane
             kernels (on linus-num's simd), small shared helpers (util: the shell-pair
             transform, density maxima, gradient sums), the study switches of the dev-switches
             feature (switches)
one_e/       the plan of the one-electron integrals (plan: OneEPlan, the S, T, dipole and
             point-charge V matrices, their derivative matrices and their gradient and Hessian
             contractions into caller buffers over a caller-owned OneEWorkspace); the kernels of
             one shell pair (mod) and their derivative blocks (deriv); effective core potentials
             (ecp) and their plan (ecp_deriv: EcpPlan, matrix, derivative matrices, gradient,
             Hessian; an EcpWorkspace with the local channels' PotentialWorkspace and the
             semi-local projection buffers, reused level after level)
eri4c/       the quartet engine (mod); batched kernels for direct Fock builds (batch); the
             integral-direct J/K plan (jk: JkPlan with caller-owned JkWorkspaces, and its J/K
             digester); derivative integrals (deriv) and the plan of their gradient and Hessian
             contractions (deriv_plan: Eri4DerivPlan, per-chunk accumulators in a caller-owned
             Eri4DerivWorkspace); first-derivative J/K (jk_deriv)
eri3c/       three- and two-center integrals for density fitting (mod); the screening
             thresholds shared with the device engines, every derivative's included
             (DfThresholds, one named field per kind of screening); the plan of the
             derivative contractions (deriv_plan: Eri3DerivPlan over an orbital and an
             auxiliary pair set, its kernels' pair layouts built on first use: the split-RI-J
             gamma, three-center and metric gradients of a Coulomb fit, whose clients are the
             RI-J gradient and the density-fitted PCM; per block of auxiliary shells, the
             Gamma-contracted gradient and Hessian and the derivative blocks of the DF
             gradient and Hessian; ordered chunks, per-thread scratch in a caller-owned
             Eri3DerivWorkspace); its kernels: split-RI-J gradient (deriv), derivative blocks
             and the Gamma-contracted Hessian (dblock), the Gamma-contracted gradient (dgrad)
eri2c.rs     the density-fitting metric's derivatives: Eri2DerivPlan over the (P, 1) pairs
             (gradient and Hessian contractions with a naux x naux γ into caller buffers,
             per-chunk accumulators in a caller-owned Eri2DerivWorkspace; the derivative
             integrals (∂_k P|Q) in one 3 x naux x naux layout)
points/      basis functions on points (ao, colloc); potential integrals of shell pairs at
             points for seminumerical exchange (grid_pot, grid_pot_deriv); integrals of
             Gaussian surface charges for continuum solvation (pcm_ints: SurfacePlan, owning
             its pair data, for SCF passes and responses (potentials, matrices, both fused,
             the Rys gradient) over a SurfaceWorkspace made for a SurfaceShape; its
             SurfaceDerivPlan over a pair set of order one or two (the McMurchie-Davidson
             gradient, point-potential derivatives, derivative matrices, Hessian);
             AuxSurfacePlan, the density-fitted PCM's closed-form (P|g) and gradient)
potential.rs the potential of a set of charge centers between basis functions, for any
             ChargeSource (point charges, Gaussian surface charges, the ECP local channel):
             PotentialPlan, generic over the source and the shell pairs it visits (gradient,
             hessian, add_deriv_matrices; per-chunk accumulators in a PotentialWorkspace) over
             the sources' Hermite tensors, the centers moving with coordinate slots, and the
             one potential block of a pair (matrix_block; a matrix and the d/dA derivative
             blocks sum the centers' tensors before one contraction); the engine of OneEPlan's
             potential, EcpPlan's local channels and SurfaceDerivPlan
raw.rs       the raw path: one shell quartet (and its first derivatives), triplet, pair or
             one-electron block into a caller's buffer, libint-style, with the layouts stated;
             the plans are the fast path for whole matrices
generated/   generated kernels (pub(crate))
reference/   plain reference integrals and contractions the tests check the kernels against
             (feature `reference`, on in the tests of linus-ints, the method crates, linus and
             linus-gpu)
```

## Concepts and where they live

Paths without a crate name are the facade's (`crates/linus/src`).

| concept | crate and module |
|---|---|
| structure (atoms, charge, spin), charges the electrons see | `structure.rs` (linus-core's `Structure`, elements and XYZ reader under linus's facade paths) |
| basis set errors as `LinusError` | linus-base's `basis` (`From<BasisError>`; the library is `linus_ints::basis`) |
| errors and their hints | linus-base's `error` |
| point groups: a structure's group and symmetry number, its abelian (D2h-family) subgroup and irreps, the group on a basis set's functions | linus-base's `symmetry` (`detect`, `abelian::AbelianGroup`, `ao::AoGroup`); the orbitals' symmetry at a converged point is linus-scf's `stability::symmetry` |
| linear algebra, timers, memory limits and probes, gradients as sums of parts | linus-base's `linalg` and `memory` (timers and gradients as sums of parts are linus-core's `timing` and `gradient`, deterministic sums linus-num's `det`); `linalg` is the only route to faer's products, solves and factorizations (thread-count determinism; clippy.toml, CONTRIBUTING.md) |
| the method families' terms: how the SCF and the responses call J/K, XC and the environment | linus-base's `terms`: `JkBuilder` (with `JkRequest`, `DensityFactors`, `JkOutput`), `ResponseJk`, `CoulombHook`, `XcPotential` (at an `XcDensity`), `ResponseKernel`, `EnvironmentTerm`, `EnvironmentResponse`, and `exchange_parts` (the method's exact exchange as the builders apply it) |
| SCF (RHF, UHF, RKS, UKS) | linus-scf: one algorithm, evaluate → observe → decide → step (`run`), with `run_scf` and its input, the XC and environment terms it takes (mod) and its records (`result`); the Fock evaluator, the only caller of the J/K, XC and environment terms, with accepted and trial evaluations (`fock`); the control policy as pure state machines on each iteration's scalars: convergence, the lowest energy, two-cycle, plateau, a DIIS blow-up, escalation, the stability trigger (the probe, the saddle check, `scf.stability` and a Hessian SCF's zero-mode check, whose search at the accepted point is its record; `control/`); the accelerators and their restarts: DIIS, ADIIS, SOSCF, damping, the level shift (`accel/`); the analyses of converged points on trial evaluations (`analysis`) and the Davidson iterations and records behind them (`stability`: one search per irrep of the converged state's abelian point group, from the irrep's lowest-gap pair with every product projected onto the irrep, `stability::symmetry`); `settings`, the state between builds (`state`), `guess` (SAD), the trace hook (`trace`); the device loop's controller over this loop's controls is the facade's `backend::gpu::scf_control` (its loop: `backend::gpu::scf_loop`), built from these units |
| XC quadrature | linus-dft: the functional resolved once (`XcFunctional`), the one factory (`XcIntegrator::build`) and the integrator (energy, potential, gradient with VV10; mod), the batch engine every pass over a grid runs through (`batch`: scheduling per batch or batch group, per-thread work, collocation in the one function-major layout every pass reads, ordered outputs) and its shared contractions (`push_vmat`, `push_grad`; `push_vco` in `kernel`), the composition with VV10 at every derivative order (`engine`: the CPKS kernel `XcKernel`, the Hessian terms `add_hessian_terms`), the functional's pointwise inputs and weighted derivatives per spin channel (`points`), second-derivative terms (`hessian`) and kernel contractions for responses (`kernel`), each written once and generic over the channel count (one: a closed shell's total density; two: the spins; the entry points take linus-core's `Channels`), VV10 (`vv10`, with the one pair primitive and the one local block of every VV10 sweep; its CPKS kernel and Hessian terms: `vv10_resp`), profiling (`profile`), test reference (`reference`) |
| J and K | linus-jk, one module per approximation: `exact/` (the SCF builder over linus-ints's JkPlan: incremental thresholds, workspaces kept per request shape; gradient, Hessian terms, the spin and hybrid density factors of its derivatives: `terms`), `df/` (the RI-JK builder `DfJk`, its exchange scratch kept from one build to the next, gradient, Hessian terms, response J/K `DfResponseJk`, RI-J gradient, shared pieces, test references), `cosx/` (seminumerical exchange, its gradient, the RIJCOSX gradient); at its root the simple builders (`IncoreJk`, `CompositeJk`, `DenseResponseJk`); the dispatch of the settings' J/K method at every derivative order is `two_electron.rs` |
| the reference determinant of derivatives and responses | linus-scf's `reference`: `Reference`, built once per SCF result and borrowed by the gradient, the Hessian, the polarizability and their CPHF/CPKS: per spin channel (linus-core's `Channels`) an `OrbitalBlock` (occupied and virtual orbitals and energies, the channel density), and the total and energy-weighted densities, each in one fixed arithmetic |
| linear response (CPHF/CPKS) and second-order properties | linus-scf's `response`: a second-order property is a pair of perturbations over one response framework. The perturbations (`perturbation`: `Perturbation`, what a static perturbation hands the response and the properties: F^k C_o at fixed density, the overlap derivative's occupied projections `Overlap`, the right-hand sides; `NuclearDisplacements` here, `ElectricField` in `properties/dipole`), the response operator, composed once per response from the method families' terms and applied in a fixed order (`operator`: `ResponseOperator` over the J/K, the XC kernel and an `EnvironmentResponse`, with a two-level response's iteration J/K), the CPHF/CPKS equations over spin channels (`problem`: `ResponseProblem`, with the induced Fock matrices and the final residuals), the solvers (`solver`: the block Krylov linear solver, and the Davidson eigensolver that the SCF's stability analysis shares), the second-order assembly (`second_order`: the response part of E^{ba} from perturbation b's first-order matrices and perturbation a's first-order orbitals `FirstOrderOrbitals`, `SecondOrder`; one reduction, GEMMs of the stacked matrices) and the stationary (Hylleraas) form, written once (`stationary`: the correction U^a . r^b that makes a property's error quadratic in the residuals, `Correction`, added by every property) |
| gradients, Hessians | `derivatives/` (`gradient`, `hessian`, `ecp`): assemblies of the core Hamiltonian's terms and the method families' terms (`terms`: one shape per derivative order, added in a fixed order); the Hessian's response part over linus-scf's `response` (the nuclear displacements, their `ResponseProblem`, `SecondOrder` with the stationary correction); what a method can compute, checked before any SCF (`capabilities`) |
| dipole moment, dipole derivatives, polarizability | `properties/`, all the electric field's code: `dipole` (the dipole integrals, once; the electric field as a `Perturbation`, `ElectricField`; the dipole moment; the dipole derivatives from a Hessian's response in their stationary form, decision D35), `polarizability` (α from the field's response in the stationary form) |
| implicit solvent (C-PCM, IEF-PCM, COSMO, SS(V)PE, SMD) | linus-solvent: the surface, the operator and its second derivatives, radii, the RI modes, SMD and its SASA, the solvent database (`db`), the SCF term (`scf`: `PcmScf`, `PcmTerm`) and its derivatives and response (`derivatives`), the models' parameters (`model`: `ResolvedPcm`); `solvent/`: the settings group and its resolution (`settings`), the SCF's term on the plan's backend (`scf`), the derivative terms (`derivatives`) |
| the CPU/GPU boundary | `backend/`: the provider trait `Backend` (one factory per method family and derivative order, each answering with the family's engine or declining with the reason), the one "device, else CPU" (`CalcContext::resolve`), the CPU's provider (`cpu`), the SCF's terms and loop as the backends set them up (`scf`), the GPU's provider (`gpu`, feature `gpu`: the device and the device session of a geometry, `session`, the SCF's density-fitting hand-off to its derivatives and its PCM device term, one owner with explicit lifetimes, emptied when the atoms move and by each new SCF; the adapters over linus-gpu's engines, `df`, `xc`, `pcm`, the device loop `scf_loop` and its controller `scf_control`, D37's exchange costing `costing`) |
| empirical dispersion | `dispersion.rs` (over `linus-disp`, whose errors it converts) |
| harmonic analysis, thermochemistry | linus-vib: pure functions on any program's Hessian (re-exported by `vib.rs`, with the analysis a Hessian carries: `FrequencyResult`, `Session::hessian_analysis`), run by the `hessian` calctype (the settings group `freq`); the constants are linus-core's `units` |
| accuracy targets | `driver/accuracy.rs`: `Targets` (the `accuracy` settings group, or a session request's own), the SCF and response thresholds they set (`scf`, `response_tol`, from the measured bounds with a factor 3 of margin), and `AccuracyRecord` (`record.accuracy`: target, thresholds, estimate, met); `CalcOptions::convergence_for` and `response_tol` apply them, the given thresholds as floors |
| the ground state of a geometry | `driver/ground.rs`: `GroundState`, one SCF at one geometry as the SCF stage leaves it (the result, what it was set up with: `ScfSetup`, its purpose, grid and memory `ScfMemory`; VV10's statistics, the solvation results with the PCM state, the DF storage), immutable and shared (`Arc`), handed explicitly to the gradient, the Hessian, the polarizability and the report, which read nothing of "the last SCF" from the session |
| the setups of one geometry | `context.rs`: `IntegralContext`, built on first use per geometry and replaced when the atoms move: one index-ordered shell-pair set at the highest derivative order a phase asked for (E is the same at every order; the Hessian frees the derivative expansions before its response), S, the XC integrators per grid, the placed auxiliary basis sets and their Coulomb metrics (linus-jk's `df::CoulombMetric`, the one builder of the factor of (P\|Q); held until the call's last phase that fits with it); the SCF, the gradient, the Hessian and the polarizability of a geometry share them (the CPU gradient and Hessian build no grid of their own) |
| a session's lineages | `driver/lineage.rs`: `LineageKey` (any JSON value's canonical text; null is the default lineage), `Lineage` (what a chain keeps from its last successful request: positions, occupied orbitals, densities, the gap at its last probe), `ScfStart` (where one SCF starts and the probes' rule: they run at a fresh start, not at a continuation unless the gap trigger fires; `GUESS_DISTANCE`, 1 bohr), `GuessRecord` (`record.scf.guess`, `record.warm_start`); `driver/session.rs`: the request (`Request`: positions, calctype, lineage, seeds, commit, accuracy targets for this request) and `Session::compute`, which starts from the lineage's state (else its first seed's) and commits it only when the whole request succeeds |
| session, settings, result | `driver/`: `Session` (its engine `CalcContext`, the last ground state, the lineages and the request running, the failures; orchestration methods only) and `MethodInfo` (mod; the SCF stage `scf_stage` returns the ground state), the settings and their grammar (`options`), the settings table every front end is generated from (`schema`), the request (`request`), the structured result (`report`), the backend resolved per family of kernels (`backend`), the device policy: the card class and every kernel family's arithmetic (`policy`), the device memory plan: every limit-derived reserve and workspace (`memory`), the host memory each phase needs and the smallest `memory_gib` that runs a calculation, one function for the plan's `memory.required_gib` and the failure before any work (`needs`: the process's baseline, the engines' own sizing functions such as `DfJk::scf_storage_needs` and `df_gradient_workspace`, the shares of the limit and the measured `RestModel`s), the run record: the decisions made at run time (`record`), the plan: the one reader of the settings, which of them take effect and the run's canonical settings (`plan`, `plan/settings`) |

## Layering

1. Each crate depends on lower layers only ("Crates"), and the compiler enforces it.
2. The method crates of layer 3 know each other only through linus-base's terms: the SCF and the
   responses (linus-scf) take trait objects, the families (linus-jk, linus-dft, linus-solvent)
   implement them, and none of them names another family, the facade or the settings. They take
   configurations (`ScfConfig`, `DfConfig`, `CosxConfig`, `ResolvedPcm`, grids), which the plan
   resolves from the settings.
3. The facade composes them: the driver resolves the settings (the plan, the only reader), the
   backend providers set up each family's engine, `derivatives/` and `properties/` assemble the
   families' terms, and the dispatch (`two_electron.rs`, `solvent/`) chooses a family's engine
   for the settings at each derivative order.
4. The front ends (CLI, Python) use the facade.

Within the facade, `CalcContext` (the session's engine: the structure, method, plan, backend,
record and the geometry's setups; never the session's state) is extended by `impl` blocks outside
`driver/`: in `derivatives/`, `properties/`, `solvent/`, `two_electron.rs` and `dispersion.rs` (the
dispatch of each order and its terms), and `Session` keeps the orchestration over its session
state there (`gradient`, `hessian_with`, `polarizability`, `ir_properties`, `frequencies`). The
backends' providers, their adapters (`backend/`) and the derivative terms take `&CalcContext`.

## Dispatch and the backend boundary

The J/K method of the settings (`JkMethod`: exact, df, rij_snk; where exact integrals are kept is
`jk.exact_storage`, direct or incore, and the hardware is the `backend` setting) is resolved in one
place, `two_electron.rs`, with one arm per derivative order:
- energies: the SCF's builder (`build_jk`) and the end of a converged SCF (`finish_scf_jk`);
- gradients: `gradient_2e`;
- Hessians: `hessian_2e`;
- responses: `response_jk`.

The arms call linus-jk's module of the approximation (`exact`, `df`, `cosx`). What a method
can do is asked of `JkMethod` itself (`aux`, `k_grid`, `max_orbital_l`,
`has_second_derivatives`), not by matching its variants elsewhere.

The traits a backend implements are linus-base's terms (`linus_base::terms`; their builds return
`Result`: a device fails with `LinusError::Backend`, the CPU builders always succeed):
- `JkBuilder`: J and K of full densities, for the SCF; a request may carry density factors
  (`DensityFactors`: symmetric C C^T, the case implemented today, or left/right pairs
  L R^T), and the densities stay authoritative;
- `ResponseJk`: J and K of first-order densities with occupied factors, projected on the
  occupied orbitals, for CPHF/CPKS. `DenseResponseJk` adapts any `JkBuilder`;
- `XcPotential`: the XC energy and potential of the SCF, from the spin densities or the
  occupied orbitals that give them (`XcDensity`; a GPU forms the density on its grid
  from the orbitals);
- `ResponseKernel`: the XC kernel of CPKS responses, dV[dD] C_o of factored first-order
  densities (`XcIntegrator::kernel_contract_occ` on the CPU);
- `EnvironmentTerm`: a density-dependent environment term of the SCF (the solvent's
  reaction field), with its own energy component and timer;
- `EnvironmentResponse`: that environment's linear response in CPHF/CPKS,
  dV[dD] C_o of factored first-order densities (the solvent's `PcmScf`).

A `JkBuilder` or `EnvironmentTerm` that builds from its own history (COSX's incremental K, the
PCM's predicted charges and incremental passes) says so after each build (`built_incrementally`)
and builds of the density alone after `reset_incremental`. The SCF counts such a build as
incremental and confirms a converged one with a full build at the same density (both loops,
from `linus_scf::control`'s `Confirm`), so a converged SCF ends on one build of its density. The
last iteration of `max_iter` is a full build in both loops, so an SCF the plateau rule accepts at
`max_iter` ends on one too.

The hardware is the `backend` setting. It is resolved once per session (the device is opened in
`Session::new`) and per family of kernels (`Family`: the SCF's J/K, XC, loop and solvent
terms, VV10, the gradient's two-electron and XC terms, the Hessian's explicit two-electron and XC
terms, the response's J/K and XC kernel). The plan places each family on the device or the CPU,
with the reason, before the work starts (`CalcContext::placement`), and `resolved.record.backend`
reports it. docs/gpu.md describes what runs where.

The CPU/GPU boundary is one provider per backend, `crate::backend`:
- `Backend`, the provider trait, has one factory per method family and derivative order. A
  factory sets its engine up and returns it as the family's own trait object (`Setup::Ready`), or
  declines with the reason (`Setup::Declined`: out of device memory, or a case the device's
  engines do not cover). The engines compute. The trait objects are the families' traits:
  - the SCF's `backend::scf::ScfJk` (a `JkBuilder` with a device's fit to the memory the other
    engines left), `ScfXc` (an `XcPotential` with VV10's statistics), its loop (`ScfLoop`) and
    the solvent's `solvent::scf::ScfEnvironment`;
  - the gradient's `derivatives::terms::GradientTerm` and the solvent's `SolventGradient`;
  - the Hessian's `HessianTerm` (the two-electron method) and linus-dft's `engine::XcHessianEngine`;
  - the response's `ResponseJk`, `xc::ResponseKernel` and VV10's `Vv10PairSweep`.
- `CalcContext::resolve` is the one "device, else CPU". The plan's placement picks the provider:
  the device's for a family placed there, else `CpuBackend`. A declined device setup is recorded
  with its reason, and the CPU's provider answers. Any other device error is the calculation's
  (`backend`, `numerical` or `internal`). A family never changes backend after its setup.
- `CpuBackend` is thin. It hands out the CPU engines the families' own modules choose: the J/K
  method's dispatch in `two_electron.rs` (`cpu_scf_jk`, `cpu_gradient_2e`, `cpu_hessian_2e`,
  `cpu_response_jk`), the XC integrator, the solvent's `PcmScf`.
- The GPU's provider is `backend::gpu` (feature `gpu`: the boundary's one conditional
  compilation, in `backend/mod.rs`; the core compiles the same with and without it). It holds
  the device and the device session of the current geometry (`session`: the SCF's
  density-fitting hand-off to its derivatives, the PCM's device term), which the core empties
  through the provider's hooks (`clear_session`, `release_handoff`). Its adapters wrap linus-gpu's
  engines in the families' traits:
  - `df`: the DF J/K of the SCF and responses, and the gradient's and the Hessian's
    two-electron terms;
  - `xc`: the SCF potential, the CPKS kernel, the gradient's and the Hessian's XC terms, VV10;
  - `pcm`: the solvent's SCF and gradient terms;
  - `scf_loop`: the device loop over the device's J/K and XC engines, which it takes back from
    the SCF's terms, or declines;
  - `costing`: decision D37's exchange costing over the device's operator plan, and decision
    D11's costs of the DF-J/K storage ladder (a shrunk device K block against streaming).

  The device loop's controller is `scf_control`, built from linus-scf's control units (public
  for it, hidden from the documentation); no crate below the facade has a `gpu` feature.
- The core traits name no GPU type. A device's own form of an environment term or a Coulomb hook
  is reached through a neutral `device_form() -> Option<&mut dyn Any>`, which only the GPU's
  adapters downcast. The kernels' coverage of the basis sets is the provider's answer
  (`Backend::uncovered`, decision D40), and the device's memory defaults come with the device
  (`Device::response_defaults`).

The gradient's schedule is its terms' own (`derivatives::gradient`, one driver for every
backend). Each `GradientTerm` says where it runs (`Lane`):
- a device's term does its host setup before the overlap, on the whole pool (the XC engine's
  layout), and its device work beside the CPU's one-electron pass;
- a CPU term runs after the pass.

A device term that declines (its setup or a call does not fit) is recorded and replaced by the
CPU's term, after the overlap. Each part is computed on its own, so the bits do not depend on the
schedule.

What each family computes in is decided once per session, when the device opens, by the
device policy (`driver/policy.rs`, `DevicePolicy`): the card class (`CardClass`, the only
reader of the device's FP64:INT8 classification, decision D27, derived with the costing's rates
from the properties the device's provider reports, `CardProps`) and the `precision` settings
(`precision.fp64` for the exchange families, `precision.fp32` for the families with FP32 paths)
give every family its `linus_core::Arith`, an FP64 emulation as a fallback chain (Ozaki scheme,
cuBLAS emulation, FP64), and the card's choices inside engines (native K's dense planes, the XC
products' engine) as engine options. A response's arithmetic is the policy's too
(`DevicePolicy::response`: what it computes in, D23 and D37), and its J/K's setup hands the CPKS
kernel what it serves (`two_electron::KernelUse`) explicitly. The device's adapters turn the
policy's decisions into engine settings: one `apply_chain` for every exchange's fallback chain
and a response's recipe (`backend::gpu::df::ResponseRecipe`: its chain, the stored operator's
precision, D37's exchange costing on FP64-poor cards in `backend::gpu::costing`) in
`backend::gpu::df`, the semilocal CPKS kernel's arithmetic in `backend::gpu::xc`. Device-only
data and logic live in `backend::gpu`, so the core compiles without the `gpu` feature with
nothing dead in it. The engines take the `Arith` of their family and never read the card or the
settings. Every device reserve and workspace derived from the device's
limit is the memory plan's (`driver/memory.rs`, `DeviceMemoryPlan`, one rule per item), and
every free-memory check is linus-core's `Budget` (`Gpu::budget`). An engine's planned bytes
can be granted (`Gpu::grant`, a `linus_gpu::pool::Grant`): counted as in use from the
decision on, drawn by the engine's allocations, the rest released on drop, so nothing
allocated in between can invalidate the decision; a draw beyond the grant is a plan
violation, recorded (`plan_violation` in the record and a warning). The response's stored
exchange operator (granted at its setup, built at the first product) and the DF tensor B are
granted; the other engines still decide on the memory free at their setup.

Every setting is declared once (`driver/schema.rs`'s `settings_group!`, used by `driver/options.rs`
and `solvent/settings.rs`): its key, type, default, tier, concern, unit, the lowest calctype it
applies to and its doc comment. The declaration makes the group's struct, whose fields are
`Option`s (explicit presence: what was given is what the D41 checks read, alike for JSON and
for settings built in Rust), the accessors that apply the defaults, the JSON reader (keys
normalized, unknown keys an `unknown_key` error with a suggestion) and the metadata table. The
JSON Schema (`linus schema settings`, Python's `settings_schema()`), `--list-settings`, the
settings tables of docs/cli.md and linus-py's `settings.py` stubs are generated from the table
(a test keeps the committed ones current). A request (`driver/request.rs`: the structure, the
calctype, the settings) is read from JSON by the library for every front end.

The settings have one reader, the session's plan (`driver/plan`): the request is a private
field of `Plan`, and the rest of the crate asks the plan for what it needs, resolved where a
rule resolves it (a purpose's auxiliary basis and grids, the SCF's thresholds); the engines,
the SCF and the derivatives get configurations, never the settings. A unit test
(`plan_is_the_only_reader`) keeps it so. Before a calculation runs, the plan reports what it
will do (`driver/plan/report.rs`, `linus --plan`): the settings it decides (the ones only the
run decides stay `auto`, listed in `deferred`), placement, arithmetic, the memory plan, the
sizes, and `driver/plan/estimate.rs`'s resource estimates (peak host and device memory, the
threads used well), each a model of the sizes fitted to measured peaks, with its error band.

What a calculation reports is generated, not reconstructed: the run record
(`driver/record.rs`, a `linus_core::RunRecord`) keeps the decisions made at run time as they
are made (each family's device setup and its fallback, the SCF's probes; what only the device
decides in its setups, the FP64 emulation that ran and its engines' workspace grants, is its
provider's record, `Backend::emulation` and `Backend::grant_violations`), and `resolved` is
built from it and the settings: `settings` is the run's canonical
settings, only the keys that take effect (`driver/plan/settings.rs`'s table of every key's
applicability, decision D41) with every `auto` as the value it took, a valid request that
reproduces the run; `reasons` says why; `record` is what happened. The report's `warnings`
(`{code, message, path}`) come from the same record and the run's results; the library prints
nothing, the CLI prints them.

The exact exchange of the method is one operator (`MethodInfo::exchange`), split for the
engines by `terms::exchange_parts`; the SCF's builder gets it in `build_jk`.

A density-fitting setup names its storage with a `df::DfPlan` (in core, compressed formats
included; one-sided; raw integrals; direct), chosen from the memory budget, and reports it as
a `df::ResolvedDf` (`resolved.record.df`); the GPU's tensor is the plans `Device` and
`HostStreamed`, and out-of-core storage would be a further one.

A new J/K approximation adds a module to linus-jk, one arm per order in the facade's
dispatch, and its answers to the capability methods. Gradients and Hessians take the method
families' derivative terms through `derivatives::terms`, each from the backend the plan puts its
family on:
- a Hessian's `HessianTerm`s, added in a fixed order into `ExplicitHessian`: the two-electron
  method's with its response J/K, the solvent's with its `EnvironmentResponse`, and the XC
  engine's (the semilocal terms of an `XcHessianEngine`, the device's or the integrator's, then
  VV10's, all in the shape `XcHessianTerms` of V^x C_o projected inside xc/);
- a gradient's `GradientTerm`s of the XC engine and the two-electron method, run by the
  gradient's one schedule (above).

The XC integrator comes from the one factory (`XcIntegrator::build`, the session's `xc_on`);
responses take `xc::engine::XcKernel`, the semilocal kernel plus VV10's. The one-electron
integrals have no term yet: the assemblies call `linus_ints` directly.

## Generated code

Generated files say so in their first lines. Edit the generator, never the output.
`#[rustfmt::skip]` keeps rustfmt off them, and .gitattributes marks them `linguist-generated`.

| files | generator |
|---|---|
| `crates/linus-ints/src/generated/` (Hermite R tensors, class-specialized and tiled ERI kernels) | `tools/intgen/intgen.py`: math (`mdmath.py`), expression IR and schedules (`ir.py`), Rust emitters (`rtensor_gen.py`, `classes_gen.py`, `tiled_gen.py`); `--check` verifies the files |
| `crates/linus-gpu/src/kernels/eri3c_gen/` (fused three-center CUDA kernels, first and second derivatives) | `tools/intgen/intgen.py cuda` (`cuda_eri3c.py`; schedules in `tools/intgen/eri3c_schedules.json`, chosen by linus-gpu's `eri3c_tune` example) |
| `crates/linus-gpu/src/kernels/pcm_gen/` (PCM surface-charge CUDA kernels and first derivatives) | `tools/intgen/intgen.py pcm` (`cuda_pcm.py`) |
| `crates/linus-ints/src/core/rys_tables.rs`, `core/rys_ref.txt` | `tools/gen_rys_tables.py` |
| `crates/linus-grid/src/*_data.rs` (Lebedev, element and preset tables) | `crates/linus-grid/tools/gen_*.py` |
| `crates/linus-disp/src/data/` (D3, D4, EEQ parameters) | `crates/linus-disp/tools/gen_data.py` |
