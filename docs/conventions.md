# Conventions

The names, units and layouts linus uses everywhere: in the command line and its JSON, the
Python package, qcdata results and the Rust code. [cli.md](cli.md) lists the settings and the
result keys; [ARCHITECTURE.md](../ARCHITECTURE.md) maps the concepts onto the code.

## Units

Atomic units in every interface, unless the community has a strong convention for a quantity; then
that convention, the same in the command line, JSON, Python and Rust. Every setting declares its unit
(`x-unit` in `linus schema settings`, shown by `linus --list-settings`). A new setting or result key
uses atomic units unless one of the conventions below applies, and its unit goes into its declaration.

| quantity | unit |
|---|---|
| energies | hartree (Eh) |
| positions | bohr (an XYZ file is read in Å, with the CODATA 2022 bohr radius 0.529177210544 Å) |
| gradients | Eh/bohr |
| Hessians | Eh/bohr² |
| dipole moments | e·bohr, about the coordinate origin (1 e·bohr = 2.541746473 D) |
| dipole derivatives (APT) | e (dμ/dX) |
| polarizabilities | e² bohr² / Eh |
| frequencies | cm⁻¹ (imaginary as negative numbers) |
| IR intensities | km/mol |
| force constants | mdyn/Å |
| masses | u |
| temperatures, pressures | K, Pa |
| standard-state concentration | mol/L |
| the qRRHO averaging moment of inertia (`freq.qrrho_bav`) | kg m² |
| solvation radii, probe radius, SMD contact smoothing | Å |
| entropies, heat capacities | Eh/K |
| times | wall-clock seconds |
| memory | `memory_gib` is in GiB (2³⁰ bytes, as qcengine's `memory`); every budget inside linus is in bytes, and only the result converts: `*_gib` in GiB, and a few older keys with their own units (`hbar_mb` in 10⁶ bytes, `xc_kernel_mb` and `response_slot_mb` in MiB, `mem_hwm_gb_*`; the units table in [cli.md](cli.md#result)) |

Why the exceptions:
- Frequencies (cm⁻¹), IR intensities (km/mol), force constants (mdyn/Å) and masses (u) are what
  programs and papers report.
- No program uses atomic units for temperature (1 a.u. = 315 775 K) or pressure (1 atm ≈ 3.4×10⁻⁹
  a.u.). Pressure is in Pa, as in Psi4, PySCF and ASE (Gaussian and ORCA use atm): it is SI and
  writes both standard states exactly, 1 atm = 101 325 Pa and 1 bar = 100 000 Pa.
- The qRRHO averaging moment is given as published (Grimme's 10⁻⁴⁴ kg m²).
- Solvation radii are tabulated in Å in the literature (Bondi, SMD).
- The text of an XYZ file is in Å, the format's own convention.
- Memory follows qcengine (GiB); times are wall-clock seconds.

## Atomic orbitals

Shells come in the order of the basis set file per atom, atoms in input order
(a frame's `results.wavefunction.basis` with the request output `orbitals` or an AO matrix, a
`qcdata.BasisSet` in qcdata's default AO convention). Within a shell:

- **Spherical (pure) functions** (the default for l ≥ 2): real solid harmonics with m = -l..l,
  except p shells, which keep the Cartesian order x, y, z. This is PySCF's (libcint's) order.
  The harmonics are unit-normalized, without a Condon-Shortley phase.
- **Cartesian components** (`cartesian: true`, `--cartesian`): lexicographic, xx, xy, xz, yy, yz, zz
  (l_x descending, then l_y descending), as libint, libcint and PySCF.
- **Normalization**: every spherical function and the x^l, y^l, z^l Cartesian components have
  unit norm; the other Cartesian components carry the norm of x^l (xy has norm 1/√3).
  PySCF's Cartesian functions differ by √(4π/(2l+1)) for l ≥ 2.
- Basis set data are Basis Set Exchange JSON files; general contractions are split into one
  shell per contracted function.

qcdata's `Wavefunction.with_convention` converts the orbitals and the AO matrices together
between these conventions and those of Molden, libint (CCA) and TREXIO (`qcdata.AOConvention`);
`linus.convert` writes Molden files and converts to and from PySCF.

## Spin

- A calculation is restricted (RHF/RKS) for closed shells and unrestricted (UHF/UKS) for open
  shells, unless `reference` says otherwise; ROHF is not implemented.
- Results name the spins `_alpha` and `_beta`: `orbital_energies_alpha`, `occupations_beta`,
  `mo_coefficients_alpha`, `hamiltonian_alpha` (qcdata's `Orbitals` and `Wavefunction` use `_a` and
  `_b`: `energies_a`, `occupations_b`, `hamiltonian_a`).
- Spin densities: for a closed shell both are half the total density.
- Occupations are per spin, 0 or 1.
- Internally a calculation works on spin *channels*: one channel for a closed shell (the alpha
  orbitals, occupation 2), two for an open shell (occupation 1 each). Per-perturbation
  quantities are laid out `x[channel][perturbation]`.

## Geometry and atoms

- `positions` (n × 3, bohr) and `numbers` (atomic numbers) in every API and JSON; `xyz` only
  for the text of an XYZ file (Å). Inside the Rust code atomic numbers are `z`.
- linus supports H to Rn (Z = 1-86). Atoms closer than 0.05 bohr are rejected as coincident.
- Atoms with effective core potentials (the def2 sets from Rb on) carry `ecp_core_electrons`;
  the nuclear charge the electrons and the other nuclei see is then Z minus those electrons.

## Matrices and arrays

- Gradients are n × 3 lists of rows. The CLI's `hessian` is one flat row-major list of (3n)²
  values, index (3a + i)·3n + 3b + j (atoms a, b; Cartesian components i, j); Python's is a
  3n × 3n array.
- Orbital coefficient matrices are nbf × nmo, one orbital per column, ordered by energy.
- Dipole derivatives are 3n × 3: row 3a + i, column the dipole component.

## Errors

One vocabulary of error kinds across the CLI (`error_kind`), Python (the exceptions' `kind`)
and qcdata (`extras["error_kind"]`):

| kind | meaning | Python | CLI exit code |
|---|---|---|---|
| `input` | settings or a molecule linus cannot run as given | `InputError` (a `ValueError`) | 2 |
| `basis` | a basis set file that cannot be read or parsed | `InputError` | 2 |
| `convergence` | the SCF or a CPHF/CPKS solve did not converge (the result so far is attached) | `ConvergenceError` | 1 |
| `unsupported` | a valid request linus does not implement | `UnsupportedError` (a `NotImplementedError`) | 1 |
| `resource` | not enough memory (or another resource) as set up | `ResourceError` (a `MemoryError`) | 1 |
| `backend` | the GPU backend failed: no usable device for `backend: "gpu"`, or a device error during the calculation (it never continues on the CPU midway) | `BackendError` | 1 |
| `numerical` | a numerical failure (non-finite energies, singular matrices) | `NumericalError` (an `ArithmeticError`) | 1 |
| `internal` | a bug in linus (a broken invariant or a panic) | `LinusError` | 1 |
| `stopped` | a runner asked the calculation to stop (`ctx.stop_requested()`); it ended at its next step with what it reached (docs/python.md, "Stopping") | `LinusError` | (never: the command line does not stop) |

Within a kind, a code says which failure it is, with qcdata's shared names (CLI `error_code`, the
exceptions' `code`): a setting's `unknown_key`, `no_effect` and `not_applicable` (kind `input`);
`host_memory` (the host's memory: the DF tensors, the in-core integrals, the solvent's matrices, a
Hessian's arrays) and `device_memory` (a GPU's) for kind `resource`; `scf_not_converged` (the
SCF ran out of iterations; the code every qc* program gives it) and `response_not_converged` (a
CPHF/CPKS solve stopped above its tolerance) for kind `convergence`; `device_lost` for a GPU fault
that poisoned the device context (kind `backend`, Python only).

## Glossary

One name per concept, in the Rust code, Python, the JSON and the documentation.

| concept | name | instead of |
|---|---|---|
| density fitting (method, options, modules) | `df` | "ri", except in literature acronyms (RI-J, RIJCOSX) and as input aliases |
| auxiliary basis | `aux` | |
| J/K method and builders | `jk` (the setting); Rust types `Jk*` (`JkBuilder`, `ResponseJk`, the density-fitting `DfJk`) | `JK*` casing, `RiJK` |
| exact four-center integrals and where they are kept | `jk: "exact"`, `jk.exact_storage` (`direct`, `incore`) | `jk: "incore"` as a method of its own (still accepted as input) |
| seminumerical exchange | `rij_snk` (the method), `cosx` (its grid schemes and module) | "sn-K" |
| the hardware the kernels run on | `backend` (`cpu`, `gpu`, `auto`); a family of kernels it resolves on its own (`jk`, `xc`, `gradient_jk`, `response_jk`, ...); "device" for the GPU itself | "accelerator" |
| what hands out a family's engine on one backend | "provider": Rust `Backend` (`crate::backend`; `CpuBackend`, the GPU's `GpuBackend`), one factory per family and derivative order, answering `Setup::Ready` (the engine) or `Setup::Declined` (the reason); `CalcContext::resolve` asks the one the plan places the family on | "try GPU, else CPU" at the call site; enums over the backends |
| exact-exchange operator a/r + b·erf(ωr)/r | `Operator`, variable `xop` (an engine splits it with `terms::exchange_parts`), function suffix `_op` for functions that take one | `_rs`, `rs`, the `(a_x, rs)` pair |
| range-separated hybrid | "RSH" in prose only | |
| derivatives | `gradient`, `hessian` | `grad_*`/`hess_*` outside module prefixes |
| linear response | `response` (settings, traits); `cphf_*` only for CPHF quantities; the CPKS kernel grid is `grid.kernel` | `cpks_grid`, `hess_grid_response`, `response.grid`, `response.grid_response` (now `grid.kernel` and `grid.response`) |
| the determinant derivatives and responses expand around | "reference": Rust `Reference` (built once per SCF result; an `OrbitalBlock` per spin channel, the total and energy-weighted densities) | per-consumer channel copies (`cfull`, `co`, `cv` rebuilt from `ScfResult`) |
| the induced Fock matrix of a response | "response operator": Rust `ResponseOperator`, composed from the method families' terms in a fixed order (J/K, XC kernel, environment); an environment's term is an `EnvironmentResponse` | `Response` with mutable members |
| the CPHF/CPKS equations of a response | `ResponseProblem` (the orbitals' zeroth-order part and the operator); its solvers are `response::solver`'s | |
| a static perturbation of the reference | "perturbation": Rust `Perturbation` (`response::perturbation`: F^k C_o at fixed density, the overlap derivative's occupied projections `Overlap`, the right-hand sides); `NuclearDisplacements`, `ElectricField` | per-property right-hand sides and S^x blocks |
| the stationary (Hylleraas) form of a second-order property | "stationary correction" U^a · r^b with the final residuals: `response::stationary` (`Correction`, added by every property) | per-property corrections |
| the first-order occupied orbitals C1 = C_v U - ½ C_o s1oo and orbital energies e1 of a response | `FirstOrderOrbitals` (`response::second_order`, with the assembly `SecondOrder` of E^{ba}) | |
| a method family's contribution at one derivative order | "term": `HessianTerm`, `GradientTerm` (with its `Lane` in the gradient's schedule; `derivatives::terms`), the XC engine's semilocal `XcHessianEngine` | hand-written accumulation per source |
| SCF settings | `scf` (`scf.convergence`, `scf.e_tol`, `scf.g_tol`, `scf.max_iter`, ...) | top-level `convergence`, `e_tol`, `g_tol`, `max_iter`, `guess`; `e_conv`, `g_conv` |
| spin | `_alpha`/`_beta` in results; "channel" for the one-or-two layout; `_restricted`/`_unrestricted` for the one- and two-channel implementations behind a channel entry point | `_a`/`_b` in linus's own results and extras; `rks`/`uks`/`rhf`/`uhf` in function names |
| geometry | `positions` (bohr), also in the Rust code | `coords`, `set_coords` |
| atoms | `numbers` | `z` in the public API |
| settings types | `*Options`: the options of an entry point (`CalcOptions`, `FrequencyOptions`, `VibOptions`, `ThermoOptions`); `*Settings`: a group of `CalcOptions` (`ScfSettings`, `DfSettings`, `SolventSettings`, ...); `*Config`: the internal parameters of an engine or builder (`ScfConfig`, `DfConfig`, `CosxConfig`); `Resolved*`: what a run used (`Resolved`, `ResolvedDf`, `ResolvedPcm`) | `*Options` for internal parameters, `*Settings` for resolved records |
| the whole result of a calculation | Rust `Report`; Python: qcdata's `Frame` (a session request) and `ProgramOutput` (`linus.compute`) | Python `ScfResult` or `CalcResult` for the whole result |
| vibrations | the `hessian` calctype carries the analysis (no separate frequencies calctype); the settings group `freq`, `FrequencyOptions`, `VibrationalAnalysis`, `Thermochemistry` (a result's `thermochemistry` is a list, one per temperature); result fields as qcdata's (`normal_modes_cartesian`, `point_group`, `zero_point_energy`) | a `frequencies` calctype, `FrequencyResult`, `FreqOptions`, `FreqResult`, `VibAnalysis`, `zpe`, `modes_cartesian` |
| one SCF at one geometry, handed to its derivatives | `GroundState` (Rust; with `ScfSetup`: what the SCF was set up with) | "the last SCF" as session state |
| the state carried from one SCF of a task to the next | `Lineage` (the warm start, the probes' state) | |
| the symmetry the stability searches run in | the converged state's abelian point group (D2h and its subgroups; `symmetry` in the records) and its irreps (`irrep`: the Mulliken symbol, `irrep_index`): linus-base's `symmetry::abelian::AbelianGroup` and `symmetry::ao::AoGroup` (the group on a basis set's functions), linus-scf's `stability::PairSymmetry` (the group on a converged state's orbital rotations) and `stability::Starts` (one start per irrep) | "symmetry block", "point group" for the abelian subgroup |
| the setups of one geometry shared by its phases | `IntegralContext` (pair sets, S, XC integrators, auxiliary basis sets, Coulomb metrics) | per-phase rebuilds |
| the device state the phases of one geometry hand on | `DeviceSession` (the GPU provider's, `backend::gpu::session`) | "hand-off" for the whole of it |
| a session (Rust): one molecule, model and settings at many geometries | `Session` (its engine, the lineages, the request running; `linus::Session`, Python's `linus.Session` over it) | `Calculator` |
| what a session computes with (its engine) | `CalcContext` (structure, method, plan, backend, record, the geometry's setups) | engines taking `&Session` |
| memory | `memory_gib` (GiB); bytes inside; new result keys in `_gib` | new `_mb`/`_gb` keys (the existing ones stay, with their units documented) |
