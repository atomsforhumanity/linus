"""The settings of a linus calculation as typed dictionaries (generated from the
library's settings table by `cargo test -p linus generated_files` with
LINUS_UPDATE_GENERATED=1; do not edit). Keys and values as in docs/cli.md;
`linus.settings_schema()` returns the JSON Schema."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

__all__ = ["JkSettings", "GridSettings", "PrecisionSettings", "SolventSettings", "ScfSettings", "ResponseSettings", "FreqSettings", "BackendSettings", "OutputSettings", "AccuracySettings", "Settings"]


class JkSettings(TypedDict, total=False):
    """The J/K approximation."""

    kind: Literal["exact", "df", "rij_snk"]
    """The approximation: `df` (density fitting of J and K with the calculation's auxiliary basis, `aux`; synonyms `ri`, `rijk`), `exact` four-center integrals, or `rij_snk` (density-fitted J with seminumerical K, ORCA's RIJCOSX; synonym `rijcosx`)."""
    aux: str
    """The auxiliary basis of `df` and of `rij_snk`'s J. `auto`: def2-universal-jkfit for energies and gradients, `cc-pvtz-jkfit|def2-universal-jkfit` for Hessians (df); def2-universal-jfit (rij_snk). `"a|b"` takes each element from the first set that has it."""
    k_grid: str | int | dict[str, Any]
    """The grid of `rij_snk`'s seminumerical exchange: `cosx:N` (ORCA-like COSX grids of DefGrid N), `gridx:A,B,C` (COSX grid levels of the first and last SCF iterations and the final energy) or any grid. `auto`: `cosx:2`."""
    screening: float
    """Screening threshold of the exact four-center integrals (derivatives use at most 1e-12)."""
    exact_storage: Literal["auto", "direct", "incore"]
    """Where `exact` keeps its integrals: `direct` (recomputed in every Fock build) or `incore` (all held in memory: tiny systems and tests). `auto`: direct."""
    df_storage: Literal["auto", "f64", "f48", "f40", "f32"]
    """Storage format of `df`'s three-index exchange tensor: `f64`, or the compressed `f48` (relative error at most 9e-13), `f40` (2e-10), `f32` (6e-8). `auto`: f64; a memory budget too small for it is a `host_memory` error that names the `memory_gib` needed (the compressed formats only when set, so memory never changes the arithmetic)."""
    df_long_range: Literal["auto", "stored", "one_sided"]
    """Long-range exchange of range-separated hybrids with `df`: `stored` (B and B^omega in memory) or `one_sided` (raw integrals of one side per build). `auto`: stored when both fit the budget."""
    incremental: bool
    """Incremental Fock builds (from the change of the density) for integral-direct exact J/K and the seminumerical exchange."""
    incremental_reset: int
    """A full rebuild after this many incremental builds."""
    grid_switch: float
    """The seminumerical exchange moves to its next (finer) grid once max|F_ai| is below this."""
    grid_switch_shift: bool
    """At a grid switch, shift the DIIS subspace's Fock matrices by the change of the exchange matrix."""


class GridSettings(TypedDict, total=False):
    """The grids of DFT."""

    xc: str | int | dict[str, Any]
    """The XC quadrature of the SCF, gradients and Hessians: a PySCF level `N`/`pyscf:N` (0-9), `linus:N` (1-4), `orca:N` (ORCA 6.0.0's DefGrid 1-3), `psi4`, or a grid's full specification `{"kind": "spec", ...}`. `auto`: pyscf:3 for energies and gradients, linus:3 for Hessians."""
    nlc: str | int | dict[str, Any]
    """The grid of VV10 nonlocal correlation (-V functionals). `auto`: by the XC grid's family: linus:1 with the default and linus grids, pyscf:3 with PySCF grids, ORCA's NL grid with `orca:N`, Psi4's VV10 grid with `psi4`."""
    kernel: str | int | dict[str, Any]
    """The grid of the XC kernel in the CPKS equations of Hessians. `auto`: by the XC grid's family: linus:1 (meta-GGAs linus:2) with the default and linus grids, the XC grid with PySCF, ORCA and Psi4 grids."""
    response: bool
    """The grids move with the atoms in gradients, Hessians and dipole derivatives (the XC, VV10 and seminumerical-exchange grids' weight derivatives and point motion, as ORCA); false: fixed grids, PySCF's and Psi4's convention, for comparisons with them."""


class PrecisionSettings(TypedDict, total=False):
    """The arithmetic."""

    fp64: Literal["auto", "native", "emulated"]
    """How the GPU reaches FP64 accuracy in the density-fitted exchange GEMMs: `native` (the FP64 units) or `emulated` (FP64 emulation on INT8 tensor cores). `auto`: native on FP64-rich cards (A100, H100/H200) and the CPU, emulated on the others (D27)."""
    fp32: Literal["auto", "off", "on"]
    """Where FP32 arithmetic may run outside the SCF loop: `off` (nowhere: FP64 throughout, VV10's pair sums too) or `on` (also the GPU's XC gradient, CPKS kernel, Hessian XC and VV10 gradient products). `auto`: linus's validated uses (the single-precision exchange of large RI responses, the GPU's CPKS and VV10 kernels with it, VV10's pair sums in double-float)."""
    response: Literal["auto", "fp32", "fp64"]
    """The arithmetic of the RI response exchange (CPHF/CPKS): `fp32` or `fp64` (synonyms `single`, `double`). `auto`: as `fp32` says (with `off`, fp64; else single precision in large RI Hessians unless `response.tol` is below 1e-6; on the GPU, the card's rule, docs/gpu.md)."""


class SolventSettings(TypedDict, total=False):
    """The implicit solvent (absent: gas phase)."""

    name: str
    """The solvent, by name (`--list-solvents`; the rest of the CRC table through `LINUS_SOLVENT_TABLE`): its permittivity and, for SMD, its descriptors."""
    eps: float
    """The static permittivity (overrides the named solvent's; at least 1)."""
    model: Literal["cpcm", "cosmo", "iefpcm", "ssvpe", "smd"]
    """The continuum model: `cpcm`, `cosmo`, `iefpcm`, `ssvpe` or `smd` (needs a solvent name with SMD descriptors)."""
    preset: Literal["pyscf", "orca"]
    """The cavity and discretization defaults: `pyscf` (PySCF 2.15's pyscf.solvent.pcm) or `orca` (ORCA 6.0.0's C-PCM)."""
    df: bool
    """The density-fitted PCM: the surface terms through the auxiliary basis of the density-fitted J (about up to 4% of the electrostatic solvation energy; no Hessians or response properties)."""
    lebedev: Literal["auto"] | int
    """Lebedev points per sphere. `auto`: 302 (SMD 590) with the pyscf preset, by point density with orca."""
    discretization: Literal["auto"] | Literal["swig", "iswig"]
    """The switching of the surface: `swig` or `iswig`. `auto`: by the preset (pyscf swig, orca iswig)."""
    radii: dict[str, float]
    """Per-element radii before scaling, by symbol or atomic number (`{"H": 1.2}`); they override the preset's table."""
    radii_scale: Literal["auto"] | float
    """The scale of the radii. `auto`: 1.2 (SMD 1.0)."""
    r_probe: float
    """A probe radius added to the scaled radii."""
    cosmo_x: float
    """COSMO's x in f = (eps - 1) / (eps + x)."""
    cds_contact_smoothing: Literal["auto"] | float
    """SMD: the window over which new sphere contacts of the CDS surface area are smoothed. `auto`: 0 (the exact area) for single points, 0.1 for gradients and Hessians."""
    scf_terms: Literal["auto"] | Literal["exact", "predicted"]
    """How the SCF evaluates the surface terms: `exact` (two passes of exact integrals per iteration) or `predicted` (one fused pass with predicted charges; the same converged result). `auto`: predicted with density-fitted J, else exact."""
    incremental: bool
    """The predicted mode's passes on the changes of the density and the charges."""
    screening: float
    """The screening threshold of the surface-charge integrals."""
    rys_gradient: Literal["auto"] | bool
    """Gradients: the surface-integral derivatives on the batched Rys kernels rather than the McMurchie-Davidson reference. `auto`: Rys in every SCF mode."""


class ScfSettings(TypedDict, total=False):
    """The SCF."""

    convergence: Literal["auto"] | Literal["loose", "normal", "tight", "verytight", "extreme", "pyscf"]
    """A preset of `e_tol` and `g_tol`: `loose` (1e-5, 1e-4), `normal` (1e-6, 1e-5), `tight` (1e-8, 5e-7), `verytight` (1e-9, 1e-8), `extreme` (1e-12, 1e-10) or `pyscf` (PySCF's defaults, on PySCF's gradient norm). `auto`: tight for energies, 1e-9 and 1e-7 for gradients, 1e-9 and 1e-9 for Hessians. The presets are absolute: one looser than the calculation's default in a threshold warns (`convergence_looser_than_default`; `tight` loosens a gradient, only `extreme` tightens a Hessian)."""
    e_tol: Literal["auto"] | float
    """The energy change between iterations below which the SCF may stop (overrides the preset's). `auto`: the preset's."""
    g_tol: Literal["auto"] | float
    """The orbital gradient max|F_ai| below which the SCF may stop (PySCF's norm with the `pyscf` preset; overrides the preset's). `auto`: the preset's."""
    max_iter: int
    """Iterations at most: the SCF's whole budget, the escalation's included."""
    guess: Literal["sad", "core"]
    """The initial guess: `sad` (superposition of atomic densities) or `core`."""
    stability: Literal["auto", "off", "always"]
    """The stability of UHF/UKS determinants: `auto` (the probes below, and a full analysis with descent when one fires or the SCF looks like a saddle point), `off` (no probe or analysis) or `always` (a full analysis of every converged determinant)."""
    stability_probe: Literal["auto"] | int
    """Davidson products of the open-shell probe of a converged UHF/UKS determinant (0: off, else at least 2; a negative mode starts the full analysis). `auto`: 2 at a session's first geometry, none at later ones unless the gap trigger fires."""
    stability_closed_probe: Literal["auto"] | int
    """Davidson products of the closed-shell probe of a converged RHF/RKS point whose HOMO-LUMO gap is below `stability_closed_gap` (towards a broken-symmetry UHF/UKS solution; a warning only, the result is unchanged; 0: off, else at least 2). `auto`: 2 for Kohn-Sham DFT at a session's first geometry, none for Hartree-Fock."""
    stability_closed_gap: float
    """The HOMO-LUMO gap below which the closed-shell probe runs."""
    stability_max: int
    """Full stability analyses per SCF at most."""
    stability_saddle_check: float
    """A converged UHF/UKS energy this far above the lowest one the SCF visited triggers a stability analysis (0: off)."""
    stability_hessian_check: int
    """Davidson products per irrep of the check of the lowest orbital-Hessian eigenvalue before an open-shell Hessian: the stability probe of every converged point of its SCF runs with at least these (a negative mode is followed, as the probe's is; a degenerate solution, or a negative mode no step lowers, is flagged with a warning; 0: off, else at least 2)."""
    diis_space: int
    """The size of the DIIS subspace."""
    diis_start: int
    """The first iteration whose Fock matrix enters DIIS."""
    diis_drop: Literal["largest_error", "oldest"]
    """Which entry leaves a full DIIS subspace: `largest_error` or `oldest`."""
    diis_unpin: float
    """With `largest_error`: once a new entry's error exceeds this times the smallest, the subspace drops its oldest entries from then on (0: never)."""
    adiis: bool
    """ADIIS while max|[F, D]| is above `adiis_until`, then DIIS."""
    adiis_until: float
    """The orbital gradient below which ADIIS hands over to DIIS."""
    level_shift: float
    """The level shift of the virtual orbitals while max|F_ai| is above `level_shift_until`."""
    level_shift_until: float
    """The orbital gradient below which the level shift stops."""
    damping: float
    """Fock damping F = (1 - d) F + d F_previous while max|F_ai| is above `damping_until` (in [0, 1))."""
    damping_until: float
    """The orbital gradient below which damping stops."""
    soscf: bool
    """Quasi-Newton orbital rotations (SOSCF) instead of DIIS once max|F_ai| is below `soscf_from`."""
    soscf_from: float
    """The orbital gradient from which SOSCF takes over."""
    two_cycle: bool
    """When the energy cycles between two determinants, quasi-Newton steps replace DIIS (once per SCF)."""
    plateau_window: int
    """An SCF whose energy and orbital gradient stayed on a plateau over this many iterations when its escalation comes due, or at `max_iter`, is accepted with the warning `loosely_converged`; under accuracy targets only when the estimates of its final orbital gradient meet them (0: off)."""
    escalate: bool
    """An SCF whose DIIS stagnates (`stagnation_window`), or that has spent 2/3 of `max_iter`, restarts once from the lowest-energy orbitals it visited, with quasi-Newton steps, for the rest of `max_iter`."""
    stagnation_window: int
    """Iterations without a new lowest energy after which the escalation starts (0: only at 2/3 of `max_iter`)."""
    lindep: float
    """Canonical orthogonalization: overlap eigenvalues at or below this (absolute) drop their functions; fewer orbitals left than occupied ones is an input error (in (0, 1))."""


class ResponseSettings(TypedDict, total=False):
    """The response equations of Hessians."""

    tol: Literal["auto"] | float
    """The max-abs residual of the Hessian's nuclear CPHF/CPKS (the Hessian's error is quadratic in it). `auto`: 1e-6, 1e-5 in two-level responses."""
    field_tol: float
    """The max-abs residual of the electric-field CPHF/CPKS of IR intensities and the polarizability."""
    iteration_aux: str
    """The auxiliary basis of a cheaper J/K for the CPKS iterations of large RI Hessians: `none`, `main` (the main auxiliary basis) or a basis set. `auto`: two levels for a large single-precision response."""


class FreqSettings(TypedDict, total=False):
    """The analysis and thermochemistry a Hessian carries."""

    analysis: Literal["harmonic", "off"]
    """The harmonic analysis: `harmonic` or `off` (the Hessian alone)."""
    thermo: Literal["rrho", "qrrho", "qrrho-hg", "quasiharmonic", "off"]
    """The thermochemistry: `rrho` (harmonic), `qrrho` (Grimme 2012's quasi-RRHO entropy), `qrrho-hg` (that and Li and Head-Gordon 2015's quasi-RRHO enthalpy), `quasiharmonic` (Truhlar 2011: real frequencies below `raise_below` raised to it) or `off`."""
    ir: bool
    """Dipole derivatives and IR intensities (from the Hessian's CPHF solutions) and the static polarizability (an electric-field CPHF)."""
    temperature: float | list[float]
    """The temperature of the thermochemistry, or a list of them (one thermochemistry each, in order)."""
    pressure: float
    """The pressure of the ideal-gas standard state."""
    concentration: float
    """A concentration standard state instead of the pressure's (1 for 1 M)."""
    scale: float
    """The factor the frequencies are scaled by in the thermochemistry."""
    rotor_cutoff: float
    """The free-rotor interpolation's crossover of the quasi-RRHO models (`qrrho`, `qrrho-hg`)."""
    raise_below: float
    """Quasiharmonic: real frequencies below this are raised to it."""
    qrrho_alpha: float
    """The exponent of Grimme's interpolation weight (quasi-RRHO models)."""
    qrrho_bav: float
    """Grimme's average moment of inertia B_av (quasi-RRHO models)."""
    imaginary_as_real: float
    """Imaginary modes smaller than this are treated as real modes of |nu| in the thermochemistry (0: none)."""
    electronic_degeneracy: Literal["auto"] | float
    """The electronic degeneracy. `auto`: the spin multiplicity."""
    mass_table: Literal["isotope", "average"]
    """The table of the default masses: `isotope` (the most abundant isotope) or `average` (standard atomic weights)."""
    masses: list[float]
    """The masses of all atoms, replacing the table's."""
    isotopes: list[tuple[int, float]]
    """Single atoms' masses, applied last: [[atom index, mass], ...] (deuterium 2.01410177812)."""
    projection: Literal["translations_rotations", "translations", "none"]
    """What is projected out of the mass-weighted Hessian: `translations_rotations`, `translations` or `none` (the translation- and rotation-like modes then flagged `external`)."""
    symmetry_number: Literal["auto"] | int
    """The rotational symmetry number. `auto`: from the detected point group."""
    symmetry_tol: float
    """The tolerance of the point-group detection and of the test for linear molecules."""
    codata: Literal["codata2022", "codata2014"]
    """The physical constants: `codata2022` or `codata2014` (as Psi4 1.11 and PySCF 2.15)."""


class BackendSettings(TypedDict, total=False):
    """The hardware."""

    kind: Literal["cpu", "gpu", "auto"]
    """`cpu`, `gpu` (every kernel family that has device kernels on an NVIDIA GPU, the others on the CPU, each with its reason; a build with the `gpu` feature) or `auto` (the GPU when the build has it and a device opens, else the CPU)."""
    devices: int | list[int]
    """The GPU to use: a device id (or a list of one)."""
    scf_loop: Literal["auto", "host", "device"]
    """Where the SCF loop's own work runs with a GPU: `host` (the CPU drives the device's Fock builds) or `device` (the whole loop on the GPU). `auto`: device, closed and open shells."""
    kernel_cache: str
    """Where a GPU backend keeps its compiled kernels between runs: a directory or `off` (memory only). `auto`: `$LINUS_GPU_CACHE`, else `$XDG_CACHE_HOME/linus/gpu-kernels`, else `~/.cache/linus/gpu-kernels`."""


class OutputSettings(TypedDict, total=False):
    """Optional outputs."""

    verbose: bool
    """The SCF's progress lines: on stderr from the command line, in the output's `logs` in process (Python), which never prints."""
    orbitals: bool
    """The MO coefficients in the result (`mo_coefficients_alpha`/`_beta`, nbf x nmo)."""
    hamiltonian: bool
    """The Fock (Kohn-Sham) matrices of the SCF's last build in the result (`hamiltonian_alpha`/`_beta`, nbf x nbf), with the overlap: the operator of one full build at the last density, whose generalized eigenvectors are the returned orbitals (H C = S C diag(eps))."""
    overlap: bool
    """The overlap matrix S of the basis functions in the result (`overlap`, nbf x nbf)."""
    core_hamiltonian: bool
    """The core Hamiltonian in the result (`core_hamiltonian`, nbf x nbf): the kinetic energy, the nuclei's attraction with their effective charges and the ECPs, whose trace with the total density is the energy's `one_electron` component."""
    molden: Literal["off", "spherical", "cartesian"]
    """A Molden file of the orbitals in the result (`files["orbitals.molden"]`; the CLI's `--molden FILE` writes it): `spherical` or `cartesian` functions, or `off`."""


class AccuracySettings(TypedDict, total=False):
    """Accuracy targets."""

    energy: float
    """The energy's (Eh)."""
    gradient: float
    """The largest gradient component's."""
    hessian: float
    """The largest Hessian element's."""


class Settings(TypedDict, total=False):
    """Every setting of a calculation."""

    method: str
    """`hf` or a functional (`--list-functionals`; libxc names, mixtures such as `"0.5*b3lyp + 0.5*hf"`). A dispersion suffix (`b3lyp-d3bj`, `-d3(bj)`, `-d3bj-atm`, `-d4`) sets `dispersion`, unless linus knows the whole name (`wb97x-d3bj`)."""
    basis: str
    """The orbital basis set (`--list-basis`); the def2 sets carry ECPs for Rb-Rn."""
    dispersion: str
    """Empirical dispersion: `d3bj`, `d3bj-atm`, `d4` or `none`. `auto`: implied by the method's name, else none. Bare `d3` is ambiguous (zero damping in Psi4 and qcdata, Becke-Johnson in ORCA) and rejected."""
    reference: Literal["auto", "restricted", "unrestricted"]
    """The determinant: `restricted` (synonyms `rhf`, `rks`) or `unrestricted` (`uhf`, `uks`). `auto`: restricted for closed shells, unrestricted for open shells (ROHF is not implemented)."""
    cartesian: bool
    """Cartesian d and higher functions (false: spherical)."""
    basis_dir: str | list[str]
    """Directories of BSE JSON basis set files searched first (a string or a list), then `$LINUS_BASIS_DIR`, a development build's source tree and the installation's sets."""
    threads: int
    """Threads of the calculation's own pool."""
    memory_gib: Literal["auto"] | float
    """The memory the calculation may use; every host budget derives from it. `auto`: the total RAM, or the cgroup limit if lower."""
    device_memory_gib: Literal["auto"] | float
    """The device memory a GPU backend may use; every device budget derives from it. `auto`: `$LINUS_GPU_MEMORY_GIB`, else 95% of the memory free when the device opens (at most that either way)."""
    jk: JkSettings | str
    """The J/K approximation."""
    grid: GridSettings | Any
    """The grids of DFT."""
    precision: PrecisionSettings | Literal["auto", "fp64", "native", "double", "emulated", "mixed"]
    """The arithmetic."""
    solvent: SolventSettings | Any
    """The implicit solvent (absent: gas phase)."""
    scf: ScfSettings
    """The SCF."""
    response: ResponseSettings
    """The response equations of Hessians."""
    freq: FreqSettings
    """The analysis and thermochemistry a Hessian carries."""
    backend: BackendSettings | Literal["cpu", "gpu", "auto"]
    """The hardware."""
    output: OutputSettings
    """Optional outputs."""
    accuracy: AccuracySettings
    """Accuracy targets."""
