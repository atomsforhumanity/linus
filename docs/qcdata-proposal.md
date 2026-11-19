# Proposal: basis sets and orbitals in qcdata

Status: adopted. qcdata's `Orbitals`, `BasisSet`, `Shell` and `AOConvention` are this design (it was
tried first as `linus.models`); linus returns them and keeps the converters to PySCF and Molden
as functions on them (`linus.convert`). This page stays as the design record. Reference: qcdata
0.23.1.

## What qcdata lacks today

A `SinglePointData` from an SCF program carries the energy, gradient, Hessian, dipole moment,
`nuclear_repulsion_energy`, the `calcinfo_*` counts and a `Wavefunction` with only

```python
scf_eigenvalues_a, scf_eigenvalues_b, scf_occupations_a, scf_occupations_b
```

Nothing says which basis functions the calculation used, and there are no orbital
coefficients or density matrices. So a qcdata result cannot be used to

- start another program's SCF from the converged orbitals (codecs keep wavefunctions in
  program-specific files, e.g. ORCA `.gbw`, Psi4 `.npy`, axtb's in-memory ids);
- compute properties afterwards (populations, densities on grids, orbital plots, IBOs);
- write a Molden file or hand orbitals to PySCF, Multiwfn or a correlated method;
- compare orbitals between programs, which needs the AO conventions to be explicit.

Energy components are also missing (linus and axtb put them in `extras`); QCSchema has
`scf_one_electron_energy`, `scf_two_electron_energy`, `scf_xc_energy`, `scf_iterations`.
They are a separate, smaller addition and not part of this proposal.

## Proposed models

All in qcdata's style: `QCDataBaseModel` (frozen, extra fields forbidden, `extras`),
numpy arrays as `SerializableVector`/`SerializableMatrix`, atomic units, enums as `StrEnum`.

### `Shell`

| field | type | meaning |
|---|---|---|
| `center` | `int` | 0-based index of the atom in the structure |
| `angular_momentum` | `int` | l |
| `pure` | `bool` | spherical (2l+1 functions) or Cartesian ((l+1)(l+2)/2) |
| `exponents` | vector | primitive exponents (bohr⁻²) |
| `coefficients` | vector | contraction coefficients of the **unnormalized** primitives r^l exp(-a r²), all normalization included: the x^l component of the contracted shell, and so every spherical function, has unit norm |

General contractions are split into one shell per contracted function (as linus, PySCF's
`nctr` aside, and most integral codes do internally). `Shell.primitive_coefficients()` gives
the coefficients of normalized primitives, the convention of basis set files.

### `AOConvention`

How the functions *within* a shell are ordered and normalized (shells are in `shells` order):

| field | values | default |
|---|---|---|
| `spherical_order` | `pyscf`: m = -l..l, p as (x, y, z) (libcint, PySCF, linus) · `increasing_m`: m = -l..l for all l, p as (y, z, x) (libint's "standard", the CCA order QCSchema adopts) · `molden`: 0, +1, -1, +2, -2, ..., p as (x, y, z) (Molden, Gaussian, ORCA order) · `trexio`: 0, +1, -1, ... for all l, p as (z, x, y) | `pyscf` |
| `cartesian_order` | `lexicographic`: xx, xy, xz, yy, yz, zz (libint, libcint, PySCF, TREXIO) · `molden`: xx, yy, zz, xy, xz, yz, ... (to g) | `lexicographic` |
| `cartesian_normalization` | `axis`: every component carries the x^l normalization (xy has norm 1/√3; libint, linus) · `unit`: each component normalized (Molden) · `pyscf`: `axis` × √(4π/(2l+1)) for l ≥ 2 (PySCF `cart=True`) | `axis` |

Real solid harmonics are the unit-normalized ones without a Condon-Shortley phase
(S₁⁺¹ = x, S₁⁻¹ = y, as PySCF, libint, Molden and TREXIO). `BasisSet.transform(C,
convention, shell_order=None)` converts AO-indexed arrays between conventions (and shell
orders): a permutation within shells plus a diagonal scaling.

### `BasisSet`

| field | type | meaning |
|---|---|---|
| `name` | `str \| None` | e.g. `"def2-svp"` (None for mixed or custom bases) |
| `shells` | `list[Shell]` | the AOs are the shells' functions in this order |
| `ao_convention` | `AOConvention` | |

plus `nao`, `shell_offsets()`, `ao_labels(symbols)`.

### `Orbitals`

| field | type | meaning |
|---|---|---|
| `basis` | `BasisSet` | what the orbitals expand in |
| `spin` | `"restricted"` \| `"unrestricted"` | restricted: one set of spatial orbitals (RHF, RKS, ROHF); the `_b` coefficients and energies are None |
| `kind` | `str` | `canonical` (default), `natural`, `localized`, ... |
| `coefficients_a`, `coefficients_b` | matrix nao × nmo | column i is orbital i |
| `energies_a`, `energies_b` | vector nmo \| None | orbital energies (Eh) |
| `occupations_a`, `occupations_b` | vector nmo \| None | per spin, 0..1; `occupations_b` None: as alpha. ROHF: same orbitals, different occupations |

plus `density()` (C n Cᵀ per spin), `with_convention()`.

### Where they go in qcdata

Smallest change: `Wavefunction.orbitals: Orbitals | None = None` (and optionally
`scf_density_a/_b`), filled only when the input asks for it (linus: keyword
`output: {"orbitals": True}`; today it goes to `extras['orbitals']`). Orbitals of a 2000-function
calculation are 32 MB of JSON, so codecs should not add them by default.

### Converters (in linus now; qcinf or qcdata later)

- `to_pyscf(structure, orbitals) -> (mol, mo_coeff, mo_energy, mo_occ)`,
  `to_pyscf_mol(structure, basis)`, `from_pyscf(mol, mo_coeff, mo_energy, mo_occ)`,
  `from_pyscf_mol(mol)`: exact (PySCF's overlap of a linus basis agrees with linus's to
  1e-15; HF energies of the converted orbitals agree to 1e-8 Eh, spherical and Cartesian,
  RHF and UHF; general contractions are split).
- `write_molden(structure, orbitals)`: spherical [5D7F]/[9G] or Cartesian, up to g; PySCF's
  Molden reader recovers the orbitals (orthonormal, same energies, same HF energy).

## How TREXIO and QCSchema represent the same

| | this proposal | TREXIO 2.x (`basis`, `ao`, `mo`) | QCSchema (`BasisSet`, `Wavefunction`) |
|---|---|---|---|
| shells | list of `Shell` objects, one per atom and shell | flat arrays: `basis.nucleus_index`, `shell_ang_mom`, `shell_factor`, `r_power` per shell; `shell_index`, `exponent`, `coefficient`, `prim_factor` per primitive | per *element/center type*: `center_data[key].electron_shells`, and `atom_map` (one key per atom) |
| general contractions | split | split (one shell per contraction) | kept: `coefficients[contraction][primitive]`, and SP shells with `angular_momentum: [0, 1]` |
| coefficient normalization | one fixed convention (raw primitives, normalized shell) | product `shell_factor` × `prim_factor` × `coefficient` × `ao.normalization`, split freely ("the only requirement is self-consistency") | coefficients of normalized primitives, as in the Basis Set Exchange; contraction normalization implicit |
| spherical/Cartesian | per shell (`pure`) | per file (`ao.cartesian`) or per shell (`ao.cartesian_shell`) | per shell (`harmonic_type`) |
| AO order in a shell | named in `AOConvention` | fixed: spherical m = 0, +1, -1, ...; Cartesian lexicographic | fixed: "CCA standard as implemented in libint" |
| per-AO normalization | from `cartesian_normalization` | explicit `ao.normalization` per AO | implicit in the CCA standard |
| MO coefficients | `coefficients_a/_b`, nao × nmo | `mo.coefficient` [mo_num][ao_num], all spin orbitals in one list with `mo.spin` (0/1) | `scf_orbitals_a/_b` (column-major, nao × nmo) |
| energies, occupations | per spin vectors | `mo.energy`, `mo.occupation` per MO | `scf_eigenvalues_a/_b`, `scf_occupations_a/_b` |
| orbital kind | `kind` | `mo.type` (free text), `mo.class` (core/inactive/active/virtual/deleted), `mo.symmetry` | `scf_*`, `localized_orbitals_*`, plus `return_results` naming which set is "the" result |
| densities, Fock | `Orbitals.density()`; optional `scf_density_a/_b` | `rdm` group | `scf_density_a/_b`, `scf_fock_a/_b`, `h_core_a/_b`, ... |
| ECPs | not yet in this draft (linus itself has def2 ECPs; see open question 4) | `ecp` group (z_core and terms per atom) | `ecp_electrons`, `ecp_potentials` per center |

Choices made here and why:

- **Per-atom shells rather than per-element data plus `atom_map`.** Simpler to validate and
  to read, works for ghost atoms and per-atom basis choices; the cost is size for large
  molecules (each shell repeats its exponents), which matters less than the MO matrix.
- **One normalization convention for coefficients.** TREXIO's factor split and QCSchema's
  implicit contraction normalization each leave room for programs to disagree; storing the
  coefficients that make an AO ``Σ cᵢ x^a y^b z^c exp(-αᵢ r²)`` (with the named
  per-component factor) is unambiguous and is what integral codes use.
- **Named AO conventions instead of one fixed order.** Every program's coefficients can be
  stored as produced, and converted when needed; a fixed order (TREXIO, QCSchema) forces
  every codec to permute and rescale on the way in.
- **`_a`/`_b` per spin**, as qcdata's `Wavefunction` and QCSchema, rather than TREXIO's
  list of spin orbitals.

## Open questions

1. Embed `BasisSet` in `Orbitals` (self-contained, as here) or put one `basis_set` on
   `SinglePointData` and let orbitals, densities and Fock matrices refer to it?
2. Fixed AO order (simpler readers; TREXIO, QCSchema) or named conventions (simpler
   writers; this draft)? If named: which set, and should ORCA's sign flips of some f/g
   functions be a convention value (`spherical_phase: "orca"`)?
3. Should `Shell.coefficients` keep the published convention (normalized primitives, as BSE
   and QCSchema) with an explicit `coefficient_normalization` field, instead of fully
   normalized raw-primitive coefficients?
4. ECPs: a per-atom `ecp` (core electrons and terms, as TREXIO) on `BasisSet`?
5. Size: nested JSON lists of float64 for nao × nmo matrices. Store large arrays as binary
   files (`files`, `.npy`) and reference them, or allow base64 arrays?
6. GHF, complex and relativistic two-component orbitals, and k-points are out of scope;
   does the model need room for them (TREXIO has `_im` arrays and `k_point`)?
7. Where do converters and file writers (PySCF, Molden, fchk, TREXIO) live: qcdata (no
   heavy dependencies, writers only), qcinf, or the programs' packages?
8. Should codecs fill `Wavefunction.orbitals` by default for small systems, or only on
   request (keyword `output.orbitals`, as linus and qccompute's `collect_wfn`)?

---

# Proposal: vibrational analysis and thermochemistry in qcdata

Status: draft, implemented in linus (`linus::vib` in Rust; `linus.vib.VibrationalAnalysis`,
`linus.vib.Thermochemistry` in Python; `extras['vibrational_analysis']` and
`extras['thermochemistry']` (a list, one entry per temperature) of a linus `hessian`
`ProgramOutput`; the analysis is tuned by the `ProgramInput`'s `keywords["freq"]`, the settings
group `freq`, and there is no separate frequencies calctype). Reference: qcdata 0.23.1.

## What qcdata has and lacks

`SinglePointData` has `freqs_wavenumber` (cm⁻¹, imaginary negative), `normal_modes_cartesian`
(unit-length Cartesian displacements, n_vib x n_atoms x 3) and `gibbs_free_energy` (Eh).
These cannot be compared or reused across programs because nothing says how they were made:

- **masses**: most abundant isotopes (Psi4, Gaussian, Q-Chem, linus's default) or standard
  atomic weights (ORCA, PySCF's `thermo`, xtb); this alone moves X-H stretches by ~0.4% and G
  by ~1e-5 Eh;
- **projection**: translations and rotations projected out or not (an off-stationary Hessian's
  low modes depend on it), and which modes were dropped as "translation/rotation";
- **the thermochemical model**: temperature, pressure or concentration standard state (1 atm vs
  1 M is 1.89 kcal/mol in G), symmetry number (and whether it was detected or given), RRHO or
  quasi-RRHO (ORCA applies Grimme's qRRHO entropy by default, Psi4 and Gaussian do not),
  frequency scale factor, and what happened to imaginary modes (dropped, inverted);
- **what `gibbs_free_energy` is**: ORCA's "Final Gibbs free energy" includes E_el, xtb's
  "G(RRHO) contrib." does not;
- reduced masses, force constants, IR intensities (and the dipole derivatives behind them),
  rotational constants, point group, ZPE and the per-term E, H, S, Cv are missing, so codecs
  put them in `extras` under different names or drop them.

## Proposed models

qcdata style: `QCDataBaseModel`, numpy arrays as `SerializableVector`/`SerializableMatrix`,
atomic units except where chemistry has a fixed convention (cm⁻¹, u, km/mol), enums as `StrEnum`.

### `VibrationalAnalysis`

| field | type | meaning |
|---|---|---|
| `masses` | vector (u) | masses used, per atom |
| `projection` | `"translations_rotations"` \| `"translations"` \| `"none"` | what was projected out of the mass-weighted Hessian |
| `frequencies` | vector (cm⁻¹) | all modes analysed, ascending, imaginary negative |
| `external` | list[bool] | modes that are translations/rotations rather than vibrations (only without full projection) |
| `normal_modes_cartesian` | n_modes x n_atoms x 3 | unit-length Cartesian displacements (as `SinglePointData`) |
| `normal_modes_mass_weighted` | n_modes x 3N \| None | orthonormal eigenvectors of the mass-weighted Hessian |
| `reduced_masses` | vector (u) | 1 / &#124;M^-1/2 L&#124;² |
| `force_constants` | vector (mdyn/Å) | μ λ, negative for imaginary modes |
| `ir_intensities` | vector (km/mol) \| None | N_A π / (3 c² 4πε0) &#124;dμ/dQ&#124;² |
| `dipole_derivatives` | 3N x 3 (e) \| None | atomic polar tensors dμ_c/dX, the input of `ir_intensities` |
| `rotor_type` | `"atom"` \| `"linear"` \| `"nonlinear"` | |
| `moments_of_inertia` | vector (u bohr²) | principal moments, ascending |
| `rotational_constants` | 3 x float \| None (cm⁻¹) | A ≥ B ≥ C; None for a zero moment |
| `point_group` | str | Schoenflies symbol of the structure with these masses |
| `symmetry_number` | int | rotational σ used, and `symmetry_number_source`: `"detected"` \| `"input"` |
| `n_imaginary` | int | imaginary vibrations (no cutoff; consumers apply theirs, e.g. -20 cm⁻¹) |

### `ThermoTerm`

`energy`, `enthalpy`, `entropy`, `cv`, `cp`, `gibbs` (Eh, Eh/K): one contribution
(electronic, translational, rotational, vibrational) or the total. Conventions to fix: the
vibrational energy includes the ZPE; pV = kT sits in the translational enthalpy (PySCF, Psi4,
linus); the electronic energy term is E_el.

### `Thermochemistry`

| field | type | meaning |
|---|---|---|
| `temperature` | float (K) | |
| `pressure` | float (Pa) \| None | ideal-gas standard state |
| `concentration` | float (mol/L) \| None | concentration standard state instead (1 M) |
| `electronic_energy` | float (Eh) | E_el the corrections were added to |
| `zero_point_energy` | float (Eh) | |
| `electronic`, `translational`, `rotational`, `vibrational`, `total` | `ThermoTerm` | `total` includes E_el |
| `enthalpy_correction`, `gibbs_correction`, `thermal_energy_correction` | float (Eh) | total minus E_el: what to add to an electronic energy at another level |
| `symmetry_number`, `electronic_degeneracy` | int, float | |
| `entropy_model` | `"rrho"` \| `"qrrho_grimme"` \| `"quasiharmonic_truhlar"` | with `qrrho_frequency` (cm⁻¹), `qrrho_alpha`, `qrrho_bav` (kg m²) |
| `enthalpy_model` | `"rrho"` \| `"qrrho_head_gordon"` | |
| `frequency_scale_factor` | float | |
| `excluded_imaginary` | vector (cm⁻¹) | imaginary modes left out (a TS's reaction coordinate) |
| `inverted_imaginary` | vector (cm⁻¹) | imaginary modes treated as real (if the program does that) |

### Where they go

`SinglePointData.vibrational_analysis: VibrationalAnalysis | None` and
`SinglePointData.thermochemistry: list[Thermochemistry]` (one per temperature/pressure;
programs such as ORCA can do several in one run). `freqs_wavenumber`,
`normal_modes_cartesian` and `gibbs_free_energy` stay as they are, defined as the vibrational
modes of `vibrational_analysis` and `thermochemistry[0].total.gibbs` (E_el included), so
existing consumers keep working. Codecs fill what their program prints and leave the rest None;
an analysis library (linus's, or a Python one in qcinf) can fill the whole model from any
`hessian` result, so that thermochemistry at other conditions or with another qRRHO choice
needs no rerun.

## How others represent the same

| | this proposal | cclib | ASE (`ase.vibrations`, `ase.thermochemistry`) | Psi4 (`vibanal_wfn` vibinfo/therminfo) |
|---|---|---|---|---|
| frequencies | `frequencies` cm⁻¹, imaginary negative, with `external` flags | `vibfreqs` (cm⁻¹, negative) | `get_frequencies()` complex | `omega` complex, `TRV` labels |
| modes | unit Cartesian + optional mass-weighted | `vibdisps` (as printed: normalized or not, program-dependent) | `get_mode(i)` | `q` (mass-weighted), `w`, `x` (normalized) |
| masses | explicit vector | absent (`atommasses` only in some parsers) | the Atoms' masses | from the molecule |
| reduced mass, force constant | `reduced_masses` u, `force_constants` mdyn/Å | `vibrmasses`, `vibfconsts` | none | `mu`, `k` |
| IR | `ir_intensities` km/mol + `dipole_derivatives` | `vibirs` | `ase.vibrations.Infrared` | `IR_intensity` |
| thermochemistry | per-term `ThermoTerm` + model fields | `zpve`, `enthalpy`, `entropy`, `freeenergy`, `temperature`, `pressure` (totals only) | `IdealGasThermo` (methods, not data) | `ZPE_*`, `E_*`, `H_*`, `S_*`, `G_*`, `Cv_*`, `Cp_*` per term |
| symmetry number | `symmetry_number` + source | none | input | `sigma` |
| qRRHO | `entropy_model`, `enthalpy_model` | none | none | none |

## Open questions

1. Frequencies of all modes with `external` flags (exact for no-projection analyses) or only
   the vibrations (simpler, what `freqs_wavenumber` holds)?
2. Units: cm⁻¹, u, mdyn/Å and km/mol are what every program prints; qcdata otherwise uses
   atomic units. Keep the chemistry units in field docs, or store atomic units
   (Eh/(bohr² u) eigenvalues, e²/u intensities) and convert in properties?
3. Should `VibrationalAnalysis` carry its Hessian (or point to `SinglePointData.hessian`) so a
   consumer can redo it with other masses?
4. Anharmonic results (VPT2 fundamentals, Raman activities, VCD) as optional fields or
   separate models?
5. Degenerate modes: the basis within a degenerate pair is arbitrary, so per-mode reduced masses
   and intensities within it are basis-dependent; document, or report per degenerate set?
