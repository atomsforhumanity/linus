# Implicit solvation (PCM)

linus has one polarizable-continuum engine (`crates/linus-solvent/`) with four models (and
SMD on top) on a smooth surface of Gaussian charges: energies and analytic gradients for
RHF/UHF/RKS/UKS with exact, RI-JK and RIJ+COSX J/K, range-separated hybrids included, and
analytic Hessians and polarizabilities wherever linus has them in the gas phase (exact and RI-JK
J/K).

## Models

The solute's electrostatic potential phi at the surface points (nuclei minus electrons) induces
charges q with K q = R phi:

| model | K | R | f(eps) |
|---|---|---|---|
| `cpcm` | S | -f I | (eps-1)/eps |
| `cosmo` | S | -f I | (eps-1)/(eps+x), x = 0.5 (`cosmo_x`) |
| `iefpcm` | S - f/(2 pi) D A S | -f (I - D A/(2 pi)) | (eps-1)/(eps+1) |
| `ssvpe` | S - f/(4 pi) (D A S + S A D^T) | as IEF-PCM | (eps-1)/(eps+1) |

The energy is E_pcm = 1/2 q_sym . phi with q_sym = (K^-1 R + R^T K^-T) phi / 2, reported as
`components.solvation_electrostatic` (included in `energy`); its Fock contribution is
-sum_g q_sym,g (ab|g). It is the reaction-field (dielectric) energy, not a solvation free energy:
for Delta G_solv subtract the gas-phase energy (and add non-electrostatic terms, e.g. SMD's CDS).

## Discretization (smooth, so energies and gradients are continuous)

Each atom carries a sphere of radius R_J (scaled vdW radius) with Lebedev points r_i = X_J + R_J
n_i. Point i holds a normalized Gaussian charge of exponent zeta_i^2, zeta_i = xi_N / (R_J
sqrt(4 pi w_i)) (York & Karplus 1999), and a switching factor F_i = prod_K f(|r_i - X_K|) that
turns it off smoothly inside other spheres (Lange & Herbert 2010): SWIG (5th-order polynomial,
C2) or ISWIG (erf, smooth). S_ij = erf(zeta_ij r_ij)/r_ij, S_ii = zeta_i sqrt(2/pi)/F_i, areas
a_i = 4 pi w_i R_J^2 F_i.

## Presets

| preset | radii | points per sphere | switching | potential at the points | reproduces |
|---|---|---|---|---|---|
| `pyscf` (default) | Bondi with H 1.1 Å (PySCF's table) x 1.2 | 302 (`lebedev` changes it) | SWIG (`discretization`) | smeared over the Gaussian | PySCF 2.15 `pyscf.solvent.pcm` exactly |
| `orca` | Bondi with H 1.1 Å x 1.2 (2.0 Å where Bondi/Mantina have none) | nearest of ORCA's grids to 5 per Å^2 (H 110, N/O 194, C/S 302) | ISWIG, F < 1e-7 dropped | at the point (Gaussians only in S) | ORCA 6.0.0 `CPCM(...)` (its "GAUSSIAN VDW" default) |

The default preset is `pyscf`: its exact reference is open source (PySCF 2.15 reproduces it to
1e-11 Eh), and it has the smaller discretization error. The `orca` preset's constants (ORCA's
sphere grids and their exponents xi_N, the point density, the switching cut, the fallback
radius) were read off ORCA 6.0.0's output and `.cpcm` files. Against dense surfaces (up to 5810
points per sphere; SWIG and ISWIG converge to the same sharp-cavity limit; la18, u1an39, ts61,
B3LYP/def2-SVP, C-PCM(THF)):

| molecule | `pyscf` points | `pyscf` E_solv error (kcal/mol) | `pyscf` max dG (Eh/bohr) | `orca` points | `orca` E_solv error (kcal/mol) | `orca` max dG (Eh/bohr) |
|---|---|---|---|---|---|---|
| la18 | 2220 | +0.072 (0.79%) | 1.0e-4 | 1589 | +0.117 (1.29%) | 1.2e-4 |
| u1an39 | 4581 | +0.130 (0.37%) | 1.2e-4 | 3670 | +0.164 (0.47%) | 1.7e-4 |
| ts61 | 6237 | +0.106 (0.72%) | 6.1e-5 | 5097 | +0.180 (1.22%) | 9.6e-5 |

`orca` reproduces ORCA 6.0.0 and uses 18-28% fewer points (the PCM cost is linear in them). Its
constants, ORCA's solvent table and the ORCA comparisons here are included with permission of
FAccTs GmbH ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#orca-faccts-gmbh)).

For exact comparisons with another program, give that program's permittivity (`eps`): programs'
own tables differ from the CRC values (ORCA 6.0.0's THF 7.565 vs 7.4257, water 80.151 vs 78.355).

## Using it

- CLI: `linus mol.xyz --method b3lyp --eps 7.4257 --solvent-model cpcm [--set solvent.preset=pyscf|orca]
  [--set solvent.lebedev=N] [--set solvent.discretization=swig|iswig]
  [--set solvent.scf_terms=exact|predicted | --set solvent.df=true] [--solvent NAME]`; JSON
  requests take `"solvent": {"eps": 7.4257, "model": "cpcm", ...}` (the driver's `SolventSettings`)
  or `"solvent": "thf"`.
- Python: the keyword `solvent` of a qcdata `ProgramInput`: `{"eps": 7.4257, "model": "cpcm"}`,
  `"thf"` or `{"eps": 78.36, "model": "iefpcm", "radii": {"H": 1.2}, "radii_scale": 1.0}`; the
  surface statistics are in the frame's or output's `record["solvation"]`, the solvation energies in
  `results.energy_components`.

Solvent names: linus ships the CRC permittivities of the SMD solvents and about 30 other common
liquids (`linus --list-solvents`, `linus.available_solvents()`); the rest of the CRC table
(1303 liquids) resolves through qcref's copy where qcref is installed (the Python package finds
it; the CLI reads `$LINUS_SOLVENT_TABLE`, the path of qcref's `data/solvents/crc.csv`), with the
same rules and values, and `resolved.record.solvent.eps_table` then names the table.

Options: `eps`, `name` (the solvent's name; needs the solvent database that comes with SMD), `model`,
`preset`, `lebedev`, `discretization`, `radii` (Å per element before scaling, by symbol or
atomic number), `radii_scale`, `r_probe` (Å), `cosmo_x`, `screening` (of the
surface-charge integrals, 1e-14), `scf_terms` (surface terms in the SCF iterations, below) and `df`,
`incremental` (predicted mode: passes on the changes, default on), `rys_gradient` (gradient
integrals on the Rys kernels, the default in every mode; off: the McMurchie-Davidson reference).

Results (`solvation`): model, eps, f(eps), preset, per-atom Lebedev sizes and radii (Å),
`electrostatic` (Eh), `npoints`, `area_angstrom2`, `surface_charge` (sum of q_sym) and
`gauss_charge` (-(eps-1)/eps Q: the value for a solute charge fully inside the cavity; the
difference measures escaped charge), `scf` (the SCF mode used), and the time spent.

## SCF modes of the surface terms (`scf_terms`, `df`)

Each SCF iteration needs the solute's electronic potential at the points (for the energy) and
the Fock term of the resulting charges; both are passes of exact integrals over all shell pairs
and points, the second depending on the first.

| `scf_terms` | per iteration | result | default |
|---|---|---|---|
| `exact` | two exact passes (potential, then the Fock term of its charges) | exact PCM (bitwise the reference path) | exact J/K |
| `predicted` | one fused exact pass: the exact potential of D_n and the Fock term of charges predicted from the previous iteration's exact potential plus the RI-J fitted change of the density; on the changes of D and of the predicted charges (`incremental`); the last build is `exact`'s two passes | exact PCM (the fixed point; one build more than `exact`) | density-fitted J (`df`, `rij-snk`) |
| `df: true` (`ri`) | no exact pass: potential and Fock term through the J auxiliary basis, riding on the J build | The density-fitted PCM, an approximation (below) | - |

`predicted` (`solvent::ri`): phi~_n = phi_{n-1} - M (c_n - c_{n-1}) with the J build's fit
coefficients c (`linus_base::terms::CoulombHook`) and M_gP = (P|g) (closed form for solid harmonic
Gaussians, Hobson's theorem: one Boys function per primitive and point). The energy of every
iteration is exact; the Fock term differs from the exact one only by the RI error of the
potential's change, which vanishes at convergence, so the SCF converges to the exact PCM
solution in the same number of iterations (energies equal to `exact` to ~1e-12 Eh, gradients to
the SCF convergence). Both terms are linear in the density and the charges, so the fused pass
runs on D_n - D_{n-1} and q~_n - q~_{n-1} (a full pass every 8), where the integral screening
drops more as the SCF converges. Each such build is incremental in the SCF's history (`fock`:
`incremental`): its Fock term is not of D_n alone. When an iterate meets the convergence criteria,
the SCF builds once more at the same density with `exact`'s two passes (the step `rebuild`, then a
`full` build), so the returned Fock matrix and orbitals are those of the exact PCM operator of the
returned density (the one-build rule of qcdata's wavefunction outputs). It costs one Fock build
(caffeine B3LYP/def2-SVP in water, 8 cores: the SCF 2.04 -> 2.23 s, +9%; its cation, UKS, +5%);
the energy is the same, the orbital energies move by up to ~1e-10 Eh and gradients by ~1e-11
Eh/bohr. Gradients, Hessians and response properties are those of the exact PCM.

`df` (the density-fitted PCM): (ab|g) ~ sum_PQ (ab|P) [V^-1]_PQ (Q|g). The energy is variational and
its analytic gradient exact (RI-J-like 3- and 2-centre derivative terms), but the Coulomb-metric fit is made
for the Coulomb energy, not for potentials at points in the density's tail: with
def2-universal-jfit the potential at the surface points is off by ~1e-3 au (rms) and the
electrostatic solvation energy up to 4% too negative (water 0.13, la18 0.30, ts113 wB97X-D3BJ/def2-TZVP 0.61 kcal/mol; PySCF's own fit
of the same density gives the same), gradients ~1e-4 Eh/bohr; with def2-universal-jkfit about
0.7%. Not available for Hessians or response properties.

Gradients: the surface-integral derivatives run on the batched Rys kernels (first-centre
derivatives of ordered shell pairs through the raised and lowered Cartesian shells,
`SurfacePlan::gradient`, 4-5x faster than the McMurchie-Davidson routine) in every SCF mode;
the McMurchie-Davidson reference stays as a dev switch (`rys_gradient: false`).


## SMD

`model: "smd"` (needs `solvent`, a name with SMD descriptors in `solvent::db`): SMD's bulk
electrostatics plus its non-electrostatic term G_CDS (`components.solvation_cds`,
`solvation.cds`; included in `energy`, its gradient and Hessian).

- Electrostatics: IEF-PCM with the `pyscf` preset (as PySCF and Gaussian), C-PCM with the
  `orca` preset (as ORCA); `solvation.model` reports the one used.
- Radii: SMD's Coulomb radii (O depends on the solvent's alpha), unscaled; PySCF's table
  (Br 2.60 Å) with the `pyscf` preset, SMD's otherwise. 590 points per sphere with the `pyscf`
  preset (PySCF's SMD discretizes its cavity with the SASA grid).
- eps: the solvent database's default (qcref/CRC at 298.15 K, as for the other models) unless
  `eps` is given; exact comparisons pass the program's value (PySCF's water 78.3553, the
  Minnesota database's 78.355).
- G_CDS (`smd::cds`): exact SASA for single points; for gradients and Hessians new sphere
  contacts are smoothed over 0.1 Å (`cds_contact_smoothing`, Å) so the surface tension term has
  no gradient jumps. `cds_contact_smoothing: 0` reproduces PySCF exactly.

## Derivatives

Gradients reuse the SCF's surface and factorized K. Analytic Hessians (RHF/UHF/RKS/UKS, exact
or RI-JK, also in two-level CPKS modes) add the explicit second derivatives of the surface and of
the solute-charge interaction, the fixed-density F^x, and the reaction field's linear response
in every induced-Fock product (`response/cphf.rs`, `Response::solvent`); polarizabilities
include the same response. The solvent terms do not depend on the functional, so they apply to
range-separated hybrid Hessians as well.

## Cost

Setup (surface, S, D, factorization) and the dense solves are small. The cost is the surface
integrals over all shell pairs x points: per SCF iteration two dependent passes in the `exact`
mode (`pcm_potential`, `pcm_fock` in the timings), one fused pass in the `predicted` mode
(`pcm_potential_fock`, about the cost of the Fock pass alone), and the gradient's derivative
integrals (`grad_solvent_ints`). Energy + gradient with 8 cores, RIJ+COSX, C-PCM(THF), `pyscf`
preset, beside ORCA 6.0.0 on the same 8 cores (8 MPI ranks; RIJCOSX with def2/J, DefGrid2,
TightSCF, `CPCM(THF)`):

| case | gas (s) | C-PCM, predicted (default) | C-PCM, exact two-pass | ORCA 6.0.0 CPCM (s) | ORCA / linus |
|---|---|---|---|---|---|
| la18 B3LYP/def2-SVP | 1.8 | 2.1 (+19%) | 2.4 (+32%) | 6.9 (+8%) | 3.2x |
| u1an39 wB97X-D3BJ/def2-TZVP | 70.4 | 78.4 (+11%) | 86.1 (+22%) | 152.0 (+7%) | 1.9x |
| ts61 wB97X-D3BJ/def2-TZVP | 157.6 | 174.8 (+11%) | 192.0 (+22%) | 333.5 (+1%) | 1.9x |
| ts113 B3LYP/def2-SVP | 95.1 | 115.6 (+22%) | 139.1 (+46%) | 176.9 (+18%) | 1.5x |
| ts113 wB97X-D3BJ/def2-TZVP, energy (production job) | 371.9 | 387.8 (+4%; PCM 43 s, one SCF iteration fewer) | 401.1 (+8%; PCM 66 s) | 830.4 (+12%) | 2.1x |

All runs the same day on `slice bench`; ORCA's percentages are its own CPCM overhead over its gas
phase run. The cost is linear in the number of points (`lebedev`; the `orca` preset uses 25-40%
fewer).
