# linus-disp

Dispersion corrections for linus: **DFT-D3 with rational (Becke-Johnson) damping**, with
the Axilrod-Teller-Muto (ATM) three-body term as an option, and **DFT-D4**. Energies,
analytic gradients and analytic Hessians of molecules, in bohr and Hartree. Results are
bitwise identical for any number of threads.

* D3(BJ) is a Rust port of s-dftd3 1.7.0 (`src/d3.rs`), including second derivatives.
* D4 uses the D4 model of **axtb-d4** (AFH's Rust D4, bit-identical to dftd4 4.3.1 at given
  charges), plus EEQ 2019 charges with first and second geometric derivatives ported from
  multicharge 0.6.0 (`src/eeq.rs`).
* Parameter tables are generated from the reference sources, literal for literal
  (`tools/gen_data.py`).

**License: LGPL-3.0-or-later** for this crate only; see [License and provenance](#license-and-provenance).

## API

```rust
use linus_disp::*;

// z: &[u32] atomic numbers, xyz: &[[f64; 3]] in bohr
let p = d3bj_params("b3lyp")?;               // ORCA `B3LYP/G D3BJ`: s6 1, s8 1.9889, a1 0.3981, a2 4.4211, s9 0
let r = d3bj(&z, &xyz, &p, &D3Options { order: Order::Hessian, ..Default::default() })?;
let (e, g, h) = (r.energy, r.gradient, r.hessian.unwrap());  // Eh; Vec<[f64; 3]> Eh/bohr; 3N x 3N row-major Eh/bohr^2

let abc = d3bj_params("wb97x-d3bj")?.with_atm(1.0);           // ORCA `wB97X-D3BJ` + `ABC`

let p4 = d4_params("r2scan-3c")?;            // s8 0, a1 0.42, a2 5.65, s9 2, ga 2, gc 1
let r4 = d4(&z, &xyz, charge, &p4, &D4Options::default())?;  // total charge enters the EEQ charges
```

```rust
pub struct DispResult { pub energy: f64, pub gradient: Vec<[f64; 3]>, pub hessian: Option<Vec<f64>> }
pub enum Order { Energy, Gradient /* default */, Hessian }
pub fn d3bj(z: &[u32], xyz_bohr: &[[f64; 3]], params: &D3Params, opts: &D3Options) -> Result<DispResult, DispError>
pub fn d4(z: &[u32], xyz_bohr: &[[f64; 3]], charge: f64, params: &D4Params, opts: &D4Options) -> Result<DispResult, DispError>
pub struct D3Params { s6, s8, a1, a2, s9, alp }            // D3Params::new(s6, s8, a1, a2), .with_atm(s9)
pub struct D4Params { s6, s8, a1, a2, s9, alp, ga, gc, wf } // D4Params::new(s6, s8, a1, a2): s9 1, alp 16, ga 3, gc 2, wf 6
pub struct D3Options { order, cutoffs: D3Cutoffs }          // s-dftd3 defaults: CN 40, two-body 60, ATM 40 bohr (hard)
pub struct D4Options { order, cutoffs: D4Cutoffs }          // dftd4 defaults: CN 30, two-body 60, ATM 40 bohr; EEQ CN 25
pub fn d3bj_params(functional: &str) -> Result<D3Params, DispError>   // two-body (s9 = 0)
pub fn d4_params(functional: &str) -> Result<D4Params, DispError>     // as the dftd4 program uses them
```

Also public: `D3Model` (`coordination_numbers`, `c6`, `compute`, `gradient_parts`),
`eeq_charges`, `d4_properties` (EEQ charges, D4 CN, two- and three-body energies),
`LINUS_FUNCTIONALS`, and the generated tables in `data`. Errors: unsupported element (D3 and
D4: H-Lr, the range of the D3 reference data and of the EEQ parameters), no parameters for a
functional, shape mismatch, singular EEQ matrix. Everything runs on the caller's rayon pool
(`pool.install(|| ...)`).

Hessians are fully analytic: D3 as `E_RR + E_RC J + J^T E_CR + J^T E_CC J + sum_k E_C,k
d2CN_k` (J = dCN/dR), with every row block computed by the task that owns its atom; D4 as
the fixed-charge Hessian of axtb-d4 plus the EEQ charge response `P Q + Q^T P^T + Q^T K Q +
sum_i V_i d2q_i` (Q = dq/dR from n+1 linear solves, the last term with one adjoint solve).

## Functional names

Lookup: the linus name (as `linus-xc` spells it; case, `-`, `_`, `/` ignored), then every
name s-dftd3 1.7.0 (D3) or dftd4 4.3.1 (D4) accepts.

| linus name | XC functional (linus-xc) | D3(BJ): s-dftd3 entry (s8, a1, a2) | D4: dftd4 entry | Note |
|---|---|---|---|---|
| `b3lyp`, `b3lyp/g` | B3LYP with VWN-RPA (ORCA `B3LYP/G`) | `b3lyp` (1.9889, 0.3981, 4.4211) | `b3lyp` | ORCA 6.0.0, s-dftd3 and dftd4 use the same parameters for both B3LYP flavours |
| `b3lyp5` | B3LYP with VWN5 (ORCA `B3LYP`) | `b3lyp` | `b3lyp` | |
| `pbe`, `pbe0`, `tpss`, `tpssh`, `blyp`, `bp86`, `scan`, `r2scan`, `cam-b3lyp`, `hf` | same names | same name | same name | ORCA 6.0.0 has no SCAN-D3BJ; ORCA prints r2SCAN's D3(BJ) set with 4 decimals but uses s-dftd3's 8-decimal one |
| `wb97x-d3bj` | wB97X-V semilocal part, VV10 dropped (ORCA `wB97X-D3BJ`) | `wb97x` (0.2641, 0.0, 5.4959) | - | s-dftd3 cites no reference for these numbers; ORCA 6.0.0 prints the same |
| `wb97m-d3bj`, `b97m-d3bj` (`b97m`) | wB97M-V / B97M-V semilocal parts | `wb97m`, `b97m` | - | s-dftd3 cites doi 10.1021/acs.jctc.8b00842 |
| `wb97x-d4`, `wb97m-d4`, `b97m-d4` | same semilocal parts (ORCA `wB97X-D4` etc.) | - | `wb97x`, `wb97m`, `b97m` | needs the `*-d3bj` XC functional of linus-xc |
| `wb97x` | Chai/Head-Gordon 2008 wB97X | none (ORCA 6.0.0 `wB97X D3BJ` aborts) | `wb97x-2008` (= ORCA `wB97X D4`) | |
| `wb97x-v`, `wb97m-v`, `b97m-v` | VV10 functionals | none | none | ORCA 6.0.0 refuses D3/D4 with VV10 |
| `r2scan-3c` | r2SCAN | - | `r2scan-3c` with s9 2, ga 2, gc 1 | dftd4's `app/driver.f90` special case; gCP and mTZVPP are not part of this crate |

D3(BJ) parameters come with s9 = 0: **ORCA 6.0.0's `D3BJ` is two-body only** (its energies
match ours without ATM, see below), and so is the s-dftd3 program unless `--atm` is given
(its Python API defaults to s9 = 1). ORCA's `! D3BJ ABC` adds the ATM term with s9 = 1,
which `with_atm(1.0)` reproduces. D4 parameters follow the dftd4 program: ATM included
(s9 = 1) where dftd4 has `bj-eeq-atm` parameters, which is also what ORCA 6.0.0 does.

## Validation

Reference programs: s-dftd3 1.7.0 and dftd4 4.3.1 (conda-forge builds, micromamba env
`linus-disp-ref`), ORCA 6.0.0. Geometries go to every program in bohr (CODATA 2022 from the
xyz files in Angstrom). Molecules: the AFH benchmark set (la18, ts61, ts113, u1an39), water,
ClCN-H, crambin (640 atoms, +2) and heavier elements: ferrocene, [Cu(NH3)4]2+, [ZnCl4]2-,
CF3Br-benzene, cisplatin, SnI4, GdCl3(H2O), UF6...Xe (`tests/molecules`).
Cases: D3(BJ) B3LYP, D3(BJ)+ATM B3LYP, D3(BJ) wB97X-D3BJ, D4 B3LYP, D4 wB97X-D4, D4 r2SCAN-3c.

### Against s-dftd3 and dftd4 (`tools/compare_ref.py`)

Worst over 15 molecules x 6 cases (90 cases; Hessians for all but crambin, where the D3
Hessians were compared and the D4 Hessian was not):

| quantity | D3(BJ) (+ATM) vs s-dftd3 | D4 vs dftd4 | target |
|---|---|---|---|
| energy | 2.2e-16 Eh | 1.1e-15 Eh (crambin, E = 1.78 Eh) | 1e-12 |
| gradient | 3.7e-18 Eh/bohr | 5.9e-16 Eh/bohr | 1e-11 |
| Hessian vs reference program | 5.4e-17 Eh/bohr^2 (s-dftd3 analytic) | 1.4e-9 (dftd4's two-body Hessian is a central difference, step 1e-4) | - |
| Hessian vs 5-point FD of our gradient (h = 1e-3 bohr) | 1.8e-12 Eh/bohr^2 | 1.3e-11 Eh/bohr^2 | 1e-8 |
| gradient vs 5-point FD of our energy | 4.1e-13 Eh/bohr | 8.0e-13 Eh/bohr | - |

Most D3 energies and gradients are bitwise identical to s-dftd3. The same comparisons for
9 molecules run in `cargo test` (`tests/reference.rs`, fixtures from
`tools/make_fixtures.py`), together with finite differences, Hessian symmetry and
translational invariance, and bitwise identity on 1, 3, 8 and 17 threads
(`tests/derivatives.rs`).

### Against ORCA 6.0.0 (`tools/orca_check.py`)

10 molecules (water, ClCN-H, la18, u1an39, ts61, CF3Br-benzene, ferrocene, [ZnCl4]2-,
cisplatin, SnI4) x the 6 cases, ORCA 6.0.0 `EnGrad` with a cheap SCF (STO-3G, def2-SVP for
elements beyond Kr, r2SCAN-3c with its own basis; `! Bohrs` on the CODATA 2022 bohr geometry);
only ORCA's printed dispersion energies and gradients are compared:

| case | energy, max abs difference | ORCA prints | dispersion gradient, max abs difference |
|---|---|---|---|
| D4 B3LYP, wB97X-D4, r2SCAN-3c | 5.0e-10 Eh | 9 decimals | 5.0e-10 Eh/bohr (9 decimals) |
| D3(BJ), D3(BJ) ABC, wB97X-D3BJ: water, ClCN-H, ZnCl4, SnI4, cisplatin | 3.0e-11 Eh | 12 decimals | see below |
| D3(BJ) ...: la18, CF3Br-benzene, ferrocene | 5.8e-10 Eh | 12 decimals | |
| D3(BJ) ...: u1an39, ts61 | 6.8e-9 Eh (ts61, 5e-8 relative) | 12 decimals | |

* **D4** matches ORCA to its printed precision, energies and gradients (ORCA 6.0.0 links
  dftd4 3.4.0). This confirms the r2SCAN-3c settings (s9 2, ga 2, gc 1) and wB97X-D4.
* **ORCA's D3(BJ) energy is two-body only by default**, and `ABC` adds exactly our ATM term.
  ORCA's D3 energies deviate from s-dftd3 by up to 6.8e-9 Eh, growing with system size; they
  come from ORCA's own element tables (ORCA prints "DFTD3 V3.1 Rev 1"; the rounded tables of
  the original dftd3 3.1 program, `autoang = 0.52917726`, reproduce la18 to 2e-13 but not the
  N/F/Cl-containing molecules). linus follows s-dftd3, the maintained reference.
* **ORCA 6.0.0's D3 gradient is not the derivative of its D3 energy.** Central differences of
  ORCA's own printed dispersion energy agree with linus-disp (and s-dftd3), but ORCA's printed
  `DISPERSION GRADIENT`, which ORCA adds to its total gradient (checked: total(D3BJ) -
  total(no D3) = printed dispersion gradient to 5e-10), differs by up to 1.7e-3 Eh/bohr (ts61,
  on K; the correct value there is 7.6e-5), 9.8e-5 on la18 and 1.1e-6 on water. Zero damping
  (`D3ZERO`) is affected too. ORCA's numbers are close to the explicit (fixed-C6) gradient
  without most of the coordination-number chain rule. D4 gradients are fine. Consequence: ORCA
  6.0.0 D3 geometry optimizations and `NumFreq` Hessians are not consistent with ORCA's D3
  energies; compare linus D3 gradients with s-dftd3 or with finite differences, not with ORCA.

### Upstream data quirk

s-dftd3 1.7.0 lists 7 reference CNs and C6 blocks for Cm but `number_of_references(96) = 6`,
so the 7th Cm reference (CN 5.6866) is never used; linus-disp does the same
(`tools/gen_data.py` checks that nothing else is dropped).

## Timing

`tools/bench.py`, 2026-11-06, AMD Threadripper PRO 9985WX (64 cores), while another
(niced) job used about 30 cores. linus-disp: in-process wall time, best of 5, B3LYP
parameters. s-dftd3 / dftd4: whole program run (includes ~2 ms start-up and input reading),
best of 3, `OMP_NUM_THREADS` as given.

| molecule (atoms) | term | linus 1 thread | linus 8 | linus 32 | s-dftd3 / dftd4 1 thread | 8 | 32 |
|---|---|---|---|---|---|---|---|
| la18 (18) | D3(BJ) gradient | 26 us | 54 us | 75 us | 2.0 ms | 2.1 ms | 3.1 ms |
| | D3(BJ) Hessian | 67 us | 145 us | 208 us | 2.6 ms | 2.8 ms | 3.8 ms |
| | D3(BJ)+ATM Hessian | 0.30 ms | 0.16 ms | 0.30 ms | 2.9 ms | 2.8 ms | 4.1 ms |
| | D4 gradient | 76 us | 118 us | 210 us | 2.5 ms | 3.4 ms | 8.6 ms |
| | D4 Hessian | 0.50 ms | 0.42 ms | 0.76 ms | 13 ms | 6.6 ms | 12 ms |
| ts113 (113) | D3(BJ) gradient | 0.42 ms | 0.17 ms | 0.23 ms | 2.6 ms | 2.3 ms | 3.2 ms |
| | D3(BJ) Hessian | 6.0 ms | 1.5 ms | 1.8 ms | 51 ms | 51 ms | 51 ms |
| | D3(BJ)+ATM gradient | 9.8 ms | 1.5 ms | 0.90 ms | 16 ms | 4.3 ms | 4.0 ms |
| | D3(BJ)+ATM Hessian | 76 ms | 12 ms | 5.6 ms | 140 ms | 64 ms | 62 ms |
| | D4 gradient | 5.4 ms | 1.9 ms | 1.8 ms | 36 ms | 11 ms | 13 ms |
| | D4 Hessian | 95 ms | 32 ms | 26 ms | 2.0 s | 0.30 s | 0.19 s |
| crambin (640) | D3(BJ) gradient | 13 ms | 2.6 ms | 2.1 ms | 24 ms | 11 ms | 10 ms |
| | D3(BJ) Hessian | 0.82 s | 0.13 s | 65 ms | | | |
| | D3(BJ)+ATM gradient | 1.5 s | 0.20 s | 60 ms | 2.1 s | 0.28 s | 99 ms |
| | D3(BJ)+ATM Hessian | 12 s | 1.5 s | 0.48 s | | | |
| | D4 gradient | 0.59 s | 0.13 s | 73 ms | 4.7 s | 0.69 s | 0.26 s |
| | D4 Hessian | 15 s | 5.5 s | 4.7 s | | | |

Pinned to the 8 physical cores of one CCD (`slice bench`, 2026-11-07, node still shared;
same script), the 8-thread times agree with the table within the noise: ts113 D3(BJ)
gradient 0.14 ms, D3(BJ) Hessian 1.5 ms, D3(BJ)+ATM Hessian 11 ms, D4 gradient 1.9 ms,
D4 Hessian 30 ms; crambin D3(BJ)+ATM Hessian 1.4 s, D4 gradient 0.12 s, D4 Hessian 7.2 s.

For scale: ORCA 6.0.0 B3LYP/def2-SVP RIJCOSX single points take 3.9 s for la18 (8 processes)
and 29 s for ts61 (16 processes) on this machine; a DFT Hessian of ts113 takes hours. The
dispersion terms are below 0.1 % of that. The ATM Hessian computes every triple three times
(once per owning atom) to stay thread-count independent. The D4 Hessian of crambin stops
scaling at about 4.5 s because the chain-rule products inside axtb-d4's `hessian()` are
serial loops (an easy rayon change upstream in axtb). Small molecules gain nothing from
threads (rayon start-up is ~50 us).

## Using it with linus

The dispersion terms do not depend on the density: add `energy`, `gradient` and `hessian` to
the SCF energy, the nuclear gradient and the Hessian. `linus_xc::Functional::dispersion()`
returns `"D3BJ"` for the ORCA-style `*-d3bj` names; `d3bj_params(name)` resolves the same
names. The `linus` crate links linus-disp through its `dispersion` feature, which is on by
default (`crates/linus/src/dispersion.rs`; without the feature, dispersion requests are rejected), so
a default build of `linus`, linus-cli or linus-py is subject to the LGPL for linus-disp;
`--no-default-features` leaves it out (the default is the maintainers' decision for each release).

## Why axtb-d4 is a crates.io dependency

axtb-d4 already has the D4 model with analytic gradients and Hessians, validated
bit-identically against dftd4 4.3.1; rewriting it would duplicate ~4,000 lines and their
validation. Options considered:

* **the crates.io release, pinned exactly** (chosen, `axtb-d4 = "=0.1.1"`): one source of truth,
  an immutable published version, and `Cargo.lock` records its checksum. Fixes arrive by
  bumping the version. Offline builds work after `cargo fetch` or with `cargo vendor`. axtb-d4
  depends only on rayon.
* git dependency pinned to a commit (used until 2026-11-10): the pinned commit existed only in
  AFH's private axtb development repository, not in the public `atomsforhumanity/axtb` (whose
  history was cut for release), so clean checkouts and CI could not build. The crates.io
  release gives bitwise-identical D4 energies, gradients and Hessians. 0.1.1 (2026-11-11) has
  0.1.0's code and declares Rust 1.89 instead of 1.99, so linus's own minimum holds again.
* path dependency: breaks for every checkout without axtb next to it, including git worktrees
  (their relative depth differs).
* vendored copy: self-contained, but a second copy of 4,000 lines that drifts from the
  validated original.

What linus-disp adds for DFT-D4 is what axtb-d4 lacks as a dftd4 replacement: the EEQ charges
of the molecule and their response (dq/dR, d2q/dR2), and the dftd4 program's parameter
handling (functional table, r2SCAN-3c). EEQ is not taken from axtb, whose EEQ (in the large
`axtb` crate, for SCF guesses) has no derivatives.

## License and provenance

linus-disp ports s-dftd3 and uses dftd4 data (both LGPL-3.0-or-later), ports multicharge and
mctc-lib code (Apache-2.0) and links axtb-d4 (LGPL-3.0-or-later), so it is
**LGPL-3.0-or-later** (`COPYING.LESSER`, `COPYING`, `LICENSE-APACHE`, `NOTICE.md`). It
depends on no linus crate. `linus-ints`, `linus-grid` and `linus-xc` do not depend on it; the
`linus` crate does, through its default `dispersion` feature, and linus-cli and linus-py
through `linus`. The sources of those crates stay MIT, but a binary that links linus-disp (any
default build of them) must meet the LGPL for it. Building them with `--no-default-features`
leaves linus-disp out (THIRD_PARTY_NOTICES.md at the repository root).

Every table is generated by `tools/gen_data.py` from the release tarballs (URLs and SHA-256 in
the script and in the headers of `src/data/*.rs`); literals keep the Fortran digits, and the
array operations of the Fortran declarations (`aatoau *`, `4/3 *`,
`sqrt(0.5 * r4r2 * sqrt(Z))`) are applied in Rust with the same IEEE operations.

| Table | Source (path in the tarball) | Reference |
|---|---|---|
| D3 reference C6 (compact `nref(i) x nref(j)` blocks) | simple-dftd3-1.7.0 `src/dftd3/reference.f90` `copy_c6` | Grimme et al., JCP 132, 154104 (2010) |
| D3 reference CN, number of references | same file, `reference_cn`, `number_of_references` | same |
| <r4>/<r2> | simple-dftd3-1.7.0 `src/dftd3/data/r4r2.f90` | same; rare gases as in s-dftd3 |
| vdW radii (ATM zero damping) | simple-dftd3-1.7.0 `src/dftd3/data/vdwrad.f90` | same |
| covalent radii (D3 and EEQ CN) | mctc-lib-0.7.0 `src/mctc/data/covrad.f90` | Pyykko and Atsumi, Chem. Eur. J. 15, 188 (2009), metals -10 % |
| CODATA 2018 constants for `aatoau` | mctc-lib-0.7.0 `src/mctc/io/codata2018.f90`, `constants.f90` | |
| D3(BJ) s6, s8, a1, a2 and names (156 sets) | simple-dftd3-1.7.0 `src/dftd3/param.f90` (`get_rational_damping`, `get_method_id`), DOIs from `citation.f90` | per entry (`reference` field) |
| D4 s6, s8, a1, a2, default s9 and names (112 sets) | dftd4-4.3.1 `src/dftd4/param.f90` (`get_d4eeq_bjatm_parameter`, `get_d4eeq_bj_parameter`, `get_functional_id`) | per entry (DOI or dftd4 fit tag such as `SAW190103`) |
| r2SCAN-3c s9, ga, gc | dftd4-4.3.1 `app/driver.f90` (hand-coded in `src/params.rs`) | Grimme et al., JCP 154, 064103 (2021) |
| EEQ chi, eta, kcnchi, rad | multicharge-0.6.0 `src/multicharge/param/eeq2019.f90` | Caldeweyher et al., JCP 150, 154122 (2019) |
| D4 reference data | axtb-d4 (generated from dftd4 4.3.1 by its own `scripts/gen_tables.py`) | same |

## Reproducing

```bash
# tables (downloads and checks the four release tarballs)
python3 -I crates/linus-disp/tools/gen_data.py --src /some/dir --fetch
# reference programs
micromamba create -n linus-disp-ref -c conda-forge --override-channels python=3.12 \
    simple-dftd3=1.7.0 dftd4=4.3.1 dftd3-python=1.7.0 dftd4-python=4.3.1 numpy
cargo build --release -p linus-disp --examples
python3 -I crates/linus-disp/tools/compare_ref.py crates/linus-disp/tests/molecules/*.xyz   # crambin: --order gradient --fd-step 0
python3 -I crates/linus-disp/tools/orca_check.py --workdir /scratch/orca crates/linus-disp/tests/molecules/la18.xyz
python3 -I crates/linus-disp/tools/make_fixtures.py     # tests/fixtures
python3 -I crates/linus-disp/tools/bench.py --threads 1,8,64 crates/linus-disp/tests/molecules/{la18,ts113,crambin}.xyz
cargo test --release -p linus-disp
```

`examples/disp_json.rs` is a JSON driver (energy, gradient, Hessian, optional
finite-difference Hessian and the D3 gradient split into its parts).
