# Third-party notices

linus is MIT-licensed (`LICENSE`, Copyright (c) 2026 Atoms for Humanity). It contains,
links or ships the third-party works below, each under its own license; the license texts are
in `LICENSES/`. This file and `LICENSES/` travel with every copy of linus: the source tree,
the Python wheel (in its `.dist-info/licenses/`) and command-line release archives.

## What a build contains

| Part | License | In the source | In the CLI and the Python wheel |
|---|---|---|---|
| linus (crates `linus`, `linus-ints`, `linus-grid`, `linus-xc`, `linus-cli`, `linus-py`) | MIT | yes | yes |
| linus-gpu (in development; GPU kernels behind its `gpu` feature) | MIT | yes | no |
| linus-disp (D3/D4 dispersion) and axtb-d4 | LGPL-3.0-or-later | yes (axtb-d4: a pinned Cargo dependency) | with the `dispersion` feature (on by default) |
| libxc 7.0.0 | MPL-2.0 | downloaded at build time | statically linked; GPU builds also embed the source of 24 functional files, compiled at run time |
| Basis Set Exchange basis sets and ECPs (`data/basis`) | BSD-3-Clause | yes | the wheel ships them |
| Data and constants derived from PySCF 2.15 | Apache-2.0 | yes | yes |
| Lebedev-Laikov tables (SciPy, Robert Parrish) | BSD-3-Clause, BSD-2-Clause | yes | yes |
| SMD CDS tables after mnsol.F (NWChem) | ECL-2.0 | yes | yes |
| ORCA 6.0.0-derived data (`data/orca`) and ORCA results (comparisons, reference values) | included with permission of FAccTs GmbH | yes | the `data/orca` presets are compiled in |
| Rust crates from crates.io (`Cargo.lock`) | MIT, Apache-2.0, BSD-2-Clause, Zlib, Unlicense (see below) | no (fetched by Cargo) | statically linked |

SPDX expression of a default binary build:
`MIT AND LGPL-3.0-or-later AND MPL-2.0 AND Apache-2.0 AND BSD-3-Clause AND BSD-2-Clause AND ECL-2.0`.
Without the `dispersion` feature (`cargo build --no-default-features` for `linus`,
`linus-cli` or `linus-py`) the LGPL-3.0-or-later part is left out, and D3/D4 dispersion
requests are refused with a message.

## linus-disp and axtb-d4 (LGPL-3.0-or-later)

`crates/linus-disp` ports s-dftd3 1.7.0 and uses dftd4 4.3.1 data (both LGPL-3.0-or-later),
ports multicharge 0.6.0 and mctc-lib 0.7.0 code (Apache-2.0), and links axtb-d4
(LGPL-3.0-or-later, https://github.com/atomsforhumanity/axtb, the crates.io release pinned
exactly in `crates/linus-disp/Cargo.toml`), so it is licensed LGPL-3.0-or-later. Authors, versions and
what is taken from each work: `crates/linus-disp/NOTICE.md`. License texts:
`LICENSES/LGPL-3.0.txt` and the GPL it supplements, `LICENSES/GPL-3.0.txt`; Apache-2.0:
`LICENSES/Apache-2.0.txt`.

The `linus` crate depends on linus-disp through its `dispersion` feature, which linus-cli and
linus-py forward and which is enabled by default. A default build of the `linus` library, the
`linus` command-line program or the Python package therefore contains LGPL-3.0 code. The
complete corresponding source of each release (linus-disp included; `Cargo.lock` pins
axtb-d4) is the linus source at that release, from which a modified linus-disp can be rebuilt
and relinked.

## libxc (MPL-2.0)

`crates/linus-xc` downloads libxc 7.0.0 (Susi Lehtola, Miguel A. L. Marques and the libxc
contributors, https://libxc.gitlab.io), checks its SHA-256, builds it unmodified and links it
statically. Source code: https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz
(SHA-256 `8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77`). License text:
`LICENSES/MPL-2.0.txt`. linus's FFI declarations (`crates/linus-xc/src`) and the glibc `cbrt`
shim (`crates/linus-xc/csrc`) are linus's own.

Builds with the GPU backend (the `gpu` feature: the `afh-linus-cuda13` wheel, the GPU container
image) also carry, unmodified apart from leaving out their third and fourth derivatives, libxc's
Maple-generated functional code and the parameter structs of 24 of its source files
(`src/maple2c/*/<name>.c` and `src/<name>.c`, extracted from the same tarball at build time by
`crates/linus-xc/build.rs`, feature `device-sources`, which lists them). linus compiles them for the
GPU at run time with NVRTC, inside a harness of its own (`crates/linus-gpu/device/xc/devfunc.cuh`,
`crates/linus-gpu/src/xc/devfunc.rs`); they remain under MPL-2.0, with their source at the address
above.

## Basis Set Exchange (BSD-3-Clause)

The basis sets and effective core potentials in `data/basis` were exported from the Basis Set
Exchange (`basis_set_exchange` 0.12, version in `data/basis/VERSION`) by
`tools/fetch_basis.py`; the Python wheel ships them. Copyright (c) 2020 The Molecular Sciences
Software Institute, Virginia Tech. License text: `LICENSES/BSD-3-Clause-BSE.txt`. Cite the
Basis Set Exchange (B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus,
J. Chem. Inf. Model. 59, 4814 (2019)) and the papers of the basis sets you use.

## PySCF (Apache-2.0)

Copyright 2014-2024 The PySCF Developers. The attribution notices of PySCF's NOTICE file
are reproduced in `LICENSES/Apache-2.0-PySCF-NOTICE.txt`; license text:
`LICENSES/Apache-2.0.txt`. The code is
linus's own; these numbers come from PySCF 2.15:

- Integration grids (`crates/linus-grid`): the `pyscf` presets reproduce PySCF's `pyscf.dft`
  grids point for point and take from it the per-level radial and angular sizes (`gen_grid`
  `RAD_GRIDS` and `ANG_ORDER`), the NWChem pruning region fractions, and the element radii as
  PySCF tabulates them (`pyscf.data.radii` `BRAGG` and `COVALENT`, with PySCF's choices for H,
  the noble gases and heavy elements; the Treutler-Ahlrichs xi values beyond Kr as PySCF and
  Psi4 use them), generated by `crates/linus-grid/tools/gen_element_tables.py`.
- PCM (`crates/linus-solvent`, `crates/linus-ints/src/points/pcm_ints.rs`): follows
  `pyscf.solvent.pcm` and takes from it the van der Waals radius table of `pyscf.data.radii`
  (`solvent/radii.rs`) and the Gaussian-charge exponent constants per Lebedev grid
  (`solvent/surface.rs`, from J. Chem. Phys. 122, 194110 (2005)). The matrices and their
  derivatives were written for linus from the published formulas.
- Solvent dielectric constants as PySCF uses them (`solvent/data/program_eps.csv`, from
  `pyscf/solvent/_solvent_data.py`), for the PySCF preset.

PySCF's `smd*.py` files (GPL-3.0 headers inherited from GPU4PySCF) were used only as a
numerical reference; no code was taken from them.

## Lebedev-Laikov quadratures (BSD-3-Clause, BSD-2-Clause)

`crates/linus-grid/src/lebedev_data.rs` is generated by
`crates/linus-grid/tools/gen_lebedev_table.py` from SciPy 1.18.1's
`scipy/integrate/_lebedev.py` (Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy
Developers; `LICENSES/BSD-3-Clause-SciPy.txt`), a translation of `getLebedevSphere` by
Robert Parrish (Copyright (c) 2010, Robert Parrish; `LICENSES/BSD-2-Clause-Parrish.txt`, also
reproduced in the header of the generated file). The values are the published Lebedev-Laikov
parameters (V. I. Lebedev, D. N. Laikov, Doklady Mathematics 59, 477 (1999)).

## SMD solvation: mnsol.F (ECL-2.0) and published data

The SMD CDS term and its exact solvent-accessible surface area were written for linus from the
paper (A. V. Marenich, C. J. Cramer, D. G. Truhlar, J. Phys. Chem. B 113, 6378 (2009)) and the
geometry. The COT distance table, the van der Waals radii table (`MNSOL_VDW` in
`crates/linus-solvent/src/smd.rs`) and the single-precision parameter convention follow the
Minnesota CDS code `mnsol.F` (Marenich, Cramer and Truhlar) as distributed with NWChem, which
was read as a reference, not translated. NWChem: Copyright (c) 1994-2020 Pacific Northwest
National Laboratory, Battelle Memorial Institute, licensed under the Educational Community
License, Version 2.0 (`LICENSES/ECL-2.0.txt`).

Solvent data (`crates/linus-solvent/src/data`, provenance in its `README.md`):

- Dielectric constants from the CRC Handbook of Chemistry and Physics, 106th ed.
  (C. Wohlfarth, "Permittivity (Dielectric Constant) of Liquids", CRC Press), vendored in part
  (the shipped solvents' rows) from AFH's qcref v0.5.1 (MIT). Cite the Handbook.
- SMD solvent descriptors from the Minnesota Solvent Descriptor Database (P. Winget,
  D. M. Dolney, D. J. Giesen, C. J. Cramer, D. G. Truhlar, University of Minnesota, version of
  2026-01-06). Cite the database and the SMD paper.

## ORCA (FAccTs GmbH)

linus contains no ORCA code or binaries. It does contain numbers obtained with ORCA 6.0.0,
included with permission of FAccTs GmbH (2026-11-17): the `orca:N` grid tables, which reproduce
ORCA's DefGrid1-3 and NL grid layouts as determined from ORCA's printed output; the `orca` PCM
preset's constants; ORCA's solvent permittivities and names (`data/orca`, compiled into the
`linus-grid` and `linus-solvent` crates); and the ORCA results in the documentation, the
validation records (`reference/validation`) and the tests, each identified as ORCA 6.0.0's.

## linus-gpu

`crates/linus-gpu` (MIT) is not linked into the CLI or the Python package. With its `gpu`
feature it uses the cudarc crate (MIT OR Apache-2.0), which loads the NVIDIA CUDA driver,
NVRTC, cuBLAS and cuSOLVER from the user's CUDA installation at run time; no NVIDIA code is in
this repository or linked at build time.

## Psi4

The `psi4` grid preset uses the Bragg-Slater radii (bohr) that Psi4 1.11 lists in
`libfock/cubature.cc` (`GetBSRadius`): published radii, numbers only, no Psi4 code.

## Rust crates

The binaries statically link the crates.io dependencies recorded in `Cargo.lock`. Each is
licensed under MIT, MIT OR Apache-2.0, Apache-2.0 OR MIT OR Zlib, Unlicense OR MIT, or
BSD-2-Clause (`numpy`, in the Python package, and `atomic-wait`); none is GPL, AGPL or of
unknown license. Their license texts and copyright notices are in the crate sources that
Cargo downloads (`cargo metadata` lists the license of each).
