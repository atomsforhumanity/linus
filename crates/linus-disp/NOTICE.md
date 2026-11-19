# linus-disp: licenses and provenance

linus-disp is licensed under the **GNU Lesser General Public License v3.0 or later**
(`COPYING.LESSER`, which supplements the GPL v3 in `COPYING`). The rest of the linus
workspace is MIT. The `linus` crate depends on linus-disp through its `dispersion` feature,
which is enabled by default, so default builds of `linus`, the `linus` command-line program
(linus-cli) and the Python package (linus-py) contain this LGPL-3.0 code and must meet the
LGPL for it; `linus`, linus-cli and linus-py built with `--no-default-features` do not (the
default is the maintainers' decision for each release).

It derives from these works:

| Work | Version | License | What linus-disp takes from it |
|---|---|---|---|
| [s-dftd3](https://github.com/dftd3/simple-dftd3) (S. Ehlert et al.) | 1.7.0 | LGPL-3.0-or-later | D3 algorithms ported to Rust (`src/d3.rs`); reference C6, reference CN, number of references, <r4>/<r2>, vdW radii, D3(BJ) functional parameters (`src/data/d3.rs`, `src/data/params.rs`, generated) |
| [dftd4](https://github.com/dftd4/dftd4) (E. Caldeweyher, S. Ehlert et al.) | 4.3.1 | LGPL-3.0-or-later | D4 functional parameters and names (`src/data/params.rs`, generated); r2SCAN-3c settings of `app/driver.f90` |
| [axtb-d4](https://github.com/atomsforhumanity/axtb) (Atoms for Humanity) | 0.1.1 (crates.io) | LGPL-3.0-or-later | the D4 model, damping, ATM term and their derivatives (Cargo dependency, not copied) |
| [multicharge](https://github.com/grimme-lab/multicharge) (S. Ehlert et al.) | 0.6.0 | Apache-2.0 | EEQ 2019 model ported to Rust (`src/eeq.rs`); EEQ parameters (`src/data/eeq.rs`, generated) |
| [mctc-lib](https://github.com/grimme-lab/mctc-lib) (S. Ehlert et al.) | 0.7.0 | Apache-2.0 | covalent radii, CODATA 2018 constants (`src/data/common.rs`, generated); counting functions and CN cap |

The Apache-2.0 license text is in `LICENSE-APACHE`. Apache-2.0 code may be included in an
LGPL-3.0 work; the original copyright and license notices of multicharge and mctc-lib are
kept by this file and by the headers of the derived source files.

The published data originate from:

* D3 reference C6 and CN, covalent radii: S. Grimme, J. Antony, S. Ehrlich, H. Krieg,
  J. Chem. Phys. 132, 154104 (2010); Pyykko and Atsumi, Chem. Eur. J. 15, 188 (2009).
* Becke-Johnson damping: S. Grimme, S. Ehrlich, L. Goerigk, J. Comput. Chem. 32, 1456
  (2011); per-functional references in `src/data/params.rs` (`reference` field).
* D4 and EEQ: E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher,
  C. Bannwarth, S. Grimme, J. Chem. Phys. 150, 154122 (2019).
