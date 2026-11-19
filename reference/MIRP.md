# MIRP: arbitrary-precision reference integrals for linus

[MIRP](https://github.com/MolSSI/MIRP) (MolSSI Integral Reference Project, B. P. Pritchard, BSD-3)
evaluates Gaussian integrals with Arb ball (interval) arithmetic, so every value carries a
rigorous error bound. Use it to check linus's ERIs and Boys function. It is a 2018 code that is
no longer developed (last release v1.0.1; open issues for overlap/kinetic/potential date from 2018).

**What it computes:** the Boys function F_m(t) and 4-centre ERIs (ab|cd) over **Cartesian**
contracted GTOs (segmented and general contractions, any l; data goes up to h).
**What it does not compute:** overlap, kinetic, nuclear attraction, multipoles, 2-/3-centre ERIs,
**no derivatives** of any kind, **no spherical (pure) functions**, no ECPs, no range-separated or
attenuated Coulomb. Branch `enuc_pot` has no commits beyond master. For S/T/V, linus needs another
reference (e.g. a high-precision mpmath Obara-Saika implementation).

## Layout (all git-ignored except this file and `mirp_ref.py`)

| Path | What |
|---|---|
| `.refenv-mirp/` | conda env (Arb 2.23 + FLINT 2.9, gcc 15, cmake 4) |
| `reference/src/MIRP/` | clone, commit `ca4d36c0d5fd8ef42cb7bca4f09667f6e11f61c8` (tag v1.0.1, 2018-02-12) |
| `reference/src/MIRP/install/{bin,lib,include}` | built programs, `libmirp.so`, headers (RPATH set, no env activation needed) |
| `reference/src/MIRP/tests/generator/{basis,geometry}/` | MIRP's own `.bas`/`.xyz` inputs |
| `reference/src/mirp-data/zenodo-1043591-v1.0/` | Zenodo `.ref.bz2` files as downloaded (1.7 GB), plus `md5sums`, `sha256sums`, record JSON |
| `reference/src/mirp-data/extracted/<name>/<name>.ref` | each file decompressed into its own directory (12.4 GB total) |
| `reference/out/mirp/` | references generated here (see below) |
| `reference/mirp_ref.py` | stdlib-only reader for `.ref`/`.ref.bz2` plus `summary` CLI (Cartesian order, normalization) |

## Build (as run on AFH's benchmark node; no source patches needed)

conda-forge still ships Arb 2.x (`arb 2.23.0`, pinned to `libflint 2.9`), which is what MIRP's CMake
expects (`find_library(arb|flint|mpfr|gmp)`), so the FLINT 3 port is unnecessary.

```bash
export MAMBA_ROOT_PREFIX=$HOME/micromamba
E=$PWD/.refenv-mirp; S=$PWD/reference/src/MIRP     # from the repository root
micromamba create -y -p $E -c conda-forge --override-channels \
  "arb=2.23.0" "libflint=2.9" mpfr gmp c-compiler cxx-compiler cmake ninja make python=3.12
micromamba install -y -p $E -c conda-forge --override-channels mpmath   # only for tests/generator/*.py
git clone https://github.com/MolSSI/MIRP $S
micromamba run -p $E bash -c "
  cmake -S $S -B $S/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_PREFIX_PATH=$E -DCMAKE_INSTALL_PREFIX=$S/install \
        -DCMAKE_INSTALL_RPATH='$S/install/lib;$E/lib' -DMIRP_OPENMP=ON
  cmake --build $S/build -j16 && cmake --install $S/build
  cd $S/build && OMP_NUM_THREADS=4 ctest"
```

Resolved: arb 2.23.0, libflint 2.9.0, mpfr 4.2.2, gmp 6.3.0, gcc/g++ 15.3.0, cmake 4.4.5,
ninja 1.13.2, python 3.12.16, mpmath 1.4.1. Notes:
- `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required because CMake 4 rejects MIRP's `cmake_minimum_required(VERSION 3.2)`.
- **ctest: 54/54 pass serially** (75 s with `OMP_NUM_THREADS=8`). MIRP declares no test
  dependencies, so under `ctest -jN` the 10 "verify the file we just created" tests can start
  before their input exists and fail (`std::bad_alloc` / 0.00 s). A re-run passes.
- `-DMIRP_OPENMP=ON` parallelises over Cartesian components. **Always set `OMP_NUM_THREADS`**
  (e.g. 8–16), otherwise it grabs all 128 threads of the shared node.
- The shipped C examples (`examples/`, `simple_gtoeri`, `read_ref`) also build against the install:
  `cmake -S $S/examples -B $S/build_examples -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_PREFIX_PATH="$E;$S/install"`.

## Programs (from `$S/install/bin`)

| Program | Purpose | Integrals (`--integral`) |
|---|---|---|
| `mirp_create_reference --integral gtoeri --basis X.bas --geometry X.xyz --outfile X.ref [--am ssss,dddd,...]` | **exact-double** ERI file for a whole molecule/basis | `gtoeri` |
| `mirp_verify_reference --integral gtoeri --file X.ref` | recompute every value and require bitwise equality | `gtoeri` |
| `mirp_create_test --infile X.inp --outfile X.dat --integral I --prec BITS --ndigits N` | arbitrary-precision values for listed shell quartets / (m,t) | `boys`, `gtoeri`, `gtoeri_single` |
| `mirp_verify_test --file X.dat --integral I --float interval --prec BITS` or `--float exact` | check MIRP's own interval or double paths against a `.dat` | same names (the `-h` text says `eri`; the code wants `gtoeri`) |

`--am` takes letters `s p d f g h i k ...` per quartet and must match the **ordered** quartet
(p,q,r,s) as written (p ≥ q, r ≥ s by shell index). Which order appears therefore depends on
shell order: `dfdf` and `fdfd` select different shell pairs.

## Precision model

- **Reference files (`create_reference`)**: inputs are doubles (parsed with `strtod`). For each
  shell quartet the working precision starts at 128 bits and grows by 64 until every integral has
  ≥ 64 bits relative accuracy (or its ball lies below `DBL_TRUE_MIN`, giving exactly `0`). The
  stored value is the ball midpoint rounded to nearest double: the true value **for those exact
  double inputs**, to within about 0.5 ulp (not guaranteed correctly rounded, but within an ulp).
  Digits cannot be chosen; output is always written as C99 hex floats (`0x1.323e82f79b97dp+2`).
- **Test files (`create_test`)**: input numbers are decimal strings, read at `--prec` bits
  (effectively exact). Output is `[mid +/- radius]` with `--ndigits` significant digits. The
  program **throws** if the achieved relative accuracy is below `(ndigits+5)·log2(10)` bits, so
  to get more digits raise `--prec` with margin for cancellation (used here: 256 bits/30 digits,
  512/40; MIRP's own test files use 2048/101).
- **Library**: `mirp_boys(arb_ptr F, int m, const arb_t t, slong prec)`, `mirp_boys_str`,
  `mirp_boys_exact(double *F, int m, double t)` (F_0..F_m), `mirp_gtoeri_single` (one
  Cartesian component over **unnormalized primitives**), and per-shell-quartet `mirp_gtoeri`,
  `_str` and `_exact` (normalized contracted shells). The shell-quartet ERI functions are
  `static inline` macros in `mirp/kernels/gtoeri.h`, so they are **not exported** from
  `libmirp.so` and exist only for C/C++ code that includes the header (see
  `$S/examples/simple_gtoeri.c`, built and run OK). A Python/Rust caller needs a small C shim.
  `mirp_boys_exact` is exported and works from ctypes:
  `lib=ctypes.CDLL("$S/install/lib/libmirp.so"); F=(ctypes.c_double*5)(); lib.mirp_boys_exact(F, ctypes.c_int(4), ctypes.c_double(2.5))`.

## Inputs: basis set, molecule, geometry

- **Geometry**: XYZ file in **Å** (line 1 count, line 2 comment, then `El x y z`). MIRP converts
  with `1/0.52917721067` (CODATA 2014) and stores **bohr**. linus should take coordinates from the
  `.ref` file, not convert again (CODATA 2018's 0.529177210903 differs by about 4e-10 relative). `reference/molecules/*.xyz` work as-is.
- **Basis**: NWChem-format `.bas` (as exported by the Basis Set Exchange). Lines starting with
  `#` or `BASIS` are skipped, `END` stops. Each `El  L` block has rows `exponent c_1 ... c_ngen`
  (several columns = one **general contraction** shell). Combined `SP` blocks are **split into
  separate s and p shells** (s first). Shells are ordered by atom (xyz order), then as in the `.bas`.
- Shipped `.bas`: sto-3g, aug-cc-pvtz, aug-cc-pvqz, roos-ano-tz, ano-rcc (BSE, Aug 2017).
  **Pitfall:** aug-cc-pv{t,q}z.bas use Fortran `D` exponents (`0.478894D-05`) for Na–Ar. The C++
  reader's `strtod` silently stops at the `D`, giving a **wrong coefficient with no error**, and
  the Python generator crashes on it. H–Ne (and so all Zenodo data) are unaffected. For other
  elements or bases, convert `D`→`E` first (as in `reference/out/mirp/basis/aug-cc-pvtz.bas`).
- To make test inputs for random shell quartets from a molecule/basis, run MIRP's generator with
  the env's Python: `$E/bin/python $S/tests/generator/generate_integral_frombasis.py --filename X.inp --basis B.bas --geometry M.xyz --ncenter 4 --ndigits 30 --ntests 20 --seed N`.

## Worked examples (run 2026-11-06; outputs in `reference/out/mirp/`)

```bash
B=$S/install/bin; G=$S/tests/generator; O=reference/out/mirp; export OMP_NUM_THREADS=8
$B/mirp_create_reference --integral gtoeri --basis $G/basis/sto-3g.bas \
    --geometry reference/molecules/water.xyz --outfile $O/gtoeri_linus-water_sto-3g.ref   # 0.2 s
$B/mirp_verify_reference --integral gtoeri --file $O/gtoeri_linus-water_sto-3g.ref
#   0 / 474 failed (100% passed)
OMP_NUM_THREADS=16 $B/mirp_create_reference --integral gtoeri --basis $G/basis/aug-cc-pvtz.bas \
    --geometry reference/molecules/water.xyz --am ffff,dfdf --outfile $O/gtoeri_linus-water_aug-cc-pvtz_ffff-dfdf.ref  # 10 s wall
$B/mirp_create_test --infile $O/boys_grid.inp --outfile $O/boys_grid.dat --integral boys --prec 512 --ndigits 40
$B/mirp_verify_test --integral boys --file $O/boys_grid.dat --float exact      # 0 / 288 failed
python3 -I reference/mirp_ref.py summary $O/gtoeri_linus-water_sto-3g.ref
```

| File in `reference/out/mirp/` | Content |
|---|---|
| `gtoeri_water_sto-3g.ref` | MIRP water/STO-3G; identical (non-comment lines) to `tests/` and to the Zenodo file |
| `gtoeri_linus-{water,nh3}_sto-3g.ref` | full ERI set at linus's asymmetric geometries (474 / 730 values) |
| `gtoeri_linus-water_aug-cc-pvtz_{dddd,ffff-dfdf}.ref` | high-l classes at linus's water geometry |
| `4center_linus-water_aug-cc-pvtz.inp`, `gtoeri_linus-water_aug-cc-pvtz_test.dat` | 20 random contracted quartets, 30 digits, 256-bit |
| `boys_grid.inp`, `boys_grid.dat` | F_m(t), m ∈ {0..4,6,8,12,16,20,24,32}, t from 0 to 1e8 incl. 1e-15, 17.5, 33; 40 digits |

## File formats

`.ref` (exact double, hex floats; `#` lines are comments):

```
5                                              <- number of shells
0 3 1 0x0p+0 0x0p+0 0x1.ebe5b05b479ebp-3       <- am nprim ngen x y z (bohr)
0x1.056b2bfdb4cc2p+7 0x1.7cf1183b60286p+4 ...  <- nprim exponents
0x1.3c10d3b7cd8c4p-3 0x1.121687abdb659p-1 ... <- one line of nprim raw coefficients per general contraction
...
0 0 0 0 0x1.323e82f79b97dp+2                   <- p q r s then all values of (pq|rs)
2 2 0 0 0x1.1da5f9579ea8bp+0 0x0p+0 0x0p+0 0x0p+0 0x1.1da5f9579ea8bp+0 ...
```

`.dat` (test): header `ntests ndigits prec`, then each shell's `am nprim ngen x y z` and
`alpha c_1..c_ngen` rows (decimal, transposed relative to `.ref`), then `idx  [value +/- radius]`.
Exact zeros print as `[+/- 1.07e-154]`. Boys `.dat` lines are `m t [F +/- r]`. The annotated
examples are in `$S/doc/examples/`.

## Conventions linus must match to compare

- **Notation**: chemist's (pq|rs), with p,q on electron 1. Values for a quartet are row-major
  `[g_p][g_q][g_r][g_s][c_p][c_q][c_r][c_s]` (last index fastest; general-contraction indices outer, Cartesian inner).
- **Cartesian order** (`mirp_iterate_gaussian`): x-power descending, then y: d = xx xy xz yy yz zz;
  f = xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz (same as libint2/psi4 "CCA"). `mirp_ref.cart_lmn(l)`.
- **Normalization** (`mirp_normalize_shell`): file coefficients are raw basis-library values,
  treated as multiplying *normalized* primitives. Each general contraction is renormalized so that
  its **x^l component** has unit self-overlap. All Cartesian components share that factor, so
  e.g. d_xy has norm 1/3 (the libint/psi4 Cartesian convention). `mirp_ref.normalized_coeff()` reproduces it
  (checked: one-centre (O1s O1s|O1s O1s) recomputed to 2e-16).
- **Spherical**: none. Validate linus's Cartesian ERIs here, and test the cart→sph transform separately.
- **Which quartets are present**: the writer loops `p, r, q≤p, s≤r` and keeps a quartet when
  `p(p+1)/2+q ≥ r(r+1)/2+r`. That line has a bug (`+r` where `+s` is meant), so **every file
  omits the canonical quartets with r = p and s ≤ q < p**, (n−1)n(n+1)/6 of them, e.g.
  (10|10), (20|20), (21|20). Compare only the quartets in the file, keyed by (p,q,r,s); never assume the canonical set.
- **Zeros**: symmetry zeros and anything below 4.9e-324 are stored as `0x0p+0`.

### Recipe

1. Read the `.ref` (`mirp_ref.read_ref`, or port its ~40 parsing lines to Rust; Rust std has no
   hex-float parser, so use a crate such as `hexf-parse` or decode mantissa/exponent by hand). Build linus's shells **from the file** (hex floats are the exact double inputs MIRP used), not from linus's basis library.
2. Expand any `ngen>1` shell into `ngen` segmented shells with the same exponents (zero coefficients
   are explicit; aug-cc-pVTZ O s is 10 primitives × 4 contractions, two of them single-primitive), or support general contractions directly.
3. Normalize as above, compute Cartesian (pq|rs) for each listed quartet, and compare element-wise.
   Report max |Δ| and max |Δ|/|ref| per angular-momentum class. Start at |Δ| ≤ 1e-13 + 1e-11·|ref|
   and tighten as linus matures. MIRP is good to ~1 ulp, so any residual is linus's error.
4. For Boys: compare against `boys_grid.dat` (40 digits; extend `boys_grid.inp` and rerun
   `mirp_create_test` for more points) or call `mirp_boys_exact` via ctypes (F_0..F_m in double,
   matches the grid, e.g. F_3(2.5) = 0.022870837329790384).

## Zenodo reference data

Record DOI 10.5281/zenodo.1043590 is the **concept** record. It resolves to record **1043591**,
version **"1.0"**, published 2017-11-21, the only version (`is_last: true`), license CC-BY-SA-4.0.
The files were made by a pre-release build (`mirp_v0.6_knl`); our v1.0.1 build reproduces them
bit for bit (see below). Fetched with `https://zenodo.org/api/records/1043591/files/<name>/content`.
All `.bz2` match the Zenodo md5 and shipped `sha256sums`, and all decompressed `.ref` match the shipped `md5sums`/`sha256sums`.

| File (`.ref.bz2`) | bz2 bytes | .ref bytes | sha256 of .bz2 (first 16 hex) | shells | max l | Cart. nbf | quartets | values |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| water_sto-3g | 2,257 | 8,434 | `8e86d3833fc935fb` | 5 | p | 7 | 100 | 474 |
| h2_roos-ano-tz | 1,015,515 | 10,507,039 | `79207e50c51c5d94` | 6 | d | 50 | 196 | 983,249 |
| h2_ano-rcc | 10,196,110 | 111,537,535 | `5318900ff54347ab` | 8 | f | 92 | 582 | 10,478,048 |
| water_roos-ano-tz | 49,046,118 | 245,808,018 | `ee6234d4c725cefa` | 10 | f | 105 | 1,375 | 17,639,299 |
| water_aug-cc-pvtz | 50,948,565 | 227,376,106 | `8ccbcbb674167a13` | 20 | f | 105 | 20,825 | 16,326,445 |
| water_ano-rcc | 660,851,044 | 3,459,072,323 | `70bfa3742958f69a` | 13 | g | 205 | 3,822 | 247,873,017 |
| water_aug-cc-pvqz | 808,205,362 | 3,944,910,686 | `cc9a3372ea27f584` | 26 | g | 215 | 58,851 | 281,794,347 |
| u2-2ang_ano-rcc | 156,270,127 | 4,413,038,173 | `59d48525ec2cfbda` | 12 | h | 608 | 39 | 405,592,010 |

(All files are prefixed `gtoeri_`. Full checksums are in `reference/src/mirp-data/zenodo-1043591-v1.0/{md5sums,sha256sums}`.)

- **ERIs only**, all in `.ref` format (exact double, hex), basis embedded in each file.
  Geometries are MIRP's `tests/generator/geometry`: water = CCCBDB HF/STO-3G (O z=0.1271,
  H y=±0.7580 z=−0.5085 Å, C2v in the yz plane, so many symmetry zeros), H2 ±0.3561 Å on z, U2 ±1.0 Å on z.
- Except U2, each file has every quartet MIRP's loop writes (= canonical − (n−1)n(n+1)/6).
  **U2 is a deliberate subset** (`--am ssss,pppp,dddd,ffff,gggg,hhhh,hsss,hshs`, written as
  `...ref.head`): 39 quartets of heavily general-contracted shells (s: 26 primitives × 12 contractions,
  up to 31.6M values in a single `gggg` quartet).
- Cross-check: `mirp_verify_reference --integral gtoeri --file reference/src/mirp-data/extracted/<name>/<name>.ref`
  with our build reproduces three files **bitwise**: h2_roos-ano-tz (983,249 values, 31 s wall,
  8 threads), h2_ano-rcc (10,478,048 values, 199 s, 8 threads), water_aug-cc-pvtz (16,326,445
  values, 240 s wall / 3586 s CPU, 16 threads). All report `0 / N failed`.
- For linus: start with water_sto-3g and h2_*, then water_aug-cc-pvtz (f, general contractions)
  and water_aug-cc-pvqz / water_ano-rcc (g). Stream them, since they are GB-sized: `mirp_ref.read_ref` yields one quartet at a time and reads `.bz2` directly.

## Known gaps and gotchas

- Only Boys and Cartesian 4-centre ERIs; no S/T/V, no derivatives, no spherical, no 2/3-centre ERIs.
- Quartet-skip bug: (pq|ps) with s ≤ q < p is never written (see above). It is not reported upstream.
- `D`-exponent basis files are misread silently (Na–Ar in the shipped aug-cc-pV{T,Q}Z).
- Reference values are exact for the **double** inputs. They are not the exact integrals for the
  basis-library decimal strings or for an exact geometry, so compare against inputs taken from the file.
- No Python package (the old `attic/mirppy` binding is not built). ctest needs serial runs. OpenMP needs `OMP_NUM_THREADS`.
- Timings (AFH's benchmark node, Threadripper PRO 9985WX, load ~20–40): water/STO-3G full 0.2–0.8 s. linus water aug-cc-pVTZ `dddd` 6 s
  wall / 96 s CPU, `ffff,dfdf` 10 s / 134 s (16 threads). Recomputing a whole aug-cc-pVTZ
  water file takes about 1 CPU-hour, so a full aug-cc-pVQZ/ano-rcc file is many CPU-hours. Use the Zenodo files for those.
