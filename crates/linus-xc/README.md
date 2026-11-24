# linus-xc

Exchange-correlation layer for linus. `Functional` is what the SCF, gradient and
Hessian code uses; `XcKernel` is the backend trait. The first backend is
**libxc 7.0.0** (MPL-2.0), built static from the unmodified upstream tarball and
called through a hand-written FFI (`src/ffi.rs`). It is pinned to exactly 7.0.0
because cuEST uses the same version. Native Rust or GPU kernels can implement
`XcKernel` later and replace libxc one component at a time.

## API

```rust
// Build from a name, a libxc name or id, or a weighted mixture.
Functional::new(spec: &str, polarized: bool) -> Result<Functional>
Functional::from_components(spec: String, polarized: bool, components: Vec<Component>,
                            hf_weight: f64, dispersion: Option<&'static str>) -> Result<Functional>

// What the functional needs
f.family() -> Family                 // Hf < Lda < Gga < Mgga (max over components)
f.needs_sigma() / f.needs_tau() / f.needs_lapl() -> bool
f.polarized() -> bool, f.nspin() -> usize, f.max_order() -> usize   // max_order is 2 (fxc)
f.components() -> &[Component]       // weight + Box<dyn XcKernel>

// Exact exchange and non-local parts the host must add
f.exact_exchange() -> ExactExchange  // {alpha, beta, omega, kernel: None|Erf|Yukawa}
f.exx_fraction() -> f64              // alpha
f.range_separation() -> Option<(alpha, beta, omega)>
f.is_hybrid() -> bool
f.vv10() -> Option<Vv10>             // {b, c, scale}
f.dispersion() -> Option<&'static str>   // "D3BJ" for ORCA-style *-D3BJ names

// Thresholds
f.dens_threshold() -> f64, f.set_dens_threshold(&mut self, t: f64)

// Evaluation on a batch of grid points
f.eval(order: usize, input: &XcInput, out: &mut XcOutput) -> Result<()>       // owned, reusable buffers
f.eval_into(order: usize, input: &XcInput, out: &mut XcOutRefs) -> Result<()> // caller's slices
XcInput { rho: &[f64], sigma: Option<&[f64]>, lapl: Option<&[f64]>, tau: Option<&[f64]> }
XcInput::lda(rho), ::gga(rho, sigma), ::mgga(rho, sigma, tau), .with_lapl(lapl)
out.zk(), out.vrho(), out.vsigma(), out.vtau(), out.vlapl(), out.v2rho2(), out.v2rhosigma(),
out.v2sigma2(), out.v2rhotau(), out.v2sigmatau(), out.v2tau2(), ... ; out.get(Out::V2rho2)
out.integrate_energy(rho, weights) -> f64     // Σ w ρ zk

// Backend trait (LibxcFunctional implements it)
trait XcKernel: Send + Sync { fn eval(&self, order, &XcInput, &mut XcOutRefs) -> Result<()>; family(),
    needs_tau(), needs_lapl(), max_order(), exact_exchange(), vv10(), dens_threshold(), ... }
LibxcFunctional::new(id: i32, polarized) / ::from_name("GGA_X_B88", polarized)
libxc_version(), libxc_id(name), libxc_key(id), libxc_available_ids()
```

Orders: 0 gives `zk`; 1 adds `vrho, vsigma, vlapl, vtau`; 2 adds the ten second
derivatives. Order 3 and above are not compiled (kxc/lxc are off).

### Conventions

* **Layout** (struct of arrays, libxc's own): one array per quantity, spin
  components interleaved per point. Unpolarized: `rho[np]`, `sigma[np]`,
  `tau[np]`, `lapl[np]` (total density, |∇ρ|², total τ). Polarized:
  `rho[2np] = (a,b)`, `sigma[3np] = (aa, ab, bb)`, `tau[2np]`, `lapl[2np]`.
  Outputs have `np * Out::dim(polarized)` entries in libxc order: `vsigma (aa,ab,bb)`,
  `v2rho2 (a_a, a_b, b_b)`, `v2rhosigma (a_aa … b_bb)`, `v2rhotau (a_a, a_b, b_a, b_b)`.
  This is the same layout as PySCF's `eval_xc` arrays.
* **Energy:** `zk` is the energy **per particle** (libxc `zk`, PySCF `exc`), so
  `E_xc = Σ_i w_i ρ_i zk_i` (`XcOutput::integrate_energy`). Derivatives are of the
  energy **per volume** `ρ·zk`: `vrho = ∂(ρ zk)/∂ρ_s`, and so on.
* **τ** = ½ Σ|∇φ|² (no missing ½), as in libxc and PySCF.
* **Exact exchange** uses libxc's convention: `K = alpha·K + beta·K_sr(omega)`, so the
  short-range fraction is `alpha + beta` and the long-range fraction is `alpha`.
  wB97X: (1.0, −0.842294, 0.3). CAM-B3LYP: (0.65, −0.46, 0.33). PySCF's
  `rsh_coeff` uses the same numbers.
* **VV10:** libxc does not evaluate it. `vv10()` gives `b`, `C` for the host
  (wB97X-V, wB97M-V, B97M-V: b = 6.0, C = 0.01).
* **Thresholds:** libxc's own. A point whose total density is below a component's
  `dens_threshold` gets exactly 0 from that component. `set_dens_threshold`
  sets it on every component; libxc propagates it to its internal mixtures.
  Errors that libxc would answer with `exit(1)` (missing output buffer,
  order not compiled in) come back as `XcError` instead.

### Names

`Functional::new` accepts any of these (case, `-`, `_` and `/` are ignored):

1. a linus alias (table below);
2. `HF`;
3. a libxc name (`GGA_X_B88`, `xc_gga_x_b88`) or id (`106`);
4. a weighted sum of the above, e.g.
   `0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP`
   or `0.5*b3lyp + 0.1*hf`.

A `-` counts as subtraction only after whitespace (`b3lyp - 0.1*hf`), so
`wB97X-V` stays one name. PySCF's `"X,C"` comma syntax is rejected on purpose:
in PySCF, `PBE` before the comma means PBE *exchange only*.

### linus names vs ORCA (and PySCF / Psi4)

| linus | libxc components | ORCA keyword | Notes |
|---|---|---|---|
| `hf` | (none), α = 1 | `HF` | |
| `slater`, `hfs` | `LDA_X` | `HFS` | PySCF's `LDA` means **this** (Slater only) |
| `lda`, `lsda`, `svwn`, `svwn5` | `LDA_X + LDA_C_VWN` (VWN5) | `LDA`/`LSD` (ORCA's LDA is VWN5), `VWN5` | PySCF `SVWN` is the same. **Psi4 and Gaussian `SVWN` are VWN-RPA.** |
| `svwn-rpa`, `svwn3`, `vwn3` | `LDA_X + LDA_C_VWN_RPA` | `VWN3` | ORCA/Gaussian/PySCF "VWN3" means the RPA fit (VWN "functional III"). **libxc's `LDA_C_VWN_3` (id 30) is a different functional:** it equals VWN5 for closed shells. |
| `bp86` | `GGA_X_B88 + GGA_C_P86` (PZ81 local part) | | Gaussian, PySCF and Psi4 BP86 |
| `bp86-vwn5` | `GGA_X_B88 + GGA_C_P86VWN` | `BP86`? | ORCA's BP86 is built on VWN5. Closest libxc match, **not checked against ORCA** |
| `blyp` | `GGA_X_B88 + GGA_C_LYP` | `BLYP` | |
| `pbe` | `GGA_X_PBE + GGA_C_PBE` | `PBE` | |
| `pbe0` (`pbeh`, `pbe1pbe`) | `HYB_GGA_XC_PBEH`, α = 0.25 | `PBE0` | |
| `b3lyp` (`b3lyp/g`) | `HYB_GGA_XC_B3LYP` (VWN-RPA), α = 0.2 | **`B3LYP/G`** | Gaussian, PySCF ≥ 2.3, Psi4 `b3lyp`, Q-Chem. libxc's `HYB_GGA_XC_B3LYP3` is **not** this one (it uses libxc's VWN_3) |
| `b3lyp5` | `HYB_GGA_XC_B3LYP5` (VWN5) | **`B3LYP`** | Turbomole `b3-lyp`, Psi4 `b3lyp5`, PySCF `B3LYP5`. Differs from `b3lyp` by about 3.7 mEh per electron (AFH orca skill) |
| `tpss` | `MGGA_X_TPSS + MGGA_C_TPSS` | `TPSS` | |
| `tpssh` | `HYB_MGGA_XC_TPSSH`, α = 0.1 | `TPSSh` | |
| `scan` | `MGGA_X_SCAN + MGGA_C_SCAN` | `SCAN` | |
| `r2scan` | `MGGA_X_R2SCAN + MGGA_C_R2SCAN` | `r2SCAN` | `r2SCAN-3c` is this plus D4, gCP and the mTZVPP basis (not in linus-xc) |
| `m06-2x` | `HYB_MGGA_X_M06_2X + MGGA_C_M06_2X`, α = 0.54 | `M062X` | |
| `wb97x` | `HYB_GGA_XC_WB97X`, (1, −0.842294, 0.3) | `wB97X` | Chai and Head-Gordon 2008 |
| `wb97x-d3` | `HYB_GGA_XC_WB97X_D3` | `wB97X-D3` | Different refit (Lin et al. 2013); D3(0) added by the host (`dispersion()`) |
| `wb97x-v` | `HYB_GGA_XC_WB97X_V` + VV10 | `wB97X-V` | |
| `wb97x-d3bj` | `HYB_GGA_XC_WB97X_V`, **VV10 dropped** | `wB97X-D3BJ` | **Not wB97X + D3BJ:** ORCA's wB97X-D3BJ is the semilocal part of wB97X-V plus D3(BJ) (Najibi and Goerigk 2018). `dispersion() = "D3BJ"` |
| `wb97m-v` | `HYB_MGGA_XC_WB97M_V` + VV10 | `wB97M-V` | |
| `wb97m-d3bj` | `HYB_MGGA_XC_WB97M_V`, VV10 dropped | `wB97M-D3BJ` | Same construction as wB97X-D3BJ |
| `cam-b3lyp` | `HYB_GGA_XC_CAM_B3LYP`, (0.65, −0.46, 0.33) | `CAM-B3LYP` | libxc's CAM-B3LYP uses VWN5 |
| `pw6b95` | `HYB_MGGA_XC_PW6B95`, α = 0.28 | `PW6B95` | |
| `b97m-v` | `MGGA_XC_B97M_V` + VV10 | `B97M-V` | |
| `b97m-d3bj` | `MGGA_XC_B97M_V`, VV10 dropped | `B97M-D3BJ` | |

The B3LYP/G vs B3LYP equivalence with ORCA is verified for closed shells (Psi4 1.11 `b3lyp`
matches ORCA 6.0.0 `B3LYP/G` to 0.004 mEh).
ORCA's open-shell VWN3 and its BP86 local part have not been checked numerically here.

## Thread safety

`Functional` and `LibxcFunctional` are `Send + Sync`. Share one `&Functional`
across rayon workers and give each worker its own `XcOutput`.

Evidence from the libxc 7.0.0 source:

* Every evaluation entry point takes `const xc_func_type *`.
* There are no writes to the functional on the evaluation path (`work_*_inc.c`,
  `mix_func.c`, the generated `maple2c` code).
* Mixtures (`xc_mix_func`) `malloc` their scratch per call.
* The only file-scope arrays on that path are read-only Chebyshev tables
  (`bessel.c`, `expint_e1.c`).

Initialization is *not* reentrant. Some init routines write function-local
`static` arrays before copying them: `hyb_gga_xc_hse.c` sets
`funcs_id[0]` differently per HSE/HJS variant, as do `hyb_mgga_xc_kcis.c`,
`hyb_gga_xc_b1wc.c`, `hyb_gga_xc_lcy_*.c` and others. So `xc_func_init` and
`xc_func_end` run under a global mutex.

`tests/threads.rs` checks two things:

* a rayon evaluation of one shared functional is bit-identical to a serial one
  (b3lyp, r2scan, wb97m-v and a mixture, both spins, order 2);
* 256 concurrent constructions of HSE/HJS variants give the right parameters.

## Build and caching

`build.rs` takes libxc from, in order:

1. **`LIBXC_DIR=<prefix>`:** a prebuilt install (`include/xc.h`, `lib{,64}/libxc.a`
   or `libxc.so`). Anything but 7.0.0 is refused, unless you set
   `LINUS_XC_ALLOW_VERSION_MISMATCH=1`. With only a `.so`, it links
   dynamically and you need `LD_LIBRARY_PATH`.
2. **Build cache:** the libxc 7.0.0 tarball
   (`https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz`,
   sha256 `8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77`,
   the same hash Spack records) is downloaded with `curl` or taken from
   `LIBXC_TARBALL=<file>` for offline builds. It is verified, then built with the
   `cmake` crate into
   `${LINUS_XC_CACHE_DIR:-${XDG_CACHE_HOME:-~/.cache}/linus-xc}/libxc-7.0.0-<key>/`.
   `<key>` hashes the tarball sha256, the CMake options, the target and the C
   compiler. A file lock serializes concurrent builds. The cache survives
   `cargo clean`, other profiles, worktrees and rust-analyzer, so libxc compiles
   once per machine and compiler. `LINUS_XC_CACHE_DIR=off` builds inside
   `OUT_DIR` instead.

CMake options: Release (`-O3 -DNDEBUG`), static with `-fPIC` (it can go into the
future Python extension), no Fortran, no tests, `DISABLE_KXC=ON`,
`DISABLE_LXC=ON`, Fermi-hole-curvature enforcement on (libxc's default), and
`ENABLE_XHOST=OFF`. With XHOST off there is no `-march=native`, so the
library is portable and free of FMA contraction, which keeps results
bit-identical to other generic builds. CMake ≥ 4 needs
`CMAKE_POLICY_VERSION_MINIMUM=3.5` for libxc 7.0.0.

Measured on a Threadripper PRO 9985WX (128 threads), gcc 15.2, cmake 4.4:

| build | wall time |
|---|---|
| cold: download 68 MB, compile libxc (454 CPU-s), crate | 16.7 s (9 s for libxc itself) |
| no-op `cargo build` | 0.02 s |
| new target dir or profile, warm cache | 1.5 s |

A 4–8 core machine needs about 1–2 min for the one-time libxc build. The
cache entry is 60 MB (36 MB `libxc.a`), plus the 65 MB tarball.

The build also compiles two small C shims:

* **`csrc/layout_check.c`:** `sizeof`/`offsetof` of the mirrored structs, from
  the very headers we link. The test `layouts_match_c_headers` compares them
  with the Rust `#[repr(C)]` definitions.
* **`csrc/glibc_cbrt.c`:** see the next section.

### The `cbrt` trap (linux-gnu)

Rust's `compiler_builtins` (linked last into every Rust binary) defines a
**weak `cbrt`**: a correctly rounded port of CORE-MATH. It also defines weak
`sqrt`, `fma`, `floor` and a few others. The static libxc objects sit in this
rlib, ahead of `compiler_builtins`, and `-lm` comes after it. So libxc's calls to
`cbrt` bind to Rust's version, while `pow`, `exp` and `log` still come from glibc.

The two `cbrt` differ in the last bit for some arguments, and libxc calls `cbrt`
everywhere. Our results were then about 1 ulp off PySCF at ordinary points,
and 1e-5 to 1e-2 relative off at points with an empty spin channel, where the
error is amplified.

`csrc/glibc_cbrt.c` fixes this with a hidden, strong `cbrt` that forwards to the
versioned `cbrt@GLIBC_2.2.5`. It must be hidden: if exported, ld.so lets the
unversioned definition satisfy the versioned reference, and `cbrt` loops
forever. `LINUS_XC_GLIBC_CBRT=0` turns the shim off. The other weak symbols
are exact or correctly rounded everywhere, so they don't matter.

## Validation

`tests/gen_reference.py` writes the JSON fixtures in `tests/data/`, so
`cargo test -p linus-xc` needs no Python. The inputs are 24 random but
physical points per spin case:

* log-uniform ρ from 1e-4 to 50, plus 1e-8 … 1e-12, plus 1e-16 (screened);
* fully polarized points (one channel exactly 0 or 1e-14);
* reduced gradient s from 0 to 3, including exactly 0;
* τ = τ_W + a·τ_unif with a from 0 to 2;
* gradients rounded so that σ is exact.

Every named functional above plus two mixtures are evaluated for both spins
at orders 0, 1 and 2, and every array is compared with
`pyscf.dft.libxc.eval_xc(deriv=2)`. The hybrid, range-separation and VV10
parameters are compared with PySCF's `hybrid_coeff`, `rsh_coeff` and `nlc_coeff`.

* **PySCF 2.15 bundles libxc 7.1.2, not 7.0.0.** So a second fixture comes
  from the same PySCF with its `libxc.so` replaced by a libxc 7.0.0 build
  (`gen_reference.py --libxc-so`). The two libxc versions have the same
  `xc_func_type` layout, so PySCF's C interface works with either.
* **Same libxc (7.0.0), all 24 points incl. edge cases:** every array of every
  functional, both spins, orders 0–2, is **bit-identical** with Ubuntu 26.04's
  glibc (max relative difference 0). The system's C math library moves the
  last bits: glibc 2.36 and 2.39 round `erf` differently (ωB97X, ≤ 5.8e-15),
  and Ubuntu 26.04's amd64v3 glibc (GitHub's ubuntu-26.04 runner) has a `cbrt`
  1–4 ulp off the x86-64 build's on 20% of arguments, which a minority spin
  channel amplifies (B97 u, ωB97 attenuation). So the test's tolerance goes by
  point class, each at least 10× the drift measured in those four
  environments: regular points 1e-11 (measured 7.2e-13), strongly polarized
  points (min ρ_s < 1e-3 ρ) 5e-9 (4.4e-10), and degenerate points (some
  ρ_s < 1e-6, σ_ss = 0 or τ_s = 0) 2e-2 (1.1e-3) with the same screened
  points. A libxc compiled for x86-64-v3 gives the baseline bits: build.rs
  turns floating-point contraction off.
* **Stock PySCF (libxc 7.1.2):** this measures version drift. Libxc 7.1.0
  regenerated all functional code (SymPy backend, cancellation-free forms, no
  `pow()` for rational powers). The test checks bulk points (every ρ_s ≥ 1e-4,
  σ_ss > 0, 13–17 points) and reports the edge points. Maximum relative
  differences at bulk points:

| quantity | spin-unpolarized | polarized |
|---|---|---|
| `zk` | ≤ 1.2e-14 (M06-2X), most ≤ 4e-15 | ≤ 9e-15 |
| first derivatives | ≤ 2e-10 (TPSS/SCAN `vtau`, PBE `vsigma` at s = 1e-3) | ≤ 1.6e-10 |
| second derivatives | ≤ 4e-11 | ≤ 4.2e-7, in near-cancelling spin-cross terms (TPSS `v2rhosigma`; BLYP `v2rhosigma(a, bb)` ≈ −4.5e-7) |

To check against another PySCF or libxc, regenerate a fixture and run
`cargo run -p linus-xc --example eval_fixture -- fixture.json ours.json`, which
writes our outputs in the fixture's schema. Set
`LINUS_XC_VALIDATION_TABLE=<dir>` to have the tests write their per-case tables.

At edge points (an exactly empty spin channel, σ = 0, ρ ≤ 1e-8) the versions
differ by orders of magnitude. libxc 7.0.0 evaluates the empty channel at
clamped threshold values. 7.1 returns 0 or regular values there.

## Performance (1 thread)

Batch of 524,288 points: log-uniform ρ from 1e-5 to 100, s from 0 to 3, all
above threshold. Threadripper PRO 9985WX, best of 5, with other jobs on the
machine (expect ±10%). Run it with:

```text
OMP_NUM_THREADS=1 .venv/bin/python crates/linus-xc/examples/bench_pyscf.py DIR
cargo run --release -p linus-xc --example bench_eval -- DIR [threads]
```

Throughput in Mpoints/s:

| functional | spin | order | linus-xc 7.0.0, 1024-pt blocks | PySCF eval_xc, same libxc 7.0.0 | PySCF eval_xc stock, libxc 7.1.2 | libxc 7.1.2 rebuilt with our flags, via PySCF |
|---|---|---|---|---|---|---|
| b3lyp | unpol | 1 | 7.16 | 6.91 | 6.38 | 7.02 |
| b3lyp | unpol | 2 | 4.74 | 4.08 | 4.55 | 5.18 |
| b3lyp | pol | 1 | 3.33 | 2.71 | 1.61 | 1.85 |
| b3lyp | pol | 2 | 2.07 | 1.57 | 1.07 | 1.13 |
| r2scan | unpol | 1 | 4.50 | 3.86 | 2.18 | 2.41 |
| r2scan | unpol | 2 | 2.55 | 2.13 | 1.04 | 1.35 |
| r2scan | pol | 1 | 2.21 | 1.85 | 0.78 | 0.87 |
| r2scan | pol | 2 | 0.79 | 0.64 | 0.34 | 0.38 |

* Block size barely matters: 128-point blocks run within about 6% of whole-batch calls.
* With rayon on 16 threads, one shared `Functional` gives 13.7–14.4× (b3lyp
  unpol order 1: 103 Mpoints/s; r2scan pol order 2: 10.8 Mpoints/s).
* Over PySCF on the same libxc, linus-xc gains 4–32%. The gain is the
  gradient-to-σ preprocessing and output merging PySCF does around libxc.
  The time is libxc's own.
* libxc 7.1.2 is **slower** than 7.0.0 for r2SCAN and polarized B3LYP:
  1.4–2.1× when built with identical flags, 1.5–2.4× as shipped in the PySCF
  wheel. Its changelog's `pow()` speedups don't make up for the regenerated,
  cancellation-free code. Unpolarized B3LYP is level at order 1 and 27% faster
  in 7.1.2 at order 2. Pinning 7.0.0 costs no speed.

## libxc 7.0.0 quirks to know

* **NaN in polarized fxc with an empty spin channel.** `MGGA_C_TPSS` and
  `MGGA_C_R2SCAN` (so TPSS, TPSSh and r2SCAN) return NaN for all of `v2rho2`,
  including the occupied channel, when one spin density is exactly 0. This hits
  UKS Hessians and TDDFT of systems with no β electrons (H atom, triplet H₂)
  and points where screening zeroes one channel. Energies and potentials are
  finite. libxc 7.1 fixed it. The Hessian code should guard such points; the
  test `libxc_700_empty_channel_nan_is_documented` pins the behaviour.
* **Empty-channel derivatives are threshold artefacts in 7.0.0.** Examples:
  TPSS-X `vrho_a` equals `vrho_b` at ρ_a = 0, and `v2rho2_aa` reaches about 1e8
  for LDA/GGA at ρ_b = 0. 7.1 returns 0 there.
* libxc calls `exit(1)` on a missing output buffer or an order that isn't
  compiled. The safe layer checks both first.
* `xc_func_init` failure leaves `info` null, so `xc_func_end` would crash; the
  wrapper only calls `xc_func_free` in that case.

## Licence

libxc is MPL-2.0 and linked unmodified. The source is at the URL above
(file-level copyleft; linus stays MIT).
