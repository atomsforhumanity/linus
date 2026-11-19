#!/usr/bin/env python3
"""Generate PySCF/libxc reference values for linus-xc's tests.

Two fixtures, both from PySCF 2.15's eval_xc (run with the dev venv):

1. Stock PySCF (bundles libxc 7.1.2): version drift vs our libxc 7.0.0.

    OMP_NUM_THREADS=1 .venv/bin/python crates/linus-xc/tests/gen_reference.py \
        crates/linus-xc/tests/data/pyscf_reference.json

2. The same PySCF with its libxc.so replaced by libxc 7.0.0 (same version as
   linus-xc, so agreement must be exact). Build a shared 7.0.0 once:

    cmake -S libxc-7.0.0 -B build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DENABLE_FORTRAN=OFF \
        -DENABLE_XHOST=OFF -DDISABLE_KXC=ON -DDISABLE_LXC=ON \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=$PWD/xc700
    cmake --build build -j --target install

   then

    OMP_NUM_THREADS=1 .venv/bin/python crates/linus-xc/tests/gen_reference.py \
        --libxc-so xc700/lib/libxc.so \
        crates/linus-xc/tests/data/pyscf_libxc700_reference.json

   With --libxc-so the script copies the pyscf package to a temporary
   directory, swaps lib/deps/lib/libxc.so (xc_func_type is unchanged between
   7.0.0 and 7.1.2, so PySCF's C interface works with either) and re-runs
   itself with that copy first on PYTHONPATH.

For every named functional below, in both spin treatments, it evaluates
pyscf.dft.libxc.eval_xc(deriv=2) on one shared set of random but physical
grid points and stores zk and all first and second derivatives in libxc's
layout (per point, spin components interleaved; NaN stored as null). The
Rust test tests/pyscf_reference.rs evaluates the same points at orders 0, 1
and 2 and compares. The PySCF spec strings here are written independently
of the linus alias table, so the test also checks the aliases.

Input design:
* rho: log-uniform 1e-4..50 plus tiny densities (1e-12..1e-8), one at 1e-16
  (below every libxc density threshold, so all outputs must be 0) and, when
  polarized, fully polarized points (one spin exactly 0 or 1e-14).
* gradients: reduced gradient s in 0..3 (incl. exactly 0 and 1e-3) with random
  directions; components are rounded to 20 significant bits so that
  sigma = gx*gx + gy*gy + gz*gz is exact in floating point (PySCF forms sigma
  in C, we store it precomputed; exact products make the result independent
  of evaluation order and FMA).
* tau = tau_W + a * tau_unif with a in 0..2 (a = 0: single-orbital limit),
  so tau >= tau_W (von Weizsaecker) always holds.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

if "--libxc-so" in sys.argv:
    i = sys.argv.index("--libxc-so")
    so = os.path.abspath(sys.argv[i + 1])
    rest = sys.argv[:i] + sys.argv[i + 2:]
    import importlib.util

    pyscf_dir = os.path.dirname(importlib.util.find_spec("pyscf").origin)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(pyscf_dir, os.path.join(tmp, "pyscf"), symlinks=True)
        shutil.copyfile(so, os.path.join(tmp, "pyscf", "lib", "deps", "lib", "libxc.so"))
        env = dict(os.environ, PYTHONPATH=tmp + os.pathsep + os.environ.get("PYTHONPATH", ""))
        sys.exit(subprocess.call([sys.executable, *rest], env=env))

import pyscf  # noqa: E402
from pyscf.dft import libxc  # noqa: E402

SEED = 20261106
NP = 24

# linus spec -> PySCF spec. Independent of crates/linus-xc/src/spec.rs.
FUNCTIONALS = [
    ("hf", "HF"),
    ("lda", "LDA_X,LDA_C_VWN"),
    ("svwn-rpa", "LDA_X,LDA_C_VWN_RPA"),
    ("bp86", "GGA_X_B88,GGA_C_P86"),
    ("blyp", "GGA_X_B88,GGA_C_LYP"),
    ("pbe", "GGA_X_PBE,GGA_C_PBE"),
    ("pbe0", "HYB_GGA_XC_PBEH"),
    ("b3lyp", "HYB_GGA_XC_B3LYP"),
    ("b3lyp5", "HYB_GGA_XC_B3LYP5"),
    ("tpss", "MGGA_X_TPSS,MGGA_C_TPSS"),
    ("tpssh", "HYB_MGGA_XC_TPSSH"),
    ("scan", "MGGA_X_SCAN,MGGA_C_SCAN"),
    ("r2scan", "MGGA_X_R2SCAN,MGGA_C_R2SCAN"),
    ("m06-2x", "HYB_MGGA_X_M06_2X,MGGA_C_M06_2X"),
    ("wb97x", "HYB_GGA_XC_WB97X"),
    ("wb97x-v", "HYB_GGA_XC_WB97X_V"),
    ("wb97m-v", "HYB_MGGA_XC_WB97M_V"),
    ("cam-b3lyp", "HYB_GGA_XC_CAM_B3LYP"),
    ("pw6b95", "HYB_MGGA_XC_PW6B95"),
    ("b97m-v", "MGGA_XC_B97M_V"),
    (
        "0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP",
        "0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88, 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP",
    ),
    (
        "0.25*HF + 0.75*MGGA_X_R2SCAN + MGGA_C_R2SCAN",
        "0.25*HF + 0.75*MGGA_X_R2SCAN, MGGA_C_R2SCAN",
    ),
]

C_TF = 0.3 * (3 * math.pi**2) ** (2 / 3)  # tau_unif = C_TF rho^(5/3) (per spin channel: see below)
K_S = 2 * (3 * math.pi**2) ** (1 / 3)  # s = |grad rho| / (K_S rho^(4/3))


def round_bits(x, bits=20):
    """Round to `bits` significant bits (exact products/sums later)."""
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 2.0**bits) / 2.0**bits, e)


def channel(rng, rho, nspin_scale):
    """Gradient (3, n) and tau (n) for one density channel.

    nspin_scale = 1 for a total density, 2 for a spin density (spin scaling:
    the uniform-gas tau of a spin channel is C_TF 2^(2/3) rho_s^(5/3) and the
    reduced gradient uses 2 rho_s).
    """
    n = rho.size
    s = rng.uniform(0.0, 3.0, n)
    s[:3] = [0.0, 1e-3, 3.0]
    dens = nspin_scale * rho
    gnorm = s * K_S * dens ** (4 / 3) / nspin_scale
    d = rng.normal(size=(3, n))
    d /= np.linalg.norm(d, axis=0)
    g = round_bits(d * gnorm)
    sigma = g[0] * g[0] + g[1] * g[1] + g[2] * g[2]
    tau_w = np.where(rho > 0, sigma / (8 * np.where(rho > 0, rho, 1.0)), 0.0)
    a = rng.uniform(0.0, 2.0, n)
    a[:2] = [0.0, 1.0]
    tau_unif = C_TF * dens ** (5 / 3) / nspin_scale
    tau = tau_w + a * tau_unif
    lapl = rng.normal(size=n) * dens
    return g, tau, lapl


def make_inputs(rng):
    def densities(n):
        r = np.exp(rng.uniform(math.log(1e-4), math.log(50.0), n))
        r[-6:] = [1e-8, 1e-10, 1e-12, 3e-11, 2e-9, 1e-16]
        return r

    rho = densities(NP)
    g, tau, lapl = channel(rng, rho, 1)
    unpol = dict(rho=rho, g=g, tau=tau, lapl=lapl)

    ra, rb = densities(NP), densities(NP)
    rng.shuffle(rb[:-1])
    # fully polarized points (one channel exactly zero or below threshold)
    rb[0], ra[1], rb[2] = 0.0, 0.0, 1e-14
    ga, ta, la = channel(rng, ra, 2)
    gb, tb, lb = channel(rng, rb, 2)
    pol = dict(rho=(ra, rb), g=(ga, gb), tau=(ta, tb), lapl=(la, lb))
    return unpol, pol


def pyscf_rho(inp, spin, xctype):
    """PySCF's (nvar, N) / (2, nvar, N) density layout."""
    def one(rho, g, tau, lapl):
        if xctype == "LDA":
            return rho[None, :]
        if xctype == "GGA":
            return np.vstack([rho, g])
        return np.vstack([rho, g, lapl, tau])

    if spin == 0:
        return one(inp["rho"], inp["g"], inp["tau"], inp["lapl"])
    return np.stack([one(inp["rho"][s], inp["g"][s], inp["tau"][s], inp["lapl"][s]) for s in (0, 1)])


def libxc_layout(inp, spin):
    """Inputs in libxc layout (what linus-xc takes)."""
    if spin == 0:
        g = inp["g"]
        return dict(
            rho=inp["rho"],
            sigma=g[0] * g[0] + g[1] * g[1] + g[2] * g[2],
            tau=inp["tau"],
            lapl=inp["lapl"],
        )
    (ra, rb), (ga, gb) = inp["rho"], inp["g"]
    saa = ga[0] * ga[0] + ga[1] * ga[1] + ga[2] * ga[2]
    sab = ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2]
    sbb = gb[0] * gb[0] + gb[1] * gb[1] + gb[2] * gb[2]
    return dict(
        rho=np.column_stack([ra, rb]).ravel(),
        sigma=np.column_stack([saa, sab, sbb]).ravel(),
        tau=np.column_stack(inp["tau"]).ravel(),
        lapl=np.column_stack(inp["lapl"]).ravel(),
    )


# Names of PySCF's eval_xc tuples, in PySCF order.
VXC_NAMES = ["vrho", "vsigma", "vlapl", "vtau"]
FXC_NAMES = ["v2rho2", "v2rhosigma", "v2sigma2", "v2lapl2", "v2tau2",
             "v2rholapl", "v2rhotau", "v2lapltau", "v2sigmalapl", "v2sigmatau"]
DIMS = {  # polarized dims; unpolarized are all 1
    "zk": 1, "vrho": 2, "vsigma": 3, "vlapl": 2, "vtau": 2, "v2rho2": 3, "v2rhosigma": 6,
    "v2sigma2": 6, "v2lapl2": 3, "v2tau2": 3, "v2rholapl": 4, "v2rhotau": 4, "v2lapltau": 4,
    "v2sigmalapl": 6, "v2sigmatau": 6,
}
PRODUCED = {
    "LDA": ["zk", "vrho", "v2rho2"],
    "GGA": ["zk", "vrho", "vsigma", "v2rho2", "v2rhosigma", "v2sigma2"],
    "MGGA": ["zk", "vrho", "vsigma", "vtau", "v2rho2", "v2rhosigma", "v2sigma2",
             "v2rhotau", "v2sigmatau", "v2tau2"],
}


def flat(a):
    """JSON-safe list; NaN/inf (libxc 7.0.0 returns NaN in a few edge cases) -> null."""
    return [float(x) if math.isfinite(x) else None for x in np.asarray(a, dtype=float).ravel()]


def evaluate(pyscf_xc, inp, spin):
    if pyscf_xc == "HF":
        n = NP
        dims = DIMS if spin else {k: 1 for k in DIMS}
        out = {k: [0.0] * (n * dims[k]) for k in PRODUCED["LDA"]}
        return "HF", out
    xctype = libxc.xc_type(pyscf_xc)
    assert not libxc.needs_laplacian(pyscf_xc)
    exc, vxc, fxc, _ = libxc.eval_xc(pyscf_xc, pyscf_rho(inp, spin, xctype), spin=spin, deriv=2)
    arrays = {"zk": exc}
    arrays.update({k: v for k, v in zip(VXC_NAMES, vxc) if v is not None})
    arrays.update({k: v for k, v in zip(FXC_NAMES, fxc) if v is not None})
    out = {}
    for k in PRODUCED[xctype]:
        a = np.asarray(arrays[k], dtype=float)
        expect = NP * (DIMS[k] if spin else 1)
        assert a.size == expect, (pyscf_xc, k, a.shape)
        out[k] = flat(a)  # (N, dim) row-major == libxc interleaved layout
    return xctype, out


def main(path):
    rng = np.random.default_rng(SEED)
    unpol, pol = make_inputs(rng)
    inputs = {}
    for label, inp, spin in (("unpolarized", unpol, 0), ("polarized", pol, 1)):
        lx = libxc_layout(inp, spin)
        inputs[label] = {"np": NP, **{k: flat(v) for k, v in lx.items()}}

    cases = []
    for name, pyscf_xc in FUNCTIONALS:
        if pyscf_xc == "HF":
            hyb, rsh, nlc = 1.0, [0.0, 0.0, 0.0], None
        else:
            hyb = float(libxc.hybrid_coeff(pyscf_xc))
            rsh = [float(x) for x in libxc.rsh_coeff(pyscf_xc)]
            nlc = libxc.nlc_coeff(pyscf_xc) if libxc.is_nlc(pyscf_xc) else None
            if nlc is not None:
                (b, c), w = nlc[0]
                nlc = [float(b), float(c), float(w)]
        for label, inp, spin in (("unpolarized", unpol, 0), ("polarized", pol, 1)):
            xctype, out = evaluate(pyscf_xc, inp, spin)
            cases.append({
                "name": name, "pyscf_xc": pyscf_xc, "xctype": xctype, "spin": label, "order": 2,
                "hybrid_coeff": hyb, "rsh_coeff_omega_alpha_beta": rsh, "nlc_b_c_weight": nlc,
                "out": out,
            })

    doc = {
        "generator": "crates/linus-xc/tests/gen_reference.py",
        "pyscf_version": pyscf.__version__,
        "libxc_version": libxc.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "inputs": inputs,
        "cases": cases,
    }
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {len(cases)} cases, pyscf {pyscf.__version__}, libxc {libxc.__version__} -> {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pyscf_reference.json")
