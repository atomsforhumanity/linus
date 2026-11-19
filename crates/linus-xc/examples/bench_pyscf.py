#!/usr/bin/env python3
"""PySCF side of the linus-xc throughput benchmark (single thread).

    OMP_NUM_THREADS=1 .venv/bin/python crates/linus-xc/examples/bench_pyscf.py DIR [NPOINTS]

Writes the batch to DIR/{unpolarized,polarized}_{rho,sigma,tau}.f64 (raw
little-endian f64, libxc layout) for `cargo run --release -p linus-xc --example
bench_eval -- DIR`, then times pyscf.dft.libxc.eval_xc on the same points and
prints points/second (best of 5). Run it with OMP_NUM_THREADS=1: PySCF's
eval_xc is OpenMP-parallel.
"""

import math
import os
import sys
import time

import numpy as np
from pyscf.dft import libxc

FUNCS = {"b3lyp": "HYB_GGA_XC_B3LYP", "r2scan": "MGGA_X_R2SCAN,MGGA_C_R2SCAN"}


def channel(rng, rho, scale):
    n = rho.size
    s = rng.uniform(0.0, 3.0, n)
    dens = scale * rho
    g = rng.normal(size=(3, n))
    g *= s * 2 * (3 * math.pi**2) ** (1 / 3) * dens ** (4 / 3) / scale / np.linalg.norm(g, axis=0)
    sigma = (g * g).sum(axis=0)
    tau = sigma / (8 * rho) + rng.uniform(0, 2, n) * 0.3 * (3 * math.pi**2) ** (2 / 3) * dens ** (5 / 3) / scale
    return g, tau


def main(out, n):
    rng = np.random.default_rng(7)
    os.makedirs(out, exist_ok=True)
    batches = {}
    rho = np.exp(rng.uniform(math.log(1e-5), math.log(1e2), n))
    g, tau = channel(rng, rho, 1)
    batches["unpolarized"] = (0, np.vstack([rho, g, np.zeros(n), tau]))
    np.ascontiguousarray(rho).tofile(f"{out}/unpolarized_rho.f64")
    (g * g).sum(axis=0).tofile(f"{out}/unpolarized_sigma.f64")
    tau.tofile(f"{out}/unpolarized_tau.f64")

    ra = np.exp(rng.uniform(math.log(1e-5), math.log(5e1), n))
    rb = ra * rng.uniform(0.0, 1.0, n)
    ga, ta = channel(rng, ra, 2)
    gb, tb = channel(rng, rb, 2)
    batches["polarized"] = (1, np.stack([np.vstack([ra, ga, np.zeros(n), ta]), np.vstack([rb, gb, np.zeros(n), tb])]))
    np.column_stack([ra, rb]).ravel().tofile(f"{out}/polarized_rho.f64")
    np.column_stack([(ga * ga).sum(0), (ga * gb).sum(0), (gb * gb).sum(0)]).ravel().tofile(f"{out}/polarized_sigma.f64")
    np.column_stack([ta, tb]).ravel().tofile(f"{out}/polarized_tau.f64")

    print(f"PySCF eval_xc, libxc {libxc.__version__}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, {n} points")
    print("| functional | spin | order | Mpoints/s |")
    print("|---|---|---|---|")
    for name, xc in FUNCS.items():
        for spin_label, (spin, r) in batches.items():
            rr = r if libxc.xc_type(xc) == "MGGA" else r[..., :4, :]
            for order in (1, 2):
                best = math.inf
                for _ in range(5):
                    t0 = time.perf_counter()
                    libxc.eval_xc(xc, rr, spin=spin, deriv=order)
                    best = min(best, time.perf_counter() - t0)
                print(f"| {name} | {spin_label} | {order} | {n / best / 1e6:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 19)
