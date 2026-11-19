#!/usr/bin/env python
"""Compare linus ECP integrals (crates/linus-ints/examples/ecp_dump.rs) with PySCF.

    .refenv/bin/python tests/python/ecp_pyscf_check.py [--bin target/release/examples/ecp_dump]
        [--what matrix,grad,deriv,hess] [--cases NAME,...] [--basis def2-svp,def2-tzvp]

PySCF gets the same basis functions in the same order as linus (built from the repository's
Basis Set Exchange JSON, segmented and sorted by angular momentum as linus does) and the same
ECP data, so the matrices compare element by element. Prints one row per (case, basis):
max |linus - PySCF| and max |PySCF| for each quantity.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ecp_entry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASIS_DIR = REPO / "data" / "basis"

# bohr; deliberately low symmetry
CASES = {
    "HI": [("I", (0.0, 0.0, 0.0)), ("H", (0.3, 0.2, 3.04))],
    "CH3I": [("I", (0.0, 0.0, 0.0)), ("C", (0.1, -0.2, 4.04)), ("H", (1.9, 0.0, 4.75)), ("H", (-1.0, 1.6, 4.80)), ("H", (-0.9, -1.7, 4.70))],
    "PdCl2PH3": [("Pd", (0.0, 0.0, 0.0)), ("Cl", (4.35, 0.2, 0.0)), ("Cl", (-4.30, 0.1, 0.3)), ("P", (0.1, 4.30, -0.2)),
                 ("H", (1.5, 5.2, -1.4)), ("H", (-1.8, 5.1, 0.4)), ("H", (1.2, 5.6, 1.6))],
    "PtCl4": [("Pt", (0.0, 0.0, 0.0)), ("Cl", (4.37, 0.0, 0.0)), ("Cl", (-4.37, 0.1, 0.0)), ("Cl", (0.0, 4.37, 0.2)), ("Cl", (0.0, -4.37, 0.0))],
    "Au2": [("Au", (0.0, 0.0, 0.0)), ("Au", (0.2, 0.3, 4.7))],
    "CsF": [("Cs", (0.0, 0.0, 0.0)), ("F", (0.3, 0.1, 4.4))],
    "RbCl": [("Rb", (0.0, 0.0, 0.0)), ("Cl", (0.2, -0.3, 5.5))],
    "HgH2": [("Hg", (0.0, 0.0, 0.0)), ("H", (0.0, 0.1, 3.1)), ("H", (0.1, 0.0, -3.1))],
    "AgCl": [("Ag", (0.0, 0.0, 0.0)), ("Cl", (0.1, 0.2, 4.3))],
    # lanthanides: def2 4f-in-core ECPs with semi-local channels up to g (local h)
    "GdF": [("Gd", (0.0, 0.0, 0.0)), ("F", (0.2, 0.1, 3.9))],
    "CeO": [("Ce", (0.0, 0.0, 0.0)), ("O", (0.1, -0.2, 3.4))],
    "CH2I": [("I", (0.0, 0.0, 0.0)), ("C", (0.05, -0.03, 3.97)), ("H", (1.75, 0.10, 4.95)), ("H", (-1.80, 0.05, 4.90))],
    "IrH3": [("Ir", (0.0, 0.0, 0.0)), ("H", (3.0, 0.0, 0.1)), ("H", (-1.5, 2.6, 0.0)), ("H", (-1.5, -2.6, 0.2))],
}


def stem(name: str) -> str:
    return name.lower().replace("*", "_st_").replace("+", "_pl_").replace("/", "_sl_").replace("(", "_lp_").replace(")", "_rp_").replace(",", "_cm_")


def element_data(basis: str, z: int):
    d = json.loads((BASIS_DIR / f"{stem(basis)}.json").read_text())["elements"][str(z)]
    shells = []
    for sh in d["electron_shells"]:
        exps = [float(x) for x in sh["exponents"]]
        am = sh["angular_momentum"]
        rows = sh["coefficients"]
        ls = am * len(rows) if len(am) == 1 else am
        for l, row in zip(ls, rows):
            keep = [(e, float(c)) for e, c in zip(exps, row) if float(c) != 0.0]
            if keep:
                shells.append((l, keep))
    shells.sort(key=lambda s: s[0])
    return shells, ecp_entry(d)


def pyscf_mol(atoms, basis: str, cart: bool) -> gto.Mole:
    bas, ecp = {}, {}
    for sym, _ in atoms:
        if sym in bas:
            continue
        shells, e = element_data(basis, gto.charge(sym))
        bas[sym] = [[l] + [[x, c] for x, c in prims] for l, prims in shells]
        if e is not None:
            ecp[sym] = e
    nel = sum(gto.charge(s) - (ecp[s][0] if s in ecp else 0) for s, _ in atoms)
    return gto.M(atom=[(s, x) for s, x in atoms], basis=bas, ecp=ecp, unit="Bohr", cart=cart, verbose=0, spin=nel % 2, max_memory=4000)


def cart_scale(mol: gto.Mole) -> np.ndarray:
    """PySCF's Cartesian functions carry no l-dependent angular factor for l >= 2; linus's have
    unit-norm x^l components: scale = 1/sqrt(<x^l|x^l>_PySCF) per shell."""
    s = mol.intor("int1e_ovlp_cart")
    f = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        n = (l + 1) * (l + 2) // 2
        i0 = len(f)
        f.extend([1.0 / np.sqrt(s[i0, i0])] * n)
    return np.array(f)


def run_linus(binary: str, atoms, basis: str, pure: bool, what, dens=None):
    z = [gto.charge(s) for s, _ in atoms]
    inp = {"z": z, "coords": [list(x) for _, x in atoms], "basis": basis, "pure": pure, "what": what}
    if dens is not None:
        inp["dens"] = dens.ravel().tolist()
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.json"
        op = Path(td) / "out.bin"
        ip.write_text(json.dumps(inp))
        r = subprocess.run([binary, str(ip), str(op)], capture_output=True, text=True, check=True)
        out = json.loads(r.stdout)
        raw = np.fromfile(op, dtype="<f8")
    return out, raw


def random_density(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, max(2, n // 3))) / np.sqrt(n)
    return c @ c.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(REPO / "target" / "release" / "examples" / "ecp_dump"))
    ap.add_argument("--what", default="matrix")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--basis", default="def2-svp,def2-tzvp")
    ap.add_argument("--cart", action="store_true")
    args = ap.parse_args()
    what = args.what.split(",")
    worst = 0.0
    for name in args.cases.split(","):
        atoms = CASES[name]
        for basis in args.basis.split(","):
            mol = pyscf_mol(atoms, basis, args.cart)
            nao = mol.nao
            dens = random_density(nao) if any(w in what for w in ("grad", "hess")) else None
            out, raw = run_linus(args.bin, atoms, basis, not args.cart, what, dens)
            assert out["nbf"] == nao, (out["nbf"], nao)
            row = [f"{name:10s} {basis:10s} nbf={nao:4d}"]
            off = 0
            if "matrix" in what:
                v = raw[: nao * nao].reshape(nao, nao)
                off = nao * nao
                ref = mol.intor("ECPscalar_cart" if args.cart else "ECPscalar")
                if args.cart:
                    f = cart_scale(mol)
                    ref = ref * f[:, None] * f[None, :]
                # PySCF screens shell pairs by their overlap (exp(-ab/(a+b) AB^2) etc., libcint's
                # check_3c_overlap), which drops semi-local (type 2) integrals between functions
                # on opposite sides of the ECP atom; report those elements separately
                screened = ref == 0.0
                dm = np.where(screened, 0.0, np.abs(v - ref))
                d = dm.max()
                worst = max(worst, d)
                i, j = np.unravel_index(dm.argmax(), v.shape)
                labels = mol.ao_labels()
                row.append(f"V: {d:.2e} (max {np.abs(ref).max():.1e}, {out['times'].get('matrix', 0) * 1e3:.1f} ms) at {labels[i].strip()}|{labels[j].strip()}; "
                           f"PySCF-screened max |V| {np.abs(np.where(screened, v, 0.0)).max():.1e}")
                if os.environ.get("ECP_DUMP_PREFIX"):
                    np.save(os.environ["ECP_DUMP_PREFIX"] + f"{name}.{basis}.linus.npy", v)
                    np.save(os.environ["ECP_DUMP_PREFIX"] + f"{name}.{basis}.pyscf.npy", ref)
            if "grad" in what or "deriv" in what or "hess" in what:
                from pyscf.grad import rhf as rhf_grad
                natm = mol.natm
                # PySCF: d/dR of ECP matrix via the hcore generator difference (hcore - (T + V))
                ecp_d = ecp_deriv_pyscf(mol)
                if "deriv" in what:
                    dv = raw[off: off + 3 * natm * nao * nao].reshape(natm, 3, nao, nao)
                    d = np.abs(dv - ecp_d).max()
                    worst = max(worst, d)
                    row.append(f"dV: {d:.2e} (max {np.abs(ecp_d).max():.1e})")
                if "grad" in what:
                    g = np.array(out["grad"])
                    gref = np.einsum("axij,ij->ax", ecp_d, dens)
                    d = np.abs(g - gref).max()
                    worst = max(worst, d)
                    row.append(f"g: {d:.2e} (max {np.abs(gref).max():.1e}, sum {np.abs(g.sum(0)).max():.1e}, {out['times'].get('grad', 0) * 1e3:.1f} ms)")
                if "hess" in what:
                    h = np.array(out["hess"]).reshape(3 * natm, 3 * natm)
                    href = ecp_hess_pyscf(mol, dens)
                    d = np.abs(h - href).max()
                    worst = max(worst, d)
                    row.append(f"H: {d:.2e} (max {np.abs(href).max():.1e}, {out['times'].get('hess', 0) * 1e3:.1f} ms)")
            print("  ".join(row), flush=True)
    print(f"worst {worst:.2e}")


def ecp_deriv_pyscf(mol: gto.Mole) -> np.ndarray:
    """d<i|U|j>/dR_A, (natm, 3, nao, nao), from PySCF's ECP derivative integrals."""
    nao = mol.nao
    out = np.zeros((mol.natm, 3, nao, nao))
    aoslices = mol.aoslice_by_atom()
    # basis-center part: ECPscalar_ipnuc = <nabla i|U|j> (electron coordinate) = -d/dA_i
    ip = mol.intor("ECPscalar_ipnuc", comp=3)
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2:]
        out[ia, :, p0:p1] -= ip[:, p0:p1]
        out[ia, :, :, p0:p1] -= ip[:, p0:p1].transpose(0, 2, 1)
    # ECP-center part: ECPscalar_iprinv with rinv at the ECP atom
    for ia in range(mol.natm):
        if not mol.has_ecp() or mol.atom_symbol(ia) not in mol._ecp:
            continue
        with mol.with_rinv_at_nucleus(ia):
            v = mol.intor("ECPscalar_iprinv", comp=3)
        out[ia] += v + v.transpose(0, 2, 1)
    return out


def ecp_hess_pyscf(mol: gto.Mole, dens: np.ndarray) -> np.ndarray:
    """sum_ij D_ij d2<i|U|j>/dR dR by central differences of PySCF analytic ECP derivatives."""
    natm = mol.natm
    h = np.zeros((3 * natm, 3 * natm))
    step = 1e-4
    coords = mol.atom_coords()
    for a in range(natm):
        for x in range(3):
            g = []
            for s in (1, -1):
                c = coords.copy()
                c[a, x] += s * step
                m2 = mol.set_geom_(c, unit="Bohr", inplace=False)
                g.append(np.einsum("axij,ij->ax", ecp_deriv_pyscf(m2), dens).ravel())
            h[3 * a + x] = (g[0] - g[1]) / (2 * step)
    return h


if __name__ == "__main__":
    main()
