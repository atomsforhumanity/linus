"""Generate PySCF reference fixtures for the linus-ints derivative tests.

    python tests/python/gen_deriv_fixtures.py

writes crates/linus-ints/tests/fixtures/pyscf_deriv_ints.json and pyscf_rhf_grad.json
(`... gen_deriv_fixtures.py erf`: pyscf_deriv_ints_erf.json, erf(0.3 r)/r first and second derivatives).

PySCF conventions: "ip" is the derivative with respect to the electron coordinate of the
marked function, i.e. minus the derivative with respect to its center. Every tensor
X[c, i, j, ...] is stored as a random projection P[c, i] = sum_{j,...} X[c, i, j, ...]
v_j ... (vectors in the fixture) over all indices but the component and one "kept" index:
compact, and sensitive to every element. The Rust tests rebuild the same projections from
linus-ints derivative blocks.
"""
import json, pathlib, sys
import numpy as np
from pyscf import gto, scf, df
import pyscf.grad.rhf as grad_rhf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/linus-ints/tests/fixtures"
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl"}


def basis_file(name):
    stem = name.lower().replace("*", "_st_").replace("+", "_pl_").replace("/", "_sl_").replace("(", "_lp_").replace(")", "_rp_").replace(",", "_cm_")
    return ROOT / "data/basis" / f"{stem}.json"


def bse_shells(name, z):
    """Shells of element z from our BSE JSON file as [l, [e, c], ...], sorted by l (stable)."""
    data = json.loads(basis_file(name).read_text())
    shells = []
    for sh in data["elements"][str(z)]["electron_shells"]:
        exps = [float(e) for e in sh["exponents"]]
        ls = sh["angular_momentum"]
        rows = sh["coefficients"]
        if len(ls) == 1:
            ls = ls * len(rows)
        for l, row in zip(ls, rows):
            prims = [[e, float(c)] for e, c in zip(exps, row) if float(c) != 0.0]
            if prims:
                shells.append([l] + prims)
    shells.sort(key=lambda s: s[0])
    return shells


def custom_basis(spec):
    """spec: {z: [(l, exps, coefs), ...]} -> PySCF basis dict (sorted by l)."""
    out = {}
    for z, shells in spec.items():
        s = sorted(shells, key=lambda x: x[0])
        out[SYM[z]] = [[l] + [[e, c] for e, c in zip(ex, co)] for l, ex, co in s]
    return out


# Same test system as crates/linus-ints/tests/common/mod.rs (test_system(true)) and aux_def().
ORB = {
    8: [(0, [5.0, 1.2], [0.4, 0.7]), (1, [1.1], [1.0]), (2, [0.9], [1.0]), (3, [0.7], [1.0])],
    1: [(0, [1.5, 0.4], [0.5, 0.6]), (1, [0.8], [1.0])],
    7: [(0, [0.9], [1.0]), (4, [0.6], [1.0])],
}
AUX = {
    8: [(0, [3.0], [1.0]), (1, [1.5], [1.0]), (2, [1.2], [1.0]), (3, [1.0], [1.0]), (4, [0.8], [1.0])],
    1: [(0, [1.0, 0.3], [0.6, 0.5]), (1, [0.9], [1.0])],
    7: [(0, [1.3], [1.0]), (2, [0.7], [1.0])],
}
Z = [8, 1, 7]
COORDS = [[0.1, -0.2, 0.05], [1.3, 0.9, -0.4], [-0.7, 1.1, 1.2]]


def make_mol(z, coords, basis, charge=0, spin=None):
    atoms = [(SYM[zi], c) for zi, c in zip(z, coords)]
    if spin is None:
        spin = (sum(z) - charge) % 2
    return gto.M(atom=atoms, unit="Bohr", basis=basis, cart=False, charge=charge, spin=spin, verbose=0)


def proj(x, keep, vecs):
    """x: (comp, d1, ..., dm); contract every axis except 0 and `keep` (1-based) with vecs."""
    y = x
    # contract from the last axis backwards
    for ax in range(x.ndim - 1, 0, -1):
        if ax == keep:
            continue
        y = np.tensordot(y, vecs[ax], axes=([ax], [0]))
    return y  # (comp, d_keep)


def rnd(n, seed):
    return np.random.default_rng(seed).uniform(-1, 1, n)


def ints_fixture():
    mol = make_mol(Z, COORDS, custom_basis(ORB))
    auxmol = df.make_auxmol(mol, custom_basis(AUX))
    n, naux = mol.nao, auxmol.nao
    v = {k: rnd(n, 10 + k) for k in range(1, 5)}
    u = {k: rnd(naux, 20 + k) for k in range(1, 3)}
    res = {}

    def put(name, x, keep, vecs):
        p = proj(np.asarray(x), keep, vecs)
        res[name] = {"shape": list(p.shape), "data": p.ravel().tolist()}

    ao2 = {2: v[2]}
    for nm in ["int1e_ipovlp", "int1e_ipkin", "int1e_ipnuc", "int1e_irp",
               "int1e_ipipovlp", "int1e_ipovlpip", "int1e_ipipkin", "int1e_ipkinip", "int1e_ipipnuc", "int1e_ipnucip"]:
        put(nm, mol.intor(nm), 1, ao2)
    for ia in range(mol.natm):
        with mol.with_rinv_at_nucleus(ia):
            for nm in ["int1e_iprinv", "int1e_ipiprinv", "int1e_iprinvip"]:
                put(f"{nm}@{ia}", mol.intor(nm), 1, ao2)
    # core Hamiltonian derivative matrices and second-derivative contraction (PySCF helpers)
    mf = scf.RHF(mol)
    g = mf.nuc_grad_method()
    hgen = grad_rhf.hcore_generator(g, mol)
    for ia in range(mol.natm):
        put(f"hcore_deriv@{ia}", hgen(ia), 1, ao2)
    dmat = rnd(n * n, 30).reshape(n, n)
    dmat = dmat + dmat.T
    hess = mf.Hessian()
    h2gen = hess.hcore_generator(mol)
    hh = np.zeros((3 * mol.natm, 3 * mol.natm))
    for ia in range(mol.natm):
        for ja in range(mol.natm):
            hh[3 * ia:3 * ia + 3, 3 * ja:3 * ja + 3] = np.einsum("xypq,pq->xy", h2gen(ia, ja), dmat)
    res["hcore_hess_contract"] = {"shape": list(hh.shape), "data": hh.ravel().tolist()}
    res["hcore_hess_dmat"] = {"shape": [n, n], "data": dmat.ravel().tolist()}
    ao4 = {2: v[2], 3: v[3], 4: v[4]}
    for nm in ["int2e_ip1", "int2e_ipip1", "int2e_ipvip1", "int2e_ip1ip2"]:
        put(nm, mol.intor(nm), 1, ao4)
    # 3c: (comp, i, j, P)
    for nm, keep in [("int3c2e_ip1", 1), ("int3c2e_ipip1", 1), ("int3c2e_ipvip1", 1), ("int3c2e_ip1ip2", 1),
                     ("int3c2e_ip2", 3), ("int3c2e_ipip2", 3)]:
        x = df.incore.aux_e2(mol, auxmol, intor=nm, aosym="s1")
        vecs = {1: v[1], 2: v[2], 3: u[1]}
        put(nm, x, keep, vecs)
    for nm in ["int2c2e_ip1", "int2c2e_ipip1", "int2c2e_ip1ip2"]:
        put(nm, auxmol.intor(nm), 1, {2: u[2]})
    fx = {
        "z": Z, "coords": COORDS,
        "basis": {str(z): [[l, e, c] for l, e, c in sh] for z, sh in ORB.items()},
        "aux": {str(z): [[l, e, c] for l, e, c in sh] for z, sh in AUX.items()},
        "nbf": n, "naux": naux,
        "v1": v[1].tolist(), "v2": v[2].tolist(), "v3": v[3].tolist(), "v4": v[4].tolist(),
        "u1": u[1].tolist(), "u2": u[2].tolist(),
        "ints": res,
    }
    return fx


def erf_fixture(omega=0.3):
    """First and second derivative integrals of erf(omega r)/r (mol.with_range_coulomb), same
    system and projections as ints_fixture: int2e_ip1/ipip1/ipvip1/ip1ip2,
    int3c2e_ip1/ip2/ipip1/ipvip1/ip1ip2/ipip2, int2c2e_ip1/ipip1/ip1ip2."""
    mol = make_mol(Z, COORDS, custom_basis(ORB))
    auxmol = df.make_auxmol(mol, custom_basis(AUX))
    n, naux = mol.nao, auxmol.nao
    v = {k: rnd(n, 10 + k) for k in range(1, 5)}
    u = {k: rnd(naux, 20 + k) for k in range(1, 3)}
    res = {}

    def put(name, x, keep, vecs):
        p = proj(np.asarray(x), keep, vecs)
        res[name] = {"shape": list(p.shape), "data": p.ravel().tolist()}

    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        for nm in ["int2e_ip1", "int2e_ipip1", "int2e_ipvip1", "int2e_ip1ip2"]:
            put(nm, mol.intor(nm), 1, {2: v[2], 3: v[3], 4: v[4]})
        for nm, keep in [("int3c2e_ip1", 1), ("int3c2e_ip2", 3), ("int3c2e_ipip1", 1), ("int3c2e_ipvip1", 1),
                         ("int3c2e_ip1ip2", 1), ("int3c2e_ipip2", 3)]:
            put(nm, df.incore.aux_e2(mol, auxmol, intor=nm, aosym="s1"), keep, {1: v[1], 2: v[2], 3: u[1]})
        for nm in ["int2c2e_ip1", "int2c2e_ipip1", "int2c2e_ip1ip2"]:
            put(nm, auxmol.intor(nm), 1, {2: u[2]})
    return {"omega": omega, "nbf": n, "naux": naux, "ints": res}


def read_xyz(path):
    lines = pathlib.Path(path).read_text().splitlines()
    nat = int(lines[0])
    zs, xyz = [], []
    for ln in lines[2:2 + nat]:
        s, x, y, zz = ln.split()[:4]
        zs.append(gto.charge(s))
        xyz.append([float(x), float(y), float(zz)])
    return zs, (np.array(xyz) / 0.529177210544).tolist()  # linus' BOHR_ANGSTROM


def rhf_case(name, xyz, basis, auxbasis=None):
    zs, coords = read_xyz(ROOT / "tests/molecules" / xyz)
    bas = {SYM[z]: bse_shells(basis, z) for z in set(zs)}
    mol = make_mol(zs, coords, bas)
    mf = scf.RHF(mol)
    if auxbasis:
        mf = mf.density_fit(auxbasis={SYM[z]: bse_shells(auxbasis, z) for z in set(zs)})
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    e = mf.kernel()
    assert mf.converged
    g = mf.nuc_grad_method().kernel()
    occ = mf.mo_occ > 0
    return {
        "name": name, "z": zs, "coords": coords, "basis": basis, "aux": auxbasis,
        "nbf": mol.nao, "nocc": int(occ.sum()),
        "c_occ": mf.mo_coeff[:, occ].ravel().tolist(), "e_occ": mf.mo_energy[occ].tolist(),
        "energy": e, "grad": g.ravel().tolist(),
    }


def grad_fixture():
    cases = [
        rhf_case("water/def2-svp", "water.xyz", "def2-svp"),
        rhf_case("methanethiol/def2-tzvp", "ch3sh.xyz", "def2-tzvp"),
        rhf_case("water/def2-svp DF(def2-universal-jkfit)", "water.xyz", "def2-svp", "def2-universal-jkfit"),
    ]
    return {"cases": cases}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["ints", "grad", "erf"]
    if "ints" in which:
        (OUT / "pyscf_deriv_ints.json").write_text(json.dumps(ints_fixture()))
    if "grad" in which:
        (OUT / "pyscf_rhf_grad.json").write_text(json.dumps(grad_fixture()))
    if "erf" in which:
        (OUT / "pyscf_deriv_ints_erf.json").write_text(json.dumps(erf_fixture()))
    for f in sorted(OUT.glob("*.json")):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
