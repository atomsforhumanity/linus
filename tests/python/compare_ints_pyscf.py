"""Compare linus-ints integrals with PySCF (libcint) using identical basis definitions.

python tests/python/compare_ints_pyscf.py <molecule.xyz> <basis> [aux] [--omega w]

With --omega, also compares the range-separated integrals erf(w r)/r and erfc(w r)/r with
PySCF's mol.with_range_coulomb(w) / (-w).
"""
import json, subprocess, sys, tempfile, pathlib
import numpy as np
from pyscf import gto, df

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ROOT, pyscf_basis, read_xyz  # noqa: E402,F401  (pyscf_basis: imported from here by run_ref.py and others)


def main():
    xyz, basis = sys.argv[1], sys.argv[2]
    args = sys.argv[1:]
    omega = None
    if "--omega" in args:
        i = args.index("--omega")
        omega = float(args[i + 1])
        del args[i:i + 2]
    sys.argv = [sys.argv[0]] + args
    aux = sys.argv[3] if len(sys.argv) > 3 else None
    atom_syms, pos = read_xyz(xyz)
    syms = sorted(set(atom_syms))
    mol = gto.M(atom=list(zip(atom_syms, pos)), unit="Angstrom", basis=pyscf_basis(basis, syms), cart=False, verbose=0, spin=sum(gto.charge(s) for s in atom_syms) % 2)
    coords = mol.atom_coords().tolist()
    z = [int(c) for c in mol.atom_charges()]
    nbf = mol.nao
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td) / "in.json"
        do_eri = nbf <= 140
        inp.write_text(json.dumps({"z": z, "coords": coords, "basis": basis, "aux": aux, "pure": True, "eri": do_eri, "omega": omega}))
        exe = ROOT / "target/release/examples/dump_ints"
        info = json.loads(subprocess.check_output([str(exe), str(inp), td]))
        assert info["nbf"] == nbf, (info, nbf)
        ld = lambda n, shape: np.fromfile(pathlib.Path(td) / n, dtype="<f8").reshape(shape)
        res = {}
        res["S"] = np.abs(ld("S.bin", (nbf, nbf)) - mol.intor("int1e_ovlp")).max()
        res["T"] = np.abs(ld("T.bin", (nbf, nbf)) - mol.intor("int1e_kin")).max()
        res["V"] = np.abs(ld("V.bin", (nbf, nbf)) - mol.intor("int1e_nuc")).max()
        r = mol.intor("int1e_r")
        res["dip"] = max(np.abs(ld(f"D{c}.bin", (nbf, nbf)) - r[i]).max() for i, c in enumerate("XYZ"))
        if do_eri:
            res["ERI"] = np.abs(ld("ERI.bin", (nbf,) * 4) - mol.intor("int2e")).max()
            if omega is not None:
                with mol.with_range_coulomb(omega):
                    lr = mol.intor("int2e")
                with mol.with_range_coulomb(-omega):
                    sr = mol.intor("int2e")
                res["ERI_LR"] = np.abs(ld("ERI_LR.bin", (nbf,) * 4) - lr).max()
                res["ERI_SR"] = np.abs(ld("ERI_SR.bin", (nbf,) * 4) - sr).max()
        if aux:
            auxmol = df.make_auxmol(mol, pyscf_basis(aux, syms))
            naux = auxmol.nao
            j2 = auxmol.intor("int2c2e")
            j3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e").reshape(nbf * nbf, naux).T
            res["J2C"] = np.abs(ld("J2C.bin", (naux, naux)) - j2).max()
            res["J3C"] = np.abs(ld("J3C.bin", (naux, nbf * nbf)) - j3).max()
        print(json.dumps({"xyz": pathlib.Path(xyz).name, "basis": basis, "aux": aux, "nbf": nbf, **{k: float(f"{v:.3e}") for k, v in res.items()}, "t_eri": info["t_eri"]}))


if __name__ == "__main__":
    main()
