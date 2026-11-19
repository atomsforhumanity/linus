"""Compare Hessians (frequencies) of one molecule from linus JSON (examples/hess output, or a
bench/ri_hess_scaling.py record) and ORCA .hess files.

  python bench/hess_compare.py XYZ REF OTHER [OTHER ...] [--json OUT]

REF and OTHER: a .json with "hessian" (3N x 3N, Eh/bohr^2) or an ORCA .hess file. Prints, per
OTHER, max |dH| and max/rms |dnu| (harmonic frequencies, cm^-1, most abundant isotopes,
translations/rotations projected; bench/ri_hess_scaling.frequencies) against REF.
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ri_hess_scaling import frequencies, read_xyz  # noqa: E402


def load(p):
    p = pathlib.Path(p)
    if p.suffix == ".hess":
        lines = p.read_text().splitlines()
        i = lines.index("$hessian")
        n3 = int(lines[i + 1])
        h = np.zeros((n3, n3))
        r, c0 = i + 2, 0
        while c0 < n3:
            cols = [int(x) for x in lines[r].split()]
            for k in range(n3):
                vals = lines[r + 1 + k].split()
                for j, c in enumerate(cols):
                    h[int(vals[0]), c] = float(vals[1 + j])
            c0 += len(cols)
            r += n3 + 1
        return h
    d = json.loads(p.read_text())
    h = np.asarray(d["hessian"])
    n3 = int(round(np.sqrt(h.size)))
    return h.reshape(n3, n3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("ref")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    atoms = read_xyz(a.xyz)
    href = load(a.ref)
    fref = frequencies(href, atoms)
    out = {"ref": a.ref, "ref_frequencies": fref.tolist(), "cmp": []}
    for o in a.others:
        h = load(o)
        f = frequencies(h, atoms)
        d = f - fref
        i = int(np.abs(d).argmax())
        rec = {"other": o, "max_abs_dH": float(np.abs(h - href).max()), "max_abs_dnu": float(np.abs(d).max()), "rms_dnu": float(np.sqrt((d ** 2).mean())),
               "nu_at_max": float(fref[i]), "frequencies": f.tolist()}
        out["cmp"].append(rec)
        print(f"{pathlib.Path(o).name:40} max|dH| {rec['max_abs_dH']:.2e}  max|dnu| {rec['max_abs_dnu']:.3f} cm-1 (at {rec['nu_at_max']:.1f})  rms {rec['rms_dnu']:.3f}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
