"""Markdown tables for bench/results/2026-11-07-openshell-scf: builds and energies old vs new over
the open-shell set (openshell_set.json), the SCF benchmark set (scf_set.json) and the SCF change
validation (validation_old_new.json), and the PySCF comparison (pyscf.json).

usage: python3 bench/openshell_summary.py [RESULTS_DIR]
"""
import json, pathlib, sys

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "bench/results/2026-11-07-openshell-scf")


def fmt_e(x):
    return "-" if x is None else f"{x:.10f}"


def openshell():
    p = R / "openshell_set.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())["results"]
    print("| case | old: builds, final max F_ai | new: builds, final max F_ai | new: stability analyses (eigenvalue, builds) | E new (Eh) | E new - old | <S^2> new |")
    print("|---|---|---|---|---|---|---|")
    for case, r in d.items():
        o, n = r.get("old", {}), r.get("new", {})

        def cell(x):
            if not x:
                return "-"
            b = x.get("builds")
            g = x.get("final_gmax")
            return f"{b}, {g:.1e}" if x.get("converged") else f"not converged ({b}), {g:.1e}" if g is not None else f"failed ({b})"
        st = n.get("stability_checks") or []
        stab = "; ".join(f"{c['eigenvalue']:+.2e}, {c['fock_builds']}" for c in st) if st else "-"
        eo, en = o.get("energy"), n.get("energy")
        both = eo is not None and en is not None and o.get("converged") and n.get("converged")
        de = ("0 (bitwise)" if en == eo else f"{en - eo:+.1e}") if both else "-"
        s2 = f"{n['s2']:.5f}" if n.get("s2") is not None else "-"
        print(f"| {case} | {cell(o)} | {cell(n)} | {stab} | {fmt_e(en)} | {de} | {s2} |")
    print()


def scf_set():
    p = R / "scf_set.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())["results"]
    print("| case | builds old | builds new | E new - old (Eh) |")
    print("|---|---|---|---|")
    for case, r in d.items():
        o, n = r.get("old", {}), r.get("new", {})
        de = ("0 (bitwise)" if n["energy"] == o["energy"] else f"{n['energy'] - o['energy']:+.1e}") if o.get("ok") and n.get("ok") else "-"
        print(f"| {case} | {o.get('builds')} | {n.get('builds')} | {de} |")
    print()


def validation():
    p = R / "validation_old_new.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())["results"]
    nb = sum(1 for r in d.values() if r.get("diff", {}).get("bitwise"))
    print(f"SCF change validation: {nb} of {len(d)} cases bitwise identical (energies and gradients)")
    for k, r in d.items():
        df = r.get("diff")
        if df and not df["bitwise"]:
            print(f"  {k}: builds {df['builds']} dE {df['de']:+.1e} max|dG| {df['max_dg']:.1e}")
        if not df:
            print(f"  {k}: failed")
    print()


def pyscf():
    p = R / "pyscf.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())["results"]
    print("| case | linus: E (Eh, no dispersion), <S^2>, builds | PySCF at linus's density: E_linus - E_PySCF, max F_ai | PySCF from linus's orbitals: dE, cycles | PySCF alone (minao): dE, <S^2>, cycles, converged |")
    print("|---|---|---|---|---|")
    for case, r in d.items():
        l, a, f, o = r["linus"], r["pyscf_at_linus_density"], r["pyscf_from_linus"], r.get("pyscf_own")
        own = f"{o['de_linus_minus_pyscf']:+.1e}, {o['s2']:.5f}, {o['cycles']}, {'yes' if o['converged'] else 'no (max F_ai %.1e)' % o['max_fai']}" if o else "-"
        print(f"| {case} | {l['energy_scf']:.10f}, {l['s2']:.5f}, {l['fock_builds']} | {a['de_linus_minus_pyscf']:+.1e}, {a['max_fai']:.1e} | {f['de_linus_minus_pyscf']:+.1e}, {f['cycles']} | {own} |")
    print()


if __name__ == "__main__":
    openshell()
    scf_set()
    validation()
    pyscf()
