"""Validate linus's vibrational analysis and thermochemistry.

Run with the reference env, which has PySCF and Psi4 (linus is imported from this source tree,
built with `maturin develop` into any venv: the abi3 extension lands in crates/linus-py/python;
it needs qcdata):

  .refenv/bin/python bench/vib/validate.py hessian MOL.xyz --method hf --basis def2-svp [--jk df]
        [--grid pyscf:3] [--fixed-grid] [--charge 0 --mult 1] [--threads 8] --out CASE.json
      A linus hessian request with ir (linus.compute): Hessian, dipole derivatives, its own analysis.
  .refenv/bin/python bench/vib/validate.py analysis CASE.json [--masses isotope|average] [--orca-dir D]
      The same Hessian through PySCF's, Psi4's and ORCA's (orca_vib, PrintThermoChem) analysis
      and thermochemistry with the masses matched, against linus's (CODATA 2014 for PySCF and
      Psi4, which use it, and linus's default 2022). Prints a table and writes CASE.analysis.json.
  .refenv/bin/python bench/vib/validate.py pyscf CASE.json [--threads 8] [--max-memory 4000]
      End to end: PySCF's own analytic Hessian at the same settings, its frequencies against
      linus's (same masses).

Distances of modes: for each linus mode, the norm of its projection onto the reference modes of
the same frequency (within 0.5 cm^-1, so degenerate pairs count as one subspace); 1 = same.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "crates" / "linus-py" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import refs  # noqa: E402

BOHR = 0.529177210544


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    syms = [ln.split()[0] for ln in lines[2:2 + n]]
    pos = np.array([[float(x) / BOHR for x in ln.split()[1:4]] for ln in lines[2:2 + n]])
    return syms, pos


def cmd_hessian(a):
    import linus
    import qcdata
    from linus._library import atomic_numbers

    syms, pos = read_xyz(a.xyz)
    kw = dict(jk=a.jk, threads=a.threads)
    if a.grid:
        kw["grid"] = {"xc": a.grid}
    # first-order grid response (linus's default) unless --fixed-grid (PySCF's convention); a setting of DFT only
    if a.method.lower() != "hf":
        kw["grid"] = {**kw.get("grid", {}), "response": not a.fixed_grid}
    if not a.no_ir:
        kw["freq"] = {"ir": True}
    structure = qcdata.Structure(symbols=syms, geometry=pos, charge=a.charge, multiplicity=a.mult)
    inp = qcdata.ProgramInput(program="linus", calctype="hessian", structure=structure,
                              model={"method": a.method, "basis": a.basis}, keywords=kw)
    t0 = time.perf_counter()
    out = linus.compute(inp)  # D42: the hessian carries the vibrational analysis and thermochemistry
    wall = time.perf_counter() - t0
    if not out.success:
        raise SystemExit(f"linus failed: {out.error.message}")
    fr, v = out.results, out.results.vibrational_analysis
    case = {
        "name": Path(a.xyz).stem, "xyz": str(a.xyz), "symbols": syms, "numbers": atomic_numbers(syms),
        "positions": pos.tolist(), "charge": a.charge, "multiplicity": a.mult,
        "settings": {"method": a.method, "basis": a.basis, **{k: v for k, v in kw.items() if k != "threads"},
                     "resolved": out.resolved.settings},
        "energy": fr.energy, "hessian": fr.hessian.tolist(),
        "gradient_max": float(np.abs(fr.gradient).max()),
        "apt": None if v.dipole_derivatives is None else np.asarray(v.dipole_derivatives).tolist(),
        "polarizability": None if fr.polarizability is None else np.asarray(fr.polarizability).tolist(),
        "linus": {"vib": v.model_dump(mode="json"), "thermo": fr.thermochemistry[0].model_dump(mode="json")},
        "wall_s": wall, "threads": a.threads, "linus_version": linus.__version__,
    }
    Path(a.out).write_text(json.dumps(case))
    print(f"{case['name']}: E {fr.energy:.10f}, max|g| {case['gradient_max']:.2e}, {v.point_group} "
          f"sigma {v.symmetry_number}, {v.n_imaginary()} imaginary, {wall:.1f} s")
    print("  freq:", np.array2string(v.frequencies, precision=2, max_line_width=150))
    if v.ir_intensities is not None:
        print("  IR  :", np.array2string(v.ir_intensities, precision=2, max_line_width=150))


def mode_overlaps(f_l, x_l, f_r, x_r, tol=0.5):
    """Per linus mode: norm of its projection on the span of the reference modes within tol cm^-1."""
    out = []
    for i, f in enumerate(f_l):
        grp = [j for j, g in enumerate(f_r) if abs(g - f) < tol]
        if not grp:
            out.append(0.0)
            continue
        q, _ = np.linalg.qr(np.asarray(x_r)[grp].T)
        out.append(float(np.linalg.norm(q.T @ x_l[i])))
    return np.array(out)


def nondegenerate(f, tol=0.5):
    f = np.asarray(f)
    return np.array([np.sum(np.abs(f - x) < tol) == 1 for x in f])


def compare_one(name, lin, ref, thermo_keys):
    """Differences linus - ref."""
    fl, fr_ = np.asarray(lin["frequencies"]), np.asarray(ref["frequencies"])
    row = {"program": name, "nmodes": [len(fl), len(fr_)]}
    if len(fl) != len(fr_):
        row["error"] = "different number of modes"
        return row
    row["dfreq_max"] = float(np.abs(fl - fr_).max()) if len(fl) else 0.0
    xl = np.asarray(lin["modes"]).reshape(len(fl), -1)
    ov = mode_overlaps(fl, xl, fr_, ref["modes"])
    row["mode_overlap_min"] = float(ov.min()) if len(ov) else 1.0
    nd = nondegenerate(fl)
    for k in ("reduced_masses", "force_constants"):
        if k in ref and nd.any():
            a, b = np.asarray(lin[k])[nd], np.asarray(ref[k])[nd]
            row[f"d{k}_max_rel"] = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
    if "ir" in ref and lin.get("ir") is not None:
        # real modes only: ORCA reports no intensity for imaginary ones
        real = fl > 0
        a, b = np.asarray(lin["ir"])[real], np.asarray(ref["ir"])[real]
        row["dir_max"] = float(np.abs(a - b).max())
        row["ir_max"] = float(np.abs(b).max())
    if "symmetry_number" in ref:
        row["symmetry_number"] = [lin["symmetry_number"], int(ref["symmetry_number"])]
    if "rotational_constants" in ref:
        a = np.array([x if x is not None else np.inf for x in lin["rotational_constants"]], dtype=float)
        b = np.asarray(ref["rotational_constants"], dtype=float)
        fin = np.isfinite(a) & np.isfinite(b) & (b < 1e5)
        row["drot_max_rel"] = float(np.max(np.abs(a[fin] - b[fin]) / b[fin])) if fin.any() else 0.0
    th = {}
    for k in thermo_keys:
        if k in ref["thermo"] and k in lin["thermo"]:
            th[k] = float(lin["thermo"][k] - ref["thermo"][k])
    row["thermo_diff"] = th
    return row


def linus_flat(va, th, apt_given):
    t = {"zpe": th.zero_point_energy}
    for term in ("electronic", "translational", "rotational", "vibrational", "total"):
        key = {"electronic": "elec", "translational": "trans", "rotational": "rot", "vibrational": "vib", "total": "tot"}[term]
        x = getattr(th, term)
        t[f"E_{key}"], t[f"H_{key}"], t[f"S_{key}"], t[f"G_{key}"] = x.energy, x.enthalpy, x.entropy, x.gibbs
    t["U_corr"] = th.thermal_energy_correction
    t["H_corr"] = th.enthalpy_correction
    t["G_corr"] = th.gibbs_correction
    t["TS_tot"] = th.temperature * th.total.entropy
    for key in ("elec", "trans", "rot", "vib"):
        t[f"TS_{key}"] = th.temperature * t[f"S_{key}"]
    vib = va.is_vibration
    return {
        "frequencies": va.frequencies[vib], "modes": va.normal_modes_cartesian[vib].reshape(int(vib.sum()), -1),
        "reduced_masses": va.reduced_masses[vib], "force_constants": va.force_constants[vib],
        "ir": None if va.ir_intensities is None else va.ir_intensities[vib],
        "symmetry_number": va.symmetry_number, "rotational_constants": list(va.rotational_constants),
        "thermo": t,
    }


def cmd_analysis(a):
    """The same Hessian through every program's analysis. Thermochemistry is compared at the
    reference's symmetry number (so the formulas are compared); the symmetry numbers detected
    are reported in `symmetry_number` [linus, reference]."""
    import linus

    case = json.loads(Path(a.case).read_text())
    z, pos = case["numbers"], np.asarray(case["positions"])
    hess, e, mult = np.asarray(case["hessian"]), case["energy"], case["multiplicity"]
    apt = None if case.get("apt") is None else np.asarray(case["apt"])
    masses = [linus.vib.atomic_mass(x, a.masses) for x in z]
    T, P = 298.15, 101325.0
    results = {"case": case["name"], "masses": a.masses, "rows": []}
    cache = {}

    def lin(cod, sigma=None, **freq):
        key = (cod, sigma)
        if key not in cache:
            cache[key] = linus.vib.harmonic_analysis(z, pos, hess, masses=masses, codata=cod, dipole_derivatives=apt,
                                                     symmetry_number=sigma)
        va = cache[key]
        th = linus.vib.thermochemistry(va, e, mult, temperature=T, pressure=P, codata=cod, **freq)[0]
        return linus_flat(va, th, apt is not None), va

    sigma_linus = lin("codata2022")[1].symmetry_number

    def add(program, cod, ref, keys, **freq):
        flat, va = lin(cod, int(ref["symmetry_number"]), **freq)
        row = compare_one(program, flat, ref, keys)
        row["codata"] = cod
        row["symmetry_number"] = [sigma_linus, int(ref["symmetry_number"])]
        results["rows"].append(row)
        return row, va

    keys_full = ["zpe"] + [f"{f}_{k}" for f in ("E", "H", "S", "G") for k in ("elec", "trans", "rot", "vib", "tot")]
    t0 = time.perf_counter()
    # PySCF: RRHO, Grimme's qRRHO entropy, plus Head-Gordon's qRRHO enthalpy
    ref = refs.pyscf_analysis(z, pos, hess, masses, e, mult, T, P)
    for cod in ("codata2014", "codata2022"):
        add("pyscf", cod, ref, keys_full)
    refq = refs.pyscf_analysis(z, pos, hess, masses, e, mult, T, P, qrrho=True)
    add("pyscf-qS", "codata2014", refq, ["S_vib", "G_tot"], thermo="qrrho")
    refq = refs.pyscf_analysis(z, pos, hess, masses, e, mult, T, P, qrrho=True, qrrho_enthalpy=True)
    add("pyscf-qSH", "codata2014", refq, ["E_vib", "H_tot", "S_vib", "G_tot"], thermo="qrrho-hg")
    # Psi4
    ref = refs.psi4_analysis(z, pos, hess, masses, e, mult, T, P, apt=apt)
    for cod in ("codata2014", "codata2022"):
        row, va = add("psi4", cod, ref, keys_full)
        row["point_group"] = [va.point_group, ref["point_group"]]
    # ORCA
    okeys = ["zpe", "U_corr", "H_corr", "G_corr", "TS_trans", "TS_rot", "TS_vib", "TS_elec", "TS_tot"]
    for q in (False, True):
        name = "orca-qS" if q else "orca"
        ref = refs.orca_analysis(z, pos, hess, masses, e, mult, T, P, apt=apt, qrrho=q,
                                 workdir=Path(a.orca_dir or Path(a.case).parent / "orcavib") / f"{Path(a.case).stem}_{a.masses}_{'qrrho' if q else 'rrho'}")
        if apt is not None and "ir_t2" in ref:
            # orca_vib prints T^2 = |dmu/dQ|^2 hbar / (2 omega) in a.u.: back to km/mol
            km = 974.8801118 * 1822.888486 * 2.0 / 219474.6313705  # km/mol per (T^2 * cm^-1)
            ref["ir"] = ref["ir_t2"] * np.abs(ref["frequencies"]) * km
        row, va = add(name, "codata2022", ref, okeys, thermo="qrrho" if q else "rrho")
        row["point_group"] = [va.point_group, ref["point_group"]]
        # ORCA's wavenumbers are a constant factor below everyone else's for the same Hessian and
        # masses; with that factor as linus's freq.scale the thermochemistry is compared alone
        vib = va.is_vibration
        ratio = np.asarray(ref["frequencies"]) / va.frequencies[vib]
        row["orca_frequency_ratio"] = [float(ratio.min()), float(ratio.max())]
        add(name + "-scaled", "codata2022", ref, okeys, thermo="qrrho" if q else "rrho", scale=float(np.median(ratio)))
    results["wall_s"] = time.perf_counter() - t0
    out = Path(a.case).with_suffix(f".analysis-{a.masses}.json")
    out.write_text(json.dumps(results, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print_rows(case["name"], a.masses, results["rows"])


def print_rows(name, masses, rows):
    print(f"\n{name} ({masses} masses): linus - reference, same Hessian")
    print(f"{'program':12s} {'codata':10s} {'n':>3s} {'max|dnu|':>9s} {'min ovl':>10s} {'mu rel':>8s} {'k rel':>8s} "
          f"{'sigma':>6s} {'rot rel':>8s} {'dIR':>8s}  thermo max |d| (Eh, Eh/K)")
    for r in rows:
        if "error" in r:
            print(f"{r['program']:12s} {r['codata']:10s} {r['error']} {r['nmodes']}")
            continue
        th = r["thermo_diff"]
        worst = max(th.items(), key=lambda kv: abs(kv[1])) if th else ("-", 0.0)
        ths = " ".join(f"{k}={v:+.1e}" for k, v in th.items() if k in ("zpe", "S_tot", "G_tot", "H_tot", "G_corr", "H_corr", "TS_tot", "S_vib"))
        print(f"{r['program']:12s} {r['codata']:10s} {r['nmodes'][0]:3d} {r['dfreq_max']:9.2e} {1 - r['mode_overlap_min']:10.1e} "
              f"{r.get('dreduced_masses_max_rel', float('nan')):8.1e} {r.get('dforce_constants_max_rel', float('nan')):8.1e} "
              f"{str(r.get('symmetry_number', '')):>6s} {r.get('drot_max_rel', float('nan')):8.1e} {r.get('dir_max', float('nan')):8.1e}  "
              f"worst {worst[0]} {worst[1]:+.1e}; {ths}")


def cmd_pyscf(a):
    """PySCF's own analytic Hessian at the case's settings; frequencies against linus's."""
    import linus
    from pyscf import dft, gto, lib, scf
    from pyscf.hessian import thermo

    lib.num_threads(a.threads)
    case = json.loads(Path(a.case).read_text())
    s = case["settings"]
    z, pos = case["numbers"], np.asarray(case["positions"])
    sys.path.insert(0, str(ROOT / "tests" / "python"))
    from compare_ints_pyscf import pyscf_basis

    syms = case["symbols"]
    mol = gto.M(atom=[(sy, p) for sy, p in zip(syms, pos)], unit="Bohr", basis=pyscf_basis(s["basis"], sorted(set(syms))),
                charge=case["charge"], spin=case["multiplicity"] - 1, verbose=0, max_memory=a.max_memory)
    method = s["method"].lower()
    mf = scf.RHF(mol) if method == "hf" else dft.RKS(mol)
    if method != "hf":
        mf.xc = method
        g = s["resolved"]["grid"]["xc"]  # a string such as "pyscf:5"
        if not str(g).startswith("pyscf:"):
            raise SystemExit(f"end-to-end vs PySCF needs a PySCF grid, not {g}")
        mf.grids.level = int(g.split(":")[1])
    jk = s["resolved"]["jk"]
    if jk["kind"] == "df":
        mf = mf.density_fit(auxbasis=pyscf_basis(jk["aux"], sorted(set(syms))))
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    t0 = time.perf_counter()
    mf.kernel()
    hobj = mf.Hessian()
    hobj.conv_tol = 1e-10
    h4 = hobj.kernel()
    wall = time.perf_counter() - t0
    n = len(z)
    hp = h4.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    masses = [linus.vib.atomic_mass(x, "isotope") for x in z]
    res = thermo.harmonic_analysis(mol, h4, mass=np.asarray(masses))
    fw = res["freq_wavenumber"]
    fp = np.where(np.abs(fw.imag) > np.abs(fw.real), -np.abs(fw.imag), fw.real).astype(float)
    vl = case["linus"]["vib"]
    fl = np.asarray(vl["frequencies"])
    hl = np.asarray(case["hessian"])
    print(f"{case['name']} {s['method']}/{s['basis']} jk={jk['kind']}: PySCF E {mf.e_tot:.10f}, linus E {case['energy']:.10f}, "
          f"dE {case['energy'] - mf.e_tot:+.1e}; max|dH| {np.abs(hl - hp).max():.1e} Eh/bohr^2; PySCF {wall:.0f} s")
    print(f"  max|dnu| {np.abs(fl - fp).max():.2e} cm^-1 over {len(fl)} modes; imaginary linus {int((fl < 0).sum())}, PySCF {int((fp < 0).sum())}")
    print("  linus:", np.array2string(fl[:8], precision=3), "...")
    print("  pyscf:", np.array2string(fp[:8], precision=3), "...")
    out = {"case": case["name"], "pyscf_energy": mf.e_tot, "linus_energy": case["energy"], "dhess_max": float(np.abs(hl - hp).max()),
           "dfreq_max": float(np.abs(fl - fp).max()), "pyscf_frequencies": fp.tolist(), "linus_frequencies": fl.tolist(),
           "pyscf_wall_s": wall, "threads": a.threads}
    Path(a.case).with_suffix(".pyscf.json").write_text(json.dumps(out, indent=1))


def cmd_orca(a):
    """End to end: ORCA's own Freq run (its .hess and property file) against linus's Hessian
    analysed with ORCA's masses (standard atomic weights) and quasi-RRHO entropy."""
    import re

    import linus

    case = json.loads(Path(a.case).read_text())
    d = Path(a.orca_run)
    text = (d / "orca.hess").read_text()
    sec = refs._hess_section(text, "vibrational_frequencies").splitlines()
    allf = np.array([float(x.split()[1]) for x in sec[2:2 + int(sec[1])]])
    modes = refs._hess_matrix(refs._hess_section(text, "normal_modes")).T
    keep = np.linalg.norm(modes, axis=1) > 0
    fo, xo = allf[keep], refs._unit_rows(modes[keep])
    ir_o = None
    if "$ir_spectrum" in text:
        rows = refs._hess_section(text, "ir_spectrum").splitlines()[2:]
        ir_o = np.array([float(r.split()[2]) for r in rows if r.strip()])[keep]
    ho = refs._hess_matrix(refs._hess_section(text, "hessian"))
    eo = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(\S+)", (d / "orca.out").read_text())[-1])
    z, pos = case["numbers"], np.asarray(case["positions"])
    apt = None if case.get("apt") is None else np.asarray(case["apt"])
    # thermochemistry at ORCA's symmetry number (the detected numbers are reported separately)
    sn = re.search(r"Point Group:\s+(\S+), Symmetry Number:\s+(\d+)", (d / "orca.out").read_text())
    va0 = linus.vib.harmonic_analysis(z, pos, np.asarray(case["hessian"]), mass_table="average")
    va = linus.vib.harmonic_analysis(z, pos, np.asarray(case["hessian"]), mass_table="average", dipole_derivatives=apt,
                                     symmetry_number=int(sn.group(2)) if sn else None)
    th = linus.vib.thermochemistry(va, case["energy"], case["multiplicity"], thermo="qrrho")[0]
    vib = va.is_vibration
    fl, xl = va.frequencies[vib], va.normal_modes_cartesian[vib].reshape(int(vib.sum()), -1)
    ov = mode_overlaps(fl, xl, fo, xo, tol=2.0)
    prop = (d / "orca.property.txt").read_text()
    g = lambda k: float(re.search(rf"&{k} \[&Type \"Double\"\]\s+(\S+)", prop, re.I).group(1))  # noqa: E731
    el = g("elEnergy")  # U, H and G of a full Freq run include E_el
    T = th.temperature
    rows = {
        "case": case["name"], "dE": case["energy"] - eo, "dhess_max": float(np.abs(np.asarray(case["hessian"]) - ho).max()),
        "dfreq_max": float(np.abs(fl - fo).max()), "mode_overlap_min": float(ov.min()),
        "imaginary": [int((fl < 0).sum()), int((fo < 0).sum())],
        "dzpe": th.zero_point_energy - g("zpe"), "dH_corr": th.enthalpy_correction - (g("enthalpyH") - el),
        "dTS": T * th.total.entropy - g("entropyS"), "dG_corr": th.gibbs_correction - (g("freeEnergyG") - el),
        "orca_frequency_ratio": float(np.median(fo / fl)),
        "linus_frequencies": fl.tolist(), "orca_frequencies": fo.tolist(),
    }
    if ir_o is not None and va.ir_intensities is not None:
        il = va.ir_intensities[vib]
        real = fl > 0  # ORCA reports no intensity for imaginary modes
        rows["dir_max"], rows["ir_max"] = float(np.abs(il - ir_o)[real].max()), float(ir_o.max())
        rows["linus_ir"], rows["orca_ir"] = il.tolist(), ir_o.tolist()
    if sn:
        rows["point_group"] = [va0.point_group, sn.group(1)]
        rows["symmetry_number"] = [va0.symmetry_number, int(sn.group(2))]
    Path(a.case).with_suffix(".orca.json").write_text(json.dumps(rows, indent=1))
    print(f"{case['name']}: dE {rows['dE']:+.1e} Eh, max|dH| {rows['dhess_max']:.1e}, max|dnu| {rows['dfreq_max']:.2e} cm^-1, "
          f"1-min overlap {1 - rows['mode_overlap_min']:.1e}, imaginary {rows['imaginary']}, "
          f"dIR {rows.get('dir_max', float('nan')):.1e} (max {rows.get('ir_max', float('nan')):.1f}) km/mol, "
          f"dZPE {rows['dzpe']:+.1e} dH {rows['dH_corr']:+.1e} dTS {rows['dTS']:+.1e} dG {rows['dG_corr']:+.1e} Eh, "
          f"sym {rows.get('point_group')} {rows.get('symmetry_number')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hessian")
    h.add_argument("xyz")
    h.add_argument("--method", default="hf")
    h.add_argument("--basis", default="def2-svp")
    h.add_argument("--jk", default="exact")
    h.add_argument("--grid")
    h.add_argument("--fixed-grid", action="store_true", help="XC Hessian terms on a fixed grid (PySCF's convention)")
    h.add_argument("--charge", type=int, default=0)
    h.add_argument("--mult", type=int, default=1)
    h.add_argument("--threads", type=int, default=8)
    h.add_argument("--no-ir", action="store_true")
    h.add_argument("--out", required=True)
    an = sub.add_parser("analysis")
    an.add_argument("case")
    an.add_argument("--masses", default="isotope", choices=["isotope", "average"])
    an.add_argument("--orca-dir", help="where the ORCA runs go (default: orcavib/ next to CASE.json)")
    p = sub.add_parser("pyscf")
    p.add_argument("case")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-memory", type=int, default=4000)
    o = sub.add_parser("orca")
    o.add_argument("case")
    o.add_argument("orca_run", help="directory of an ORCA Freq run (orca.out, orca.hess, orca.property.txt)")
    a = ap.parse_args()
    {"hessian": cmd_hessian, "analysis": cmd_analysis, "pyscf": cmd_pyscf, "orca": cmd_orca}[a.cmd](a)


if __name__ == "__main__":
    main()
