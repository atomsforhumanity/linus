"""Psi4 1.11 + PCMSolver reference energies (IEF-PCM and C-PCM, GePol cavity) for linus's implicit solvation.

For water, NH3, acetate and L-lactide (la18) at HF and B3LYP (Psi4 `b3lyp` = VWN3 = ORCA `B3LYP/G`) / def2-SVP, exact
J/K (scf_type pk), solvents THF (eps = 7.4257, eps_inf = 1.9718 (n = 1.4050^2), explicit) and water (PCMSolver's built-in
`Water`): total energy, gas-phase energy at the same settings, PCM polarization energy, cavity (spheres, tesserae,
area, volume, radii) as PCMSolver reports it. Also probes which PCM derivatives Psi4 offers (gradient, Hessian) and
records the error text or the method Psi4 picks.

Writes bench/results/2026-11-07-pcm/psi4/<mol>_<method>_<solver>_<solvent>.json plus derivatives.json and
cavity_settings.json. Raw Psi4 outputs stay in --work.

usage: systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 ~/.local/bin/slice ccd2 \
           .refenv/bin/python bench/pcm_psi4.py --threads 4
"""
import argparse, json, os, pathlib, re, time, traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = ROOT / "bench/molecules"
OUT = ROOT / "bench/results/2026-11-07-pcm/psi4"
from bench_paths import SCRATCH  # noqa: E402  (env override BENCH_SCRATCH: bench/bench_paths.py)
WORK = SCRATCH / "psi4-pcm"
BOHR = 0.529177210544
SYSTEMS = {"water": (0, 1, "pcm/water"), "nh3": (0, 1, "pcm/nh3"), "acetate": (-1, 1, "pcm/acetate"), "la18": (0, 1, "la18")}

CAVITY = dict(Type="GePol", RadiiSet="Bondi", Scaling="True", Area=0.3, Mode="Implicit")
SOLVENTS = {
    # explicit THF permittivities: eps = 7.4257 (Minnesota solvent database, as gpu4pyscf/PySCF), eps_inf = n^2 = 1.4050^2
    "thf": "ProbeRadius = 2.9\n Green<inside> {\n  Type = Vacuum\n }\n Green<outside> {\n  Type = UniformDielectric\n  Eps = 7.4257\n  EpsDyn = 1.97402500\n }",
    "water": "Solvent = Water",
}


def read_xyz(path):
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + n]]


def pcm_input(solver, solvent, cavity=CAVITY):
    cav = "\n".join(f" {k} = {v}" for k, v in cavity.items())
    return f"""
Units = Angstrom
Medium {{
 SolverType = {solver}
 {SOLVENTS[solvent]}
}}
Cavity {{
{cav}
}}
"""


def molecule(psi4, name):
    charge, mult, geo = SYSTEMS[name]
    atoms = read_xyz(MOLS / f"{geo}.xyz")
    s = f"{charge} {mult}\nunits bohr\nno_reorient\nno_com\nsymmetry c1\n" + "\n".join(
        f"{a} {c[0] / BOHR:.12f} {c[1] / BOHR:.12f} {c[2] / BOHR:.12f}" for a, c in atoms)
    return psi4.geometry(s, name=name.replace("+", "p")), atoms, charge, mult


def parse_pcm_block(text):
    """Cavity numbers from PCMSolver's report in the Psi4 output."""
    d = {}
    for key, pat, conv in [("cavity_type", r"Cavity type:\s+(.+)", str), ("radii_set", r"Atomic radii set:[ \t]*(.*)", str),
                           ("average_tessera_area_ang2", r"Average tesserae area\s*=\s*(\S+)", float),
                           ("solvent_probe_radius_ang", r"Solvent probe radius\s*=\s*(\S+)", float), ("nspheres", r"Number of spheres\s*=\s*(\d+)", int),
                           ("nfinite_elements", r"Number of finite elements\s*=\s*(\d+)", int),
                           ("pruned", r"Number of finite elements\s*=\s*\d+ \((\d+) pruned\)", int),
                           ("solver_type", r"Solver Type:\s+(.+)", str), ("correction", r"Correction\s*=\s*(\S+)", float),
                           ("permittivity", r"Permittivity\s*=\s*(\S+)", float), ("solvent", r"Solvent name:\s+(.+)", str),
                           ("constants", r"Using (CODATA \S+) set of constants", str), ("pcmsolver_version", r"PCMSolver (\S+\+\S+)", str),
                           ("pcm_polarization_energy", r"PCM Polarization Energy =\s+(\S+)", float)]:
        m = re.findall(pat, text)
        if m:
            try:
                d[key] = conv(m[-1].strip())
            except ValueError:
                d[key] = m[-1].strip()
    sph = re.findall(r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", text, flags=re.M)
    d["sphere_radius_alpha_by_element"] = {e: [float(r), float(al)] for _, e, r, al, *_ in sph}
    d["solvent_block"] = re.findall(r"(?s)============ Medium(.*?)\n\s*\n", text)[-1:] if "============ Medium" in text else None
    return d


def cavity_npz(path):
    """PCMSolver's cavity.npz (written to the cwd): tesserae count, total area, divergence-theorem volume (bohr)."""
    import numpy as np
    if not pathlib.Path(path).exists():
        return None
    z = np.load(path)
    w = z["weights"].ravel()
    c = z["centers"].T
    n = z["normals"].T
    n = n / np.linalg.norm(n, axis=1)[:, None]
    return dict(n_tesserae=int(z["elements"][0]), area_bohr2=float(w.sum()), volume_bohr3=float((w * np.einsum("ij,ij->i", c, n)).sum() / 3),
                sphere_radii_bohr=sorted(set(round(float(r), 6) for r in z["elRadius"].ravel())))


def run_case(psi4, a, mol, method, solver, solvent):
    wd = pathlib.Path(a.work) / f"{mol}_{method}_{solver}_{solvent}"
    wd.mkdir(parents=True, exist_ok=True)
    os.chdir(wd)
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_output_file(str(wd / "psi4.out"), False)
    psi4.core.IOManager.shared_object().set_default_path(str(wd))
    m, atoms, charge, mult = molecule(psi4, mol)
    base = dict(basis="def2-svp", scf_type="pk", e_convergence=1e-10, d_convergence=1e-8, maxiter=200, reference="rhf" if mult == 1 else "uhf")
    psi4.set_options(base)
    t0 = time.perf_counter()
    e_gas = psi4.energy(method, molecule=m)
    t_gas = time.perf_counter() - t0
    psi4.core.clean()
    psi4.set_options(dict(pcm=True, pcm_scf_type="total"))
    psi4.pcm_helper(pcm_input(solver, solvent))
    t0 = time.perf_counter()
    e_pcm = psi4.energy(method, molecule=m)
    t_pcm = time.perf_counter() - t0
    epol = psi4.variable("PCM POLARIZATION ENERGY")
    text = (wd / "psi4.out").read_text(errors="replace")
    rec = dict(case=f"{mol}_{method}_{solver}_{solvent}", mol=mol, charge=charge, mult=mult, method=method, basis="def2-SVP",
               psi4_version=psi4.__version__, options={**base, "pcm": True, "pcm_scf_type": "total"}, pcm_input=pcm_input(solver, solvent),
               dft_grid="Psi4 default (75 radial, 302 spherical)" if method != "hf" else None,
               geometry_bohr=[[s] + [x / BOHR for x in c] for s, c in atoms], units="Eh",
               energy=e_pcm, gas_energy=e_gas, delta_solv=e_pcm - e_gas, pcm_polarization_energy=epol,
               pcmsolver=parse_pcm_block(text), cavity=cavity_npz(wd / "cavity.npz"), wall_gas=t_gas, wall_pcm=t_pcm,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (OUT / f"{rec['case']}.json").write_text(json.dumps(rec, indent=1))
    print(f"{rec['case']:36s} E={e_pcm:.10f} gas={e_gas:.10f} dG={rec['delta_solv'] * 627.509:8.3f} kcal/mol Epol={epol:.8f} "
          f"nts={rec['pcmsolver'].get('nfinite_elements')}", flush=True)
    return rec


def probe_derivatives(psi4, a):
    """What PCM derivatives does Psi4 1.11 have? Try an analytic gradient and a Hessian on water/HF/def2-SVP with PCM."""
    res = {}
    for kind in ["gradient", "hessian"]:
        for method in ["hf", "b3lyp"]:
            wd = pathlib.Path(a.work) / f"deriv_{kind}_{method}"
            wd.mkdir(parents=True, exist_ok=True)
            os.chdir(wd)
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.set_output_file(str(wd / "psi4.out"), False)
            m, *_ = molecule(psi4, "water")
            psi4.set_options(dict(basis="def2-svp", scf_type="pk", e_convergence=1e-10, d_convergence=1e-8, pcm=True, pcm_scf_type="total"))
            psi4.pcm_helper(pcm_input("CPCM", "water"))
            t0 = time.perf_counter()
            try:
                fn = getattr(psi4, kind)
                r = fn(method, molecule=m, dertype=1) if kind == "hessian" else fn(method, molecule=m)
                text = (wd / "psi4.out").read_text(errors="replace")
                how = sorted(set(l.strip() for l in text.splitlines() if re.search(r"finite.difference|displacement|analytic|Using finite", l, flags=re.I)))[:12]
                res[f"{kind}_{method}"] = dict(ok=True, wall=time.perf_counter() - t0, how=how, shape=list(r.np.shape) if hasattr(r, "np") else None)
            except Exception as e:
                res[f"{kind}_{method}"] = dict(ok=False, wall=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}"[:1500])
            print(kind, method, json.dumps(res[f"{kind}_{method}"])[:600], flush=True)
            # hessian with dertype=1 failing -> also try the default (Psi4 picks dertype itself)
            if kind == "hessian" and not res[f"{kind}_{method}"]["ok"] and method == "hf":
                t0 = time.perf_counter()
                try:
                    psi4.core.clean()
                    r = psi4.hessian(method, molecule=m)
                    text = (wd / "psi4.out").read_text(errors="replace")
                    how = sorted(set(l.strip() for l in text.splitlines() if re.search(r"finite.difference|displacement|analytic|Using finite", l, flags=re.I)))[:12]
                    res["hessian_hf_default_dertype"] = dict(ok=True, wall=time.perf_counter() - t0, how=how)
                except Exception as e:
                    res["hessian_hf_default_dertype"] = dict(ok=False, wall=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}"[:1500])
                print("hessian default", json.dumps(res["hessian_hf_default_dertype"])[:600], flush=True)
    (OUT / "derivatives.json").write_text(json.dumps(dict(psi4_version=psi4.__version__, system="water HF and B3LYP/def2-SVP, CPCM(water) GePol",
                                                          results=res), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="8 GB")
    ap.add_argument("--mols", default="water,nh3,acetate,la18")
    ap.add_argument("--methods", default="hf,b3lyp")
    ap.add_argument("--solvers", default="IEFPCM,CPCM")
    ap.add_argument("--solvents", default="thf,water")
    ap.add_argument("--skip-derivatives", action="store_true")
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--reparse", action="store_true", help="only re-read the PCMSolver report of finished runs into their JSON")
    a = ap.parse_args()
    if a.reparse:
        for f in sorted(OUT.glob("*_*PCM_*.json")):
            rec = json.loads(f.read_text())
            out = pathlib.Path(a.work) / rec["case"] / "psi4.out"
            if out.exists():
                rec["pcmsolver"] = parse_pcm_block(out.read_text(errors="replace"))
                f.write_text(json.dumps(rec, indent=1))
        return
    import psi4
    OUT.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.work).mkdir(parents=True, exist_ok=True)
    psi4.set_num_threads(a.threads)
    psi4.set_memory(a.memory)
    (OUT / "cavity_settings.json").write_text(json.dumps(dict(psi4_version=psi4.__version__, pcmsolver=psi4.addons("pcmsolver"), cavity=CAVITY,
                                                              solvents=SOLVENTS, note="Cavity block as passed to pcm_helper (Units = Angstrom)"), indent=1))
    for mol in a.mols.split(","):
        for method in a.methods.split(","):
            for solver in a.solvers.split(","):
                for solvent in a.solvents.split(","):
                    try:
                        run_case(psi4, a, mol, method, solver, solvent)
                    except Exception:
                        print(f"{mol} {method} {solver} {solvent} FAILED\n{traceback.format_exc()[-1500:]}", flush=True)
    if not a.skip_derivatives:
        probe_derivatives(psi4, a)


if __name__ == "__main__":
    main()
