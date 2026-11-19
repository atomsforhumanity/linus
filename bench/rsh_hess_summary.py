"""Summary of the range-separated hybrid Hessian benchmark (bench/rsh_hess_bench.sh runs):
wall time, peak memory and frequency errors against the exact-J/K fine-grid references.

  python bench/rsh_hess_summary.py RUNS_DIR [RUNS_DIR ...] REF_DIR OUT.json

each RUNS_DIR (a set, labelled by its name) holds <mol>_<basis>_<prog>_<n>.json records of
bench/ri_hess_scaling.py of the functional of the references; REF_DIR the
reference Hessians <mol>_<basis>.json (linus exact J/K, PySCF level 5 grid, first-order grid
response). Prints a table and writes OUT.json (no host names).
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ri_hess_scaling import frequencies, read_xyz  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    *sets, refs, out = map(pathlib.Path, sys.argv[1:])
    rows = []
    for f in sorted(f for runs in sets for f in runs.glob("*.json")):
        d = json.loads(f.read_text())
        mol, basis = pathlib.Path(d["xyz"]).stem, d.get("basis", "def2-svp")
        ref = refs / f"{mol}_{basis}.json"
        row = {"set": f.parent.name, "run": f.name, "prog": d["prog"], "mol": mol, "basis": basis, "ok": d.get("ok"), "wall_s": d.get("wall"),
               "maxrss_gb": d.get("maxrss_gb") or d.get("maxrss_gb_largest_process"), "loadavg_before": d.get("loadavg_before"),
               "slice": d.get("slice"), "locked": d.get("locked"), "cphf_iterations": d.get("cphf_iterations"), "env": d.get("env")}
        if d.get("hessian") and ref.exists():
            atoms = read_xyz(ROOT / "bench/molecules" / f"{mol}.xyz")
            fr = frequencies(json.loads(ref.read_text())["hessian"], atoms)
            dn = frequencies(d["hessian"], atoms) - fr
            row.update(max_abs_dnu=float(np.abs(dn).max()), rms_dnu=float(np.sqrt((dn ** 2).mean())), nu_at_max=float(fr[int(np.abs(dn).argmax())]))
        if d.get("timers"):
            row["timers"] = d["timers"]
        if d.get("orca_timings"):
            row["orca_timings"] = d["orca_timings"]
        rows.append(row)
        print(f"{f.parent.name:14} {f.name:32} wall {row['wall_s'] or float('nan'):8.1f} s  rss {row['maxrss_gb'] or float('nan'):6.2f} GB  "
              f"max|dnu| {row.get('max_abs_dnu', float('nan')):6.3f}  rms {row.get('rms_dnu', float('nan')):6.3f} cm^-1")
    out.write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
