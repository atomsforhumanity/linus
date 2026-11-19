"""ORCA 6.0.0 side of the end-to-end workflows, through qccompute.

    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 -p OOMPolicy=continue \
        taskset -c 32-39 <venv>/bin/python bench/workflows/orca_workflows.py minima|ts|sp

Level B3LYP/G D3BJ def2-SVP def2/J RIJCOSX DefGrid2 TightSCF (`common.SVP_ORCA`), 8 MPI ranks;
the affinity of this process (taskset) is inherited by the ranks. Each stage writes
bench/results/2026-11-07-workflows/orca_<stage>_<mol>.json and is skipped when that file exists.

- minima: Opt (TightOpt) of la18 and u1an39 from the perturbed starts, then Freq.
- ts: OptTS of ts61 from bench/molecules/ts61.xyz (calc_hess true, TightOpt), Freq at the TS,
  IRC both ways from the Freq Hessian (FileInput: the ORCA codec has no IRC calculation type),
  Opt (TightOpt) and Freq of both IRC ends.
- sp: wB97X-D3BJ/def2-TZVP RIJCOSX single points on ORCA's optimized structures.

ORCA is licensed software: only small JSON summaries are kept, never ORCA's own files.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from qccompute import compute  # noqa: E402
from qcdata import CalcType, FileInput, Model, ProgramInput, Structure  # noqa: E402

EXE = {"orca": str(C.ORCA / "orca")}


def orca_input(calctype: CalcType, s: Structure, method: str, basis: str = "def2-SVP", **keywords) -> ProgramInput:
    kw = {"pal": C.ORCA_PAL, "maxcore": C.ORCA_MAXCORE, **keywords}
    return ProgramInput(program="orca", calctype=calctype, structure=s, model=Model(method=method, basis=basis),
                        keywords=kw)


def run(inp, **kwargs):
    t = time.perf_counter()
    out = compute(inp, raise_exc=False, environment=C.orca_env(), executables=EXE, **kwargs)
    return out, time.perf_counter() - t


def thermo_from_log(log: str) -> dict:
    """ORCA's own thermochemistry lines (Eh), if present."""
    pats = {
        "zpe": r"Zero point energy\s+\.\.\.\s+(-?\d+\.\d+) Eh",
        "total_enthalpy": r"Total Enthalpy\s+\.\.\.\s+(-?\d+\.\d+) Eh",
        "total_entropy_correction": r"Total entropy correction\s+\.\.\.\s+(-?\d+\.\d+) Eh",
        "final_gibbs": r"Final Gibbs free energy\s+\.\.\.\s+(-?\d+\.\d+) Eh",
        "g_minus_eel": r"G-E\(el\)\s+\.\.\.\s+(-?\d+\.\d+) Eh",
        "electronic_energy": r"Electronic energy\s+\.\.\.\s+(-?\d+\.\d+) Eh",
    }
    out = {}
    for k, p in pats.items():
        m = re.findall(p, log or "")
        if m:
            out[k] = float(m[-1])
    return out


def opt_summary(out, wall: float) -> dict:
    res = out.results
    frames = list(getattr(res, "frames", None) or [])
    log = out.logs or ""
    d = {
        "success": out.success,
        "converged": "HURRAY" in log,
        "not_converged_message": "did not converge" in log,
        "n_steps": len(frames),
        "energies": [f.results.energy if f.results is not None else None for f in frames],
        "final_energy": frames[-1].results.energy if frames and frames[-1].results is not None else None,
        "final_geometry_bohr": (res.final_structure.geometry.tolist() if getattr(res, "final_structure", None) else None),
        "wall_s": wall,
        "traceback": None if out.success else (out.traceback or "")[-3000:],
    }
    return d


def freq_summary(out, wall: float) -> dict:
    res = out.results
    d = {
        "success": out.success,
        "energy": getattr(res, "energy", None),
        "freqs_wavenumber": list(map(float, res.freqs_wavenumber)) if getattr(res, "freqs_wavenumber", None) is not None else None,
        "hessian": res.hessian.tolist() if getattr(res, "hessian", None) is not None else None,
        "orca_thermo": thermo_from_log(out.logs or ""),
        "wall_s": wall,
        "traceback": None if out.success else (out.traceback or "")[-3000:],
    }
    if d["freqs_wavenumber"] is not None:
        d["n_imaginary"] = sum(1 for f in d["freqs_wavenumber"] if f < -20.0)
    return d


def env_record() -> dict:
    return {"affinity": C.affinity(), "orca_pal": C.ORCA_PAL, "ccd_busy_cores": C.ccd_load(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def minima(names: list[str]) -> None:
    for name in names:
        key = f"orca_min_{name}"
        if C.load(key):
            print(f"{key}: done", flush=True)
            continue
        rec = {"program": "orca 6.0.0", "molecule": name, "level": C.SVP_ORCA + " / def2-SVP", "threads": C.THREADS,
               "before": env_record()}
        t0 = time.perf_counter()
        s = C.structure(name, perturbed=True)
        out, wall = run(orca_input(CalcType.optimization, s, C.SVP_ORCA + " TightOpt", geom={"maxiter": 200}))
        rec["opt"] = opt_summary(out, wall)
        print(f"{name}: opt {rec['opt']['n_steps']} steps, E={rec['opt']['final_energy']}, {wall:.0f} s", flush=True)
        if rec["opt"]["final_geometry_bohr"] is not None:
            final = C.with_geometry(s, rec["opt"]["final_geometry_bohr"])
            out, wall = run(orca_input(CalcType.hessian, final, C.SVP_ORCA))
            rec["freq"] = freq_summary(out, wall)
            print(f"{name}: freq n_imag={rec['freq'].get('n_imaginary')}, {wall:.0f} s", flush=True)
        rec["total_wall_s"] = time.perf_counter() - t0
        rec["after"] = env_record()
        C.save(key, rec)


IRC_INPUT = """! {method} def2-SVP IRC
%pal nprocs {pal} end
%maxcore {maxcore}
%irc
  maxiter 100
  inithess read
  hess_filename "ts.hess"
end
* xyzfile {charge} {mult} ts.xyz
"""


def read_xyz_frames(text: str) -> list[list[list[float]]]:
    """Frames of a (multi-)xyz file, positions in bohr."""
    lines = text.splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        rows = lines[i + 2:i + 2 + n]
        frames.append([[float(x) / C.BOHR_TO_ANGSTROM for x in r.split()[1:4]] for r in rows])
        i += 2 + n
    return frames


def ts(name: str = "ts61", level: str = "svp") -> None:
    method = C.LEVELS[level][1]
    key = f"orca_ts_{name}" + ("" if level == "svp" else f"_{level}")
    rec = C.load(key) or {"program": "orca 6.0.0", "molecule": name, "level": method + " / def2-SVP",
                          "threads": C.THREADS, "before": env_record(), "walls": {}}
    s0 = C.structure(name)
    t_all = time.perf_counter()

    if "optts" not in rec:
        out, wall = run(orca_input(CalcType.transition_state, s0, method + " TightOpt",
                                   geom={"calc_hess": True, "maxiter": 200}))
        rec["optts"] = opt_summary(out, wall)
        print(f"OptTS: {rec['optts']['n_steps']} steps, converged={rec['optts']['converged']}, "
              f"E={rec['optts']['final_energy']}, {wall:.0f} s", flush=True)
        C.save(key, rec)
    ts_key = "optts"
    if not rec["optts"]["converged"]:
        # AFH TS protocol (failure table: "max steps, energy drifting down"): a fresh exact
        # Hessian at the current point and a restart. (The ORCA skill's recalc_hess 5 would cost
        # an analytic Hessian every 5 steps; an unconverged OptTS still ends "TERMINATED NORMALLY".)
        if "optts2" not in rec:
            last = C.with_geometry(s0, rec["optts"]["final_geometry_bohr"])
            out, wall = run(orca_input(CalcType.transition_state, last, method + " TightOpt",
                                       geom={"calc_hess": True, "maxiter": 200}))
            rec["optts2"] = opt_summary(out, wall)
            print(f"OptTS (continued from a fresh Hessian): {rec['optts2']['n_steps']} steps, "
                  f"converged={rec['optts2']['converged']}, E={rec['optts2']['final_energy']}, {wall:.0f} s", flush=True)
            C.save(key, rec)
        ts_key = "optts2"
    rec["ts_from"] = ts_key
    ts_s = C.with_geometry(s0, rec[ts_key]["final_geometry_bohr"])

    hess_text = None
    if "freq_ts" not in rec or "irc" not in rec:
        out, wall = run(orca_input(CalcType.hessian, ts_s, method), collect_files=True)
        rec["freq_ts"] = freq_summary(out, wall)
        hess_text = next((v for k, v in out.results.files.items() if k.endswith(".hess")), None)
        print(f"Freq(TS): n_imag={rec['freq_ts'].get('n_imaginary')}, lowest {rec['freq_ts']['freqs_wavenumber'][:3]}, "
              f"{wall:.0f} s", flush=True)
        C.save(key, rec)

    if "irc" not in rec:
        assert hess_text is not None, "no ORCA .hess file from the TS frequencies"
        text = IRC_INPUT.format(method=method, pal=C.ORCA_PAL, maxcore=C.ORCA_MAXCORE, charge=s0.charge, mult=s0.multiplicity)
        inp = FileInput(program="orca", files={"irc.inp": text, "ts.xyz": C.xyz_text(ts_s, f"{name} ORCA TS"),
                                               "ts.hess": hess_text}, cmdline_args=["irc.inp"])
        out, wall = run(inp, collect_files=True)
        files = out.results.files if out.results is not None else {}
        log = out.logs or ""
        ends = {}
        for tag, fname in (("forward", "irc_IRC_F.xyz"), ("backward", "irc_IRC_B.xyz")):
            if fname in files:
                ends[tag] = read_xyz_frames(files[fname])[-1]
        full = read_xyz_frames(files["irc_IRC_Full_trj.xyz"]) if "irc_IRC_Full_trj.xyz" in files else []
        summary = log[log.rfind("IRC PATH SUMMARY"):][:6000] if "IRC PATH SUMMARY" in log else None
        rec["irc"] = {"success": out.success, "wall_s": wall, "n_frames_full": len(full),
                      "ends_bohr": ends, "path_summary": summary,
                      "n_gradients": len(re.findall(r"ORCA SCF GRADIENT CALCULATION", log)),
                      "traceback": None if out.success else (out.traceback or "")[-3000:]}
        print(f"IRC: success={out.success}, frames={len(full)}, ends={list(ends)}, {wall:.0f} s", flush=True)
        C.save(key, rec)

    for tag in ("forward", "backward"):
        if f"opt_{tag}" in rec:
            continue
        end = C.with_geometry(s0, rec["irc"]["ends_bohr"][tag])
        out, wall = run(orca_input(CalcType.optimization, end, method + " TightOpt", geom={"maxiter": 300}))
        rec[f"opt_{tag}"] = opt_summary(out, wall)
        print(f"Opt({tag}): {rec[f'opt_{tag}']['n_steps']} steps, E={rec[f'opt_{tag}']['final_energy']}, {wall:.0f} s",
              flush=True)
        C.save(key, rec)
        final = C.with_geometry(s0, rec[f"opt_{tag}"]["final_geometry_bohr"])
        out, wall = run(orca_input(CalcType.hessian, final, method))
        rec[f"freq_{tag}"] = freq_summary(out, wall)
        print(f"Freq({tag}): n_imag={rec[f'freq_{tag}'].get('n_imaginary')}, {wall:.0f} s", flush=True)
        C.save(key, rec)

    rec["walls"]["this_invocation_s"] = time.perf_counter() - t_all
    rec["total_wall_s"] = sum(rec[k]["wall_s"] for k in ("optts", "optts2", "freq_ts", "irc", "opt_forward", "freq_forward",
                                                           "opt_backward", "freq_backward") if k in rec)
    rec["after"] = env_record()
    C.save(key, rec)


def single_points() -> None:
    jobs = []
    for name in ("la18", "u1an39"):
        r = C.load(f"orca_min_{name}")
        if r and r["opt"]["final_geometry_bohr"]:
            jobs.append((f"{name}_min", C.with_geometry(C.structure(name), r["opt"]["final_geometry_bohr"])))
    r = C.load("orca_ts_ts61")
    if r:
        s0 = C.structure("ts61")
        for tag, k in (("ts", r.get("ts_from", "optts")), ("forward", "opt_forward"), ("backward", "opt_backward")):
            if k in r and r[k]["final_geometry_bohr"]:
                jobs.append((f"ts61_{tag}", C.with_geometry(s0, r[k]["final_geometry_bohr"])))
    for label, s in jobs:
        key = f"orca_sp_{label}"
        if C.load(key):
            continue
        out, wall = run(orca_input(CalcType.energy, s, C.TZVP_ORCA, basis="def2-TZVP"))
        rec = {"program": "orca 6.0.0", "label": label, "level": C.TZVP_ORCA + " / def2-TZVP", "threads": C.THREADS,
               "success": out.success, "energy": getattr(out.results, "energy", None), "wall_s": wall,
               "geometry_bohr": s.geometry.tolist(), "env": env_record()}
        print(f"SP {label}: E={rec['energy']}, {wall:.0f} s", flush=True)
        C.save(key, rec)


def single_points_pcm() -> None:
    """wB97X-D3BJ/def2-TZVP CPCM(THF) on ORCA's minima (linus: linus_workflows.py sp_pcm)."""
    for name in ("la18", "u1an39"):
        r = C.load(f"orca_min_{name}")
        key = f"orca_sppcm_{name}_min"
        if not r or C.load(key):
            continue
        s = C.with_geometry(C.structure(name), r["opt"]["final_geometry_bohr"])
        out, wall = run(orca_input(CalcType.energy, s, f"{C.TZVP_ORCA} {C.PCM_ORCA}", basis="def2-TZVP"))
        log = out.logs or ""
        cpcm = re.findall(r"CPCM Dielectric\s*:\s*(-?\d+\.\d+)", log)
        rec = {"program": "orca 6.0.0", "label": f"{name}_min", "level": f"{C.TZVP_ORCA} {C.PCM_ORCA} / def2-TZVP",
               "threads": C.ORCA_PAL, "success": out.success, "energy": getattr(out.results, "energy", None),
               "cpcm_dielectric_eh": float(cpcm[-1]) if cpcm else None, "wall_s": wall,
               "geometry_bohr": s.geometry.tolist(), "env": env_record()}
        print(f"SP CPCM(THF) {name}: E={rec['energy']}, CPCM dielectric {rec['cpcm_dielectric_eh']}, {wall:.0f} s",
              flush=True)
        C.save(key, rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["minima", "ts", "sp", "sp_pcm"])
    ap.add_argument("--mols", nargs="*", default=["la18", "u1an39"])
    ap.add_argument("--ts-mol", default="ts61")
    ap.add_argument("--level", default="svp", choices=sorted(C.LEVELS))
    a = ap.parse_args()
    if a.stage == "minima":
        minima(a.mols)
    elif a.stage == "ts":
        ts(a.ts_mol, a.level)
    elif a.stage == "sp_pcm":
        single_points_pcm()
    else:
        single_points()


if __name__ == "__main__":
    main()
