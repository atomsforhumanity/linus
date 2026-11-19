# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""linus side of the end-to-end workflows, through qccompute (geomeTRIC, Sella) and qcopt.

    systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 -p OOMPolicy=continue \
        taskset -c 24-31 <venv>/bin/python bench/workflows/linus_workflows.py <stage> [options]

Level B3LYP-D3(BJ)/def2-SVP (`b3lyp-d3bj`, VWN-RPA B3LYP as ORCA's B3LYP/G), RI-JK with
def2-universal-jkfit (Hessians: linus's "auto" aux), grid "auto" (PySCF level 3 for energies and
gradients, linus level 3 for Hessians), 8 threads. Every step of an optimizer starts from the
previous step's orbitals (qccompute ``propagate_wfn=True``; qcopt through
`linus.evaluator.QcoptEvaluator`). Stages (each writes bench/results/2026-11-07-workflows/
linus_<stage>_<...>.json and is skipped when that file exists):

- min --mol M --opt geometric|sella|qcopt: minimization from the perturbed start, ORCA's TightOpt
  criteria (`common`).
- freq --mol M: Hessian, frequencies and thermochemistry (RRHO and quasi-RRHO) at the geomeTRIC
  minimum.
- ts: Sella TS search on ts61 from linus's analytic Hessian (fmax 0.001 eV/Å), frequencies at
  the TS, Sella IRC both ways, geomeTRIC minimization and frequencies of both ends.
- ts_qcopt: the same with qcopt (find_ts with the frequency check, IRC with minimized ends from
  the kept TS Hessian) and `linus.evaluator.QcoptEvaluator`, then frequencies of both ends.
- sp: wB97X-D3BJ/def2-TZVP single points, RI-JK and RIJ+COSX, on linus's optimized structures.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # numpy inside the optimizers: small matrices
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402
from qccompute import compute  # noqa: E402
from qcdata import CalcType, ProgramInput, ProgramSpec  # noqa: E402

import linus  # noqa: E402
from linus import vib  # noqa: E402

GRAD_KW = {"jk": "df", "threads": C.THREADS}
HESS_KW = {"jk": "df", "threads": C.THREADS}
THERMO = {"temperature": 298.15, "pressure": 101325.0}
# the level of the optimizations, Hessians and TS stages (--level, common.LEVELS)
MODEL: dict = dict(C.SVP_LINUS)


def spec(calctype: CalcType, model=None, **kw) -> ProgramSpec:
    return ProgramSpec(program="linus", calctype=calctype, model=model or MODEL,
                       keywords=kw or (HESS_KW if calctype == CalcType.hessian else GRAD_KW))


def env_record() -> dict:
    return {"affinity": C.affinity(), "ccd_busy_cores": C.ccd_load(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def frame_rows(frames) -> list[dict]:
    rows = []
    for f in frames:
        res = f.results
        ex = (res.extras or {}) if res is not None else {}
        rows.append({"energy": getattr(res, "energy", None), "scf_iterations": ex.get("scf_iterations"),
                     "scf_guess": ex.get("scf_guess"), "linus_total_s": (ex.get("timings") or {}).get("total"),
                     "wall_s": f.execution.wall_time if getattr(f, "execution", None) else None,
                     "hessian": getattr(res, "hessian", None) is not None, "success": f.success})
    return rows


def series_summary(out, wall: float) -> dict:
    res = out.results
    frames = list(getattr(res, "frames", None) or [])
    rows = frame_rows(frames)
    linus_s = sum(r["linus_total_s"] or 0.0 for r in rows)
    child_s = sum(r["wall_s"] or 0.0 for r in rows)
    d = {
        "success": out.success, "n_steps": len(frames), "wall_s": wall,
        "final_energy": rows[-1]["energy"] if rows else None,
        "final_geometry_bohr": frames[-1].structure.geometry.tolist() if frames else None,
        "linus_s": linus_s, "child_wall_s": child_s,
        "child_overhead_s": child_s - linus_s, "driver_overhead_s": wall - child_s,
        "scf_iterations": [r["scf_iterations"] for r in rows], "scf_guess": [r["scf_guess"] for r in rows],
        "steps": rows, "traceback": None if out.success else (out.traceback or "")[-3000:],
    }
    return d


def thermo_both(numbers, positions, hessian, energy) -> dict:
    """RRHO and quasi-RRHO (Grimme entropy, 100 cm^-1) thermochemistry of one Hessian."""
    va = vib.harmonic_analysis(numbers, positions, hessian)
    out = {"frequencies": va.vibrational_frequencies.tolist(), "n_imaginary": int(va.n_imaginary)}
    for tag, opts in (("rrho", {}), ("qrrho", {"thermo": "qrrho"})):
        th = vib.thermochemistry(va, energy, multiplicity=1, **THERMO, **opts)[0]  # one temperature
        out[tag] = {"zpe": th.zero_point_energy, "gibbs": th.gibbs_free_energy, "enthalpy": th.enthalpy,
                    "gibbs_correction": th.gibbs_correction}
    return out


def hessian_at(s) -> tuple[dict, object]:
    inp = ProgramInput(program="linus", calctype=CalcType.hessian, structure=s, model=MODEL, keywords=HESS_KW)
    t = time.perf_counter()
    out = compute(inp, raise_exc=False)
    wall = time.perf_counter() - t
    res = out.results
    d = {"success": out.success, "wall_s": wall, "energy": getattr(res, "energy", None),
         "traceback": None if out.success else (out.traceback or "")[-3000:]}
    if out.success:
        ex = res.extras
        d.update({"freqs_wavenumber": list(map(float, res.freqs_wavenumber)),
                  "n_imaginary": int(sum(1 for f in res.freqs_wavenumber if f < -20.0)),
                  "scf_iterations": ex.get("scf_iterations"), "timings": ex.get("timings"),
                  "hessian": np.asarray(res.hessian).tolist()})
        d["thermo"] = thermo_both([str(x) for x in s.symbols], s.geometry, np.asarray(res.hessian), res.energy)
    return d, out


# -- minimizations --------------------------------------------------------------------------


def minimize(s, optimizer: str) -> dict:
    t = time.perf_counter()
    if optimizer == "geometric":
        inp = ProgramInput(program="geometric", calctype=CalcType.optimization, structure=s,
                           keywords={"maxiter": 300, **C.GEOMETRIC_TIGHTOPT}, subprograms=[spec(CalcType.gradient)])
        out = compute(inp, raise_exc=False, propagate_wfn=True)
    elif optimizer == "sella":
        inp = ProgramInput(program="sella", calctype=CalcType.optimization, structure=s,
                           keywords={"fmax": C.SELLA_MIN_FMAX, "steps": 500}, subprograms=[spec(CalcType.gradient)])
        out = compute(inp, raise_exc=False, propagate_wfn=True)
    elif optimizer == "qcopt":
        import qcopt

        from linus.evaluator import QcoptEvaluator

        g = {"program": "linus", "model": MODEL, "keywords": GRAD_KW}
        out = qcopt.minimize(s, g, evaluator=QcoptEvaluator(), convergence={"gmax": 1e-4, "grms": 3e-5},
                             maxiter=300)
    else:
        raise ValueError(optimizer)
    return series_summary(out, time.perf_counter() - t)


def stage_min(name: str, optimizer: str) -> None:
    key = f"linus_min_{name}_{optimizer}"
    if C.load(key):
        print(f"{key}: done")
        return
    rec = {"program": f"linus {linus.__version__}", "molecule": name, "optimizer": optimizer,
           "level": "b3lyp-d3bj/def2-svp RI-JK", "threads": C.THREADS, "before": env_record()}
    rec["opt"] = minimize(C.structure(name, perturbed=True), optimizer)
    o = rec["opt"]
    print(f"{name} {optimizer}: {o['n_steps']} steps, E={o['final_energy']}, {o['wall_s']:.1f} s "
          f"(linus {o['linus_s']:.1f}, child overhead {o['child_overhead_s']:.1f}, driver {o['driver_overhead_s']:.1f}), "
          f"iterations {o['scf_iterations']}", flush=True)
    rec["after"] = env_record()
    C.save(key, rec)


def stage_freq(name: str) -> None:
    key = f"linus_freq_{name}"
    if C.load(key):
        return
    m = C.load(f"linus_min_{name}_geometric")
    s = C.with_geometry(C.structure(name), m["opt"]["final_geometry_bohr"])
    rec = {"program": f"linus {linus.__version__}", "molecule": name, "before": env_record()}
    rec["freq"], _ = hessian_at(s)
    f = rec["freq"]
    print(f"{name} freq: n_imag={f.get('n_imaginary')}, lowest {f.get('freqs_wavenumber', [])[:4]}, {f['wall_s']:.1f} s",
          flush=True)
    rec["after"] = env_record()
    C.save(key, rec)


# -- transition state -----------------------------------------------------------------------


def stage_ts(name: str = "ts61", level: str = "svp") -> None:
    """TS workflow of `name` at `level` (common.LEVELS); results in linus_ts_<name>[_<level>].json.
    The IRC of small systems (<= 10 atoms) starts from the analytic Hessian as well."""
    key = f"linus_ts_{name}" + ("" if level == "svp" else f"_{level}")
    rec = C.load(key) or {"program": f"linus {linus.__version__}", "molecule": name, "level": MODEL,
                          "before": env_record()}
    s0 = C.structure(name)

    if "tsopt" not in rec:
        t = time.perf_counter()
        inp = ProgramInput(program="sella", calctype=CalcType.transition_state, structure=s0,
                           keywords={"fmax": C.SELLA_TS_FMAX, "steps": 1000},
                           subprograms=[spec(CalcType.gradient), spec(CalcType.hessian)])
        out = compute(inp, raise_exc=False, propagate_wfn=True)
        rec["tsopt"] = series_summary(out, time.perf_counter() - t)
        o = rec["tsopt"]
        print(f"Sella TS: success={o['success']}, {o['n_steps']} steps, E={o['final_energy']}, {o['wall_s']:.0f} s",
              flush=True)
        C.save(key, rec)
    ts = C.with_geometry(s0, rec["tsopt"]["final_geometry_bohr"])

    if "freq_ts" not in rec:
        rec["freq_ts"], _ = hessian_at(ts)
        f = rec["freq_ts"]
        print(f"freq(TS): n_imag={f.get('n_imaginary')}, lowest {f.get('freqs_wavenumber', [])[:3]}, {f['wall_s']:.0f} s",
              flush=True)
        C.save(key, rec)

    if "irc" not in rec:
        t = time.perf_counter()
        small = len(s0.symbols) <= 10
        inp = ProgramInput(program="sella", calctype=CalcType.irc, structure=ts,
                           keywords={"fmax": 0.1, "steps": 300, "keep_going": True},
                           subprograms=[spec(CalcType.gradient)] + ([spec(CalcType.hessian)] if small else []))
        out = compute(inp, raise_exc=False, propagate_wfn=True)
        d = series_summary(out, time.perf_counter() - t)
        res = out.results
        frames = list(getattr(res, "frames", None) or [])
        d["ts_index"] = getattr(res, "ts_index", None)
        d["ends_bohr"] = {"reverse": frames[0].structure.geometry.tolist(),
                          "forward": frames[-1].structure.geometry.tolist()} if frames else {}
        rec["irc"] = d
        print(f"IRC: success={d['success']}, {d['n_steps']} frames, ts_index={d['ts_index']}, {d['wall_s']:.0f} s",
              flush=True)
        C.save(key, rec)

    for tag in ("reverse", "forward"):
        if f"opt_{tag}" not in rec:
            end = C.with_geometry(s0, rec["irc"]["ends_bohr"][tag])
            rec[f"opt_{tag}"] = minimize(end, "geometric")
            o = rec[f"opt_{tag}"]
            print(f"opt({tag}): {o['n_steps']} steps, E={o['final_energy']}, {o['wall_s']:.0f} s", flush=True)
            C.save(key, rec)
        if f"freq_{tag}" not in rec:
            m = C.with_geometry(s0, rec[f"opt_{tag}"]["final_geometry_bohr"])
            rec[f"freq_{tag}"], _ = hessian_at(m)
            f = rec[f"freq_{tag}"]
            print(f"freq({tag}): n_imag={f.get('n_imaginary')}, {f['wall_s']:.0f} s", flush=True)
            C.save(key, rec)

    rec["total_wall_s"] = sum(rec[k]["wall_s"] for k in ("tsopt", "freq_ts", "irc", "opt_reverse", "freq_reverse",
                                                       "opt_forward", "freq_forward") if k in rec)
    rec["after"] = env_record()
    C.save(key, rec)


def stage_ts_qcopt() -> None:
    """The same TS workflow with qcopt and `linus.evaluator.QcoptEvaluator`: find_ts from the
    analytic Hessian with the frequency check (qcopt's default convergence, 0.001 eV/Å on the
    largest atomic gradient, as the AFH TS protocol), then the IRC from the result, whose
    starting Hessian is the evaluator's kept TS Hessian, with both ends minimized; then
    frequencies of both ends."""
    import qcopt

    from linus.evaluator import QcoptEvaluator

    key = "linus_tsqcopt_ts61"
    rec = C.load(key) or {"program": f"linus {linus.__version__} + qcopt", "molecule": "ts61", "before": env_record()}
    s0 = C.structure("ts61")
    g = {"program": "linus", "model": MODEL, "keywords": GRAD_KW}
    ev = QcoptEvaluator()
    if "tsopt" not in rec:
        t = time.perf_counter()
        out = qcopt.find_ts(s0, g, hessian=True, frequencies=True, maxiter=500, evaluator=ev)
        d = series_summary(out, time.perf_counter() - t)
        d["extras"] = {k: v for k, v in (out.results.extras or {}).items() if isinstance(v, (int, float, str, list))}
        rec["tsopt"] = d
        print(f"qcopt TS: success={d['success']}, {d['n_steps']} frames, E={d['final_energy']}, {d['wall_s']:.0f} s, "
              f"extras {d['extras']}", flush=True)
        C.save(key, rec)
    ts = C.with_geometry(s0, rec["tsopt"]["final_geometry_bohr"])
    if "irc" not in rec:
        t = time.perf_counter()
        out = qcopt.irc(ts, g, hessian=True, minimize_ends=True, evaluator=ev)
        d = series_summary(out, time.perf_counter() - t)
        frames = list(getattr(out.results, "frames", None) or [])
        d["ends_bohr"] = {"reverse": frames[0].structure.geometry.tolist(),
                          "forward": frames[-1].structure.geometry.tolist()} if frames else {}
        d["end_energies"] = {"reverse": frames[0].results.energy, "forward": frames[-1].results.energy} if frames else {}
        d["extras"] = {k: v for k, v in (out.results.extras or {}).items() if isinstance(v, (int, float, str, list))}
        rec["irc"] = d
        print(f"qcopt IRC: success={d['success']}, {d['n_steps']} frames, {d['wall_s']:.0f} s", flush=True)
        C.save(key, rec)
    for tag in ("reverse", "forward"):
        if f"freq_{tag}" not in rec and rec["irc"].get("ends_bohr"):
            rec[f"freq_{tag}"], _ = hessian_at(C.with_geometry(s0, rec["irc"]["ends_bohr"][tag]))
            print(f"freq({tag}): n_imag={rec[f'freq_{tag}'].get('n_imaginary')}, {rec[f'freq_{tag}']['wall_s']:.0f} s",
                  flush=True)
            C.save(key, rec)
    rec["total_wall_s"] = sum(rec[k]["wall_s"] for k in ("tsopt", "irc", "freq_reverse", "freq_forward") if k in rec)
    rec["after"] = env_record()
    C.save(key, rec)


# -- single points --------------------------------------------------------------------------


def sp_jobs(source: str) -> list[tuple[str, object]]:
    jobs = []
    if source == "linus":
        for name in ("la18", "u1an39"):
            r = C.load(f"linus_min_{name}_geometric")
            if r:
                jobs.append((f"{name}_min", C.with_geometry(C.structure(name), r["opt"]["final_geometry_bohr"])))
        r = C.load("linus_ts_ts61")
        if r:
            for tag, k in (("ts", "tsopt"), ("reverse", "opt_reverse"), ("forward", "opt_forward")):
                if k in r:
                    jobs.append((f"ts61_{tag}", C.with_geometry(C.structure("ts61"), r[k]["final_geometry_bohr"])))
    else:  # ORCA's structures, for a same-geometry energy comparison
        for name in ("la18", "u1an39"):
            r = C.load(f"orca_min_{name}")
            if r:
                jobs.append((f"{name}_min", C.with_geometry(C.structure(name), r["opt"]["final_geometry_bohr"])))
        r = C.load("orca_ts_ts61")
        if r:
            for tag, k in (("ts", r.get("ts_from", "optts")), ("forward", "opt_forward"), ("backward", "opt_backward")):
                if k in r:
                    jobs.append((f"ts61_{tag}", C.with_geometry(C.structure("ts61"), r[k]["final_geometry_bohr"])))
    return jobs


def stage_sp(source: str) -> None:
    for label, s in sp_jobs(source):
        for jk in ("df", "rij_snk"):
            key = f"linus_sp_{source}geom_{label}_{jk}"
            if C.load(key):
                continue
            inp = ProgramInput(program="linus", calctype=CalcType.energy, structure=s, model=C.TZVP_LINUS,
                               keywords={"jk": jk, "threads": C.THREADS})
            t = time.perf_counter()
            out = compute(inp, raise_exc=False)
            wall = time.perf_counter() - t
            res = out.results
            rec = {"program": f"linus {linus.__version__}", "label": label, "geometry_source": source, "jk": jk,
                   "level": "wb97x-d3bj/def2-tzvp", "threads": C.THREADS, "success": out.success,
                   "energy": getattr(res, "energy", None), "wall_s": wall,
                   "scf_iterations": (res.extras or {}).get("scf_iterations") if out.success else None,
                   "timings": (res.extras or {}).get("timings") if out.success else None,
                   "options": (res.extras or {}).get("options") if out.success else None,
                   "traceback": None if out.success else (out.traceback or "")[-3000:], "env": env_record()}
            print(f"SP {source} {label} {jk}: E={rec['energy']}, {wall:.1f} s", flush=True)
            C.save(key, rec)


def stage_sp_pcm() -> None:
    """wB97X-D3BJ/def2-TZVP in THF (C-PCM), RI-JK and RIJ+COSX, on ORCA's minima (as ORCA's
    CPCM(THF) single points, orca_workflows.py sp_pcm)."""
    for label, s in sp_jobs("orca"):
        if not label.endswith("_min"):
            continue
        for jk in ("df", "rij_snk"):
            key = f"linus_sppcm_orcageom_{label}_{jk}"
            if C.load(key):
                continue
            inp = ProgramInput(program="linus", calctype=CalcType.energy, structure=s, model=C.TZVP_LINUS,
                               keywords={"jk": jk, "threads": C.THREADS, **C.PCM_LINUS})
            t = time.perf_counter()
            out = compute(inp, raise_exc=False)
            wall = time.perf_counter() - t
            ex = (out.results.extras or {}) if out.success else {}
            rec = {"program": f"linus {linus.__version__}", "label": label, "jk": jk, "solvent": C.PCM_LINUS,
                   "level": "wb97x-d3bj/def2-tzvp", "success": out.success,
                   "energy": getattr(out.results, "energy", None), "wall_s": wall,
                   "energy_components": ex.get("energy_components"), "solvation": ex.get("solvation"),
                   "scf_iterations": ex.get("scf_iterations"), "timings": ex.get("timings"),
                   "traceback": None if out.success else (out.traceback or "")[-3000:], "env": env_record()}
            print(f"SP C-PCM(THF) {label} {jk}: E={rec['energy']}, {wall:.1f} s", flush=True)
            C.save(key, rec)


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["min", "freq", "ts", "ts_qcopt", "sp", "sp_pcm"])
    ap.add_argument("--mol", nargs="*", default=["la18"])
    ap.add_argument("--opt", nargs="*", default=["geometric"])
    ap.add_argument("--source", default="linus", choices=["linus", "orca"])
    ap.add_argument("--ts-mol", default="ts61")
    ap.add_argument("--level", default="svp", choices=sorted(C.LEVELS))
    a = ap.parse_args()
    MODEL = dict(C.LEVELS[a.level][0])
    if a.stage == "min":
        for m in a.mol:
            for o in a.opt:
                stage_min(m, o)
    elif a.stage == "freq":
        for m in a.mol:
            stage_freq(m)
    elif a.stage == "ts":
        stage_ts(a.ts_mol, a.level)
    elif a.stage == "ts_qcopt":
        stage_ts_qcopt()
    elif a.stage == "sp_pcm":
        stage_sp_pcm()
    else:
        stage_sp(a.source)


if __name__ == "__main__":
    main()
