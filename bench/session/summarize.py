#!/usr/bin/env python3
"""Tables of the session benchmark from results/raw/*.json: results/summary.json and results/tables.md.

Per (system, platform, mode): each step's wall time is the median over the repeats; the first step, the median of
the remaining steps and the total (the sum of the per-step medians) come from those, with the per-repeat totals
beside them. The per-step breakdown is the median, over steps 2..N, of each step's median over the repeats.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics

import common as C

med = statistics.median
# the modes whose steps start from the last step's orbitals (the others start from SAD)
WARM = ("D", "W")


def load():
    runs = collections.defaultdict(list)
    for f in sorted((C.RESULTS / "raw").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("placeholder"):  # a skipped job (run_all.py keeps existing files)
            continue
        d["repeat"] = int(f.stem.rsplit("_r", 1)[1])
        runs[(d["system"], d["platform"], d["mode"])].append(d)
    return runs


def stage(step, plat) -> dict:
    b = C.breakdown(step["timings"], plat)
    b["outside"] = step["wall"] - b["total"]
    b["wall"] = step["wall"]
    return b


def per_step(reps, key):
    n = min(r["nsteps"] for r in reps)
    return [med(key(r["steps"][i]) for r in reps) for i in range(n)]


KEYS = ("wall", "total", "outside", "scf_setup", "scf_iter", "gradient", "setup", "grad_setup", "ints_pairs", "df",
        "grid_xc", "guess", "dev_loop", "probes") + tuple("t_" + k for k in C.SETUP_KEYS)


def stats(reps, plat) -> dict:
    n = min(r["nsteps"] for r in reps)
    wall = per_step(reps, lambda s: s["wall"])
    its = per_step(reps, lambda s: s["iterations"])
    out = dict(nsteps=n, repeats=len(reps), first=wall[0], rest_median=med(wall[1:]) if n > 1 else None,
               total=sum(wall), totals_per_repeat=[sum(s["wall"] for s in r["steps"][:n]) for r in reps],
               iterations_median=med(its), iterations_median_rest=med(its[1:]) if n > 1 else None,
               iterations_total=sum(its))
    bd = {}
    for k in KEYS:
        def get(s, k=k):
            return stage(s, plat).get(k, 0.0)
        series = per_step(reps, get)
        bd[k] = dict(first=series[0], rest=med(series[1:]) if n > 1 else None)
    out["breakdown"] = bd
    # every setup-like timer (sub-timers included), median over steps 2..N
    names = sorted({k for r in reps for st in r["steps"] for k in st["timings"]
                    if k in C.SETUP_KEYS or "setup" in k or k in ("grid", "pairs", "one_electron", "guess")})
    if n > 1:
        out["setup_timers_rest"] = {k: med(per_step(reps, lambda s, k=k: s["timings"].get(k, 0.0))[1:]) for k in names}
        out["prepare_rest"] = med(per_step(reps, lambda s: s.get("prepare", 0.0))[1:])
        out["drop_rest"] = med(per_step(reps, lambda s: s.get("drop", 0.0))[1:])
    rus = per_step(reps, lambda s: s["rusage"]["minflt"])
    out["minflt_rest"] = med(rus[1:]) if n > 1 else None
    out["stime_rest"] = med(per_step(reps, lambda s: s["rusage"]["stime"])[1:]) if n > 1 else None
    out["utime_rest"] = med(per_step(reps, lambda s: s["rusage"]["utime"])[1:]) if n > 1 else None
    probes = [s.get("closed_shell_probe") or {} for s in reps[0]["steps"]]
    out["closed_shell_probe_ran"] = sum(1 for p in probes if p.get("products"))
    sp = [s.get("stability_probe") for s in reps[0]["steps"]]
    out["stability_probe_steps"] = sum(1 for p in sp if p)
    return out


def energies(runs) -> dict:
    """Max |E| and max |gradient| differences between modes (and repeats) per system and platform, and between
    platforms (mode D, repeat 0)."""
    out = {}
    by = collections.defaultdict(dict)
    for (system, plat, mode), reps in runs.items():
        for r in reps:
            by[(system, plat)][(mode, r["repeat"])] = r
    for (system, plat), rs in by.items():
        keys = list(rs)
        n = min(r["nsteps"] for r in rs.values())
        de = dg = 0.0
        de_sad = dg_sad = 0.0
        for i in range(n):
            es = {k: rs[k]["steps"][i]["energy"] for k in keys}
            gs = {k: rs[k]["steps"][i]["gradient"] for k in keys}
            de = max(de, max(es.values()) - min(es.values()))
            sad = [v for k, v in es.items() if k[0] not in WARM]
            if sad:
                de_sad = max(de_sad, max(sad) - min(sad))
            for a in keys:
                for b in keys:
                    if a < b:
                        d = max(abs(x - y) for ra, rb in zip(gs[a], gs[b]) for x, y in zip(ra, rb))
                        dg = max(dg, d)
                        if a[0] not in WARM and b[0] not in WARM:
                            dg_sad = max(dg_sad, d)
        out[f"{system}/{plat}"] = dict(max_dE=de, max_dE_sad_modes=de_sad, max_dgrad=dg, max_dgrad_sad_modes=dg_sad,
                                          runs=len(keys))
    for system in {k[0] for k in by}:
        if (system, "cpu") in by and (system, "gpu") in by:
            a = by[(system, "cpu")].get(("D", 0))
            b = by[(system, "gpu")].get(("D", 0))
            if a and b:
                n = min(a["nsteps"], b["nsteps"])
                out[f"{system}/cpu-vs-gpu"] = dict(max_dE=max(abs(a["steps"][i]["energy"] - b["steps"][i]["energy"])
                                                             for i in range(n)))
    return out


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def traj_table() -> str:
    L = []
    for system in C.SYSTEMS:
        f = C.traj_path(system)
        if not f.exists():
            continue
        t = json.loads(f.read_text())
        L.append(f"| {system} | B3LYP/{t['basis']} | {len(t['symbols'])} | {t['charge']}, {t['multiplicity']} | "
                 f"{t['nsteps']} | {t['stop']} (final gradient RMS {t['steps'][-1]['grms']:.1e}) |")
    return "\n".join(L)


def tables(summary) -> str:
    L = []
    L.append("| system | platform | mode | steps | first step (s) | median later step (s) | total (s) | totals per repeat (s) | median SCF it. | total / D |")
    L.append("|---|---|---|---:|---:|---:|---:|---|---:|---:|")
    for key, v in summary["runs"].items():
        system, plat, mode = key.split("/")
        d = summary["runs"].get(f"{system}/{plat}/D")
        ratio = v["total"] / d["total"] if d else None
        L.append(f"| {system} | {plat} | {mode} | {v['nsteps']} | {fmt(v['first'])} | {fmt(v['rest_median'], 3)} | "
                 f"{fmt(v['total'], 1)} | {', '.join(f'{t:.1f}' for t in v['totals_per_repeat'])} | "
                 f"{fmt(v['iterations_median'], 1)} | {fmt(ratio)} |")
    L.append("")
    L.append("Per later step (median of steps 2..N), seconds: wall = outside linus's timers + linus total; "
             "total = SCF setup + SCF loop + gradient.")
    L.append("")
    L.append("| system | platform | mode | wall | outside linus | SCF setup | SCF iterations | gradient | setup (SCF + gradient) | of it: ints + pairs | DF tensor + metric | grid + XC setup | SAD guess | device loop setup | probes |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, v in summary["runs"].items():
        system, plat, mode = key.split("/")
        b = {k: x["rest"] for k, x in v["breakdown"].items()}
        if b["wall"] is None:
            continue
        L.append(f"| {system} | {plat} | {mode} | {b['wall']:.3f} | {b['outside']:.3f} | {b['scf_setup']:.3f} | "
                 f"{b['scf_iter']:.3f} | {b['gradient']:.3f} | {b['setup']:.3f} | {b['ints_pairs']:.3f} | "
                 f"{b['df']:.3f} | {b['grid_xc']:.3f} | {b['guess']:.3f} | {b['dev_loop']:.3f} | {b['probes']:.3f} |")
    L.append("")
    L.append("| system / platform | runs | max ΔE, all modes (Eh) | max ΔE, SAD modes A-C (Eh) | max Δgradient, all modes (Eh/bohr) | max Δgradient, A-C (Eh/bohr) |")  # noqa: E501
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, v in summary["energies"].items():
        if "max_dgrad" in v:
            L.append(f"| {k} | {v['runs']} | {v['max_dE']:.1e} | {v['max_dE_sad_modes']:.1e} | {v['max_dgrad']:.1e} | "
                     f"{v['max_dgrad_sad_modes']:.1e} |")
        else:
            L.append(f"| {k} (mode D, repeat 0) | 2 | {v['max_dE']:.1e} | | | |")
    return "\n".join(L) + "\n"


def ratios(summary) -> str:
    """Each mode's total against D's, and the per-step differences between neighbouring modes (those run)."""
    pairs = (("A", "B"), ("B", "C"), ("C", "D"), ("B", "D"), ("W", "D"), ("A", "W"))
    L = ["| system | platform | A / D | B / D | C / D | W / D | " + " | ".join(f"({a} - {b}) per step (s)" for a, b in pairs)
         + " |", "|---|---|---:|---:|---:|---:|" + "---:|" * len(pairs)]
    seen = []
    for key in summary["runs"]:
        system, plat, _ = key.split("/")
        if (system, plat) in seen:
            continue
        seen.append((system, plat))
        r = {m: summary["runs"].get(f"{system}/{plat}/{m}") for m in C.MODES}
        if not r["D"]:
            continue
        n, d = r["D"]["nsteps"], r["D"]["total"]
        ratio = [fmt(r[m]["total"] / d) if r[m] else "–" for m in ("A", "B", "C", "W")]
        diff = [f"{(r[a]['total'] - r[b]['total']) / n:.3f}" if r[a] and r[b] else "–" for a, b in pairs]
        L.append(f"| {system} | {plat} | " + " | ".join(ratio + diff) + " |")
    return "\n".join(L) + "\n"


def deeper(summary) -> str:
    """Mode D's later steps: the setup that remains per geometry, by kind (medians of steps 2..N, seconds)."""
    L = ["| system | platform | step wall | SCF iterations + gradient compute | setup per step | of it: ints + pairs | "
         "DF tensor + metric | grid + XC setup | device loop setup | minor faults per step | kernel CPU time per step |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, v in summary["runs"].items():
        system, plat, mode = key.split("/")
        if mode != "D" or v["nsteps"] < 2:
            continue
        b = {k: x["rest"] for k, x in v["breakdown"].items()}
        L.append(f"| {system} | {plat} | {b['wall']:.3f} | {b['wall'] - b['setup'] - b['outside']:.3f} | "
                 f"{b['setup']:.3f} ({100 * b['setup'] / b['wall']:.0f} %) | {b['ints_pairs']:.3f} | {b['df']:.3f} | "
                 f"{b['grid_xc']:.3f} | {b['dev_loop']:.3f} | {v['minflt_rest']:.0f} | {v['stime_rest']:.3f} |")
    return "\n".join(L) + "\n"


NSYS_KINDS = {  # CUDA API calls by what they do (seconds inside the call; synchronizing calls include the waits)
    "synchronization (waits for the device)": ("cuStreamSynchronize", "cuEventSynchronize", "cuCtxSynchronize",
                                               "cudaDeviceSynchronize", "cudaStreamSynchronize", "cuStreamWaitEvent",
                                               "cuEventQuery"),
    "copies and memsets (D2H copies include waits)": ("cuMemcpy", "cuMemset", "cudaMemcpy", "cudaMemset"),
    "launches": ("cuLaunch", "cudaLaunch"),
    "device memory alloc/free (pool, runtime)": ("cuMemAllocAsync", "cuMemFreeAsync", "cuMemAlloc", "cuMemFree",
                                                 "cuMemPool", "cudaMalloc", "cudaFree"),
    "page-locked host memory (register/alloc/free)": ("cuMemHostAlloc", "cuMemFreeHost", "cuMemHostRegister",
                                                      "cuMemHostUnregister", "cuMemAllocHost", "cudaHostAlloc",
                                                      "cudaFreeHost", "cudaMallocHost"),
    "modules and kernels (load/unload/get)": ("cuModule", "cuLibrary", "cuKernelGet", "cuFuncSetAttribute",
                                              "cuFuncGetAttribute"),
    "context, events, streams (create/destroy/set)": ("cuCtx", "cuDevicePrimaryCtx", "cuStreamCreate",
                                                      "cuStreamDestroy", "cuEventCreate", "cuEventDestroy", "cuInit",
                                                      "cudaEventCreate", "cudaEventDestroy", "cudaStreamCreate"),
}


def nsys_table() -> str:
    """Per-step CUDA API seconds by kind, from results/nsys/<system>_gpu_<mode>_cuda_api.csv."""
    import csv
    import io

    d = C.RESULTS / "nsys"
    if not d.exists():
        return ""
    L = ["| replay under nsys | steps | " + " | ".join(NSYS_KINDS) + " | other |",
         "|---|---:|" + "---:|" * (len(NSYS_KINDS) + 1)]
    for f in sorted(d.glob("*_cuda_api.csv")):
        stem = f.name.replace("_cuda_api.csv", "")
        rp = d / f"{stem}_replay.json"
        if not rp.exists():
            continue
        n = json.loads(rp.read_text())["nsteps"]
        text = f.read_text()
        start = text.find("Time (%)")
        if start < 0:
            continue
        rows = list(csv.DictReader(io.StringIO(text[start:])))
        tot = dict.fromkeys(NSYS_KINDS, 0.0)
        other = 0.0
        for r in rows:
            name, ns = r["Name"], float(r["Total Time (ns)"])
            kind = next((k for k, pre in NSYS_KINDS.items() if name.startswith(pre)), None)
            if kind:
                tot[kind] += ns * 1e-9
            else:
                other += ns * 1e-9
        L.append(f"| {stem} | {n} | " + " | ".join(f"{v / n:.4f}" for v in tot.values()) + f" | {other / n:.4f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    runs = load()
    order = {s: i for i, s in enumerate(C.SYSTEMS)}
    keys = sorted(runs, key=lambda k: (order[k[0]], k[1], k[2]))
    summary = dict(runs={"/".join(k): stats(runs[k], k[1]) for k in keys}, energies=energies(runs))
    C.dump(C.RESULTS / "summary.json", summary)
    text = traj_table() + "\n\n" + tables(summary) + "\n" + ratios(summary) + "\n" + deeper(summary) + "\n" \
        + nsys_table()
    (C.RESULTS / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
