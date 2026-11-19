"""Tables for the linus vs ORCA / PySCF / Psi4 finding from bench/run_bench.py results.

  python bench/summarize_bench.py TAG [TAG2 ...] [--out bench/results/TAG] [--layouts pinned,unpinned]

Reads bench/results/TAG/<layout>/*.json, the accuracy references in bench/results/TAG/refs/
and blas_validation.json; writes summary.md and summary.html (snippets to paste) next to them
and prints the Markdown. Several tags are merged (later tags win for the same case/entry).

Per layout and calculation type:
- headline: per molecule and level, linus's time and every program's time ratio to linus
  (program / linus, > 1 means linus is faster), PySCF and Psi4 "as installed" and with the
  fair BLAS (conda OpenBLAS 0.3.34 for all their BLAS calls);
- detail: every program variant and approach with median (min-max) time, iterations, final
  orbital gradient as each program reports it, accuracy, peak memory and BLAS kernel.
Ratios compare like with like: an RI-JK run against linus RI-JK, RIJCOSX against linus
RIJCOSX (or linus's fastest approach where linus lacks that one, marked *).
Accuracy (needs `run_bench.py refs`): energies against (a) the same program with exact
integrals on the same grid (ΔE_int: the RI/COSX error alone) and (b) linus with exact
integrals on a fine grid (ΔE_tot: integral + grid + convergence error); gradients the same
as max |ΔG|; Hessians as harmonic-frequency deviations from the fine-grid exact-integral
Hessian (translations and rotations projected out, same masses for every program).
"""
import argparse, glob, html, json, math, os, pathlib, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "bench/results"
PROGS = ["linus", "orca", "pyscf", "psi4"]
FAIR = {"pyscf": "pyscf-fastblas", "psi4": "psi4-openblas"}
AP_NAME = {"rijk": "RI-JK", "rijcosx": "RIJCOSX", "rij": "RI-J", "exact": "exact"}
GRID_LABEL = {"linus-lg2": ", linus:2 grid", "linus-pg3": ", pyscf:3 grid", "linus-auto": ", linus defaults (grid/aux auto)"}  # linus without a label: orca:2
MASS = {"H": 1.00782503223, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957, "F": 18.99840316273, "K": 38.9637064864,
        "S": 31.9720711744, "Cl": 34.968852682, "Br": 78.9183376, "P": 30.97376199842, "Li": 7.0160034366, "Na": 22.989769282}
AU2CM = 5140.48767  # sqrt(Eh / (bohr^2 amu)) in cm^-1


def load(tags, layouts=None):
    cases, refs, blas = {}, {}, {}
    for tag in tags:
        base = RES / tag
        for f in sorted(base.glob("*/*.json")):
            lay = f.parent.name
            if lay == "refs":
                d = json.loads(f.read_text())
                refs.setdefault((d["mol"], d["level"], d["calc"]), {}).update(d["refs"])
                continue
            if layouts and lay not in layouts:
                continue
            d = json.loads(f.read_text())
            key = (lay, d["mol"], d["level"], d["calc"])
            if key in cases:
                cases[key]["entries"].update(d["entries"])
            else:
                cases[key] = d
        if (base / "blas_validation.json").exists():
            blas[tag] = json.loads((base / "blas_validation.json").read_text())
    return cases, refs, blas


def frequencies(hess, symbols, coords):
    import numpy as np
    h = np.asarray(hess, float)
    n = len(symbols)
    m = np.array([MASS[s] for s in symbols])
    x = np.asarray(coords, float)
    com = (m[:, None] * x).sum(0) / m.sum()
    r = x - com
    sm = np.repeat(np.sqrt(m), 3)
    hmw = h / np.outer(sm, sm)
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = np.sqrt(m)
        vecs.append(v.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1
        v = np.cross(e, r) * np.sqrt(m)[:, None]
        vecs.append(v.ravel())
    q, _ = np.linalg.qr(np.array(vecs).T)
    p = np.eye(3 * n) - q @ q.T
    w = np.linalg.eigvalsh(p @ hmw @ p)
    w = np.sort(w)
    keep = np.argsort(np.abs(w))[6:]
    w = np.sort(w[keep])
    return np.sign(w) * np.sqrt(np.abs(w)) * AU2CM


def max_abs_diff(a, b):
    import numpy as np
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max()) if a.shape == b.shape else None


def first_ok_run(entry):
    return next((r for r in entry.get("runs", []) if r.get("ok") and not r.get("warmup")), None)


def accuracy(case, entry, refs, mol_geom):
    """ΔE_int, ΔE_tot (Eh), max|ΔG| (Eh/bohr), frequency deviations (cm^-1)."""
    s = entry["summary"]
    out = {}
    if s.get("status") != "ok":
        return out
    rr = refs.get((case["mol"], case["level"], case["calc"]), {})
    prog = entry["program"]
    # a linus variant on another grid has its own exact-integral reference on that grid (no fallback)
    own = rr.get(f"own:{entry['variant']}") if entry['variant'] in GRID_LABEL else rr.get(f"own:{prog}")
    fine = rr.get("fine:linus")
    nodisp = prog in case.get("level_def", {}).get("no_dispersion", [])
    e = s.get("energy")
    if own and own.get("ok") and e is not None:
        out["dE_int"] = e - own["energy"]
    if fine and fine.get("ok") and e is not None and not nodisp:
        out["dE_tot"] = e - fine["energy"]
    if own and own.get("ok") and fine and fine.get("ok") and not nodisp:
        # grid error alone (exact integrals on the program's grid vs the fine grid); the matched-
        # accuracy pairing adds |integral| and |grid| errors so that their cancellation earns no credit
        out["dE_grid"] = own["energy"] - fine["energy"]
        if "dE_int" in out:
            out["dE_budget"] = abs(out["dE_int"]) + abs(out["dE_grid"])
    run = first_ok_run(entry)
    if run is not None and run.get("gradient") is not None:
        if own and own.get("gradient") is not None:
            out["dG_int"] = max_abs_diff(run["gradient"], own["gradient"])
        if fine and fine.get("gradient") is not None and not nodisp:
            out["dG_tot"] = max_abs_diff(run["gradient"], fine["gradient"])
    if run is not None and run.get("hessian") is not None and fine and fine.get("hessian") is not None and mol_geom:
        try:
            f0 = frequencies(fine["hessian"], *mol_geom)
            f1 = frequencies(run["hessian"], *mol_geom)
            d = f1 - f0
            out["dfreq_max"] = float(abs(d).max())
            out["dfreq_rms"] = float((d ** 2).mean() ** 0.5)
            out["dH_max"] = max_abs_diff(run["hessian"], fine["hessian"])
        except Exception as ex:  # noqa: BLE001
            out["freq_error"] = str(ex)
    return out


def geom(mol):
    sys.path.insert(0, str(ROOT / "bench"))
    from run_bench import molecule
    try:
        m = molecule(mol)
        return m["symbols"], m["coords"]
    except StopIteration:
        return None


def fmt_t(t):
    return "-" if t is None else (f"{t:.1f}" if t >= 10 else f"{t:.2f}")


def fmt_e(x, unit=1e-6):
    """Energy-like error in micro-units, signed."""
    if x is None:
        return "-"
    v = x / unit
    return f"{v:+.1f}" if abs(v) >= 0.1 else f"{v:+.2g}"


def fmt_g(x):
    return "-" if x is None else f"{x:.1e}"


def blas_str(s, variant=None):
    if variant == "linus":
        return "faer (Rust)"
    b = s.get("blas_binding") or []
    if b:
        return ", ".join(sorted({os.path.basename(x["binds_to"] or "?") for x in b}))
    bl = s.get("blas") or []
    return ", ".join(f"{x.get('version')} {x.get('core')}" for x in bl if x) or ""


def linus_ref(entries, approach):
    """linus time for a program column: same approach if linus ran it, else linus's fastest (starred)."""
    lin = {e["approach"]: e["summary"] for k, e in entries.items() if e["variant"] == "linus" and e["summary"].get("status") == "ok"}
    if approach in lin:
        return lin[approach]["median"], False
    if lin:
        return min(v["median"] for v in lin.values()), True
    return None, False


def headline(cases, layout, calc):
    rows = []
    keys = sorted((k for k in cases if k[0] == layout and k[3] == calc), key=lambda k: (k[2], k[1]))
    cols = [("orca", "rijcosx"), ("orca", "rijk"), ("orca", "rij"), ("orca", "exact"), ("pyscf", None), ("pyscf-fastblas", None),
            ("psi4", None), ("psi4-openblas", None)]
    present = set()
    for k in keys:
        for e in cases[k]["entries"].values():
            if e["summary"].get("status") == "ok":
                present.add((e["variant"], e["approach"]))
    use = [c for c in cols if any(v == c[0] and (c[1] is None or a == c[1]) for v, a in present)]
    head = ["molecule", "level", "linus s (approach)"] + [f"{v}{' ' + AP_NAME[a] if a else ''} ×" for v, a in use]
    for k in keys:
        ents = cases[k]["entries"]
        lin = [(e["summary"]["median"], e["approach"], e["variant"]) for e in ents.values()
               if e["program"] == "linus" and e["summary"].get("status") == "ok"]
        lin_txt = " / ".join(f"{fmt_t(t)} ({AP_NAME[a]}{GRID_LABEL.get(v, '')})"
                             for t, a, v in sorted(lin, key=lambda x: (x[2], x[1]))) or \
            next((e["summary"].get("status") + ": " + (e["summary"].get("error") or e["summary"].get("reason") or "")[:90] for e in ents.values() if e["variant"] == "linus"), "-")
        row = [k[1], k[2], lin_txt]
        for v, a in use:
            cand = [e for e in ents.values() if e["variant"] == v and (a is None or e["approach"] == a)]
            e = next((e for e in cand if e["summary"].get("status") == "ok"), None) or next((e for e in cand if e["summary"].get("status") != "skipped"), None)
            if e is None or e["summary"].get("status") != "ok":
                row.append("-" if e is None else e["summary"].get("status", "-"))
                continue
            t = e["summary"]["median"]
            ref, star = linus_ref(ents, e["approach"])
            row.append(f"{t / ref:.2f}{'*' if star else ''}" if ref else fmt_t(t))
        rows.append(row)
    return head, rows


def detail(cases, refs, layout, calc):
    rows = []
    keys = sorted((k for k in cases if k[0] == layout and k[3] == calc), key=lambda k: (k[2], k[1]))
    if calc == "hessian":
        acc_head = ["max|Δν| cm⁻¹", "RMS Δν", "max|ΔH|"]
    elif calc == "gradient":
        acc_head = ["ΔE_int µEh", "ΔE_tot µEh", "max|ΔG|_int", "max|ΔG|_tot"]
    else:
        acc_head = ["ΔE_int µEh", "ΔE_grid µEh", "ΔE_tot µEh"]
    head = ["molecule", "level", "program", "approach", "median s (min-max)", "× linus", "iter", "final orb. grad."] + acc_head + ["peak GB", "BLAS (dgemm_ binds to)"]
    for k in keys:
        case = cases[k]
        g = geom(case["mol"]) if calc == "hessian" else None
        order = {p: i for i, p in enumerate(["linus", "linus-lg2", "orca", "pyscf", "pyscf-fastblas", "pyscf-pthreads1", "pyscf-pthreads", "pyscf-haswell", "psi4", "psi4-openblas", "psi4-pthreads"])}
        for key, e in sorted(case["entries"].items(), key=lambda x: (order.get(x[1]["variant"], 9), x[1]["approach"])):
            s = e["summary"]
            if s.get("status") == "skipped":
                continue
            if s.get("status") != "ok":
                rows.append([k[1], k[2], e["variant"], AP_NAME.get(e["approach"], e["approach"]), s.get("status"), "", "", (s.get("error") or "")[:80]]
                            + [""] * len(acc_head) + ["", ""])
                continue
            ref, star = linus_ref(case["entries"], e["approach"])
            ratio = f"{s['median'] / ref:.2f}{'*' if star else ''}" if ref and e["program"] != "linus" else ""
            og = s.get("final_gmax")
            og_txt = f"{og:.1e} (max F_ai)" if og is not None else (f"{s['final_orbital_gradient']:.1e} (ORCA)" if s.get("final_orbital_gradient") is not None
                                                                  else (f"{s['final_orbital_gradient_rms']:.1e} (Psi4 rms)" if s.get("final_orbital_gradient_rms") is not None else "-"))
            acc = accuracy(case, e, refs, g)
            if calc == "hessian":
                accs = [f"{acc['dfreq_max']:.2f}" if "dfreq_max" in acc else "-", f"{acc['dfreq_rms']:.2f}" if "dfreq_rms" in acc else "-", fmt_g(acc.get("dH_max"))]
            elif calc == "gradient":
                accs = [fmt_e(acc.get("dE_int")), fmt_e(acc.get("dE_tot")), fmt_g(acc.get("dG_int")), fmt_g(acc.get("dG_tot"))]
            else:
                accs = [fmt_e(acc.get("dE_int")), fmt_e(acc.get("dE_grid")), fmt_e(acc.get("dE_tot"))]
            spread = f"{fmt_t(s['median'])} ({fmt_t(s['min'])}-{fmt_t(s['max'])})" if s["n_ok"] > 1 else fmt_t(s["median"])
            rows.append([k[1], k[2], e["variant"], AP_NAME.get(e["approach"], e["approach"]), spread, ratio, str(s.get("iterations") or "-"), og_txt]
                        + accs + [f"{s.get('peak_memory_gb', 0):.1f}", blas_str(s, e["program"])])
    return head, rows


METRIC = {"energy": ("dE_budget", "|ΔE_int|+|ΔE_grid| µEh", 1e-6), "gradient": ("dG_tot", "max|ΔG|_tot Eh/bohr", 1.0),
          "hessian": ("dfreq_max", "max|Δν| cm⁻¹", 1.0)}
MATCH_TOL = 1.25
FLOOR = {"energy": 1e-6, "gradient": 1e-6, "hessian": 0.05}  # 1 µEh, 1e-6 Eh/bohr, 0.05 cm^-1


def matched(cases, refs, layout, calc):
    """Each program run against the fastest linus run (any grid or approach) of matching accuracy by
    the calculation's metric against the fine-grid exact-integral reference: energies by the error
    budget |ΔE_int| + |ΔE_grid| (no credit for cancellation), gradients by max |ΔG|, Hessians by the
    max frequency deviation. Matching = linus's error at most MATCH_TOL times the program's, or both
    below FLOOR (convergence noise)."""
    key, label, unit = METRIC[calc]
    rows = []
    for k in sorted((k for k in cases if k[0] == layout and k[3] == calc), key=lambda k: (k[2], k[1])):
        case = cases[k]
        g = geom(case["mol"]) if calc == "hessian" else None
        acc = {}
        for ek, e in case["entries"].items():
            if e["summary"].get("status") == "ok":
                a = accuracy(case, e, refs, g)
                if a.get(key) is not None:
                    acc[ek] = abs(a[key]), e["summary"]["median"], e, a
        lin = [(t, m, e) for m, t, e, _ in acc.values() if e["program"] == "linus"]
        for ek, (m, t, e, a) in sorted(acc.items()):
            if e["program"] == "linus":
                continue
            # matched: linus at most 25% less accurate, or both below the floor where the error is convergence noise
            ok = [x for x in lin if x[1] <= max(MATCH_TOL * m, FLOOR[calc])]
            extra = f" (rms {a['dfreq_rms']:.2f})" if calc == "hessian" else ""
            if ok:
                tl, ml, el = min(ok, key=lambda x: x[0])
                rows.append([k[1], k[2], f"{e['variant']} {AP_NAME[e['approach']]}", fmt_t(t), f"{m / unit:.3g}{extra}",
                             f"{el['variant']} {AP_NAME[el['approach']]}", fmt_t(tl), f"{ml / unit:.3g}", f"{t / tl:.2f}"])
            else:
                best = min(lin, key=lambda x: x[1]) if lin else None
                rows.append([k[1], k[2], f"{e['variant']} {AP_NAME[e['approach']]}", fmt_t(t), f"{m / unit:.3g}{extra}",
                             f"none as accurate (best: {best[2]['variant']} {AP_NAME[best[2]['approach']]} {best[1] / unit:.3g})" if best else "-",
                             "-", "-", "-"])
    head = ["molecule", "level", "program", "s", label, "fastest linus of matching accuracy (≤ 1.25x the error)", "linus s", f"linus {label}", "× linus"]
    return head, rows


def blas_effect(cases):
    """Fair-BLAS variant vs as installed: time ratio and energy difference per case."""
    rows = []
    for k in sorted(cases, key=lambda k: (k[0], k[3], k[2], k[1])):
        ents = cases[k]["entries"]
        for base, fair in FAIR.items():
            for ap in ("rijk", "rij", "exact"):
                a = ents.get(f"{base}:{ap}")
                b = ents.get(f"{fair}:{ap}")
                if not a or not b or a["summary"].get("status") != "ok" or b["summary"].get("status") != "ok":
                    continue
                sa, sb = a["summary"], b["summary"]
                ra, rb = first_ok_run(a), first_ok_run(b)
                dg = max_abs_diff(ra.get("gradient"), rb.get("gradient")) if ra and rb else None
                dh = max_abs_diff(ra.get("hessian"), rb.get("hessian")) if ra and rb else None
                rows.append([k[0], k[1], k[2], k[3], base, AP_NAME[ap], fmt_t(sa["median"]), fmt_t(sb["median"]), f"{sb['median'] / sa['median']:.2f}",
                             f"{abs(sb['energy'] - sa['energy']):.1e}", fmt_g(dg) if dg is not None else (fmt_g(dh) if dh is not None else "-"),
                             blas_str(sa), blas_str(sb)])
    head = ["layout", "molecule", "level", "calc", "program", "approach", "as installed s", "fair BLAS s", "fair / installed", "|ΔE| Eh",
            "max|ΔG| or |ΔH|", "BLAS as installed", "BLAS fair"]
    return head, rows


def provenance(cases):
    linus, cpu, busy, waits, starts = set(), set(), [], [], []
    for c in cases.values():
        for h in c.get("history", []):
            env = h.get("environment_start") or {}
            li = env.get("linus") or {}
            if li:
                linus.add(f"case start: sources {(li.get('source_commit') or li.get('commit') or '?')[:10]}"
                          f"{' (dirty)' if li.get('source_dirty') else ''}, binary sha256 {(li.get('sha256') or '?')[:12]}")
            cpu.add(env.get("cpu_model"))
            starts.append(env.get("time"))
        for e in c["entries"].values():
            for r in e.get("runs", []):
                lb = r.get("linus_build")
                if lb:
                    linus.add(f"linus runs: sources {(lb.get('source_commit') or '?')[:10]}{' (dirty)' if lb.get('source_dirty') else ''}, "
                              f"branch head {(lb.get('commit') or '?')[:10]}, binary sha256 {(lb.get('sha256') or '?')[:12]}")
                if r.get("cpu_busy_before") is not None and not r.get("warmup"):
                    busy.append(r["cpu_busy_before"])
                if r.get("lock_wait_s") is not None:
                    waits.append(r["lock_wait_s"])
    starts = sorted(x for x in starts if x)
    return [f"linus build: {'; '.join(sorted(linus))}", f"CPU: {', '.join(sorted(x for x in cpu if x))}",
            f"runs: {starts[0] if starts else '?'} to {starts[-1] if starts else '?'}",
            f"busy fraction of the layout's CPUs just before each timed run: median {statistics.median(busy):.2f}, max {max(busy):.2f}" if busy else "",
            f"lock wait (not timed): total {sum(waits):.0f} s" if waits else ""]


def md_table(head, rows):
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")  # noqa: E731 - pipes inside cells (|ΔE|)
    out = ["| " + " | ".join(esc(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(esc(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def html_table(head, rows):
    h = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table>\n<thead><tr>{h}</tr></thead>\n<tbody>{b}</tbody>\n</table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--layouts", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cases, refs, blas = load(a.tags, a.layouts.split(",") if a.layouts else None)
    md, ht = [f"# linus vs ORCA 6.0.0 / PySCF 2.15 / Psi4 1.11: {' + '.join(a.tags)}", ""], []
    md += ["Times are medians of the timed repetitions (wall time of the whole program run, warm-up excluded). "
           "Ratios are program time / linus time (> 1: linus faster); * = linus lacks that approach, compared with linus's fastest. "
           "ΔE_int: against the same program with exact integrals on the same grid; ΔE_tot: against linus exact integrals on a fine grid "
           "(µEh; '-' = no reference yet). Final orbital gradients are as each program reports them (linus and PySCF: max |F_ai|; "
           "ORCA: 'Last Orbital Gradient'; Psi4: RMS orbital gradient).", ""]
    md += ["## Provenance", ""] + [f"- {x}" for x in provenance(cases) if x] + [""]
    layouts = sorted({k[0] for k in cases}, key=lambda x: ["pinned", "unpinned", "spread", "scale16", "scale32", "scale64"].index(x)
                     if x in ["pinned", "unpinned", "spread", "scale16", "scale32", "scale64"] else 99)
    for lay in layouts:
        for calc in ("energy", "gradient", "hessian"):
            if not any(k[0] == lay and k[3] == calc for k in cases):
                continue
            h1, r1 = headline(cases, lay, calc)
            h2, r2 = detail(cases, refs, lay, calc)
            hm, rm = matched(cases, refs, lay, calc)
            title = f"{lay} layout, {calc}"
            md += [f"## {title}: time ratio to linus", "", md_table(h1, r1), ""]
            ht += [f"<h2>{html.escape(title)}: time ratio to linus</h2>", html_table(h1, r1)]
            if rm:
                md += [f"### {title}: at matched accuracy", "", md_table(hm, rm), ""]
                ht += [f"<h3>{html.escape(title)}: at matched accuracy</h3>", html_table(hm, rm)]
            md += [f"### {title}: detail", "", md_table(h2, r2), ""]
            ht += [f"<h3>{html.escape(title)}: detail</h3>", html_table(h2, r2)]
    skipped = {}
    for k, c in cases.items():
        for e in c["entries"].values():
            s = e["summary"]
            if s.get("status") in ("skipped", "rejected", "oom", "timeout", "failed"):
                why = (s.get("reason") or s.get("error") or s.get("status") or "").strip().splitlines()[-1][:160] if (s.get("reason") or s.get("error")) else s.get("status")
                skipped.setdefault((s["status"], why), []).append(f"{k[0]}/{k[1]}/{k[2]}/{k[3]}:{e['variant']}:{e['approach']}")
    if skipped:
        rows = [[st, why, len(v), ", ".join(v[:4]) + (" ..." if len(v) > 4 else "")] for (st, why), v in sorted(skipped.items())]
        md += ["## Not run or failed", "", md_table(["status", "reason", "entries", "examples"], rows), ""]
        ht += ["<h2>Not run or failed</h2>", html_table(["status", "reason", "entries", "examples"], rows)]
    h3, r3 = blas_effect(cases)
    if r3:
        md += ["## Fair BLAS: effect on PySCF and Psi4", "", md_table(h3, r3), ""]
        ht += ["<h2>Fair BLAS: effect on PySCF and Psi4</h2>", html_table(h3, r3)]
    for tag, b in blas.items():
        rows = [[k, f"{v.get('gflops', 0):.1f}", str(v.get("core")), v.get("version") or "",
                 (v.get("error") or "")[:60] or (f"exit {v['returncode']}" + (" (segfault)" if v["returncode"] == -11 else "") if v.get("returncode") else "")]
                for k, v in b.get("libraries", {}).items()]
        md += [f"## DGEMM per library ({tag}, n = 2000)", "", md_table(["library@threads", "GF/s", "kernel", "version", "error"], rows), ""]
        rows = [[v, json.dumps(d.get("ld_debug_bindings")), f"{d.get('energy')}"] for v, d in b.get("variants", {}).items()]
        md += ["Dynamic-linker bindings (LD_DEBUG=bindings) of the variants, water B3LYP/def2-SVP:", "",
               md_table(["variant", "dgemm_/dsyevd_ bound to", "energy"], rows), ""]
    sys.path.insert(0, str(ROOT / "bench"))
    from run_bench import scrub_host  # never publish the host name
    text = scrub_host("\n".join(md))
    out = pathlib.Path(a.out) if a.out else RES / a.tags[-1]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text)
    (out / "summary.html").write_text(scrub_host("\n".join(ht)))
    print(text)


if __name__ == "__main__":
    main()
