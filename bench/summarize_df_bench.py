"""Print Markdown tables from run_df_bench.py result files.

  python3 bench/summarize_df_bench.py results.json [more.json ...]

Runs from several files are merged; for a configuration (program, molecule, basis, threads)
present in more than one file only the last file's runs are used (reruns). Dropped: runs
marked superseded (contended attempts that were retried), failed runs, ORCA runs that
converged in <= 3 cycles (restarted from an earlier .gbw by ORCA's AutoStart), and
range-separated linus runs not done with def2-universal-jkfit.
"""
import json, pathlib, sys

NAUX_JK = {"la18": 902, "u1an39": 2556, "ts61": 3436, "ts113": 5520}


def med(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def f(x, nd=3):
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def load(paths):
    by_key = {}
    metas = []
    for i, p in enumerate(paths):
        d = json.loads(pathlib.Path(p).read_text())
        metas.append(d["meta"])
        for r in d["runs"]:
            if not r.get("ok") or r.get("superseded"):
                continue
            if r["program"].startswith("orca") and (r.get("cycles") or 0) <= 3:
                continue
            if r["program"] == "linus-rsh" and r.get("naux") != NAUX_JK[r["mol"]]:
                continue
            k = (r["program"], r["mol"], r["basis"], r["threads"])
            by_key.setdefault(k, {}).setdefault(i, []).append(r)
    runs = {k: v[max(v)] for k, v in by_key.items()}
    return metas, runs


def main():
    paths = sys.argv[1:] or ["bench/results/df_jk_8t_bench.json"]
    metas, runs = load(paths)
    g = lambda prog, m, b, t, key: med([r.get(key) for r in runs.get((prog, m, b, t), [])])
    n = lambda prog, m, b, t: len(runs.get((prog, m, b, t), []))
    mols = ["la18", "u1an39", "ts61", "ts113"]
    bases = ["def2-svp", "def2-tzvp"]
    threads = sorted({k[3] for k in runs})
    print(f"CPU: {metas[0]['cpu']}; slice {metas[0].get('slice')}; git {', '.join(sorted({m['git'] for m in metas}))}; medians; seconds\n")
    print("### Per-iteration J/K (s)\n")
    print("| mol | basis | thr | nbf | linus RI-J (def2/J, in-mem) | linus RI-J direct | ORCA Split-RI-J | linus RI-K (jkfit) | ORCA COSX | linus J+K (jkfit) | PySCF J+K (jkfit) | runs (linus/ORCA/PySCF) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in mols:
        for b in bases:
            for t in threads:
                if not any(n(p, m, b, t) for p in ("linus-jk", "linus-j", "linus-jdir", "orca", "pyscf")):
                    continue
                nbf = g("linus-j", m, b, t, "nbf") or g("linus-jk", m, b, t, "nbf")
                print(f"| {m} | {b[5:].upper()} | {t} | {nbf or '-'} | {f(g('linus-j', m, b, t, 't_j'))} | {f(g('linus-jdir', m, b, t, 't_j'))} | "
                      f"{f(g('orca', m, b, t, 't_rij'))} | {f(g('linus-jk', m, b, t, 't_k'))} | {f(g('orca', m, b, t, 't_cosx'))} | "
                      f"{f(g('linus-jk', m, b, t, 't_jk'))} | {f(g('pyscf', m, b, t, 't_jk'))} | {n('linus-jk', m, b, t)}/{n('orca', m, b, t)}/{n('pyscf', m, b, t)} |")
    print("\n### Setup and memory\n")
    print("| mol | basis | thr | linus J+K setup (s) | 3c ints / B=A L^-T (s) | B (GiB) | peak RSS (GiB) | rows kept / packed | linus J-only setup (s) | PySCF DF build (s) | PySCF cderi (GiB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in mols:
        for b in bases:
            for t in threads:
                if not any(n(p, m, b, t) for p in ("linus-jk", "linus-j", "pyscf")):
                    continue
                rows, dense = g("linus-jk", m, b, t, "rows"), g("linus-jk", m, b, t, "rows_dense_packed")
                print(f"| {m} | {b[5:].upper()} | {t} | {f(g('linus-jk', m, b, t, 't_setup'))} | {f(g('linus-jk', m, b, t, 't_ints'))} / {f(g('linus-jk', m, b, t, 't_fit'))} | "
                      f"{f(g('linus-jk', m, b, t, 'store_gb'), 2)} | {f(g('linus-jk', m, b, t, 'peak_rss_gb'), 2)} | {f(rows / dense, 2) if rows else '-'} | "
                      f"{f(g('linus-j', m, b, t, 't_setup'))} | {f(g('pyscf', m, b, t, 't_setup'))} | {f(g('pyscf', m, b, t, 'cderi_gb'), 2)} |")
    if any(k[0] in ("linus-rsh", "orca-rsh") for k in runs):
        print("\n### Range-separated exchange (omega = 0.3; linus: J + K + K_LR per call; ORCA: wB97X-D3BJ RIJCOSX per cycle)\n")
        print("| mol | basis | thr | linus J+K (s) | linus J+K+K_LR (s) | linus B^w setup (s) | linus B+B^w (GiB) | ORCA RI-J | ORCA COSX |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m in mols:
            for b in bases:
                for t in threads:
                    if not (n("linus-rsh", m, b, t) or n("orca-rsh", m, b, t)):
                        continue
                    print(f"| {m} | {b[5:].upper()} | {t} | {f(g('linus-rsh', m, b, t, 't_jk'))} | {f(g('linus-rsh', m, b, t, 't_jk_klr'))} | {f(g('linus-rsh', m, b, t, 't_lr_setup'))} | "
                          f"{f(g('linus-rsh', m, b, t, 'store_gb'), 2)} | {f(g('orca-rsh', m, b, t, 't_rij'))} | {f(g('orca-rsh', m, b, t, 't_cosx'))} |")
    if any(k[0].startswith("scf-") for k in runs):
        print("\n### End-to-end B3LYP SCF (linus `scf` example, XC grid PySCF level 3; ORCA B3LYP/G RIJCOSX DefGrid2 TightSCF)\n")
        print("| mol | basis | thr | program | wall (s) | cycles | J/K setup (s) | J/K total (s) | J/K per cycle (s) | XC total (s) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m in mols:
            for b in bases:
                for t in threads:
                    for p in ("scf-old", "scf-ri", "orca"):
                        if not n(p, m, b, t) or not (n("scf-ri", m, b, t) or n("scf-old", m, b, t)):
                            continue
                        if p == "orca":
                            jk = (g(p, m, b, t, "t_rij_total") or 0) + (g(p, m, b, t, "t_cosx_total") or 0)
                            cyc = g(p, m, b, t, "cycles")
                            print(f"| {m} | {b[5:].upper()} | {t} | ORCA | {f(g(p, m, b, t, 'wall'))} | {cyc:.0f} | - | {f(jk)} | {f(jk / cyc)} | {f(g(p, m, b, t, 't_xc_total'))} |")
                        else:
                            print(f"| {m} | {b[5:].upper()} | {t} | linus {'DfJk' if p == 'scf-ri' else 'old DfJK'} | {f(g(p, m, b, t, 'wall'))} | {g(p, m, b, t, 'cycles'):.0f} | "
                                  f"{f(g(p, m, b, t, 't_jk_setup'))} | {f(g(p, m, b, t, 't_jk_total'))} | {f(g(p, m, b, t, 't_jk_per_cycle'))} | {f(g(p, m, b, t, 't_xc_total'))} |")
    if any(k[0] == "old" for k in runs):
        print("\n### Previous DfJK (same density, J+K per call)\n")
        for k, rs in sorted(runs.items()):
            if k[0] == "old":
                print(f"- {k[1]} {k[2]} {k[3]} threads: setup {f(med([r['t_setup'] for r in rs]))} s, J+K {f(med([r['t_jk'] for r in rs]))} s")


if __name__ == "__main__":
    main()
