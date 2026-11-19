"""Tables from bench/scf_compare.py results: Fock builds, SCF wall time and phases per program.

usage: python bench/scf_compare_table.py bench/results/TAG [more dirs...]
Medians over repetitions. linus: SCF time = scf_total (guess, J/K, XC, DIIS, diagonalization;
"total" adds setup: RI tensors, grids, guess, final COSX energy). Wall times measured by the
harness are not used (before the in-process lock they included waiting for the bench lock). ORCA: "Total time" of its SCF
TIMINGS block (includes its SCF preparation), Fock formation, and the final orbital gradient it
prints. "before": main c32d31c at its defaults (e_tol 1e-10, max |F_ai| < 1e-8) and at the
tight thresholds. PySCF: Fock builds and time (setup + cycles) until linus's criteria hold on its trace
("matched"), and until its own default stop at conv_tol 1e-8 ("default").
"""
import json, pathlib, statistics, sys


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f(x, fmt="{:.2f}"):
    return "-" if x is None else fmt.format(x)


rows = []
for d in sys.argv[1:]:
    for p in sorted(pathlib.Path(d).glob("*.json")):
        r = json.loads(p.read_text())
        if "runs" not in r or "linus_conv" not in r:
            continue
        L = [x for x in r["runs"].get("linus", []) if x.get("ok")]
        O = [x for x in r["runs"].get("orca", []) if x.get("ok")]
        P = [x for x in r["runs"].get("pyscf", []) if x.get("ok")]
        B = [x for x in r["runs"].get("base", []) if x.get("ok")]
        BT = [x for x in r["runs"].get("base-tight", []) if x.get("ok")]
        row = dict(case=f"{r['mol']} {r['method']}/{r['basis']} {r['jk']}", nbf=L[0]["nbf"] if L else None)
        if L:
            row.update(l_b=L[0]["builds"], l_scf=med([x["phases"]["scf"] for x in L]), l_jk=med([x["phases"]["jk"] for x in L]),
                       l_xc=med([x["phases"]["xc"] for x in L]), l_dd=med([x["phases"]["diag"] + x["phases"]["diis"] for x in L]),
                       l_setup=med([x["phases"]["setup"] for x in L]), l_total=med([x["timings"].get("total") for x in L]), l_wall=med([x["wall"] for x in L]), l_g=L[0]["final_gmax"])
        if O:
            row.update(o_c=O[0]["cycles"], o_scf=med([x["phases"].get("Total time") for x in O]),
                       o_fock=med([x["phases"].get("Fock matrix formation") for x in O]), o_wall=med([x["wall"] for x in O]),
                       o_g=O[0].get("last_orbital_gradient"), o_run=med([x.get("total_run_time") for x in O]))
        if P:
            row.update(p_mb=P[0]["matched"]["builds"], p_mt=med([x["matched"]["time"] for x in P]),
                       p_db=P[0]["pyscf_default"]["builds"], p_dt=med([x["pyscf_default"]["time"] for x in P]))
        if B:
            row.update(b_b=B[0]["builds"], b_scf=med([x["phases"]["scf"] for x in B]))
        if BT:
            row.update(bt_b=BT[0]["builds"], bt_scf=med([x["phases"]["scf"] for x in BT]))
        rows.append(row)

print("| case | nbf | before: default builds / SCF s | before: tight builds / SCF s | linus builds | linus SCF s (J/K, XC, diag+DIIS) | linus total s | linus final max F_ai | ORCA cycles | ORCA SCF s (Fock) / run s | ORCA final orb. grad | PySCF matched builds / s | PySCF default builds / s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['case']} | {r.get('nbf', '-')} | {r.get('b_b', '-')} / {f(r.get('b_scf'))} | {r.get('bt_b', '-')} / {f(r.get('bt_scf'))} | {r.get('l_b', '-')} | {f(r.get('l_scf'))} ({f(r.get('l_jk'))}, {f(r.get('l_xc'))}, {f(r.get('l_dd'))}) | "
          f"{f(r.get('l_total'))} | {f(r.get('l_g'), '{:.1e}')} | {r.get('o_c', '-')} | {f(r.get('o_scf'))} ({f(r.get('o_fock'))}) / {f(r.get('o_run'))} | {f(r.get('o_g'), '{:.1e}')} | "
          f"{r.get('p_mb', '-')} / {f(r.get('p_mt'))} | {r.get('p_db', '-')} / {f(r.get('p_dt'))} |")
