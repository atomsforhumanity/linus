"""Parse the SCF trace of ORCA 6 outputs (RIJCOSX single points): per iteration energy, dE, MaxDP,
DIIS error or SOSCF MaxGrad, damping and time, and the events ORCA prints (AO-DIIS on, SOSCF
initialized, incremental Fock formation restarted = the COSX switch to GridX2), the stopping rule
("Energy Check" or "Gradient check") and the final GridX3 exchange-energy change.

usage: python3 bench/orca_scf_trace.py OUT.json case=path/to/orca.out [...]
"""
import json, re, sys


def parse(path):
    its, events, stop, phase = [], [], None, "diis"
    final = {}
    for l in open(path):
        if "S-O-S-C-F" in l:
            phase = "soscf"
        m = re.match(r"^\s+(\d+)\s+(-\d+\.\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?\s*$", l)
        if m and phase in ("diis", "soscf") and "ORBITAL" not in l:
            g = m.groups()
            if phase == "diis" and g[7] is not None:
                its.append(dict(iter=int(g[0]), energy=float(g[1]), de=float(g[2]), rmsdp=float(g[3]), maxdp=float(g[4]), diis_err=float(g[5]), damp=float(g[6]), time=float(g[7])))
            elif phase == "soscf":
                its.append(dict(iter=int(g[0]), energy=float(g[1]), de=float(g[2]), rmsdp=float(g[3]), maxdp=float(g[4]), max_grad=float(g[5]), time=float(g[6])))
            continue
        n = len(its)
        if "Turning on AO-DIIS" in l:
            events.append(dict(after_iter=n, event="AO-DIIS on"))
        elif "Initializing SOSCF" in l:
            events.append(dict(after_iter=n, event="SOSCF start"))
        elif "Restarting incremental Fock" in l:
            events.append(dict(after_iter=n, event="incremental Fock restart (COSX GridX2)"))
        elif "signals convergence" in l:
            stop = l.strip().strip("*").strip()
            phase = "done"
        elif "Exchange energy change after final integration" in l:
            final["gridx3_dex"] = float(l.split(":")[1].split()[0])
        elif re.match(r"\s+Last (Energy change|Orbital Gradient|MAX-Density change)", l):
            k = l.split("...")[0].strip()
            final[k] = float(l.split("...")[1].split()[0])
    return dict(cycles=len(its), iterations=its, events=events, stop=stop, final=final)


out = sys.argv[1]
res = {"conditions": "ORCA 6.0.0 `! <functional> def2-TZVP def2/J RIJCOSX DefGrid2 TightSCF Bohrs`, %pal nprocs 8, %maxcore 2000; untimed runs (shared CCD); "
                     "the SCF trace parsed from orca.out by bench/orca_scf_trace.py", "cases": {}}
for kv in sys.argv[2:]:
    case, path = kv.split("=", 1)
    res["cases"][case] = parse(path)
    c = res["cases"][case]
    last = c["iterations"][-1]
    print(f"{case:24s} cycles {c['cycles']:2d}  events {[(e['after_iter'], e['event'][:12]) for e in c['events']]}  stop {c['stop']}  final grad {last.get('max_grad', last.get('diis_err')):.1e} dE {last['de']:.1e}")
open(out, "w").write(json.dumps(res, indent=1))
