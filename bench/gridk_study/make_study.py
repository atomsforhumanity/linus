#!/usr/bin/env python3
"""Compensated grid-K's accuracy study as a run of CPU pod tasks (bench/gridk_study/README.md):
gridk_study (crates/linus-dev/examples: fixed-density exchange errors of exact, DF and the
four grid-K variants) for the species of GMTKN55 subsets and the repository's molecules.

usage: make_study.py <run dir> <gridk_study binary> <bases,comma> --image IMAGE
       [--gmtkn-tars DIR] [--subsets A,B|paper16] [--mols la18,u1an39] [--grad] [--max-atoms N]
       [--method b3lyp] [--grids gridx:1,gridx:2] [--shards N] [--mem 24Gi]
Writes <run dir>/ (run.yaml, tasks.jsonl, bin/, src/data/basis, study_task.py, reactions.json);
--gmtkn-tars is the directory of GMTKN55's per-subset tar files (<subset>.tar)."""
import argparse
import io
import json
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MOLS = [REPO / "crates/linus-disp/tests/molecules", REPO / "bench/molecules", REPO / "tests/molecules"]
PAPER16 = "BHDIV10 BHPERI BHROT27 PX13 ACONF ICONF PCONF21 SCONF CDIE20 DARC ISO34 PArel AHB21 CHB6 IL16 S66"


def reactions(tars, subset):
    """GMTKN55 .res lines: (species, stoichiometry, reference kcal/mol)."""
    with tarfile.open(tars / f"{subset}.tar") as t:
        txt = t.extractfile(f"{subset}/.res").read().decode()
    out = []
    for ln in txt.splitlines():
        p = ln.split()
        if not p or not p[0].startswith("tmer"):
            continue
        i = p.index("x")
        species = [x.split("/$")[0] for x in p[1:i]]
        k = len(species)
        coefs = [float(c) for c in p[i + 1:i + 1 + k]]
        w = next(i for i, x in enumerate(p) if x.startswith("$"))
        out.append({"species": species, "coefs": coefs, "ref": float(p[w + 1])})
    return out


BOHR = 0.529177210903


def load_tar(tars, subset):
    """{species: (symbols, xyz in angstrom, charge, multiplicity)} from a GMTKN55 subset tar."""
    out = {}
    files = {}
    with tarfile.open(tars / f"{subset}.tar") as t:
        for m in t.getmembers():
            if m.isfile():
                files[m.name] = t.extractfile(m).read().decode(errors="replace")
    for name, txt in files.items():
        parts = name.split("/")
        if len(parts) != 3 or parts[2] != "coord":
            continue
        sp = parts[1]
        sym, xyz, on = [], [], False
        for ln in txt.splitlines():
            q = ln.strip()
            if q.startswith("$coord"):
                on = True
                continue
            if q.startswith("$"):
                if on:
                    break
                continue
            if on and q:
                v = q.split()
                xyz.append([float(c) * BOHR for c in v[:3]])
                e = v[3]
                sym.append(e[0].upper() + e[1:].lower())
        chg = int(float(files.get(f"{subset}/{sp}/.CHRG", "0").split()[0] or 0)) if files.get(f"{subset}/{sp}/.CHRG", "").strip() else 0
        uhf = int(float(files.get(f"{subset}/{sp}/.UHF", "0").split()[0] or 0)) if files.get(f"{subset}/{sp}/.UHF", "").strip() else 0
        out[sp] = (sym, xyz, chg, uhf + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("binary")
    ap.add_argument("bases")
    ap.add_argument("--image", required=True, help="the container image the pods run")
    ap.add_argument("--gmtkn-tars", default="", help="GMTKN55's <subset>.tar files")
    ap.add_argument("--subsets", default="")
    ap.add_argument("--mols", default="")
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--max-atoms", type=int, default=1000)
    ap.add_argument("--method", default="b3lyp")
    ap.add_argument("--grids", default="gridx:1,gridx:2,gridx:3")
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--mem", default="24Gi")
    a = ap.parse_args()
    run = Path(a.run)
    tars = Path(a.gmtkn_tars)
    (run / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy(a.binary, run / "bin" / "gridk_study")
    if not (run / "src/data/basis").exists():
        shutil.copytree(REPO / "data/basis", run / "src/data/basis")
    shutil.copy(HERE / "study_task.py", run / "study_task.py")
    species = []
    rx = {}
    subsets = [s for s in (PAPER16.split() if a.subsets == "paper16" else a.subsets.split(",")) if s]
    for s in subsets:
        rs = reactions(tars, s)
        geo = load_tar(tars, s)
        names = sorted({n for r in rs for n in r["species"]})
        if any(len(geo[n][0]) > a.max_atoms for n in names):
            # keep only the reactions whose species all fit
            rs = [r for r in rs if all(len(geo[n][0]) <= a.max_atoms for n in r["species"])]
            names = sorted({n for r in rs for n in r["species"]})
        rx[s] = rs
        for n in names:
            sym, xyz, chg, mult = geo[n]
            txt = f"{len(sym)}\n{s} {n}\n" + "".join(
                f"{e} {x:.10f} {y:.10f} {z:.10f}\n" for e, (x, y, z) in zip(sym, xyz))
            species.append({"id": f"{s}.{n}", "xyz": txt, "charge": chg, "mult": mult, "natoms": len(sym)})
    for m in [x for x in a.mols.split(",") if x]:
        src = next(d / f"{m}.xyz" for d in MOLS if (d / f"{m}.xyz").exists())
        txt = src.read_text()
        import re
        mc = re.search(r"charge=(-?\d+)", txt.splitlines()[1])
        species.append({"id": f"mol.{m}", "xyz": txt, "charge": int(mc.group(1)) if mc else 0, "mult": 1,
                        "natoms": int(txt.split()[0])})
    tasks = []
    for b in a.bases.split(","):
        for sp in species:
            tasks.append({"id": __import__("re").sub(r"[^A-Za-z0-9._-]", "_", f"{sp['id']}.{b}"), "species": sp["id"], "xyz": sp["xyz"], "charge": sp["charge"],
                          "mult": sp["mult"], "basis": b, "method": a.method, "grids": a.grids,
                          "grad": a.grad, "natoms": sp["natoms"]})
    # big first, so the shards finish together
    tasks.sort(key=lambda t: -t["natoms"])
    (run / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (run / "reactions.json").write_text(json.dumps(rx))
    (run / "run.yaml").write_text(f"""image: {a.image}
description: "compensated grid-K accuracy study: fixed-density exchange errors, {a.method}, {a.bases}"
command: python3 $RUN_DIR/study_task.py
threads_per_task: 16
cpus_per_pod: 16
memory_per_pod: "{a.mem}"
shards: {a.shards}
task_timeout: 14400
max_attempts: 2
""")
    print(run, len(species), "species", len(tasks), "tasks")


if __name__ == "__main__":
    main()
