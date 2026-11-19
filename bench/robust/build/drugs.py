#!/usr/bin/env python
"""Drug-like molecules from SMILES: PubChem lookup, RDKit 3D, variants, xtb inputs.

    python drugs.py fetch          # drug_names.txt -> drug_smiles.tsv (PubChem PUG REST)
    python drugs.py embed OUTDIR   # drug_smiles.tsv -> OUTDIR/<id>.xyz + OUTDIR/inputs.jsonl (pre-xtb)

Source of the structures: PubChem (NCBI; https://pubchem.ncbi.nlm.nih.gov), looked up by name
(build/drug_names.txt) through PUG REST, isomeric SMILES and CID recorded in drug_smiles.tsv.
PubChem compound records are public-domain facts; the geometries are ours (RDKit ETKDGv3 +
MMFF94, then GFN2-xTB on the cluster, build/xtb_opt.py), so this set may be public.

Variants (each optimized in its own state by GFN2-xTB):
- neutral molecule (largest fragment of a salt);
- "protonated": the most basic aliphatic amine protonated (cation), molecules with one;
- "deprotonated": a carboxylic acid deprotonated (anion), molecules with one;
- "radical": an H atom removed from an sp3 carbon (doublet), every fourth molecule;
- "radical_cation": the neutral molecule's geometry with one electron removed (doublet,
  vertical; no optimization), every sixth molecule.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import safe_id  # noqa: E402

TSV = HERE / "drug_smiles.tsv"
ALLOWED = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se"}
MAX_HEAVY = 40


def fetch() -> int:
    names = [ln.strip() for ln in (HERE / "drug_names.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    done = {}
    if TSV.exists():
        for ln in TSV.read_text().splitlines()[1:]:
            p = ln.split("\t")
            done[p[0]] = p
    out = ["name\tcid\tsmiles"]
    for name in names:
        if name in done:
            out.append("\t".join(done[name]))
            continue
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + urllib.parse.quote(name)
               + "/property/SMILES,IsomericSMILES/JSON")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                props = json.load(r)["PropertyTable"]["Properties"][0]
            smi = props.get("SMILES") or props.get("IsomericSMILES")
            out.append(f"{name}\t{props['CID']}\t{smi}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: {exc}", file=sys.stderr)
        time.sleep(0.25)  # PubChem asks for at most 5 requests per second
    TSV.write_text("\n".join(out) + "\n")
    print(f"{len(out) - 1} of {len(names)} names resolved -> {TSV}")
    return 0


def embed(outdir: Path) -> int:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    outdir.mkdir(parents=True, exist_ok=True)
    rows = [ln.split("\t") for ln in TSV.read_text().splitlines()[1:]]
    chooser = rdMolStandardize.LargestFragmentChooser()
    inputs = []

    def conf3d(mol, seed=0xF00D):
        mh = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed
        cids = AllChem.EmbedMultipleConfs(mh, numConfs=8, params=ps)
        if not cids:
            ps.useRandomCoords = True
            cids = AllChem.EmbedMultipleConfs(mh, numConfs=8, params=ps)
        if not cids:
            return None
        if AllChem.MMFFHasAllMoleculeParams(mh):
            res = AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=2000)
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(mh, maxIters=2000)
        best = min(range(len(res)), key=lambda i: res[i][1])
        return mh, cids[best]

    def emit(mid, mh, cid, charge, mult, variant, name, cnum, smiles, optimize=True):
        pos = mh.GetConformer(cid).GetPositions()
        syms = [a.GetSymbol() for a in mh.GetAtoms()]
        xyz = [str(len(syms)), f"{mid} charge={charge} mult={mult}"]
        xyz += [f"{s:2s} {x:16.10f} {y:16.10f} {z:16.10f}" for s, (x, y, z) in zip(syms, pos)]
        (outdir / f"{mid}.xyz").write_text("\n".join(xyz) + "\n")
        inputs.append({"id": mid, "name": name, "cid": cnum, "smiles": smiles, "variant": variant,
                       "charge": charge, "multiplicity": mult, "optimize": optimize, "xyz": f"{mid}.xyz"})

    amine = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-[SX4]);!$(N#*);!$(N=*);!$(N-N);!$(N-O)]")
    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    csp3h = Chem.MolFromSmarts("[CX4;!H0]")
    for i, (name, cid, smi) in enumerate(rows):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  {name}: SMILES does not parse", file=sys.stderr)
            continue
        mol = chooser.choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        syms = {a.GetSymbol() for a in mol.GetAtoms()}
        if not syms <= ALLOWED or mol.GetNumHeavyAtoms() > MAX_HEAVY or mol.GetNumHeavyAtoms() < 3:
            print(f"  {name}: skipped ({mol.GetNumHeavyAtoms()} heavy atoms, {sorted(syms - ALLOWED)})")
            continue
        if Chem.GetFormalCharge(mol) != 0:
            print(f"  {name}: skipped (net formal charge {Chem.GetFormalCharge(mol)} after neutralizing)")
            continue
        base = safe_id(name.replace(" ", "-"))
        r = conf3d(mol)
        if r is None:
            print(f"  {name}: no 3D embedding", file=sys.stderr)
            continue
        mh, c = r
        csmi = Chem.MolToSmiles(mol)
        emit(f"drug.{base}", mh, c, 0, 1, "neutral", name, cid, csmi)
        if mol.HasSubstructMatch(amine):
            m2 = Chem.RWMol(mol)
            idx = m2.GetSubstructMatch(amine)[0]
            a = m2.GetAtomWithIdx(idx)
            a.SetFormalCharge(1)
            a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
            a.SetNoImplicit(True)
            m2 = m2.GetMol()
            Chem.SanitizeMol(m2)
            r2 = conf3d(m2)
            if r2:
                emit(f"drug.{base}.protonated", *r2, 1, 1, "protonated", name, cid, Chem.MolToSmiles(m2))
        if mol.HasSubstructMatch(acid):
            m3 = Chem.RWMol(mol)
            o = m3.GetSubstructMatch(acid)[2]
            a = m3.GetAtomWithIdx(o)
            a.SetFormalCharge(-1)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(True)
            m3 = m3.GetMol()
            Chem.SanitizeMol(m3)
            r3 = conf3d(m3)
            if r3:
                emit(f"drug.{base}.deprotonated", *r3, -1, 1, "deprotonated", name, cid, Chem.MolToSmiles(m3))
        if i % 4 == 0 and mol.HasSubstructMatch(csp3h):
            # remove one H from the first sp3 carbon carrying one (on the embedded geometry)
            cidx = mh.GetSubstructMatch(csp3h)[0]
            hs = [n.GetIdx() for n in mh.GetAtomWithIdx(cidx).GetNeighbors() if n.GetSymbol() == "H"]
            rw = Chem.RWMol(mh)
            rw.RemoveAtom(hs[0])
            rw.GetAtomWithIdx(cidx).SetNumRadicalElectrons(1)
            rw.GetAtomWithIdx(cidx).SetNoImplicit(True)
            rad = rw.GetMol()
            emit(f"drug.{base}.radical", rad, c, 0, 2, "radical", name, cid, Chem.MolToSmiles(Chem.RemoveHs(rad, sanitize=False)))
        if i % 6 == 0:
            inputs.append({**inputs[[x["id"] for x in inputs].index(f"drug.{base}")], "id": f"drug.{base}.radical_cation",
                           "variant": "radical_cation", "charge": 1, "multiplicity": 2, "optimize": False,
                           "vertical_of": f"drug.{base}"})
    (outdir / "inputs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs))
    from collections import Counter

    print(len(inputs), "inputs", Counter(x["variant"] for x in inputs))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit(fetch() if cmd == "fetch" else embed(Path(sys.argv[2])))
