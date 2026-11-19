# ROBUST molecule sets: sources and licences

One JSON object per line and molecule (format: `bench/robust/build/common.py`): symbols,
geometry in angstrom, charge, multiplicity, derived tags (size, shell, charge, ECP, metal row),
source and geometry level. 3712 systems in all.

| File | Systems | What | Source | Licence / release |
|---|---|---|---|---|
| `internal/gmtkn55.jsonl` | 2386 | every structure of the 55 GMTKN55 subsets (deduplicated across subsets; aliases kept), 286 open-shell, 309 charged, 81 with ECP elements (Rb-Rn) | L. Goerigk et al., *Phys. Chem. Chem. Phys.* 19, 32184 (2017), doi:10.1039/C7CP04913G; per-subset tar files from the GMTKN55 website (Turbomole `coord`, bohr; `.CHRG`, `.UHF`) | no licence stated: research use with citation; **internal** (excluded from the release cut) |
| `internal/tmqm.jsonl` | 442 | 230 mononuclear transition-metal complexes (all 30 metals, 5-36 atoms, charges -1/0/+1, closed shell) and 212 vertical open-shell variants (cation and anion doublets, triplets) at the same geometry | tmQM 2024 release, D. Balcells, B. B. Skjelstad, *J. Chem. Inf. Model.* 60, 6135 (2020), doi:10.1021/acs.jcim.0c01041; `tmQM/tmQM_X1.xyz.gz` from the authors' repository (GFN2-xTB geometries) | repository licence MIT, structures derived from the Cambridge Structural Database: **internal** until that is settled |
| `drugs.jsonl` | 589 | 287 drug-like molecules (3-40 heavy atoms; H, B, C, N, O, F, Si, P, S, Se, Cl, Br, I) plus 131 protonated (cations), 57 deprotonated (anions), 66 H-abstraction radicals, 48 vertical radical cations | names in `build/drug_names.txt`, structures (CID, SMILES) from PubChem PUG REST (`build/drug_smiles.tsv`); geometries ours: RDKit ETKDGv3 + MMFF94, then GFN2-xTB (xtb 6.7.1, `--opt tight`) | PubChem records are public-domain facts; geometries ours: may be public |
| `inorganic.jsonl` | 212 | 133 built 3d/4d/5d complexes (aqua and ammine ions, halides, carbonyls, oxo anions, metallocenes, cyanides, atoms and diatomics) in common spin states, 79 heavy main-group molecules (Rb-Rn) | ours: ideal polyhedra (`build/inorganic.py`), then GFN2-xTB | ours: may be public |
| `hard.jsonl` | 83 | known-hard SCF cases at fixed idealized geometries: stretched bonds, small gaps (C2, Cr2, Be2, O3, polyacenes up to hexacene, polyenes), symmetric radical cations, multiply charged anions, hypervalent molecules, open-shell atoms | ours (`build/hard.py`) | ours: may be public |

GFN2-xTB geometries: 780 optimized, 21 single atoms; 17 changed their bonding during the
optimization (a proton moving within a protonated amine or an amino acid, water leaving an aqua
ion, CsOH, I2-) and carry the tag `topology_changed`. They are still valid geometries for the SCF
and parity tests; they are not minima of the intended structure.

Rebuild (scratch paths are examples):

```bash
python bench/robust/build/gmtkn55.py <dir with the 55 GMTKN55 tar files>
python bench/robust/build/tmqm.py <tmQM_X1.xyz.gz>
python bench/robust/build/drugs.py fetch                      # PubChem -> drug_smiles.tsv
python bench/robust/build/drugs.py embed <scratch>/drugs       # RDKit (needs rdkit)
python bench/robust/build/inorganic.py <scratch>/inorganic
python bench/robust/build/make_xtb_run.py <rundir> <scratch>/drugs <scratch>/inorganic   # the xtb runs' inputs (cluster tooling)
python bench/robust/build/finish_xtb.py <rundir>/output <scratch>/drugs <scratch>/inorganic
python bench/robust/build/hard.py
```
