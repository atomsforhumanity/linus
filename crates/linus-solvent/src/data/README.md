# Solvent data

Read by `crates/linus-solvent/src/db.rs` (compiled in with `include_str!`).

| File | Content | Source | License |
|---|---|---|---|
| `crc.csv`, `crc.meta.json` | 207 of qcref's 1303 liquids (the SMD solvents, the keys of `aliases.csv` and `program_eps.csv`, and the common liquids listed in `tools/sync_qcref_solvents.py`): tabulated static dielectric constant, its temperature, and the polynomial fit epsilon(T) = a + bT + cT² + dT³ with its range | Rows copied verbatim, in qcref's order (the metadata as is), from qcref v0.5.1 (commit f2b9e44, `src/qcref/data/solvents/`), AFH's reference-data package. qcref extracted it from C. Wohlfarth, "Permittivity (Dielectric Constant) of Liquids", CRC Handbook of Chemistry and Physics, 106th ed., CRC Press. | qcref: MIT. The numbers are factual data from the CRC Handbook; cite the Handbook. |
| `smd_minnesota.csv` | 179 SMD solvents (incl. water): n, n25, alpha, beta, gamma, epsilon, phi, psi as printed, keyed to the CRC keys (five solvents the CRC table lacks get linus keys and names) | Minnesota Solvent Descriptor Database, P. Winget, D. M. Dolney, D. J. Giesen, C. J. Cramer, D. G. Truhlar, https://comp.chem.umn.edu/solvation/mnsddb.pdf, version dated 1999, updated 2010-06-06, 2021-05-22 (DMSO n) and 2026-01-06 (water). The SMD paper (Marenich, Cramer, Truhlar, J. Phys. Chem. B 113, 6378 (2009)) defines SMD with these descriptors. | Published scientific data (University of Minnesota); cite the database and the SMD paper. |
| `program_eps.csv` | Epsilon as PySCF 2.15 uses it, for the preset that reproduces PySCF | PySCF 2.15 `pyscf/solvent/_solvent_data.py` (Apache-2.0) | PySCF: Apache-2.0. |
| `aliases.csv` | Name variants: Minnesota names, PySCF names and abbreviations, common abbreviations and formulas | as listed in the `source` column | — |

The other CRC liquids resolve at run time through qcref's full table, with the same rules: the
Python package registers qcref's `data/solvents/crc.csv` at import where qcref is installed, and
`$LINUS_SOLVENT_TABLE` names a table for the CLI (or overrides it).

Regenerate:
- `crc.*`: `tools/sync_qcref_solvents.py` (needs `import qcref`), `--check` to compare.
- the rest: `tools/gen_solvent_db.py` (inputs described in its docstring).

ORCA 6.0.0's own solvent tables (its permittivities, CPCM refractive indices and the names it
accepts, which ORCA does not publish) are `data/orca/solvent_eps.csv` and
`data/orca/solvent_aliases.csv` of the source tree (`data/orca/README.md`), also compiled in;
`tools/gen_solvent_db_orca.py` regenerates them.

Default epsilon (db.rs): the CRC tabulated value if it was measured at exactly 298.15 K; else
the CRC fit at 298.15 K when 298.15 K is within the fit range widened by 30 K and the value is
>= 1; else the tabulated value at its temperature; else (five SMD solvents not in the CRC table)
Minnesota. Other programs' values are explicit overrides
(`EpsConvention`), never the default.
