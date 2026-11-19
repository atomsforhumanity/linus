# ORCA preset data

Parameters of ORCA 6.0.0's grids and models that ORCA does not publish, determined from ORCA
6.0.0's printed output, for linus's ORCA-compatible presets. They are compiled into every build
(`include_str!`); the presets never read files at run time. Included with permission of FAccTs
GmbH (ORCA licence, 2026-11-17): the DefGrid-compatible grids reproduce ORCA 6.0.0's DefGrid
layouts as determined from ORCA's printed output.

| File | Read by | Content | Generator |
|---|---|---|---|
| `grids.json` | `linus_grid::orca` (`orca:N` grids, `GridSpec::orca_nl`) | DefGrid1/2/3: per-element radial xi and shell counts, shell counts by row, pruning-region boundaries, the NL-grid shell counts | `crates/linus-grid/tools/gen_orca_tables.py` from the probes in `crates/linus-grid/tools/orca_probe/` |
| `pcm.json` | `linus_solvent::surface::OrcaPcm` (the `orca` PCM preset) | C-PCM sphere grids and exponent constants xi_N, point density, switching cut, fallback radius | read off ORCA's output and `.cpcm` files (`bench/pcm_orca.py facts`) |
| `solvent_eps.csv` | `linus_solvent::db` | permittivities of ORCA's SMD and CPCM-only tables, CPCM refractive indices | `tools/gen_solvent_db.py --orca` (`tools/gen_solvent_db_orca.py`) |
| `solvent_aliases.csv` | `linus_solvent::db` | solvent names ORCA accepts that no Minnesota, PySCF or common name claims, resolved by running ORCA | as above |

The JSON values are the generators' literals; linus parses them with correctly rounded float
parsing (serde_json `float_roundtrip`), so every value has the bits of the literal.
