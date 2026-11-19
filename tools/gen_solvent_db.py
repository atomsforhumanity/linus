#!/usr/bin/env python3
"""Generate linus's SMD solvent tables (crates/linus-solvent/src/data/) from their sources.

Inputs (see crates/linus-solvent/src/data/README.md for provenance):
  --minnesota  JSON parsed from the Minnesota Solvent Descriptor Database PDF
               (https://comp.chem.umn.edu/solvation/mnsddb.pdf, version of 2026-01-06):
               {name: {n, n25, alpha, beta, gamma, eps, phi, psi, raw: {...}}}
  PySCF 2.15 (pyscf.solvent._solvent_data) is imported for its names and epsilon.
  The CRC table (crc.csv, vendored from qcref) gives the keys.
  --orca, --orca-alias (optional): ORCA 6.0.0's solvent table and names, written by
               tools/gen_solvent_db_orca.py into data/orca/solvent_eps.csv and
               solvent_aliases.csv.

Outputs: smd_minnesota.csv, program_eps.csv (PySCF's epsilon), aliases.csv (Minnesota, PySCF and
common names). The generated CSVs are committed; this script documents how they were made and
regenerates them.
"""
import argparse
import csv
import json
import os
import re

DATA = os.path.join(os.path.dirname(__file__), '..', 'crates', 'linus-solvent', 'src', 'data')

# Minnesota name -> CRC (qcref) key where the names differ
MANUAL_CRC = {
    'a-chlorotoluene': 'CHLOROMETHYL_BENZENE', 'benzyl alcohol': 'BENZYL_ALCOHOL', 'butyl ethanoate': 'BUTYL_ACETATE',
    'dibutyl ether': 'DIBUTYL_ETHER', 'dimethyl sulfoxide (DMSO)': 'DIMETHYL_SULFOXIDE', 'ethyl ethanoate': 'ETHYL_ACETATE',
    'ethyl methanoate': 'ETHYL_FORMATE', 'methyl ethanoate': 'METHYL_ACETATE', 'methyl methanoate': 'METHYL_FORMATE',
    'pentyl ethanoate': 'PENTYL_ACETATE', 'propyl ethanoate': 'PROPYL_ACETATE',
    'N-methylformamide (E/Z mixture)': 'N_METHYLFORMAMIDE', 'tributylphosphate': 'TRIBUTYL_PHOSPHATE',
    'ethyl phenyl ether': 'ETHOXYBENZENE', 'diphenyl ether': 'DIPHENYL_ETHER', 'carbon disulfide': 'CARBON_DISULFIDE',
    'carbon tetrachloride': 'TETRACHLOROMETHANE', 'chloroform': 'TRICHLOROMETHANE', 'dichloromethane': 'DICHLOROMETHANE',
    'bromoform': 'TRIBROMOMETHANE', 'iodomethane': 'IODOMETHANE', 'mesitylene': 'TRIMETHYLBENZENE_1_3_5',
    'isopropylbenzene': 'ISOPROPYLBENZENE', 'p-isopropyltoluene': 'ISOPROPYL_4_METHYLBENZENE_1',
    'perfluorobenzene': 'HEXAFLUOROBENZENE', 'tetralin': 'TETRAHYDRONAPHTHALENE_1_2_3_4',
    'cis-decalin': 'CIS_DECAHYDRONAPHTHALENE', 'trans-decalin': 'TRANS_DECAHYDRONAPHTHALENE', 'o-cresol': 'O_CRESOL',
    'm-cresol': 'M_CRESOL', 'thiophenol': 'BENZENETHIOL', 'formic acid': 'FORMIC_ACID', 'acetic acid': 'ACETIC_ACID',
    'butanal': 'BUTANAL', 'butanone': 'BUTANONE_2', 'butanonitrile': 'BUTANENITRILE', 'propanonitrile': 'PROPANENITRILE',
    'butanoic acid': 'BUTANOIC_ACID', 'pentanoic acid': 'PENTANOIC_ACID', 'hexanoic acid': 'HEXANOIC_ACID',
    'propanoic acid': 'PROPANOIC_ACID', 'methyl benzoate': 'METHYL_BENZOATE', 'methyl butanoate': 'METHYL_BUTANOATE',
    'methyl propanoate': 'METHYL_PROPANOATE', 'nitrobenzene': 'NITROBENZENE', 'o-nitrotoluene': 'NITROTOLUENE_2',
    'o-dichlorobenzene': 'O_DICHLOROBENZENE', 'o-chlorotoluene': 'CHLOROTOLUENE_2', 'diethyl ether': 'DIETHYL_ETHER',
    'diethyl sulfide': 'DIETHYL_SULFIDE', 'dimethyl disulfide': 'DIMETHYL_DISULFIDE', 'diisopropyl ether': 'DIISOPROPYL_ETHER',
    'ethanethiol': 'ETHANETHIOL', '2-propen-1-ol': 'ALLYL_ALCOHOL', '1-hexyne': 'HEXYNE_1', 'water': 'WATER',
}
# SMD solvents that are not in the CRC table: linus keys
NON_CRC = {
    'E-2-pentene': ('E_2_PENTENE', 'E-2-Pentene'),
    'decalin (cis/trans mixture)': ('DECALIN', 'Decalin (cis/trans mixture)'),
    'cis-1,2-dimethylcyclohexane': ('CIS_1_2_DIMETHYLCYCLOHEXANE', 'cis-1,2-Dimethylcyclohexane'),
    'tetrahydrothiophene-S,S-dioxide': ('SULFOLANE', 'Sulfolane (tetrahydrothiophene 1,1-dioxide)'),
    'xylene (mixture)': ('XYLENE', 'Xylene (mixture)'),
}
# Common names and abbreviations (lower case) -> key
COMMON = {
    'h2o': 'WATER', 'thf': 'TETRAHYDROFURAN', 'dmso': 'DIMETHYL_SULFOXIDE', 'dmf': 'N_N_DIMETHYLFORMAMIDE',
    'dma': 'N_N_DIMETHYLACETAMIDE', 'dmac': 'N_N_DIMETHYLACETAMIDE', 'dcm': 'DICHLOROMETHANE',
    'methylene chloride': 'DICHLOROMETHANE', 'ccl4': 'TETRACHLOROMETHANE', 'chcl3': 'TRICHLOROMETHANE',
    'mecn': 'ACETONITRILE', 'acn': 'ACETONITRILE', 'meoh': 'METHANOL', 'etoh': 'ETHANOL', 'ipa': 'PROPANOL_2',
    'isopropyl alcohol': 'PROPANOL_2', 'et2o': 'DIETHYL_ETHER', 'ether': 'DIETHYL_ETHER', 'etoac': 'ETHYL_ACETATE',
    'ethyl acetate': 'ETHYL_ACETATE', 'dce': 'DICHLOROETHANE_1_2', 'dioxane': 'DIOXANE_1_4', 'nmp': 'N_METHYL_2_PYRROLIDINONE',
    '2-methf': 'METHYLTETRAHYDROFURAN_2', '2-methyltetrahydrofuran': 'METHYLTETRAHYDROFURAN_2', 'dme': 'DIMETHOXYETHANE_1_2',
    'hmpa': 'HEXAMETHYLPHOSPHORIC_TRIAMIDE', 'pc': 'PROPYLENE_CARBONATE', 'meno2': 'NITROMETHANE', 'phme': 'TOLUENE',
    'phcl': 'CHLOROBENZENE', 'phno2': 'NITROBENZENE', 'tfe': 'TRIFLUOROETHANOL_2_2_2', 'sulfolane': 'SULFOLANE',
    'xylene': 'XYLENE', 'xylenes': 'XYLENE', 'decalin': 'DECALIN', 'ethylene glycol': 'ETHANEDIOL_1_2',
    'acetic acid': 'ACETIC_ACID', 'isopropanol': 'PROPANOL_2', 'cyclohexane': 'CYCLOHEXANE',
    'ch2cl2': 'DICHLOROMETHANE', 'ch3cn': 'ACETONITRILE',
}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minnesota', required=True)
    ap.add_argument('--orca', help='ORCA 6.0.0 runs (tools/gen_solvent_db_orca.py)')
    ap.add_argument('--orca-alias', help='ORCA name resolutions')
    a = ap.parse_args()
    from pyscf.solvent import _solvent_data as psd

    crc_rows = list(csv.DictReader(open(os.path.join(DATA, 'crc.csv'), encoding='utf-8')))
    crc = {r['key']: r for r in crc_rows}
    by_name = {}
    for r in crc_rows:
        by_name.setdefault(norm(r['name']), []).append(r['key'])
    mn = json.load(open(a.minnesota))

    # Minnesota -> key
    mn_key = {}
    for name in mn:
        if name in NON_CRC:
            mn_key[name] = NON_CRC[name][0]
            continue
        key = MANUAL_CRC.get(name)
        if key is None:
            cands = by_name.get(norm(name), [])
            if not cands and name.startswith('n-'):
                cands = by_name.get(norm(name[2:]), [])
            if not cands:
                m = re.match(r'^([0-9,]+)-(.*)$', name)
                if m:
                    k = re.sub(r'[^A-Z0-9]+', '_', m.group(2).upper()).strip('_') + '_' + m.group(1).replace(',', '_')
                    cands = [k] if k in crc else []
            if not cands:
                k = re.sub(r'[^A-Z0-9]+', '_', name.upper()).strip('_')
                cands = [k] if k in crc else []
            assert len(cands) == 1, (name, cands)
            key = cands[0]
        assert key in crc, (name, key)
        mn_key[name] = key

    # PySCF names -> Minnesota names
    mnn = {norm(k): k for k in mn}
    pyscf_manual = {'benzylalcohol': 'benzyl alcohol', 'butylethanoate': 'butyl ethanoate', 'butylether': 'dibutyl ether',
                    'diethylether': 'diethyl ether', 'diethylsulfide': 'diethyl sulfide',
                    'dimethyldisulfide': 'dimethyl disulfide', 'dimethylsulfoxide': 'dimethyl sulfoxide (DMSO)',
                    'diphenylether': 'diphenyl ether', 'ethylethanoate': 'ethyl ethanoate',
                    'ethylmethanoate': 'ethyl methanoate', 'ethylphenylether': 'ethyl phenyl ether',
                    'formicacid': 'formic acid', 'hexanoicacid': 'hexanoic acid', 'methylbenzoate': 'methyl benzoate',
                    'methylbutanoate': 'methyl butanoate', 'methylethanoate': 'methyl ethanoate',
                    'methylmethanoate': 'methyl methanoate', 'methylpropanoate': 'methyl propanoate'}
    py_key = {}
    for name in psd.solvent_db:
        if not name:
            continue
        m = pyscf_manual.get(name) or mnn.get(norm(name))
        assert m, name
        py_key[name] = mn_key[m]
    for abbr, name in psd._alias.items():
        py_key[abbr] = py_key[name]

    # smd_minnesota.csv
    with open(os.path.join(DATA, 'smd_minnesota.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['key', 'minnesota_name', 'name_if_not_crc', 'n', 'n25', 'alpha', 'beta', 'gamma', 'eps', 'phi', 'psi'])
        for name in sorted(mn, key=lambda s: mn_key[s]):
            raw = mn[name]['raw']
            w.writerow([mn_key[name], name, NON_CRC.get(name, (None, ''))[1], raw['n'], raw['n25'] or '', raw['alpha'],
                        raw['beta'], raw['gamma'], raw['eps'], raw['phi'], raw['psi']])

    # program_eps.csv: epsilon as PySCF uses it (ORCA's: data/orca/, below)
    py_eps = {}
    for name, v in psd.solvent_db.items():
        if name:
            py_eps[py_key[name]] = v[5]
    keys = sorted(py_eps)
    with open(os.path.join(DATA, 'program_eps.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['key', 'pyscf_2_15'])
        for k in keys:
            w.writerow([k, repr(py_eps[k])])

    # aliases.csv: alias -> key, with where the alias comes from
    aliases = {}

    def add(alias, key, src):
        n = norm(alias)
        if not n:
            return
        prev = aliases.get(n)
        if prev and prev[1] != key:
            raise SystemExit(f'alias conflict {alias!r}: {prev} vs {key} ({src})')
        if not prev:
            aliases[n] = (alias, key, src)

    for name, key in mn_key.items():
        add(name, key, 'minnesota')
    for name, key in py_key.items():
        add(name, key, 'pyscf')
    for name, key in COMMON.items():
        assert key in crc or key in [v[0] for v in NON_CRC.values()], key
        add(name, key, 'common')
    # aliases must not shadow a different CRC solvent's name or key
    for r in crc_rows:
        for s in (r['name'], r['key']):
            n = norm(s)
            if n in aliases and aliases[n][1] != r['key']:
                raise SystemExit(f'alias {aliases[n]} shadows CRC {r["key"]} ({s})')
    with open(os.path.join(DATA, 'aliases.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['alias', 'key', 'source'])
        for n in sorted(aliases, key=lambda n: (aliases[n][1], n)):
            w.writerow(aliases[n])
    print(f'{len(mn_key)} SMD solvents, {len(keys)} program rows, {len(aliases)} aliases')
    if a.orca:
        # ORCA 6.0.0's table and names (only names no Minnesota, PySCF or common name claims)
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gen_solvent_db_orca
        gen_solvent_db_orca.write(a, mn, mn_key, mnn, aliases, norm)


if __name__ == '__main__':
    main()
