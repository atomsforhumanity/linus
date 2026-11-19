"""ORCA 6.0.0's solvent table and names (data/orca/). Called by tools/gen_solvent_db.py --orca
--orca-alias:

  --orca       JSON of ORCA 6.0.0 runs, one per solvent name (SMD and CPCM-only), with the
               epsilon/refractive index/SMD descriptors ORCA printed
  --orca-alias JSON of every solvent-like string in ORCA's library resolved through ORCA

Writes data/orca/solvent_eps.csv (key, ORCA's SMD epsilon, CPCM epsilon and CPCM refractive
index) and data/orca/solvent_aliases.csv (the ORCA names that no Minnesota, PySCF or common name
claims), both compiled into linus_solvent::db.
"""
import csv
import json
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'orca')


def write(a, mn, mn_key, mnn, aliases, norm):
    """`aliases`: the public {normalized: (alias, key, source)} of gen_solvent_db.py."""
    # ORCA: canonical SMD names (resolved through ORCA) -> Minnesota -> key
    orca_manual = {'DECALIN': 'decalin (cis/trans mixture)', 'DIBUTYLETHER': 'dibutyl ether',
                   'DIMETHYLSULFOXIDE': 'dimethyl sulfoxide (DMSO)', 'DIPHENYLETHER': 'diphenyl ether',
                   'ETHYL ACETATE': 'ethyl ethanoate', 'N-METHYLFORMAMIDE': 'N-methylformamide (E/Z mixture)',
                   'XYLENE': 'xylene (mixture)'}
    orca_runs = json.load(open(a.orca))['runs']
    alias_runs = json.load(open(a.orca_alias))
    o_smd = {r['name']: r for r in orca_runs if r['mode'] == 'smd' and not r.get('error')}
    o_cpcm = {r['name']: r for r in orca_runs if r['mode'] == 'cpcm' and not r.get('error')}
    canon_key = {}
    for oname in list(o_smd) + ['ETHYL ACETATE']:
        m = orca_manual.get(oname) or mnn.get(norm(oname))
        assert m, oname
        canon_key[oname] = mn_key[m]
    orca_key = dict(canon_key)
    orca_cpcm_eps = {}
    for r in alias_runs:
        if r['mode'] == 'smd' and r['solvent']:
            orca_key[r['string']] = canon_key[r['solvent']]
    # CPCM-only ORCA names
    for oname, key in (('AMMONIA', 'AMMONIA'), ('PHENOL', 'PHENOL')):
        orca_key[oname] = key
    for r in alias_runs:
        if r['mode'] == 'cpcm' and r['eps'] and r['string'] in orca_key and r['eps'] != 'INFINITY':
            orca_cpcm_eps.setdefault(orca_key[r['string']], (float(r['eps']), float(r['refrac'])))
    orca_smd_eps = {canon_key[o]: r['eps'] for o, r in o_smd.items()}
    # ORCA rejects 'ETHYL ACETATE' itself but resolves 'ETHYL ETHANOATE' to it
    ea = [r for r in alias_runs if r['mode'] == 'smd' and r['solvent'] == 'ETHYL ACETATE']
    assert ea
    orca_smd_eps['ETHYL_ACETATE'] = mn['ethyl ethanoate']['eps']

    os.makedirs(OUT, exist_ok=True)
    keys = sorted(set(orca_smd_eps) | set(orca_cpcm_eps))
    with open(os.path.join(OUT, 'solvent_eps.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['key', 'orca_6_0_0_smd', 'orca_6_0_0_cpcm', 'orca_6_0_0_cpcm_refrac'])
        for k in keys:
            c = orca_cpcm_eps.get(k)
            w.writerow([k, orca_smd_eps.get(k, ''), c[0] if c else '', c[1] if c else ''])
    rows = {}
    for name, key in orca_key.items():
        n = norm(name)
        if n and n not in aliases and n not in rows:
            rows[n] = (name, key, 'orca')
    with open(os.path.join(OUT, 'solvent_aliases.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['alias', 'key', 'source'])
        for n in sorted(rows, key=lambda n: (rows[n][1], n)):
            w.writerow(rows[n])
    print(f'ORCA: {len(keys)} program rows, {len(rows)} aliases (data/orca)')
