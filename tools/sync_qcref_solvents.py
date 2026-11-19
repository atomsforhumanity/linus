#!/usr/bin/env python3
"""Copy the part of qcref's CRC solvent table that linus ships, or check that the copy is current.

    python tools/sync_qcref_solvents.py           # copy (needs `import qcref`)
    python tools/sync_qcref_solvents.py --check   # exit 1 if the copy differs

linus ships the CRC rows of the solvents it names elsewhere (the SMD solvents of
smd_minnesota.csv, the keys of aliases.csv and program_eps.csv) and of the common liquids in
COMMON below: about 200 of qcref's 1303. The rows are copied verbatim and in qcref's order, so
the shipped permittivities are qcref's. The other CRC liquids resolve through qcref's full table
at run time where it is installed (the Python package finds it; the CLI reads
$LINUS_SOLVENT_TABLE). A name that a shipped and a dropped solvent share would resolve
differently with and without that table, so the tool refuses such a set.

Run it with a Python that has qcref (e.g. the reference environment's, .refenv/bin/python), then
update the version/commit lines in crates/linus-solvent/src/data/README.md and run
`cargo test -p linus --release solvent` (with QCREF_PYTHON=<that python> the Rust test
vendored_crc_matches_qcref compares the rows too).
"""
import argparse
import csv
import io
import os
import re
import shutil
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'crates', 'linus-solvent', 'src', 'data')

# Common liquids beyond the SMD solvents and the named ones (CRC keys): carbonates and lactones
# of electrolytes, glycols and glymes, fluorinated alcohols, ethers and esters of process
# chemistry, small acids, liquefied gases used as solvents, phenol.
COMMON = (
    "ACETALDEHYDE", "ACETIC_ANHYDRIDE", "ACRYLONITRILE", "AMMONIA", "DIETHYL_CARBONATE",
    "DIMETHYL_CARBONATE", "ETHYLENE_CARBONATE", "GAMMA_BUTYROLACTONE", "GLYCEROL",
    "DIETHYLENE_GLYCOL", "TRIETHYLENE_GLYCOL", "TETRAETHYLENE_GLYCOL", "PROPANEDIOL_1_2",
    "PROPANEDIOL_1_3", "DIMETHOXYETHANE_1_2", "DIETHYLENE_GLYCOL_DIMETHYL_ETHER",
    "TETRAETHYLENE_GLYCOL_DIMETHYL_ETHER", "METHYLTETRAHYDROFURAN_2", "TRIFLUOROETHANOL_2_2_2",
    "HEXAFLUORO_2_PROPANOL_1_1_1_3_3_3", "TRIFLUOROACETIC_ACID", "PROPANOIC_ACID", "ETHYL_LACTATE",
    "HYDROGEN_FLUORIDE", "SULFUR_DIOXIDE", "HYDRAZINE", "HYDROGEN_PEROXIDE", "HYDROGEN_CYANIDE",
    "METHYLAMINE", "DIMETHYL_ETHER", "PHENOL", "QUINOLINE",
)


def rows(name):
    with open(os.path.join(DATA, name), newline='') as f:
        return list(csv.DictReader(f))


def kept_keys():
    named = set()
    for name in ('smd_minnesota.csv', 'aliases.csv', 'program_eps.csv'):
        named |= {r['key'] for r in rows(name)}
    return named | set(COMMON)


def normalize(name):
    """db.rs's `normalize`: lower-case ASCII alphanumerics."""
    return re.sub(r'[^0-9a-z]', '', name.lower())


def shared_names(crc_rows, keep):
    """Name forms (db.rs's index: keys, CRC and Minnesota names, aliases) that a kept and a dropped
    solvent share."""
    names = {r['key']: [r['key'], r['name']] for r in crc_rows}
    for r in rows('smd_minnesota.csv'):
        names.setdefault(r['key'], [r['key'], r['name_if_not_crc']]).append(r['minnesota_name'])
    for r in rows('aliases.csv'):
        names[r['key']].append(r['alias'])
    claims = {}
    for key, ns in names.items():
        for n in ns:
            if normalize(n):
                claims.setdefault(normalize(n), set()).add(key)
    shipped = keep | {r['key'] for r in rows('smd_minnesota.csv')}
    return {f: sorted(ks) for f, ks in claims.items() if ks & shipped and ks - shipped}


def subset(text, keep):
    """The header and the rows of `text` (qcref's crc.csv) whose key is in `keep`, verbatim."""
    lines = text.splitlines(keepends=True)
    key = next(csv.reader(io.StringIO(lines[0]))).index('key')
    out = [lines[0]] + [ln for ln in lines[1:] if next(csv.reader(io.StringIO(ln)))[key] in keep]
    found = {next(csv.reader(io.StringIO(ln)))[key] for ln in out[1:]}
    missing = sorted(set(COMMON) - found)
    if missing:
        sys.exit(f'COMMON keys not in qcref: {missing}')
    shared = shared_names(list(csv.DictReader(io.StringIO(text))), found)
    if shared:
        sys.exit(f'names shared by a kept and a dropped solvent (keep both): {shared}')
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='compare only; exit 1 on a difference')
    a = ap.parse_args()
    import qcref
    src = os.path.join(os.path.dirname(qcref.__file__), 'data', 'solvents')
    version = getattr(qcref, '__version__', '?')
    with open(os.path.join(src, 'crc.csv'), newline='') as f:
        table = subset(f.read(), kept_keys())
    differ = []
    for name in ('crc.csv', 'crc.meta.json'):
        ours_path = os.path.join(DATA, name)
        ours = open(ours_path, newline='').read() if os.path.exists(ours_path) else None
        theirs = table if name == 'crc.csv' else open(os.path.join(src, name), newline='').read()
        if theirs != ours:
            differ.append(name)
            if not a.check:
                if name == 'crc.csv':
                    with open(ours_path, 'w', newline='') as f:
                        f.write(theirs)
                else:
                    shutil.copyfile(os.path.join(src, name), ours_path)
    n = len(table.splitlines()) - 1
    state = (f'differs: {", ".join(differ)}' if differ else 'vendored copy is current') if a.check else \
        (f'updated {", ".join(differ)}' if differ else 'nothing to update')
    print(f'qcref {version}: {n} of its rows shipped; {state}')
    if a.check:
        sys.exit(1 if differ else 0)


if __name__ == '__main__':
    main()
