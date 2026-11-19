"""The radial tables of the orca:N grids from ORCA 6.0.0 radial-grid measurements
(tools/orca_probe/radcampaign.py output): the "radial_xi" and "radial_shells" entries of
data/orca/grids.json (until 2026-11-08 this wrote src/orca_data.rs).

usage: python gen_orca_tables.py <campaign.json> [<campaign.json> ...]   # paste into grids.json

Each campaign row is one (element, AngularGrid, IntAcc) with the fitted number of radial shells n_r
and scale xi of ORCA 6.0.0's "OptM3 with GC (2021)" radial grid; a row is accepted only when the fit
reproduces every probe to < 1e-10 electrons and the next-best n_r is worse by > 100x. Only the DefGrid
settings (AngularGrid 3/4/6 with IntAcc 4.159/4.388/4.959) are tabulated, and only H-Kr: from Rb on
ORCA puts a def2-ECP on the probe atom, whose valence-only density fixes n_r and xi only ambiguously.
"""
import json
import sys

DEFGRID = {3: 4.159, 4: 4.388, 6: 4.959}
ZMAX = 36


def main():
    rows = []
    for f in sys.argv[1:]:
        d = json.load(open(f))
        rows += d['table'] if isinstance(d, dict) else d
    xi = [[0.0] * 3 for _ in range(ZMAX + 1)]
    nr = [[0] * 3 for _ in range(ZMAX + 1)]
    for r in rows:
        if 'error' in r or r['z'] > ZMAX or r['g'] not in DEFGRID or abs(r['intacc'] - DEFGRID[r['g']]) > 1e-9:
            continue
        if r['rms'] > 1e-10 or r['next_rms'] < 100 * r['rms']:
            print(f'// rejected {r["sym"]} g={r["g"]}: rms {r["rms"]:.1e}, next {r["next_rms"]:.1e}', file=sys.stderr)
            continue
        k = [3, 4, 6].index(r['g'])
        xi[r['z']][k] = r['xi']
        nr[r['z']][k] = r['n_r']
    # the "radial_xi" and "radial_shells" entries of data/orca/grids.json (12 decimals,
    # as written there; linus-grid parses them with correctly rounded float parsing)
    rows_xi = ',\n'.join('      [' + ', '.join(f'{v:.12f}' for v in xi[z]) + ']' for z in range(ZMAX + 1))
    rows_nr = ',\n'.join('      [' + ', '.join(str(v) for v in nr[z]) + ']' for z in range(ZMAX + 1))
    print('"radial_xi": {"angular_grids": [3, 4, 6], "by_z": [\n' + rows_xi + '\n    ]},')
    print('"radial_shells": {"defgrids": [1, 2, 3], "by_z": [\n' + rows_nr + '\n    ]},')

main()
