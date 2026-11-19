"""countscan.py <out.json> <elements> <gs> <intacc_from> <intacc_to> [extra]: ORCA single-atom point counts.

One-s-primitive basis (alpha 1: no diffuse/steep/core-polarization flags), ! NoIter, default
(adaptive) pruning, GridPrintLevel 3: '# of grid points (after initial pruning)' per (element, g, IntAcc).
IntAcc steps of 1/15 (one radial shell each).
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import radprobe

TMP = radprobe.TMP
ALPHA = float(os.environ.get('PROBE_ALPHA', '1.0'))


def count(sym, g, intacc, extra='', basis_block=None, charge=None, mult=None):
    z = radprobe.ZS[sym]
    nel = 2 if z >= 2 else 1
    if charge is None:
        charge, mult = z - nel, (1 if nel == 2 else 2)
    bb = basis_block if basis_block is not None else f'%basis NewGTO {sym} S 1 1 {ALPHA} 1.0 end end'
    kw = '' if basis_block is not None else ''
    inp = (f'! B3LYP/G NoRI HCore NoIter UKS {kw}\n%method AngularGrid {g} IntAcc {intacc:.6f} GridPrintLevel 3 {extra} end\n'
           f'{bb}\n* xyz {charge} {mult}\n{sym} 0 0 0\n*\n')
    TMP.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP) as d:
        (pathlib.Path(d) / 'a.inp').write_text(inp)
        out = subprocess.run([radprobe.ORCA, 'a.inp'], cwd=d, env=radprobe.ENV, capture_output=True, text=True).stdout
    m = re.search(r'# of grid points \((?:after initial pruning|initial unpruned grid)\)\s+\.\.\.\s+(\d+)', out)
    msgs = [k for k in ('Core-polarized basis detected', 'Diffuse basis detected', 'Steep s-basis detected') if k in out]
    return dict(sym=sym, g=g, intacc=intacc, npts=int(m.group(1)) if m else None, msgs=msgs, extra=extra,
                err=None if m else out[-800:])


if __name__ == '__main__':
    els = sys.argv[2].split(',')
    gs = [int(x) for x in sys.argv[3].split(',')]
    a, b = float(sys.argv[4]), float(sys.argv[5])
    extra = sys.argv[6] if len(sys.argv) > 6 else ''
    n = int(round((b - a) * 15))
    ias = [a + k / 15 for k in range(n + 1)]
    jobs = [(e, g, ia) for e in els for g in gs for ia in ias]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: count(*j, extra), jobs))
    bad = [r for r in res if r['npts'] is None]
    print(len(res), 'runs', len(bad), 'failed', (bad[0]['err'][-300:] if bad else ''))
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
