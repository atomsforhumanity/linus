"""radcampaign.py <out.json> <elements> <g:intacc,...>: ORCA radial grid (n_r, xi) per element and AngularGrid.

Probes: one s primitive per atom (radprobe.run), 18 exponents 0.01..3000; fit with the Perez-Jorda
Gauss-Chebyshev + Treutler-Ahlrichs M3 model (radfit 'pjm3'), probes flagged 'Steep s-basis detected'
excluded (ORCA doubles the radial grid for them).
"""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import radfit
import radprobe

ALPHAS = [0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.15, 30, 100, 200, 400, 700, 1000, 1500, 2000, 3000]


def fit(data):
    best = []
    for n in range(8, 160):
        xi, s = radfit.fit_xi(n, data, 'pjm3')
        best.append((s, n, xi))
    best.sort()
    s, n, xi = best[0]
    s2 = best[1]
    return n, xi, math.sqrt(s / len(data)), math.sqrt(s2[0] / len(data)), s2[1]


if __name__ == '__main__':
    out = sys.argv[1]
    els = sys.argv[2].split(',')
    settings = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in sys.argv[3].split(',')]
    extra = 'BFCut 1e-30 ' + (sys.argv[4] if len(sys.argv) > 4 else '')
    jobs = [(e, g, ia, a) for e in els for g, ia in settings for a in ALPHAS]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: radprobe.run(j[0], j[1], j[2], j[3], extra), jobs))
    table = []
    for e in els:
        for g, ia in settings:
            d = [r for r in res if r['sym'] == e and r['g'] == g and r['intacc'] == ia and r['n_total'] is not None]
            steep = [r['alpha'] for r in d if 'Steep s-basis detected' in r['msgs']]
            diffuse = [r['alpha'] for r in d if 'Diffuse basis detected' in r['msgs']]
            use = [r for r in d if 'Steep s-basis detected' not in r['msgs']]
            if len(use) < 4:
                print(json.dumps(dict(sym=e, g=g, intacc=ia, error='too few probes', nprobes=len(use))), flush=True)
                continue
            n, xi, rms, rms2, n2 = fit(use)
            row = dict(sym=e, z=radprobe.ZS[e], g=g, intacc=ia, n_r=n, xi=xi, rms=rms, next_rms=rms2, next_n=n2,
                       nprobes=len(use), steep_from=min(steep) if steep else None, diffuse_upto=max(diffuse) if diffuse else None,
                       npts=sorted({r['npts'] for r in use}))
            table.append(row)
            print(json.dumps(row), flush=True)
    json.dump(dict(table=table, probes=res), open(out, 'w'), indent=1)
