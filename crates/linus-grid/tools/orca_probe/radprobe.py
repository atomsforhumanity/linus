"""radprobe.py <out.json> <Sym> <g> <intacc> [alphas]: ORCA radial-grid probe.

One atom, a basis of one s primitive (exponent alpha), two electrons (charge Z-2), RKS B3LYP/G:
the density is spherical, so ORCA's integrated N(Total) = sum_i W_i rho(r_i) depends only on its
radial grid (nodes r_i, weights W_i incl. 4 pi r^2), not on the angular grids or the pruning.
Prints N(Total) for each alpha.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

# ORCA_DIR: the ORCA 6.0.0 directory; REFENV: an environment with Open MPI (reference/README.md)
ORCA_DIR = os.environ.get('ORCA_DIR', os.path.expanduser('~/opt/orca-6.0.0'))
REFENV = os.environ.get('REFENV', str(pathlib.Path(__file__).resolve().parents[4] / '.refenv'))
ORCA = f'{ORCA_DIR}/orca'
ENV = dict(os.environ)
ENV['PATH'] = f'{ORCA_DIR}:{REFENV}/bin:' + ENV['PATH']
ENV['LD_LIBRARY_PATH'] = f'{ORCA_DIR}/lib:{REFENV}/lib'
ENV['OMPI_MCA_hwloc_base_binding_policy'] = 'none'
SYMS = ('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr '
        'Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu '
        'Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn').split()
ZS = {s: i for i, s in enumerate(SYMS) if i}


def ecp_core(z):
    # electrons replaced by ORCA's automatic def2-ECP
    return 0 if z <= 36 else 28 if z <= 54 else 46 if z <= 71 else 60


TMP = pathlib.Path(os.environ.get('PROBE_TMP', tempfile.gettempdir())) / 'orca_probe'


def run(sym, g, intacc, alpha, extra='', grid_kw=''):
    z = ZS[sym]
    nel = 2 if z >= 2 else 1
    charge, mult = z - ecp_core(z) - nel, (1 if nel == 2 else 2)
    meth = f'%method AngularGrid {g} IntAcc {intacc} GridPrintLevel 3 {extra} end' if g else ''
    inp = (f'! B3LYP/G NoRI HCore {"UKS" if nel == 1 else "RKS"} {grid_kw}\n{meth}\n'
           f'%basis NewGTO {sym} S 1 1 {alpha:.10f} 1.0 end end\n'
           f'%scf MaxIter 50 end\n* xyz {charge} {mult}\n{sym} 0 0 0\n*\n')
    TMP.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP) as d:
        (pathlib.Path(d) / 'a.inp').write_text(inp)
        out = subprocess.run([ORCA, 'a.inp'], cwd=d, env=ENV, capture_output=True, text=True).stdout
    n = re.findall(r'N\(Total\)\s+:\s+(\S+)', out)
    npts = re.findall(r'Total number of grid points\s+\.\.\.\s+(\d+)', out)
    msgs = [k for k in ('Core-polarized basis detected', 'Diffuse basis detected', 'Steep s-basis detected') if k in out]
    return dict(sym=sym, z=z, g=g, intacc=intacc, alpha=alpha, nel=nel, n_total=float(n[-1]) if n else None,
                npts=int(npts[0]) if npts else None, msgs=msgs, extra=extra, grid_kw=grid_kw,
                err=None if n else out[-1500:])


ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 300, 1000]

if __name__ == '__main__':
    sym, g, ia = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    alphas = [float(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 and sys.argv[5] else ALPHAS
    extra = sys.argv[6] if len(sys.argv) > 6 else ''
    gkw = sys.argv[7] if len(sys.argv) > 7 else ''
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda a: run(sym, g, ia, a, extra, gkw), alphas))
    for r in res:
        print(r['sym'], r['g'], r['intacc'], r['alpha'], r['n_total'], r['npts'], r['msgs'], (r['err'] or '')[-300:])
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
