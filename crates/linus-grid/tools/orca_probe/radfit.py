"""radfit.py <probe.json> [scheme]: fit ORCA's radial grid (n_r, xi) to N(Total) of one-s-primitive probes.

schemes: gc2m3 (Gauss-Chebyshev 2nd kind, Treutler-Ahlrichs M3), gc1m3 (1st kind), gc2m4 (M4, alpha 0.6)
Uses the probes without the steep-s doubling. Prints the best xi per n_r and the best overall.
"""
import json
import math
import sys

import numpy as np


def grid(n, xi, scheme='gc2m3'):
    i = np.arange(1, n + 1)
    if scheme.startswith('pj'):
        # Perez-Jorda et al. modified Gauss-Chebyshev (2nd kind), Krack-Koster
        t = i * math.pi / (n + 1)
        s_ = np.sin(t)
        x = (n + 1 - 2 * i) / (n + 1) + 2 / math.pi * (1 + 2 / 3 * s_ ** 2) * np.cos(t) * s_
        wx = 16 / (3 * (n + 1)) * s_ ** 4
    elif scheme.startswith('gc2'):
        t = i * math.pi / (n + 1)
        x, wx = np.cos(t), math.pi / (n + 1) * np.sin(t)
    else:
        t = (2 * i - 1) * math.pi / (2 * n)
        x, wx = np.cos(t), math.pi / n * np.sin(t)
    c = xi / math.log(2)
    l = np.log(2 / (1 - x))
    if scheme.endswith('m3'):
        r, dr = c * l, c / (1 - x)
    else:
        a = 0.6
        p = (1 + x) ** a
        r, dr = c * p * l, c * (a * p / (1 + x) * l + p / (1 - x))
    return r, 4 * math.pi * r * r * dr * wx


def nint(r, w, alpha, nel=2):
    rho = nel * (2 * alpha / math.pi) ** 1.5 * np.exp(-2 * alpha * r * r)
    return float(np.sum(w * rho))


def resid(n, xi, data, scheme):
    r, w = grid(n, xi, scheme)
    return np.array([nint(r, w, d['alpha'], d['nel']) - d['n_total'] for d in data])


def fit_xi(n, data, scheme):
    lo, hi = math.log(0.2), math.log(8.0)
    f = lambda lx: float(np.sum(resid(n, math.exp(lx), data, scheme) ** 2))
    # coarse scan then golden section
    xs = np.linspace(lo, hi, 400)
    v = [f(x) for x in xs]
    k = int(np.argmin(v))
    a, b = xs[max(k - 1, 0)], xs[min(k + 1, len(xs) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(100):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    x = (a + b) / 2
    return math.exp(x), f(x)


if __name__ == '__main__':
    data = [d for d in json.load(open(sys.argv[1])) if d['n_total'] is not None and 'Steep s-basis detected' not in d['msgs']]
    scheme = sys.argv[2] if len(sys.argv) > 2 else 'gc2m3'
    nmin = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    best = []
    for n in range(nmin, nmax + 1):
        xi, s = fit_xi(n, data, scheme)
        best.append((s, n, xi))
    best.sort()
    for s, n, xi in best[:5]:
        r = resid(n, xi, data, scheme)
        print(f'{scheme} n_r={n} xi={xi:.6f} rms={math.sqrt(s / len(data)):.2e} max={np.abs(r).max():.2e}')
    s, n, xi = best[0]
    for d, e in zip(data, resid(n, xi, data, scheme)):
        print(f"   alpha={d['alpha']:<10g} N_orca={d['n_total']:.12f} resid={e:+.2e}")
