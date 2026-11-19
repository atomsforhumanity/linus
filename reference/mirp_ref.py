"""Minimal, dependency-free reader for MIRP reference files (*.ref, *.ref.bz2).

See reference/MIRP.md for the format and conventions. Run as a script for a summary:

    python -I reference/mirp_ref.py summary FILE.ref[.bz2] [--max-quartets N]
"""

from __future__ import annotations

import bz2
import math
import sys
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Shell:
    am: int
    nprim: int
    ngen: int
    xyz: tuple[float, float, float]  # bohr
    alpha: list[float]  # nprim exponents
    coeff: list[list[float]]  # [ngen][nprim], raw (un-normalized) coefficients


def ncart(am: int) -> int:
    return (am + 1) * (am + 2) // 2


def cart_lmn(am: int) -> list[tuple[int, int, int]]:
    """Cartesian order used by MIRP (mirp_iterate_gaussian): xx, xy, xz, yy, yz, zz, ..."""
    return [(am - a, a - c, c) for a in range(am + 1) for c in range(a + 1)]


def _open(path: str):
    return bz2.open(path, "rt") if path.endswith(".bz2") else open(path, "rt")


def _tokens(f, header: list[str]) -> Iterator[str]:
    for line in f:
        s = line.strip()
        if s.startswith("#"):
            header.append(s)
            continue
        yield from s.split()


def read_ref(path: str):
    """Return (header_comment_lines, shells, quartet_iterator).

    The iterator yields (p, q, r, s, values); values is a flat list of length
    prod(ngen_i) * prod(ncart_i), laid out [g1][g2][g3][g4][c1][c2][c3][c4] (last fastest).
    The file stays open until the iterator is exhausted.
    """
    f = _open(path)
    header: list[str] = []
    tok = _tokens(f, header)
    nshell = int(next(tok))
    shells = []
    for _ in range(nshell):
        am, nprim, ngen = int(next(tok)), int(next(tok)), int(next(tok))
        xyz = tuple(float.fromhex(next(tok)) for _ in range(3))
        alpha = [float.fromhex(next(tok)) for _ in range(nprim)]
        coeff = [[float.fromhex(next(tok)) for _ in range(nprim)] for _ in range(ngen)]
        shells.append(Shell(am, nprim, ngen, xyz, alpha, coeff))
    size = [ncart(s.am) * s.ngen for s in shells]

    def quartets():
        try:
            for t in tok:
                p, q, r, s = int(t), int(next(tok)), int(next(tok)), int(next(tok))
                n = size[p] * size[q] * size[r] * size[s]
                yield p, q, r, s, [float.fromhex(next(tok)) for _ in range(n)]
        finally:
            f.close()

    return header, shells, quartets()


def normalized_coeff(sh: Shell) -> list[list[float]]:
    """Double-precision version of mirp_normalize_shell: coefficients that multiply
    *unnormalized* primitives x^l y^m z^n exp(-a r^2), such that the x^am component of
    each general contraction has unit norm (all Cartesian components share the factor)."""
    l = sh.am
    m = l + 1.5
    nf = math.pi**1.5
    for i in range(1, l + 1):
        nf *= (2 * i - 1) / 2.0
    out = []
    for c in sh.coeff:
        tot = 0.0
        for ci, ai in zip(c, sh.alpha):
            for cj, aj in zip(c, sh.alpha):
                tot += ci * cj * (ai * aj) ** (m / 2) / (ai + aj) ** m
        k = 1.0 / math.sqrt(tot * nf)
        out.append([ci * ai ** (m / 2) * k for ci, ai in zip(c, sh.alpha)])
    return out


def summary(path: str, max_quartets: int | None = None) -> None:
    header, shells, it = read_ref(path)
    n = len(shells)
    am = [s.am for s in shells]
    nbf_cart = sum(ncart(s.am) * s.ngen for s in shells)
    print(f"file: {path}")
    for h in header[:3]:
        print(f"  {h[:200]}")
    print(f"shells: {n}  max_am: {max(am)}  max_nprim: {max(s.nprim for s in shells)}"
          f"  max_ngen: {max(s.ngen for s in shells)}  nbf(cartesian): {nbf_cart}")
    centers = sorted({s.xyz for s in shells})
    print(f"centers (bohr): {len(centers)}")
    for c in centers:
        print("  " + " ".join(f"{x:+.12f}" for x in c))
    npair = n * (n + 1) // 2
    canon = npair * (npair + 1) // 2
    seen = 0
    nval = nzero = 0
    vmax = 0.0
    vmin = math.inf
    seen_set = set()
    for p, q, r, s, vals in it:
        seen += 1
        seen_set.add((p, q, r, s))
        nval += len(vals)
        for v in vals:
            a = abs(v)
            if a == 0.0:
                nzero += 1
            else:
                vmax = max(vmax, a)
                vmin = min(vmin, a)
        if max_quartets and seen >= max_quartets:
            print(f"(stopped after {seen} quartets)")
            break
    print(f"quartets in file: {seen}  canonical unique (p>=q, r>=s, pq>=rs): {canon}")
    print(f"integrals: {nval}  exact zeros: {nzero}  |nonzero| range: {vmin:.3e} .. {vmax:.3e}")
    if not max_quartets:
        missing = [(p, q, r, s) for p in range(n) for q in range(p + 1)
                   for r in range(n) for s in range(r + 1)
                   if p * (p + 1) // 2 + q >= r * (r + 1) // 2 + s and (p, q, r, s) not in seen_set]
        print(f"canonical quartets absent from file: {len(missing)}  e.g. {missing[:5]}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "summary":
        sys.exit(__doc__)
    mq = int(sys.argv[sys.argv.index("--max-quartets") + 1]) if "--max-quartets" in sys.argv else None
    summary(sys.argv[2], mq)
