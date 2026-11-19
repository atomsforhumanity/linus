"""Layer (a) of the integral generator: the McMurchie-Davidson mathematics, in exact arithmetic.

Nothing here knows about a target language, a loop order or where a value lives. It derives
- Hermite and Cartesian index sets (`herm_tuv`, `herm_idx`, `cart_components`), in linus-ints'
  orders;
- the real solid harmonics over Cartesian monomials (`solid_harmonic_coefs`, exact rationals,
  mirroring core/solid_harmonics.rs) and their normalized coefficients (`sph_coefs`, exact up to a
  final correctly rounded square root);
- the structural nonzero patterns of the SHARK E-matrices E[r][ab] of a spherical pair class
  (`e_pattern`) and of its first derivatives with respect to either centre (`e_pattern_deriv`);
- the Hermite Coulomb recursion R^(n)_{t+1_d} = X_d R^(n+1)_t + t_d R^(n+1)_{t-1_d} as a set of
  rules per level (`r_recursion`), for a chosen direction rule.

The IR (ir.py) turns these into expression graphs; the emitters (rtensor_gen.py, classes_gen.py,
tiled_gen.py, cuda_eri3c.py) print them under a schedule.
"""
from decimal import Decimal, getcontext
from fractions import Fraction
from math import comb

# ---------------------------------------------------------------- Hermite / Cartesian indices


def nherm(l):
    """Hermite functions (t, u, v) with t + u + v <= l."""
    return (l + 1) * (l + 2) * (l + 3) // 6


def herm_idx(t, u, v):
    n = t + u + v
    return n * (n + 1) * (n + 2) // 6 + (n - t) * (n - t + 1) // 2 + (n - t - u)


def herm_tuv(l):
    """Hermite indices of degree <= l in linus-ints' order (degree, then t, u descending)."""
    out = []
    for n in range(l + 1):
        for t in range(n, -1, -1):
            for u in range(n - t, -1, -1):
                out.append((t, u, n - t - u))
    for i, x in enumerate(out):
        assert herm_idx(*x) == i
    return out


def cart_components(l):
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def ncart(l):
    return (l + 1) * (l + 2) // 2


def nfunc(l):
    """Functions of a shell in linus's convention: s and p Cartesian, l >= 2 spherical."""
    return 1 if l == 0 else (3 if l == 1 else 2 * l + 1)


# ---------------------------------------------------------------- solid harmonics


def solid_harmonic_coefs(l, m):
    """Exact (unnormalized) coefficients of S_lm over Cartesian monomials (core/solid_harmonics.rs)."""
    comps = cart_components(l)
    c = [Fraction(0)] * len(comps)
    am = abs(m)
    vm2 = 1 if m < 0 else 0
    for t in range((l - am) // 2 + 1):
        for u in range(t + 1):
            vmax2 = 2 * ((am - vm2) // 2) + vm2
            v2 = vm2
            while v2 <= vmax2:
                sign = 1 if (t + (v2 - vm2) // 2) % 2 == 0 else -1
                coef = sign * Fraction(1, 4 ** t) * comb(l, t) * (comb(l - t, am + t) if am + t <= l - t else 0) \
                    * comb(t, u) * comb(am, v2)
                px, py, pz = 2 * t + am - 2 * u - v2, 2 * u + v2, l - 2 * t - am
                if px >= 0 and py >= 0 and pz >= 0:
                    c[comps.index((px, py, pz))] += coef
                v2 += 2
    return c


def _dfact(n):
    """Double factorial, (-1)!! = 1."""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def cart_metric(l):
    """Overlap of the Cartesian components (x^l normalization), exact (core/solid_harmonics.rs)."""
    comps = cart_components(l)
    norm = _dfact(2 * l - 1)
    m = []
    for a in comps:
        row = []
        for b in comps:
            s = [a[k] + b[k] for k in range(3)]
            if any(x % 2 for x in s):
                row.append(Fraction(0))
            else:
                v = 1
                for x in s:
                    v *= _dfact(x - 1)
                row.append(Fraction(v, norm))
        m.append(row)
    return m


def sph_coefs(l):
    """Normalized real solid harmonics as rows over the Cartesian components: rows m = -l..l for
    l >= 2, the identity (x, y, z) for l <= 1, as linus-ints' `cart2sph`. The norm c^T M c is an
    exact rational; each coefficient c_i / sqrt(N) is rounded once, correctly, to a double (cart2sph
    computes it in floating point, so the two agree to an ulp or two)."""
    nc = ncart(l)
    if l <= 1:
        return [[1.0 if i == j else 0.0 for j in range(nc)] for i in range(nc)]
    getcontext().prec = 60
    met = cart_metric(l)
    rows = []
    for m in range(-l, l + 1):
        c = solid_harmonic_coefs(l, m)
        n2 = sum(c[i] * met[i][j] * c[j] for i in range(nc) for j in range(nc))
        row = []
        for x in c:
            if x == 0:
                row.append(0.0)
                continue
            q = x * x / n2
            v = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
            row.append(float(v) if x > 0 else -float(v))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- E-matrix patterns


def shell_transform_pattern(l, pure=True):
    """For each function of the shell, the Cartesian components with nonzero weight."""
    comps = cart_components(l)
    if l <= 1 or not pure:
        return [[i] for i in range(len(comps))]
    rows = []
    for m in range(-l, l + 1):
        c = solid_harmonic_coefs(l, m)
        rows.append([i for i, x in enumerate(c) if x != 0])
    return rows


def e_pattern(la, lb):
    """Structural nonzeros (r, ab) of E[r][ab] for a pair class, sorted by (r, ab)."""
    ca, cb = cart_components(la), cart_components(lb)
    ta, tb = shell_transform_pattern(la), shell_transform_pattern(lb)
    na, nb = len(ta), len(tb)
    tuv = herm_tuv(la + lb)
    nz = set()
    for mu in range(na):
        for nu in range(nb):
            for ia in ta[mu]:
                for ib in tb[nu]:
                    mx = [ca[ia][k] + cb[ib][k] for k in range(3)]
                    for r, (t, u, v) in enumerate(tuv):
                        if t <= mx[0] and u <= mx[1] and v <= mx[2]:
                            nz.add((r, mu * nb + nu))
    return sorted(nz)


def e_pattern_deriv(la, lb, centre, k):
    """Structural nonzeros (r, ab) of the Hermite expansion (degree la + lb + 1) of d/dX_k of the
    product a b, X the centre of a (centre 0) or of b (centre 1), sorted by (r, ab).

    d/dA_k x_A^i exp(-alpha r_A^2) = 2 alpha x_A^(i + 1_k) - i_k x_A^(i - 1_k): the derivative
    product has the Cartesian components i +- 1_k of the differentiated factor, and its Hermite
    expansion is supported on t <= (i +- 1_k) + j componentwise (as for E)."""
    ca, cb = cart_components(la), cart_components(lb)
    ta, tb = shell_transform_pattern(la), shell_transform_pattern(lb)
    na, nb = len(ta), len(tb)
    tuv = herm_tuv(la + lb + 1)
    nz = set()
    for mu in range(na):
        for nu in range(nb):
            for ia in ta[mu]:
                for ib in tb[nu]:
                    a, b = list(ca[ia]), list(cb[ib])
                    d = a if centre == 0 else b
                    shifts = [1] + ([-1] if d[k] > 0 else [])
                    for s in shifts:
                        d[k] += s
                        mx = [a[q] + b[q] for q in range(3)]
                        d[k] -= s
                        for r, (t, u, v) in enumerate(tuv):
                            if t <= mx[0] and u <= mx[1] and v <= mx[2]:
                                nz.add((r, mu * nb + nu))
    return sorted(nz)


def e_pattern_deriv2(la, lb, k, l):
    """Structural nonzeros (r, ab) of the Hermite expansion (degree la + lb + 2) of
    d^2/dA_k dA_l of the product a b (both derivatives on the centre of a), sorted by (r, ab).

    Each derivative shifts the differentiated factor's Cartesian component by +-1 along its
    direction (e_pattern_deriv); the second-derivative product has the components i + s_k + s_l
    of a for the shifts that keep them non-negative, and its expansion is supported on
    t <= (i + s_k + s_l) + j componentwise."""
    ca, cb = cart_components(la), cart_components(lb)
    ta, tb = shell_transform_pattern(la), shell_transform_pattern(lb)
    na, nb = len(ta), len(tb)
    tuv = herm_tuv(la + lb + 2)
    nz = set()
    for mu in range(na):
        for nu in range(nb):
            for ia in ta[mu]:
                for ib in tb[nu]:
                    b = cb[ib]
                    for s1 in (1, -1):
                        a1 = list(ca[ia])
                        a1[k] += s1
                        if a1[k] < 0:
                            continue
                        for s2 in (1, -1):
                            a2 = list(a1)
                            a2[l] += s2
                            if a2[l] < 0:
                                continue
                            mx = [a2[q] + b[q] for q in range(3)]
                            for r, (t, u, v) in enumerate(tuv):
                                if t <= mx[0] and u <= mx[1] and v <= mx[2]:
                                    nz.add((r, mu * nb + nu))
    return sorted(nz)


# ---------------------------------------------------------------- Hermite Coulomb recursion


def r_direction(t, u, v, rule):
    """(axis d, index i) along which R_tuv is recursed:
    - "min": the smallest positive index, ties to the lower axis (avoids the (i - 1) R_{i-2} term
      when that index is 1; rtensor.rs, tiled.rs);
    - "first": the first positive index in x, y, z order (linus-ints' eri3c `r_prog`)."""
    comps = [(t, 0), (u, 1), (v, 2)]
    pos = [c for c in comps if c[0] > 0]
    if rule == "min":
        i, d = min(pos, key=lambda c: (c[0], c[1]))
    elif rule == "first":
        i, d = pos[0]
    else:
        raise ValueError(rule)
    return d, i


def r_recursion(L, rule="min", targets=None):
    """The rules of R^(n)_tuv = X_d R^(n+1)_{tuv - 1_d} + (i - 1) R^(n+1)_{tuv - 2_d} needed for the
    level-0 values `targets` (default: every t + u + v <= L), with the direction `rule`.

    Returns (need, rule): need[n] the set of (t, u, v) needed at level n, rule[(n, t, u, v)] =
    (d, i, terms) with terms the one or two level-(n + 1) indices (lower first)."""
    need = {0: set(herm_tuv(L)) if targets is None else set(targets)}
    rules = {}
    for n in range(0, L):
        nxt = set()
        for (t, u, v) in need[n]:
            if t + u + v == 0:
                continue
            d, i = r_direction(t, u, v, rule)
            lo1 = [t, u, v]
            lo1[d] -= 1
            terms = [tuple(lo1)]
            if i >= 2:
                lo2 = [t, u, v]
                lo2[d] -= 2
                terms.append(tuple(lo2))
            rules[(n, t, u, v)] = (d, i, terms)
            nxt.update(terms)
        need[n + 1] = nxt
    return need, rules


# ---------------------------------------------------------------- Boys function (linus-ints' eri3c scheme)

# Table at T_i = 0.1 i (i < BOYS_ROWS) with F_0..F_{BOYS_COLS-2} and exp(-T_i) in the last column;
# 8-term Taylor sum for the top order, exp(-T) = exp(-T_i) exp(T_i - T) by a 9-term series, downward
# recursion; T >= BOYS_TMAX: asymptotic form. The same table and operation order as eri3c/mod.rs.
BOYS_ROWS = 1172
BOYS_COLS = 27
BOYS_TMAX = 117.0
BOYS_TAYLOR = 8
INV_INT = [0.0, 1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0, 1.0 / 7.0, 0.125, 1.0 / 9.0]
INV_FACT = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0,
            1.0 / 40320.0, 1.0 / 362880.0]
# 2 pi^(5/2), the Coulomb prefactor of linus-ints (eri4c::ERI_PREF)
ERI_PREF = 2.0 * 17.493418327624862
