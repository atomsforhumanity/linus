"""Pieces shared by the CUDA emitters (layer c: cuda_eri3c.py, cuda_pcm.py, cuda_jengine.py): exact C
literals, the printer of IR nodes as C, and the Boys stage, which turns one pair of primitives into
the inputs of the Hermite R recursion,

    s_n = pref (-2 alpha)^n F_n(alpha rr),  n = 0..lt,

by linus-ints' eri3c scheme (mdmath: the Boys table, an 8-term Taylor sum of the top order, exp(-T)
by a 9-term series, downward recursion; the asymptotic form from a cutoff on). The emitters differ
only in how they form alpha and the prefactor, in the cutoff, in the asymptotic branch's form and in
whether the s_n accumulate; those are the stage's parameters.
"""
import math

from mdmath import BOYS_COLS, BOYS_TMAX, INV_FACT, INV_INT


def lit(v):
    """A C literal that parses to exactly v."""
    s = repr(float(v))
    return s if any(c in s for c in ".enai") else s + ".0"


class CPrinter:
    """Prints IR nodes as C: fma(a, b, c), a * b, a + b, a - b, -a, literals."""

    def __init__(self):
        self.names = {}

    def bind(self, n, name):
        self.names[n.id] = name

    def ref(self, n):
        nm = self.names.get(n.id)
        return nm if nm is not None else self.defn(n)

    def defn(self, n):
        a = n.args
        if n.op == "in":
            return n.value
        if n.op == "const":
            return lit(n.value)
        if n.op == "mul":
            return f"{self.ref(a[0])} * {self.ref(a[1])}"
        if n.op == "add":
            return f"{self.ref(a[0])} + {self.ref(a[1])}"
        if n.op == "sub":
            return f"{self.ref(a[0])} - {self.ref(a[1])}"
        if n.op == "neg":
            return f"-{self.ref(a[0])}"
        if n.op == "fma":
            return f"fma({self.ref(a[0])}, {self.ref(a[1])}, {self.ref(a[2])})"
        raise ValueError(n.op)


def boys_tstar(m):
    """The smallest T (in steps of 1/2) from which the asymptotic F_m is exact to 1e-17 relative:
    F_m(T) = F_m^asym(T) (1 - Gamma(m + 1/2, T) / Gamma(m + 1/2)) and the ratio falls with T."""
    def rel(t):
        a, g, gg = 0.5, math.sqrt(math.pi) * math.erfc(math.sqrt(t)), math.sqrt(math.pi)
        for _ in range(m):
            g = a * g + t ** a * math.exp(-t)
            gg *= a
            a += 1.0
        return g / gg
    t = 1.0
    while rel(t) >= 1e-17:
        t += 0.5
    return min(t, BOYS_TMAX)


def boys_stage(lt, ind, prologue, tmax, asym="div", accumulate=False):
    """Statements giving s_n = pref (-2 alpha)^n F_n(T), T = alpha rr, n = 0..lt, indented by
    `ind`, with rr and btab (the Boys table, BOYS_COLS per row) in scope:

    - `prologue`: the lines (indented relative to `ind`) that define the doubles `al` (alpha) and
      `pref` from what is in scope; each emitter forms them its own way. T = al * rr follows.
    - `tmax`: below it, the table scheme (the row nearest T, the Taylor sum of F_lt, exp(-T),
      downward recursion to F_0); from it on, the asymptotic form F_0 = sqrt(pi / T) / 2,
      F_m = F_{m-1} (2m - 1) / (2T). BOYS_TMAX is the table's end (linus-ints' cutoff);
      boys_tstar(lt) is where the asymptotic F_lt is already exact to 1e-17 relative.
    - `asym`, the asymptotic branch's operations: "div", ti = 1 / T and F_0 = 0.5 sqrt(pi ti)
      (linus-ints' order); "rsqrt", ri = rsqrt(T), ti = ri ri and F_0 = (sqrt(pi) / 2) ri.
    - `accumulate`: add to s0..s{lt} already in scope (s_n += ...) instead of declaring them
      (const double s_n = ...)."""
    if asym not in ("div", "rsqrt"):
        raise ValueError(asym)
    o = [f"{ind}{line}" for line in prologue]
    o.append(f"{ind}const double t = al * rr;")
    fv = ", ".join(f"f{m}" for m in range(lt + 1))
    o.append(f"{ind}double {fv};")
    o.append(f"{ind}if (t < {lit(tmax)}) {{")
    o.append(f"{ind}    const int i = (int)(t * 10.0 + 0.5);")
    o.append(f"{ind}    const double dt = i * 0.1 - t;")
    o.append(f"{ind}    const double *row = btab + i * {BOYS_COLS};")
    o.append(f"{ind}    double acc = row[{lt + 7}];")
    for k in range(6, -1, -1):
        o.append(f"{ind}    acc = row[{lt + k}] + acc * dt * {lit(INV_INT[k + 1])};")
    o.append(f"{ind}    f{lt} = acc;")
    ser = lit(INV_FACT[8])
    for k in range(7, -1, -1):
        ser = f"{lit(INV_FACT[k])} + dt * ({ser})"
    o.append(f"{ind}    const double e = row[{BOYS_COLS - 1}] * ({ser});")
    for m in range(lt - 1, -1, -1):
        o.append(f"{ind}    f{m} = (2.0 * t * f{m + 1} + e) * {lit(1.0 / (2 * m + 1))};")
    o.append(f"{ind}}} else {{")
    if asym == "div":
        o.append(f"{ind}    const double ti = 1.0 / t;")
        o.append(f"{ind}    f0 = 0.5 * sqrt({lit(math.pi)} * ti);")
    else:
        o.append(f"{ind}    const double ri = rsqrt(t), ti = ri * ri;")
        o.append(f"{ind}    f0 = {lit(0.5 * math.sqrt(math.pi))} * ri;")
    for m in range(1, lt + 1):
        o.append(f"{ind}    f{m} = f{m - 1} * ({lit((2 * m - 1) * 0.5)} * ti);")
    o.append(f"{ind}}}")
    o.append(f"{ind}const double m2a = -2.0 * al;")
    o.append(f"{ind}double sc = pref;")
    for n in range(lt + 1):
        o.append(f"{ind}s{n} += sc * f{n};" if accumulate else f"{ind}const double s{n} = sc * f{n};")
        if n < lt:
            o.append(f"{ind}sc *= m2a;")
    return o


# The grid-K engine's Boys stage (linus-gpu's kernels::gridk, written in Rust) at its orders: the
# lines of boys_stage between `t` and the scaling, for each lt = 0..2 L_MAX, the table cutoff
# BOYS_TMAX and linus-ints' asymptotic form. gridk's unit test compares its emitter with this text,
# so the two cannot drift (REVIEW GRIDK-18); `intgen.py --check` keeps it up to date.
GRIDK_L_MAX = 4


def gridk_boys_reference():
    out = ["# boys_stage core (tools/intgen/cuda_common.py) at the grid-K engine's orders;",
           "# generated by tools/intgen/intgen.py, compared by linus-gpu's kernels::gridk tests"]
    for lt in range(2 * GRIDK_L_MAX + 1):
        lines = boys_stage(lt, "", ["const double al = 0.0, pref = 0.0;"], BOYS_TMAX)
        first = next(i for i, ln in enumerate(lines) if ln.startswith("double f"))
        last = next(i for i, ln in enumerate(lines) if ln.startswith("const double m2a"))
        out.append(f"== lt {lt}")
        out.extend(lines[first:last])
    return "\n".join(out) + "\n"
