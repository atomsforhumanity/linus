"""CUDA emitter (layer c): fused three-center kernels (k|mn) and their first derivatives for
linus-gpu, McMurchie-Davidson as linus-ints' eri3c (same Boys table and scheme), for the Coulomb
operator 1/r and the long-range operator erf(omega r)/r.

For an orbital pair primitive (exponent p, centre P, Hermite expansion E[mn][t'] of the product,
nonzeros only) and an auxiliary shell K (centre A, primitives a_b with coefficients c_b (2a_b)^-l
(-1)^l):

    s_n   = sum_b pref_b (-2 alpha_b)^n F_n(alpha_b |P - A|^2)       (Boys, per auxiliary primitive)
    R_h   = Hermite recursion of the s_n along X = P - A              (once per (pair primitive, K):
                                                                       R is linear in the s_n, and
                                                                       X is the same for all of K)
    H[k][t'] = sum_c S_k[c] R_{c+t'}                                  (S: solid harmonics of l_K)
    (k|mn) += sum_t' E[mn][t'] H[k][t']

The first derivatives use the same H at one Hermite degree more with the six expansions of
d/dA_x..z and d/dB_x..z of the product (linus-ints' dblock: E_A, and E_B = shift(E) - E_A), so one
kernel produces (d_X mn|k) for both orbital centres; the auxiliary centre follows by translational
invariance, d_C = -(d_A + d_B).

The operator (linus_ints::operator) enters only the Boys stage: for erf(omega r)/r, alpha_b ->
alpha_b k_b in the Boys argument and in (-2 alpha)^n, and the prefactor gains sqrt(k_b), with
k_b = omega^2 / (omega^2 + alpha_b) (Neese, SHARK, eq. 51-54). The recursion is linear in the s_n
and independent of alpha, so the summation over the auxiliary primitives and everything after it
are the same. Each source holds the step as the macro ERI3C_OPERATOR(al, pref) and its kernel
arguments as ERI3C_OPERATOR_ARGS, defined by operator.cuh (`operator_header`), which the host
pastes in front of every kernel: empty for 1/r (the compiled Coulomb kernels are exactly the code
without the hook), the erf step with omega^2 as one more argument when ERI3C_ERF is defined. One
source, schedule and tuning record per class and kind serve both operators.

Every kernel is one fused pass (Boys, R, H, contraction, store) under a CudaSchedule (ir.py):
- mapping "thread": one thread per (pair, auxiliary shell), straight-line code, accumulators in
  registers (acc "reg") or read-modify-write of the output per pair primitive (acc "global"),
  contraction order H or G;
- mapping "block": one block per (pair, tile of auxiliary shells); phase A computes R of a stage
  of (pair primitive, auxiliary shell) jobs (one thread per job, straight-line), phase A2 forms
  H in shared memory (all threads, table-driven), phase B contracts (one thread per (auxiliary
  shell, output row), table-driven over the row's E nonzeros, nk accumulators in registers).
Each output element has one writing thread (deterministic). Sources are C-style CUDA in the
CUDA/HIP common subset; the warp size is never assumed.

Device data (packed by linus-gpu): per pair primitive a record [p, Px, Py, Pz, q, E nonzeros of
every set in pattern order]; per pair [row0, sA, sB, swap] (output row of function (ia, ib) =
row0 + ia sA + ib sB; swap: the kernel's first shell is the pair's second, so derivative planes
A and B trade places) and the atoms of its shells; per auxiliary primitive
[a, c (2a)^-l (-1)^l / a, Ax, Ay, Az, q] (1/a folded in, so the prefactor
2 pi^2.5 / (p a sqrt(p + a)) costs one rsqrt per primitive triple); per auxiliary shell its first
column and atom. Output element (row, col + k) of plane s at
out[s * pstride + (row - rbase) * ld + col + k] (rbase: the first row of the launched pair range).
Derivative kernels with a non-null `gam` store nothing and instead add
sum gam[row][col + k] d/dX (row|k) per atom X to the fixed-point accumulator `gacc` (detsum.h).
The erf kernels take omega^2 after `out`.

CLI (through intgen.py):
    intgen.py cuda                       production sources (schedules from eri3c_schedules.json)
    intgen.py cuda --candidates DIR      every candidate schedule per class for the tuner
    intgen.py cuda --record R.jsonl [--arch sm_90]
                                         merge tuner results (into the architecture's own record
                                         with --arch), then regenerate

Per-architecture schedules: eri3c_schedules.json's "schedules" are the default (tuned on sm_120)
and "arch" holds a record per architecture tuned on its own ({"sm_90": {"schedules": {...}}}).
Where an architecture's recorded schedule of a class differs from the default's, the generator
emits that variant as <key>.<arch>.cu beside the default <key>.cu and lists it in
eri3c_gen/arch.rs (ARCH_KERNELS, keyed by 10 major + minor); linus-gpu takes it on devices of
that compute capability (`eri3c_class::gen_kernel_for`). The default sources and mod.rs do not
depend on the architecture records.
"""
import json
import pathlib

from ir import CudaSchedule, Graph, r_graph
from mdmath import (BOYS_COLS, BOYS_TMAX, ERI_PREF, INV_FACT, INV_INT, cart_components, e_pattern,
                    e_pattern_deriv, e_pattern_deriv2, herm_idx, herm_tuv, ncart, nfunc, nherm, r_direction,
                    sph_coefs)

MAX_L_ORB = 4
MAX_L_AUX = 6
KINDS = ("e", "d", "h")
# second-derivative components (k, l), k <= l, in linus-ints' order (core::deriv::D2_COMPS)
D2_COMPS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def unit3(d):
    return tuple(1 if q == d else 0 for q in range(3))


GEN_DIR = "crates/linus-gpu/src/kernels/eri3c_gen"
SCHEDULE_FILE = "tools/intgen/eri3c_schedules.json"
REC_HEAD = 5          # p, Px, Py, Pz, q
AUX_REC = 6           # a, coef, Ax, Ay, Az, q
SMEM_LIMIT = 96 * 1024
CU_HEADER = "// @generated by tools/intgen/intgen.py cuda -- do not edit by hand."
SET_NAMES = {"e": ["E"], "d": ["Ax", "Ay", "Az", "Bx", "By", "Bz"],
             "h": [f"X1{k}{l}" for k, l in D2_COMPS] + [f"X2{k}{l}" for k in range(3) for l in range(3)]
             + [f"X3{k}{l}" for k, l in D2_COMPS]}
KIND_TEXT = {"e": "integrals", "d": "first derivatives", "h": "second derivatives"}
# the kind in the generated Rust tables (linus-gpu's eri3c_class::Eri3cKind)
KIND_RUST = {"e": "Eri3cKind::Integrals", "d": "Eri3cKind::Derivatives", "h": "Eri3cKind::Hessian"}


def lit(v):
    """A C literal that parses to exactly v."""
    s = repr(float(v))
    return s if any(c in s for c in ".enai") else s + ".0"


def classes():
    return [(la, lb, lk) for la in range(MAX_L_ORB + 1) for lb in range(la + 1) for lk in range(MAX_L_AUX + 1)]


def kernel_name(cls, kind, sched):
    la, lb, lk = cls
    return f"eri3c{kind}_{la}{lb}{lk}_{sched.tag().replace('-', '_')}"


# ---------------------------------------------------------------- class description (math -> IR)


class ClassInfo:
    """Everything about (la lb | lk) of one kind that does not depend on the schedule."""

    def __init__(self, la, lb, lk, kind):
        self.la, self.lb, self.lk, self.kind = la, lb, lk, kind
        lab = la + lb
        self.hdeg = lab + {"e": 0, "d": 1, "h": 2}[kind]
        self.lt = self.hdeg + lk
        self.na, self.nb, self.nk = nfunc(la), nfunc(lb), nfunc(lk)
        self.nab = self.na * self.nb
        # value sets (the E nonzeros of a pair primitive's record, each over its own Hermite
        # degree) and output sets (value set, Hermite shift): e and d output their value sets; h
        # (second derivatives, linus-ints' hess_3c) has the value sets E_AA,kl (6, D2_COMPS),
        # E_A,k (3) and E, and the 21 outputs X1_kl = E_AA,kl, X2_kl = S_l E_A,k (6 + 3 k + l),
        # X3_kl = S_k S_l E
        if kind == "e":
            self.vsets = [e_pattern(la, lb)]
        elif kind == "d":
            self.vsets = [e_pattern_deriv(la, lb, c, x) for c in (0, 1) for x in range(3)]
        else:
            self.vsets = ([e_pattern_deriv2(la, lb, k, l) for k, l in D2_COMPS]
                          + [e_pattern_deriv(la, lb, 0, k) for k in range(3)] + [e_pattern(la, lb)])
        if kind == "h":
            self.osets = ([(c, (0, 0, 0)) for c in range(6)]
                          + [(6 + k, unit3(l)) for k in range(3) for l in range(3)]
                          + [(9, tuple(unit3(k)[q] + unit3(l)[q] for q in range(3))) for k, l in D2_COMPS])
        else:
            self.osets = [(s, (0, 0, 0)) for s in range(len(self.vsets))]
        # output patterns (r', mn) over Herm(hdeg): the value set's r shifted
        tuv_v = herm_tuv(self.hdeg)
        self.sets = [sorted((herm_idx(*(tuv_v[r][q] + sh[q] for q in range(3))), mn) for r, mn in self.vsets[v])
                     for v, sh in self.osets]
        # E nonzero j of value set s lives at record offset REC_HEAD + soff[s] + j
        self.soff = []
        o = 0
        for p in self.vsets:
            self.soff.append(o)
            o += len(p)
        self.nnz = o
        self.rec = REC_HEAD + self.nnz
        tuv_h = herm_tuv(self.hdeg)
        self.tp_used = sorted({r for p in self.sets for r, _ in p})
        self.tpos = {t: i for i, t in enumerate(self.tp_used)}
        self.nht = len(self.tp_used)
        self.S = sph_coefs(lk)
        self.carts = cart_components(lk)
        self.cused = sorted({c for row in self.S for c, x in enumerate(row) if x != 0.0})
        tuv = herm_tuv(self.lt)
        self.r_targets = sorted({herm_idx(self.carts[c][0] + tuv_h[t][0], self.carts[c][1] + tuv_h[t][1],
                                          self.carts[c][2] + tuv_h[t][2])
                                 for c in self.cused for t in self.tp_used})
        self.r_targets_tuv = [tuv[h] for h in self.r_targets]
        self.rslot = {h: i for i, h in enumerate(self.r_targets)}
        self.nr = len(self.r_targets)
        # output rows (set, mn) and their E nonzeros [(record offset, t' position)] in t' order
        self.rows = []
        for s, (v, sh) in enumerate(self.osets):
            by = {}
            for j, (r, mn) in enumerate(self.vsets[v]):
                rs = herm_idx(*(tuv_v[r][q] + sh[q] for q in range(3)))
                by.setdefault(mn, []).append((REC_HEAD + self.soff[v] + j, rs))
            for mn in range(self.nab):
                self.rows.append((s, mn, sorted(by.get(mn, []), key=lambda x: x[1])))

    def tables(self):
        """The block kernels' run-time tables (the host builds the same in Rust):
        hp, hc, hr: per (k, t' position) the S_k[c] != 0 and the R slot of c + t' (CSR);
        rp, re, rt: per row (set, mn) the record offsets of its E nonzeros and their t' positions."""
        hp, hc, hr = [0], [], []
        for k in range(self.nk):
            for t in self.tp_used:
                for c in range(ncart(self.lk)):
                    if self.S[k][c] != 0.0:
                        hc.append(self.S[k][c])
                        hr.append(self.ridx(c, t))
                hp.append(len(hc))
        rp, re, rt = [0], [], []
        for s, mn, terms in self.rows:
            for off, t in terms:
                re.append(off)
                rt.append(self.tpos[t])
            rp.append(len(re))
        return hp, hc, hr, rp, re, rt

    def pattern_hash(self):
        """FNV-1a over the sets' (t', mn) lists and the R slots' Hermite indices: the host checks
        its own derivation against it."""
        h = 0xcbf29ce484222325

        def mix(x):
            nonlocal h
            h ^= x & 0xffffffffffffffff
            h = (h * 0x100000001b3) & 0xffffffffffffffff
        for p in self.sets:
            mix(len(p))
            for r, mn in p:
                mix(r * 65536 + mn)
        if self.kind == "h":
            # the record's value sets and the output sets' sources
            for p in self.vsets:
                mix(len(p))
                for r, mn in p:
                    mix(r * 65536 + mn)
            for v, sh in self.osets:
                mix(v * 4096 + sh[0] * 256 + sh[1] * 16 + sh[2])
        for x in self.r_targets:
            mix(x)
        return h

    def ridx(self, c, t):
        """Slot in r_targets of R_{c + t'} (c a Cartesian component index of lk, t a Hermite index
        of degree <= hdeg)."""
        a, b = self.carts[c], herm_tuv(self.hdeg)[t]
        return self.rslot[herm_idx(a[0] + b[0], a[1] + b[1], a[2] + b[2])]

    def tag(self):
        L = "spdfghi"
        return f"({L[self.la]}{L[self.lb]}|{L[self.lk]})"

    def contraction_fmas(self):
        """FMAs of the E contraction per (pair primitive, auxiliary shell) in H order."""
        return self.nk * sum(len(p) for p in self.sets)


def build_rh(ci, rule, h_too=True):
    """IR of R (level-0 values needed) and, optionally, H[k][t'] for the used t'."""
    g = Graph(fold=True)
    xyz = [g.inp("px"), g.inp("py"), g.inp("pz")]
    _, nodes = r_graph(g, ci.lt, rule, lambda n: g.inp(f"s{n}"), xyz, targets=ci.r_targets_tuv)
    R = [nodes[(0,) + t] for t in ci.r_targets_tuv]
    H = {}
    if h_too:
        for k in range(ci.nk):
            for t in ci.tp_used:
                terms = [(g.const(ci.S[k][c]), R[ci.ridx(c, t)]) for c in range(ncart(ci.lk)) if ci.S[k][c] != 0.0]
                H[(k, t)] = g.chain(terms)
    return g, nodes, R, H


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


# ---------------------------------------------------------------- shared source pieces

SIGNATURE = """(long long npair, int nauxsh, const int *__restrict__ pr_p0, const double *__restrict__ pr_d,
    const long long *__restrict__ pr_out, const int *__restrict__ ax_p0, const double *__restrict__ ax_d,
    const int *__restrict__ ax_col, const double *__restrict__ btab, long long ld, long long rbase, long long pstride,
    double thresh, int ntile, const int *__restrict__ hp, const double *__restrict__ hc,
    const int *__restrict__ hr, const int *__restrict__ rp, const int *__restrict__ re,
    const int *__restrict__ rt, const double *__restrict__ gam, const int *__restrict__ pr_atom,
    const int *__restrict__ ax_atom, unsigned long long *__restrict__ gacc, long long natom3,
    double *__restrict__ out ERI3C_OPERATOR_ARGS)"""

OPERATOR_FILE = "operator.cuh"
HESSIAN_FILE = "hessian.cuh"


def operator_header():
    """operator.cuh: the operator step of the Boys stage (ERI3C_OPERATOR) and its kernel arguments
    (ERI3C_OPERATOR_ARGS), 1/r unless ERI3C_ERF is defined. linus-gpu pastes it in front of every
    generated kernel (crate::eri3c)."""
    return "\n".join([
        CU_HEADER,
        "// The two-electron operator of the generated three-center kernels (linus_ints::operator).",
        "// Only the Boys stage depends on it: ERI3C_OPERATOR(al, pref) turns the Coulomb",
        "// alpha = p a / (p + a) and prefactor of one primitive triple into the operator's, and",
        "// ERI3C_OPERATOR_ARGS are the operator's kernel arguments (after `out`).",
        "// - 1/r (ERI3C_ERF not defined): nothing changes.",
        "// - erf(omega r)/r (ERI3C_ERF defined; omega2 = omega^2 > 0): alpha -> alpha k in the Boys",
        "//   argument and in (-2 alpha)^n, prefactor * sqrt(k), k = omega^2 / (omega^2 + alpha).",
        "#ifdef ERI3C_ERF",
        "#define ERI3C_OPERATOR_ARGS , double omega2",
        "#define ERI3C_OPERATOR(al, pref) { const double kw_ = omega2 / (omega2 + (al)); (al) = (al) * kw_; (pref) = (pref) * sqrt(kw_); }",
        "#else",
        "#define ERI3C_OPERATOR_ARGS",
        "#define ERI3C_OPERATOR(al, pref)",
        "#endif",
    ]) + "\n"


def boys_rn(lt, ind):
    """Statements accumulating s_n += pref (-2 alpha)^n F_n(T) for one auxiliary primitive (bd);
    needs p, rr, ipref = ERI_PREF / p and s0..s{lt} in scope. The Boys scheme of linus-ints' eri3c,
    with the operator's step (operator_header) on alpha and the prefactor."""
    o = []
    o.append(f"{ind}const double ae = bd[0];")
    o.append(f"{ind}const double sp = p + ae;")
    o.append(f"{ind}const double rs = rsqrt(sp);")
    o.append(f"{ind}double al = p * ae * (rs * rs);")
    o.append(f"{ind}double pref = ipref * rs * bd[1];   // bd[1] = c (2a)^-l (-1)^l / a")
    o.append(f"{ind}ERI3C_OPERATOR(al, pref);")
    o.append(f"{ind}const double t = al * rr;")
    fv = ", ".join(f"f{m}" for m in range(lt + 1))
    o.append(f"{ind}double {fv};")
    o.append(f"{ind}if (t < {lit(BOYS_TMAX)}) {{")
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
    o.append(f"{ind}    const double ti = 1.0 / t;")
    o.append(f"{ind}    f0 = 0.5 * sqrt({lit(3.141592653589793)} * ti);")
    for m in range(1, lt + 1):
        o.append(f"{ind}    f{m} = f{m - 1} * ({lit((2 * m - 1) * 0.5)} * ti);")
    o.append(f"{ind}}}")
    o.append(f"{ind}const double m2a = -2.0 * al;")
    o.append(f"{ind}double sc = pref;")
    for n in range(lt + 1):
        o.append(f"{ind}s{n} += sc * f{n};")
        if n < lt:
            o.append(f"{ind}sc *= m2a;")
    return o


def rn_loop(ci, ind, kvar):
    """The loop over the primitives of auxiliary shell `kvar` accumulating s_n; sets `any`."""
    lt = ci.lt
    o = [f"{ind}double " + ", ".join(f"s{n} = 0.0" for n in range(lt + 1)) + ";",
         f"{ind}bool any = false;",
         f"{ind}for (int b = ax_p0[{kvar}]; b < ax_p0[{kvar} + 1]; ++b) {{",
         f"{ind}    const double *bd = ax_d + {AUX_REC} * (long long)b;",
         f"{ind}    if (qp * bd[5] < thresh) continue;",
         f"{ind}    any = true;"]
    o += boys_rn(lt, ind + "    ")
    o.append(f"{ind}}}")
    return o


def r_stmts(g, nodes, pr, ind):
    """`const double` statements of every R node of the graph that is used (DCE from the level-0
    targets), level by level."""
    o = []
    for key, node in nodes.items():
        if node.op in ("in", "const") or node.id in pr.names:
            continue
        name = "r{}_{}_{}_{}".format(*key)
        o.append(f"{ind}const double {name} = {pr.defn(node)};")
        pr.bind(node, name)
    return o


def out_index(ci, s, mn, k, colvar="col"):
    """C expression of the output element of set s, function pair mn (kernel order), function k."""
    ia, ib = divmod(mn, ci.nb)
    plane = ""
    if ci.kind == "d":
        c, x = divmod(s, 3)
        plane = f"((swp ^ {c}) * 3 + {x}) * pstride + "
    elif ci.kind == "h":
        plane = f"{s} * pstride + "
    row = "row0" + (f" + {ia} * sA" if ia else "") + (f" + {ib} * sB" if ib else "")
    return f"{plane}({row}) * ld + {colvar} + {k}"


# ---------------------------------------------------------------- Gamma-contracted gradients

GAMMA_DOC = """    // Gamma epilogue (gam != 0): instead of storing, g_X += sum_{row, k} gam[row][col + k] d/dX (row|k)
    // for X the two pair centres and, by translational invariance, the auxiliary centre, summed
    // exactly into the fixed-point accumulator gacc (detsum.h, n = natom3, element 3 atom + x; an
    // invalid value poisons it)"""


def thread_gamma_epilogue(ci, accn):
    """Gamma epilogue of a thread-mapped derivative kernel: per-thread sums, global det_add."""
    o = [GAMMA_DOC, "    if (gam != 0) {"]
    for s in range(len(ci.sets)):
        o.append(f"        double g{s} = 0.0;")
        for mn in range(ci.nab):
            ia, ib = divmod(mn, ci.nb)
            row = "row0" + (f" + {ia} * sA" if ia else "") + (f" + {ib} * sB" if ib else "")
            o.append(f"        {{ const double *gp = gam + ({row}) * ld + col;")
            for k in range(ci.nk):
                o.append(f"          g{s} = fma(gp[{k}], {accn(s, mn, k)}, g{s});")
            o.append("        }")
    o.append("        const int at0 = pr_atom[2 * ip + swp], at1 = pr_atom[2 * ip + 1 - swp], atc = ax_atom[ks];")
    for x in range(3):
        o.append(f"        det_add(gacc, natom3, 3 * at0 + {x}, g{x});")
        o.append(f"        det_add(gacc, natom3, 3 * at1 + {x}, g{3 + x});")
        o.append(f"        det_add(gacc, natom3, 3 * atc + {x}, -(g{x} + g{3 + x}));")
    o.append("        return;")
    o.append("    }")
    return o


def block_gamma_epilogue(ci, nkt, rt, rpt, nt):
    """Gamma epilogue of a block-mapped derivative kernel: exact integer sums in shared memory
    (A and B of the block's pair, C per auxiliary shell of the tile), then one set of global
    atomics per block."""
    ne = 6 + 3 * nkt
    o = [GAMMA_DOC,
         "    if (gam != 0) {",
         f"        __shared__ unsigned long long gs[3 * {ne}];",
         f"        for (int e = tid; e < 3 * {ne}; e += {nt}) gs[e] = 0ull;",
         "        __syncthreads();",
         "        double g[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};",
         "        if (ks < nauxsh) {",
         "            const long long col = ax_col[ks];",
         "            #pragma unroll",
         f"            for (int i = 0; i < {rpt}; ++i) {{",
         f"                const int r = rw + i * {rt};",
         f"                if (r >= {len(ci.rows)}) break;",
         f"                const int s = r / {ci.nab}, mn = r - s * {ci.nab};",
         f"                const int ia = mn / {ci.nb}, ib = mn - ia * {ci.nb};",
         "                const double *gp = gam + (row0 + ia * sA + ib * sB) * ld + col;",
         "                double v = 0.0;",
         "                #pragma unroll",
         f"                for (int k = 0; k < {ci.nk}; ++k) v = fma(gp[k], acc[i][k], v);",
         "                #pragma unroll",
         "                for (int t = 0; t < 6; ++t) if (t == s) g[t] += v;",
         "            }",
         "        }",
         "        #pragma unroll",
         "        for (int t = 0; t < 9; ++t) {",
         "            const double v = t < 6 ? g[t] : -(g[t - 6] + g[t - 3]);",
         "            if (v != 0.0) {",
         "                long long h, m, q;",
         "                if (det_split(v, h, m, q)) det_poison(gacc, natom3);",
         "                const int e = t < 6 ? t : 6 + 3 * kl + (t - 6);",
         f"                atomicAdd(gs + e, (unsigned long long)h);",
         f"                atomicAdd(gs + {ne} + e, (unsigned long long)m);",
         f"                atomicAdd(gs + {2 * ne} + e, (unsigned long long)q);",
         "            }",
         "        }",
         "        __syncthreads();",
         f"        for (int e = tid; e < {ne}; e += {nt}) {{",
         "            int at, x;",
         "            if (e < 6) {",
         "                at = pr_atom[2 * ip + ((e / 3) ^ swp)];",
         "                x = e % 3;",
         "            } else {",
         f"                const int kk = tile * {nkt} + (e - 6) / 3;",
         "                if (kk >= nauxsh) continue;",
         "                at = ax_atom[kk];",
         "                x = (e - 6) % 3;",
         "            }",
         "            const long long i = 3 * (long long)at + x;",
         "            for (int w = 0; w < 3; ++w) {",
         f"                const unsigned long long v = gs[w * {ne} + e];",
         "                if (v != 0ull) atomicAdd(gacc + w * natom3 + i, v);",
         "            }",
         "        }",
         "        return;",
         "    }"]
    return o


# ---------------------------------------------------------------- Gamma-contracted second derivatives

HESS_DOC = """    // Hessian epilogue (gam != 0): instead of storing, the 21 components X_c = sum_{row, k}
    // gam[row][col + k] (X_c row|k) (X1 = E_AA, X2 = S E_A, X3 = S S E) give the centre blocks of the
    // second derivative (hessian.cuh), added exactly to the fixed-point accumulator gacc
    // (natom3 = 3 natom rows and columns; an invalid value poisons it: detsum.h)"""


def hessian_header():
    """hessian.cuh: the map from the 21 Gamma-contracted components of a (pair, auxiliary shell)
    to the Hessian's centre blocks (linus-ints' hess_3c), in fixed point: the components arrive as
    det_split words and are combined with integer coefficients (exact), then added with integer
    atomics. The accumulator holds M with H = (M + M^T) / 2: the diagonal centre blocks' upper
    triangles (off-diagonal entries twice) and the off-diagonal centre blocks twice."""
    o = [CU_HEADER,
         "// The centre blocks of a three-center second derivative (linus-ints' hess_3c) from the 21",
         "// components X1_kl (D2 order), X2_kl ([k][l]), X3_kl (D2 order) of one (pair, auxiliary shell)",
         "// as fixed-point words x[c][w] (w = the det_split words h, m, q):",
         "//   AA = X1, BB = X3 - X2^T - X2 + X1, CC = X3, AB = X2 - X1, AC = -X2, BC = X2 - X3",
         "// (A, B the pair's centres in the kernel's shell order, C the auxiliary centre), added to acc",
         "// (n3 x n3 elements of three words, row-major, n3 = 3 natom) as M with H = (M + M^T) / 2: the",
         "// diagonal blocks' upper triangles (off-diagonal entries twice), the off-diagonal blocks twice.",
         "// Word arithmetic wraps (unsigned): integer combinations of exact fixed-point values are exact.",
         "__device__ __forceinline__ void eri3c_hess_word(unsigned long long *acc, long long e, unsigned long long v)",
         "{",
         "    if (v != 0ull) atomicAdd(acc + e, v);",
         "}",
         "",
         "__device__ __forceinline__ void eri3c_hess_add(unsigned long long *acc, long long n3, int a, int b, int c,",
         "    const unsigned long long x[21][3])",
         "{",
         "    const long long nn = n3 * n3;",
         "    const long long ra = 3LL * a, rb = 3LL * b, rc = 3LL * c;",
         "    for (int w = 0; w < 3; ++w) {",
         "        unsigned long long *aw = acc + w * nn;",
         "        for (int k = 0; k < 3; ++k) {",
         "            for (int l = 0; l < 3; ++l) {",
         "                const int lo = k < l ? k : l, hi = k < l ? l : k;",
         "                const int d = lo == 0 ? hi : (lo == 1 ? 2 + hi : 5);",
         "                const unsigned long long x1 = x[d][w], x3 = x[15 + d][w];",
         "                const unsigned long long x2 = x[6 + 3 * k + l][w], x2t = x[6 + 3 * l + k][w];",
         "                if (k <= l) {",
         "                    const unsigned long long f = k == l ? 1ull : 2ull;",
         "                    eri3c_hess_word(aw, (ra + k) * n3 + ra + l, f * x1);",
         "                    eri3c_hess_word(aw, (rb + k) * n3 + rb + l, f * (x3 - x2t - x2 + x1));",
         "                    eri3c_hess_word(aw, (rc + k) * n3 + rc + l, f * x3);",
         "                }",
         "                eri3c_hess_word(aw, (ra + k) * n3 + rb + l, 2ull * (x2 - x1));",
         "                eri3c_hess_word(aw, (ra + k) * n3 + rc + l, 0ull - 2ull * x2);",
         "                eri3c_hess_word(aw, (rb + k) * n3 + rc + l, 2ull * (x2 - x3));",
         "            }",
         "        }",
         "    }",
         "}"]
    return "\n".join(o) + "\n"


def thread_hess_epilogue(ci, accn):
    """Hessian epilogue of a thread-mapped second-derivative kernel: per-thread sums of the 21
    components, split once, combined and added to the global accumulator (hessian.cuh)."""
    o = [HESS_DOC, "    if (gam != 0) {", "        unsigned long long x[21][3];"]
    for s in range(len(ci.sets)):
        o.append("        {")
        o.append("          double g = 0.0;")
        for mn in range(ci.nab):
            ia, ib = divmod(mn, ci.nb)
            row = "row0" + (f" + {ia} * sA" if ia else "") + (f" + {ib} * sB" if ib else "")
            o.append(f"          {{ const double *gp = gam + ({row}) * ld + col;")
            for k in range(ci.nk):
                o.append(f"            g = fma(gp[{k}], {accn(s, mn, k)}, g);")
            o.append("          }")
        o.append("          long long h, m, q;")
        o.append("          if (det_split(g, h, m, q)) det_poison(gacc, natom3 * natom3);")
        o.append(f"          x[{s}][0] = (unsigned long long)h;")
        o.append(f"          x[{s}][1] = (unsigned long long)m;")
        o.append(f"          x[{s}][2] = (unsigned long long)q;")
        o.append("        }")
    o.append("        eri3c_hess_add(gacc, natom3, pr_atom[2 * ip + swp], pr_atom[2 * ip + 1 - swp], ax_atom[ks], x);")
    o.append("        return;")
    o.append("    }")
    return o


def block_hess_epilogue(ci, nkt, rt, rpt, nt):
    """Hessian epilogue of a block-mapped second-derivative kernel: exact integer sums of the 21
    components per auxiliary shell of the tile in shared memory, then per shell the combination
    and the global atomics (hessian.cuh)."""
    ne = 21 * nkt
    return [HESS_DOC,
            "    if (gam != 0) {",
            f"        __shared__ unsigned long long gs[3 * {ne}];",
            f"        for (int e = tid; e < 3 * {ne}; e += {nt}) gs[e] = 0ull;",
            "        __syncthreads();",
            "        double g[21];",
            "        #pragma unroll",
            "        for (int t = 0; t < 21; ++t) g[t] = 0.0;",
            "        if (ks < nauxsh) {",
            "            const long long col = ax_col[ks];",
            "            #pragma unroll",
            f"            for (int i = 0; i < {rpt}; ++i) {{",
            f"                const int r = rw + i * {rt};",
            f"                if (r >= {len(ci.rows)}) break;",
            f"                const int s = r / {ci.nab}, mn = r - s * {ci.nab};",
            f"                const int ia = mn / {ci.nb}, ib = mn - ia * {ci.nb};",
            "                const double *gp = gam + (row0 + ia * sA + ib * sB) * ld + col;",
            "                double v = 0.0;",
            "                #pragma unroll",
            f"                for (int k = 0; k < {ci.nk}; ++k) v = fma(gp[k], acc[i][k], v);",
            "                #pragma unroll",
            "                for (int t = 0; t < 21; ++t) if (t == s) g[t] += v;",
            "            }",
            "        }",
            "        #pragma unroll",
            "        for (int t = 0; t < 21; ++t) {",
            "            if (g[t] != 0.0) {",
            "                long long h, m, q;",
            "                if (det_split(g[t], h, m, q)) det_poison(gacc, natom3 * natom3);",
            "                const int e = kl * 21 + t;",
            "                atomicAdd(gs + e, (unsigned long long)h);",
            f"                atomicAdd(gs + {ne} + e, (unsigned long long)m);",
            f"                atomicAdd(gs + {2 * ne} + e, (unsigned long long)q);",
            "            }",
            "        }",
            "        __syncthreads();",
            f"        for (int e = tid; e < {nkt}; e += {nt}) {{",
            f"            const int kk = tile * {nkt} + e;",
            "            if (kk >= nauxsh) continue;",
            "            unsigned long long x[21][3];",
            "            bool any = false;",
            "            for (int t = 0; t < 21; ++t) {",
            "                for (int w = 0; w < 3; ++w) {",
            f"                    x[t][w] = gs[w * {ne} + e * 21 + t];",
            "                    any = any || x[t][w] != 0ull;",
            "                }",
            "            }",
            "            if (any) eri3c_hess_add(gacc, natom3, pr_atom[2 * ip + swp], pr_atom[2 * ip + 1 - swp], ax_atom[kk], x);",
            "        }",
            "        return;",
            "    }"]


# ---------------------------------------------------------------- mapping "thread"


def emit_thread(ci, sched):
    g, nodes, R, H = build_rh(ci, sched.r_rule, h_too=(sched.order == "H"))
    name = kernel_name((ci.la, ci.lb, ci.lk), ci.kind, sched)
    lb = f"__launch_bounds__({sched.block}" + (f", {sched.minblocks})" if sched.minblocks else ")")
    o = [CU_HEADER, f"// {ci.tag()} {KIND_TEXT[ci.kind]}, schedule {sched.tag()}",
         f"extern \"C\" __global__ void {lb} {name}{SIGNATURE}",
         "{",
         "    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
         "    if (tid >= npair * nauxsh) return;",
         "    const long long ip = tid / nauxsh;",
         "    const int ks = (int)(tid - ip * nauxsh);",
         f"    const double ax = ax_d[{AUX_REC} * (long long)ax_p0[ks] + 2], ay = ax_d[{AUX_REC} * (long long)ax_p0[ks] + 3], az = ax_d[{AUX_REC} * (long long)ax_p0[ks] + 4];",
         "    const long long *po = pr_out + 4 * ip;",
         "    const long long row0 = po[0] - rbase, sA = po[1], sB = po[2];",
         "    const int swp = (int)po[3];",
         "    const long long col = ax_col[ks];"]
    nacc = len(ci.sets) * ci.nab * ci.nk
    accn = lambda s, mn, k: f"a{s}_{mn}_{k}"
    if sched.acc == "reg":
        o.append("    double " + ", ".join(f"{accn(s, mn, k)} = 0.0" for s in range(len(ci.sets))
                                          for mn in range(ci.nab) for k in range(ci.nk)) + ";")
    else:
        o.append("    bool first = true;")
    o.append("    for (int a = pr_p0[ip]; a < pr_p0[ip + 1]; ++a) {")
    o.append(f"        const double *pd = pr_d + (long long)a * {ci.rec};")
    o.append("        const double p = pd[0], qp = pd[4];")
    o.append(f"        const double ipref = {lit(ERI_PREF)} / p;")
    o.append("        const double px = pd[1] - ax, py = pd[2] - ay, pz = pd[3] - az;")
    o.append("        const double rr = px * px + py * py + pz * pz;")
    o += rn_loop(ci, "        ", "ks")
    o.append("        if (!any) continue;")
    pr = CPrinter()
    o += r_stmts(g, nodes, pr, "        ")
    rows_by_set = {}
    for s, mn, terms in ci.rows:
        rows_by_set[(s, mn)] = terms
    contrib = []   # (s, mn, k, list of statements producing the update)
    if sched.order == "H":
        for k in range(ci.nk):
            for t in ci.tp_used:
                node = H[(k, t)]
                if node.id not in pr.names and node.op not in ("in", "const"):
                    o.append(f"        const double h{k}_{t} = {pr.defn(node)};")
                    pr.bind(node, f"h{k}_{t}")
            for s in range(len(ci.sets)):
                for mn in range(ci.nab):
                    terms = rows_by_set[(s, mn)]
                    if not terms:
                        continue
                    if sched.acc == "reg":
                        an = accn(s, mn, k)
                        for off, t in terms:
                            o.append(f"        {an} = fma(pd[{off}], {pr.ref(H[(k, t)])}, {an});")
                    else:
                        o.append(f"        {{ double v = pd[{terms[0][0]}] * {pr.ref(H[(k, terms[0][1])])};")
                        for off, t in terms[1:]:
                            o.append(f"          v = fma(pd[{off}], {pr.ref(H[(k, t)])}, v);")
                        idx = out_index(ci, s, mn, k)
                        o.append(f"          out[{idx}] = first ? v : out[{idx}] + v; }}")
    else:  # G order: per (set, mn): g_c = sum_t' R[c+t'] E[mn][t'], then S
        for s in range(len(ci.sets)):
            for mn in range(ci.nab):
                terms = rows_by_set[(s, mn)]
                if not terms:
                    continue
                o.append("        {")
                for c in ci.cused:
                    expr = f"pd[{terms[0][0]}] * {pr.ref(R[ci.ridx(c, terms[0][1])])}"
                    o.append(f"            double g{c} = {expr};")
                    for off, t in terms[1:]:
                        o.append(f"            g{c} = fma(pd[{off}], {pr.ref(R[ci.ridx(c, t)])}, g{c});")
                for k in range(ci.nk):
                    cs = [(c, ci.S[k][c]) for c in ci.cused if ci.S[k][c] != 0.0]
                    if sched.acc == "reg":
                        an = accn(s, mn, k)
                        for c, x in cs:
                            o.append(f"            {an} = " + (f"{an} + g{c};" if x == 1.0 else f"fma({lit(x)}, g{c}, {an});"))
                    else:
                        v = " + ".join((f"g{c}" if x == 1.0 else f"{lit(x)} * g{c}") for c, x in cs)
                        idx = out_index(ci, s, mn, k)
                        o.append(f"            {{ const double v = {v}; out[{idx}] = first ? v : out[{idx}] + v; }}")
                o.append("        }")
    if sched.acc == "global":
        o.append("        first = false;")
    o.append("    }")
    if sched.acc == "reg":
        if ci.kind == "d":
            o += thread_gamma_epilogue(ci, accn)
        elif ci.kind == "h":
            o += thread_hess_epilogue(ci, accn)
        for s in range(len(ci.sets)):
            for mn in range(ci.nab):
                for k in range(ci.nk):
                    o.append(f"    out[{out_index(ci, s, mn, k)}] = {accn(s, mn, k)};")
    else:
        o.append("    if (first) {")
        for s in range(len(ci.sets)):
            for mn in range(ci.nab):
                for k in range(ci.nk):
                    o.append(f"        out[{out_index(ci, s, mn, k)}] = 0.0;")
        o.append("    }")
    o.append("}")
    meta = {"mapping": "thread", "threads": sched.block, "smem": 0, "ktile": 0, "nacc": nacc, "prelude": ""}
    return "\n".join(o) + "\n", meta


# ---------------------------------------------------------------- mapping "block"


def block_geometry(ci, sched):
    """(ktile, ipchunk, row threads, threads per block, shared bytes, rows per thread) of a block
    schedule, or None when it does not fit."""
    nkt = sched.ktile
    nrows = len(ci.rows)
    rt = max(1, min(nrows, sched.block // nkt))
    nt = nkt * rt
    ipc = max(1, min(sched.ipchunk, rt))
    nj = nkt * ipc
    smem = 8 * nj * (ci.nr + ci.nk * ci.nht)
    # derivative kernels add the Gamma epilogue's static array (3 words x (6 + 3 ktile)), second
    # derivatives the Hessian epilogue's (3 words x 21 ktile)
    static = {"e": 0, "d": 24 * (6 + 3 * nkt), "h": 24 * 21 * nkt}[ci.kind]
    if nt > 1024 or smem + static > SMEM_LIMIT:
        return None
    return nkt, ipc, rt, nt, smem, -(-nrows // rt)


def rfun_name(lt, lk, rule):
    return f"eri3c_r{lt}_{lk}{rule[0]}"


def rfun_file(lt, lk, rule):
    return f"r{lt}_{lk}{rule[0]}.cuh"


def rfun_source(lt, lk, rule):
    """The straight-line R recursion as a device function shared by every block kernel with the
    same total degree lt, auxiliary l and direction rule: R_h for h of degree lk..lt (the
    contiguous slots of ClassInfo.r_targets) from s[n] = R^(n)_000, written to rs[slot * stride].
    Inlined, it is the code the kernel would hold itself; shared, the sources stop repeating it
    (2.7 MB of distinct recursions instead of 6.6 MB)."""
    g = Graph(fold=True)
    xyz = [g.inp("px"), g.inp("py"), g.inp("pz")]
    lo = nherm(lk - 1) if lk > 0 else 0
    targets = herm_tuv(lt)[lo:]
    _, nodes = r_graph(g, lt, rule, lambda n: g.inp(f"s[{n}]"), xyz, targets=targets)
    pr = CPrinter()
    o = [CU_HEADER,
         f"// R_h, h of degree {lk}..{lt}, direction rule \"{rule}\", from s[n] = R^(n)_000 (n = 0..{lt})",
         f"__device__ __forceinline__ void {rfun_name(lt, lk, rule)}(const double *s, double px, double py, double pz, double *rs, int stride)",
         "{"]
    o += r_stmts(g, nodes, pr, "    ")
    for i, t in enumerate(targets):
        o.append(f"    rs[{i} * stride] = {pr.ref(nodes[(0,) + tuple(t)])};")
    o.append("}")
    return "\n".join(o) + "\n"


def rprog(lt, rule):
    """The universal R recursion program (linus-ints' eri3c r_prog): per Hermite index h >= 1 of
    degree <= lt, axis | j1 << 2 | j2 << 13 | c << 24 for R_h = X_axis R'_j1 + c R'_j2 (R' the
    level above; j2 = 0 and c = 0 for a one-term step)."""
    tuv = herm_tuv(lt)
    out = [0]
    for h in range(1, nherm(lt)):
        t = list(tuv[h])
        d, i = r_direction(*t, rule)
        lo1 = list(t)
        lo1[d] -= 1
        j1 = herm_idx(*lo1)
        j2, c = 0, 0
        if i >= 2:
            lo2 = list(t)
            lo2[d] -= 2
            j2, c = herm_idx(*lo2), i - 1
        out.append(d | j1 << 2 | j2 << 13 | c << 24)
    return out


def r_loop_stmts(ci, sched, nj, ind):
    """Phase A's R by the recursion program over two per-thread arrays (no dead-value removal)."""
    lt = ci.lt
    nh = nherm(lt)
    h0 = nherm(ci.lk - 1) if ci.lk > 0 else 0
    o = [f"{ind}double sn[{lt + 1}] = {{{', '.join(f's{n}' for n in range(lt + 1))}}};",
         f"{ind}double ra[{nh}], rb[{nh}];",
         f"{ind}double *prev = ra, *cur = rb;",
         f"{ind}prev[0] = sn[{lt}];",
         f"{ind}for (int n = {lt - 1}; n >= 0; --n) {{",
         f"{ind}    const int m = {lt} - n, cnt = (m + 1) * (m + 2) * (m + 3) / 6;",
         f"{ind}    cur[0] = sn[n];",
         f"{ind}    for (int h = 1; h < cnt; ++h) {{",
         f"{ind}        const unsigned int st = rprog[h];",
         f"{ind}        const unsigned int ax = st & 3u;",
         f"{ind}        const double x = ax == 0u ? px : (ax == 1u ? py : pz);",
         f"{ind}        cur[h] = x * prev[(st >> 2) & 2047u] + (double)(st >> 24) * prev[(st >> 13) & 2047u];",
         f"{ind}    }}",
         f"{ind}    double *sw = prev; prev = cur; cur = sw;",
         f"{ind}}}",
         f"{ind}for (int i = 0; i < {ci.nr}; ++i) rs[i * {nj} + tid] = prev[{h0} + i];"]
    return o


def emit_block(ci, sched):
    geo = block_geometry(ci, sched)
    if geo is None:
        return None, None
    nkt, ipc, rt, nt, smem, rpt = geo
    nj = nkt * ipc
    g, nodes, R, _ = build_rh(ci, sched.r_rule, h_too=False)
    name = kernel_name((ci.la, ci.lb, ci.lk), ci.kind, sched)
    o = [CU_HEADER, f"// {ci.tag()} {KIND_TEXT[ci.kind]}, schedule {sched.tag()}",
         "// tables (built by the host from the same patterns, see ClassInfo.tables): hp/hc/hr the H terms",
         "// (per (k, t'): S_k[c] and the R slot of c + t'), rp/re/rt the rows (per (set, mn): record",
         "// offsets of the E nonzeros and their t' positions)",
         *([f"__device__ const unsigned int rprog[{nherm(ci.lt)}] = {{{', '.join(map(str, rprog(ci.lt, sched.r_rule)))}}};"]
           if sched.r_code == "loop" else []),
         f"extern \"C\" __global__ void __launch_bounds__({nt}" + (f", {sched.minblocks})" if sched.minblocks else ")") + f" {name}{SIGNATURE}",
         "{",
         "    extern __shared__ double smem[];",
         f"    double *rs = smem;                  // R of the stage's jobs: rs[slot * {nj} + job]",
         f"    double *hs = smem + {ci.nr * nj};   // H: hs[(k * {ci.nht} + t') * {nj} + job]",
         "    const long long ip = blockIdx.x / ntile;",
         "    const int tile = (int)(blockIdx.x - ip * ntile);",
         "    const int tid = threadIdx.x;",
         f"    const int kl = tid % {nkt}, rw = tid / {nkt};",
         f"    const int ks = tile * {nkt} + kl;",
         "    const long long *po = pr_out + 4 * ip;",
         "    const long long row0 = po[0] - rbase, sA = po[1], sB = po[2];",
         "    const int swp = (int)po[3];",
         f"    double acc[{rpt}][{ci.nk}];",
         f"    #pragma unroll",
         f"    for (int i = 0; i < {rpt}; ++i) {{",
         f"        #pragma unroll",
         f"        for (int k = 0; k < {ci.nk}; ++k) acc[i][k] = 0.0;",
         "    }",
         "    const int a0 = pr_p0[ip], a1 = pr_p0[ip + 1];",
         f"    for (int as = a0; as < a1; as += {ipc}) {{",
         f"        // phase A: one thread per job (pair primitive as + job / {nkt}, auxiliary shell tile * {nkt} + job % {nkt})",
         f"        if (tid < {nj}) {{",
         f"            const int a = as + tid / {nkt};",
         f"            const int kj = tile * {nkt} + tid % {nkt};",
         "            bool done = false;",
         "            if (a < a1 && kj < nauxsh) {",
         f"                const double *pd = pr_d + (long long)a * {ci.rec};",
         f"                const double *xd = ax_d + {AUX_REC} * (long long)ax_p0[kj];",
         "                const double p = pd[0], qp = pd[4];",
         f"                const double ipref = {lit(ERI_PREF)} / p;",
         "                const double px = pd[1] - xd[2], py = pd[2] - xd[3], pz = pd[3] - xd[4];",
         "                const double rr = px * px + py * py + pz * pz;"]
    o += rn_loop(ci, "                ", "kj")
    o.append("                if (any) {")
    if sched.r_code == "loop":
        o += r_loop_stmts(ci, sched, nj, "                    ")
    else:
        o.append(f"                    const double sn[{ci.lt + 1}] = {{{', '.join(f's{n}' for n in range(ci.lt + 1))}}};")
        o.append(f"                    {rfun_name(ci.lt, ci.lk, sched.r_rule)}(sn, px, py, pz, rs + tid, {nj});")
    o.append("                    done = true;")
    o.append("                }")
    o.append("            }")
    o.append("            if (!done) {")
    o.append(f"                for (int i = 0; i < {ci.nr}; ++i) rs[i * {nj} + tid] = 0.0;")
    o.append("            }")
    o.append("        }")
    o.append("        __syncthreads();")
    o.append("        // phase A2: H[k][t'] = sum_c S_k[c] R[c + t'] for every job, all threads")
    o.append(f"        for (int x = tid; x < {ci.nk * ci.nht * nj}; x += {nt}) {{")
    o.append(f"            const int j = x % {nj}, kt = x / {nj};")
    o.append(f"            double h = 0.0;")
    o.append(f"            for (int q = hp[kt]; q < hp[kt + 1]; ++q) h = fma(hc[q], rs[hr[q] * {nj} + j], h);")
    o.append("            hs[x] = h;")
    o.append("        }")
    o.append("        __syncthreads();")
    o.append("        // phase B: thread (auxiliary shell kl, rows rw, rw + row threads, ...)")
    o.append(f"        for (int jp = 0; jp < {ipc}; ++jp) {{")
    o.append("            const int a = as + jp;")
    o.append("            if (a >= a1) break;")
    o.append(f"            const double *pd = pr_d + (long long)a * {ci.rec};")
    o.append(f"            const double *hj = hs + jp * {nkt} + kl;")
    o.append("            #pragma unroll")
    o.append(f"            for (int i = 0; i < {rpt}; ++i) {{")
    o.append(f"                const int r = rw + i * {rt};")
    o.append(f"                if (r >= {len(ci.rows)}) break;")
    o.append(f"                for (int q = rp[r]; q < rp[r + 1]; ++q) {{")
    o.append(f"                    const double e = pd[re[q]];")
    o.append(f"                    const double *h = hj + rt[q] * {nj};")
    o.append("                    #pragma unroll")
    o.append(f"                    for (int k = 0; k < {ci.nk}; ++k) acc[i][k] = fma(e, h[k * {ci.nht * nj}], acc[i][k]);")
    o.append("                }")
    o.append("            }")
    o.append("        }")
    o.append("        __syncthreads();")
    o.append("    }")
    if ci.kind == "d":
        o += block_gamma_epilogue(ci, nkt, rt, rpt, nt)
    elif ci.kind == "h":
        o += block_hess_epilogue(ci, nkt, rt, rpt, nt)
    o.append("    if (ks >= nauxsh) return;")
    o.append("    const long long col = ax_col[ks];")
    o.append("    #pragma unroll")
    o.append(f"    for (int i = 0; i < {rpt}; ++i) {{")
    o.append(f"        const int r = rw + i * {rt};")
    o.append(f"        if (r >= {len(ci.rows)}) break;")
    o.append(f"        const int s = r / {ci.nab}, mn = r - s * {ci.nab};")
    o.append(f"        const int ia = mn / {ci.nb}, ib = mn - ia * {ci.nb};")
    if ci.kind == "d":
        o.append("        const long long plane = (long long)(((swp ^ (s / 3)) * 3 + s % 3)) * pstride;")
    elif ci.kind == "h":
        o.append("        const long long plane = (long long)s * pstride;")
    else:
        o.append("        const long long plane = 0;")
    o.append("        double *dst = out + plane + (row0 + ia * sA + ib * sB) * ld + col;")
    o.append("        #pragma unroll")
    o.append(f"        for (int k = 0; k < {ci.nk}; ++k) dst[k] = acc[i][k];")
    o.append("    }")
    o.append("}")
    meta = {"mapping": "block", "threads": nt, "smem": smem, "ktile": nkt, "nacc": rpt * ci.nk,
            "prelude": rfun_file(ci.lt, ci.lk, sched.r_rule) if sched.r_code == "inline" else ""}
    return "\n".join(o) + "\n", meta


def emit(cls, kind, sched):
    """(source, launch metadata) of one kernel, or (None, None) if the schedule does not fit."""
    ci = ClassInfo(*cls, kind)
    if sched.mapping == "thread":
        return emit_thread(ci, sched)
    return emit_block(ci, sched)


# ---------------------------------------------------------------- costs and candidates


def costs(cls, kind, rule="first"):
    """FP64 operation counts (from the IR after CSE and folding): per auxiliary primitive (Boys and
    prefactor, from the source), and per (pair primitive, auxiliary shell) (R recursion, H, E
    contraction in H order)."""
    ci = ClassInfo(*cls, kind)
    g, nodes, R, H = build_rh(ci, rule)
    r_ops = g.op_count(R)
    h_ops = g.op_count(list(H.values()))
    hr = {k: h_ops[k] - r_ops[k] for k in ("fma", "mul", "add", "flops")}
    contr = ci.contraction_fmas()
    lt = ci.lt
    # Boys: Taylor 7 x 3 + 9 x 2 series + 1, downward 4 lt (2 t f + e) * c; prefactor ~5 + rsqrt
    # (counted as 1, as are the asymptotic branch's operations); s_n accumulation 2 (lt + 1) + scaling lt
    # (the erf operator's step adds ERF_PRIM_FLOPS: an add, a division, a square root, two multiplies)
    prim = 21 + 19 + 4 * lt + 8 + 2 * (lt + 1) + lt + 8
    return {"prim_flops": prim, "r_flops": r_ops["flops"], "h_flops": hr["flops"],
            "contraction_flops": 2 * contr, "pair_shell_flops": r_ops["flops"] + hr["flops"] + 2 * contr,
            "nr": ci.nr, "nnz": ci.nnz, "nht": ci.nht}


# FP64 operations the erf operator's step (operator_header) adds per primitive triple
ERF_PRIM_FLOPS = 5

R_INLINE_MAX_LT = 8       # phase A's R straight-line up to this degree by default
THREAD_MAX_TERMS = 2500    # thread-mapped kernels only while the straight-line contraction is short


def candidates(cls, kind):
    """Schedules worth timing for a class: register-resident thread kernels while their code and
    accumulators stay small, block kernels with a few tile shapes (R inline or as a loop)."""
    ci = ClassInfo(*cls, kind)
    out = []
    nacc = len(ci.sets) * ci.nab * ci.nk
    if ci.lt <= 9 and ci.nk * ci.nnz <= THREAD_MAX_TERMS:
        if nacc <= 160:
            for order in ("H", "G"):
                out.append(CudaSchedule(mapping="thread", order=order, acc="reg", block=128))
        if kind == "e":
            out.append(CudaSchedule(mapping="thread", order="H", acc="global", block=128))
    codes = ["inline"] if ci.lt <= 8 else (["inline", "loop"] if ci.lt <= 12 else ["loop"])
    for code in codes:
        for kt, ipc, blk in ((32, 1, 256), (32, 2, 512), (16, 1, 256), (64, 1, 512), (8, 1, 256), (4, 1, 256), (2, 1, 128)):
            s = CudaSchedule(mapping="block", ktile=kt, ipchunk=ipc, block=blk, r_code=code)
            if block_geometry(ci, s) is not None:
                out.append(s)
    return out


def default_schedule(cls, kind):
    """The schedule used without a tuning record: the widest fitting block tile, R inline up to
    R_INLINE_MAX_LT."""
    ci = ClassInfo(*cls, kind)
    code = "inline" if ci.lt <= R_INLINE_MAX_LT else "loop"
    for s in candidates(cls, kind):
        if s.mapping == "block" and s.r_code == code:
            return s
    return candidates(cls, kind)[-1]


# ---------------------------------------------------------------- outputs


def load_schedules(root):
    p = pathlib.Path(root) / SCHEDULE_FILE
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("schedules", {})


def load_arch_schedules(root):
    """{arch: {class key: record}} of the architectures tuned on their own (e.g. "sm_90")."""
    p = pathlib.Path(root) / SCHEDULE_FILE
    if not p.exists():
        return {}
    arch = json.loads(p.read_text()).get("arch", {})
    return {a: v.get("schedules", {}) for a, v in arch.items()}


def arch_number(arch):
    """sm_90 -> 90 (10 major + minor, as linus-gpu's ARCH_KERNELS key)."""
    if not arch.startswith("sm_") or not arch[3:].isdigit():
        raise ValueError(f"architecture {arch!r}: use sm_XY")
    return int(arch[3:])


ARCH_FILE = "arch.rs"


def gen_arch_table(arch_entries):
    """crates/linus-gpu/src/kernels/eri3c_gen/arch.rs: the per-architecture variants."""
    o = ["// @generated by tools/intgen/intgen.py cuda -- do not edit by hand.",
         "//! Per-architecture variants of the generated three-center kernels: the classes whose",
         "//! schedule tuned on that architecture (tools/intgen/eri3c_schedules.json, \"arch\") differs",
         "//! from the default's, keyed by 10 major + minor of the compute capability. linus-gpu takes",
         "//! them on devices of that architecture (`eri3c_class::gen_kernel_for`), the defaults",
         "//! (`super::eri3c_gen::KERNELS`) elsewhere.",
         "#![allow(clippy::all, unused_imports)]",
         "",
         "use super::eri3c_class::{Eri3cKind, GenKernel, Mapping};",
         "",
         f"/// (architecture, kernel), by architecture, then kind and class.",
         f"pub static ARCH_KERNELS: [(u32, GenKernel); {len(arch_entries)}] = ["]
    order = sorted(arch_entries, key=lambda e: (e[0], KINDS.index(e[2]), e[1]))
    for arch, cls, kind, fname, sched, meta, cost in order:
        pre = f'include_str!("{meta["prelude"]}")' if meta["prelude"] else '""'
        rec = kernel_record(cls, kind, sched, meta, cost, f'include_str!("{fname}")', pre)
        o.append(f"    ({arch_number(arch)}, {rec}),")
    o.append("];")
    return "\n".join(o) + "\n"


def class_key(cls, kind):
    return f"{kind}{cls[0]}{cls[1]}{cls[2]}"


def kernel_record(cls, kind, sched, meta, cost, source_expr, prelude_expr='""'):
    """One `GenKernel` literal of the generated table."""
    ci = ClassInfo(*cls, kind)
    la, lb, lk = cls
    mp = "Mapping::Thread" if meta["mapping"] == "thread" else "Mapping::Block"
    return (f"GenKernel {{ kind: {KIND_RUST[kind]}, la: {la}, lb: {lb}, lk: {lk}, entry: \"{kernel_name(cls, kind, sched)}\", "
            f"schedule: \"{sched.tag()}\", prelude: {prelude_expr}, source: {source_expr}, mapping: {mp}, threads: {meta['threads']}, "
            f"smem: {meta['smem']}, ktile: {meta['ktile']}, rec: {ci.rec}, nr: {ci.nr}, nht: {ci.nht}, "
            f"nnz: {ci.nnz}, nrows: {len(ci.rows)}, hash: {ci.pattern_hash():#018x}, "
            f"prim_flops: {cost['prim_flops']}, pair_shell_flops: {cost['pair_shell_flops']} }}")


def gen_rust_tables(entries):
    """crates/linus-gpu/src/kernels/eri3c_gen/mod.rs: per class and kind the kernel source, entry
    point, schedule, launch data and the shape of its pair records."""
    o = ["// @generated by tools/intgen/intgen.py cuda -- do not edit by hand.",
         "//! Generated three-center kernels (tools/intgen/cuda_eri3c.py): one CUDA source per class",
         "//! (l_a l_b | l_k), l_a >= l_b, and kind (`Eri3cKind`: integrals, first derivatives, second",
         "//! derivatives), each with",
         "//! the schedule the tuner recorded in tools/intgen/eri3c_schedules.json (or the default),",
         "//! the shape of its pair records and a hash of the E patterns it was generated for. A",
         "//! kernel's source is its `prelude` (a shared R recursion, r*.cuh) followed by `source`.",
         "#![allow(clippy::all)]",
         "",
         "use super::eri3c_class::{Eri3cKind, GenKernel, Mapping};",
         "",
         "/// operator.cuh: the operator step of the Boys stage (1/r, or erf(omega r)/r with",
         "/// `ERI3C_ERF` defined), pasted in front of every kernel.",
         f"pub static OPERATOR_H: &str = include_str!(\"{OPERATOR_FILE}\");",
         "",
         "/// hessian.cuh: the second-derivative kernels' epilogue (the centre blocks of the Hessian in",
         "/// fixed point), pasted in front of every second-derivative kernel after the accumulator.",
         f"pub static HESSIAN_H: &str = include_str!(\"{HESSIAN_FILE}\");",
         "",
         f"/// Every generated kernel, integrals first, then by class.",
         f"pub static KERNELS: [GenKernel; {len(entries)}] = ["]
    for (cls, kind), (fname, sched, meta, cost) in sorted(entries.items(), key=lambda e: (KINDS.index(e[0][1]), e[0][0])):
        pre = f'include_str!("{meta["prelude"]}")' if meta["prelude"] else '""'
        o.append("    " + kernel_record(cls, kind, sched, meta, cost, f'include_str!("{fname}")', pre) + ",")
    o.append("];")
    return "\n".join(o) + "\n"


def eligible(sched, kind):
    """Derivative kernels need register or block accumulators (the Gamma epilogue)."""
    return not (kind != "e" and sched.mapping == "thread" and sched.acc == "global")


# Code size breaks near-ties: among the recorded candidates within SIZE_TOLERANCE of the fastest
# eligible one, the production source takes the shortest (straight-line R and contractions
# multiply source size for a few percent of speed).
SIZE_TOLERANCE = 0.05


def recorded_schedule(entry, kind, cls=None):
    """The schedule of a tuning record entry for production: the shortest source among the
    eligible candidates within SIZE_TOLERANCE of the fastest (the fastest without `cls`), or
    None."""
    if not entry:
        return None
    times = sorted((v[0] if isinstance(v, list) else v, t) for t, v in entry.get("candidates", {}).items())
    times = [(ms, CudaSchedule.parse(t)) for ms, t in times]
    times = [(ms, s) for ms, s in times if eligible(s, kind)]
    if not times:
        s = CudaSchedule.parse(entry["schedule"])
        return s if eligible(s, kind) else None
    best = times[0][0]
    near = [s for ms, s in times if ms <= best * (1 + SIZE_TOLERANCE)]
    if cls is None or len(near) == 1:
        return near[0]
    sized = []
    for s in near:
        src, _ = emit(cls, kind, s)
        if src is not None:
            sized.append((len(src), s))
    return min(sized, key=lambda x: x[0])[1] if sized else near[0]


def production(root, only=None):
    """{path: text} of the production sources: one .cu per class and kind with its recorded (or
    default) schedule, and mod.rs."""
    rec = load_schedules(root)
    files = {}
    entries = {}
    gen = pathlib.Path(root) / GEN_DIR
    for cls in classes():
        for kind in KINDS:
            key = class_key(cls, kind)
            sched = recorded_schedule(rec.get(key), kind, cls) or default_schedule(cls, kind)
            src, meta = emit(cls, kind, sched)
            if src is None:
                sched = default_schedule(cls, kind)
                src, meta = emit(cls, kind, sched)
            fname = f"{key}.cu"
            files[gen / fname] = src
            if meta["prelude"]:
                ci = ClassInfo(*cls, kind)
                files[gen / meta["prelude"]] = rfun_source(ci.lt, ci.lk, sched.r_rule)
            entries[(cls, kind)] = (fname, sched, meta, costs(cls, kind, sched.r_rule))
    files[gen / OPERATOR_FILE] = operator_header()
    files[gen / HESSIAN_FILE] = hessian_header()
    files[gen / "mod.rs"] = gen_rust_tables(entries)
    # the per-architecture variants: classes whose schedule tuned on the architecture differs
    # from the default's
    arch_entries = []
    for arch, rec_a in sorted(load_arch_schedules(root).items()):
        arch_number(arch)
        for cls in classes():
            for kind in KINDS:
                sched = recorded_schedule(rec_a.get(class_key(cls, kind)), kind, cls)
                if sched is None or sched.tag() == entries[(cls, kind)][1].tag():
                    continue
                src, meta = emit(cls, kind, sched)
                if src is None:
                    continue
                fname = f"{class_key(cls, kind)}.{arch}.cu"
                files[gen / fname] = src
                if meta["prelude"]:
                    ci = ClassInfo(*cls, kind)
                    files[gen / meta["prelude"]] = rfun_source(ci.lt, ci.lk, sched.r_rule)
                arch_entries.append((arch, cls, kind, fname, sched, meta, costs(cls, kind, sched.r_rule)))
    files[gen / ARCH_FILE] = gen_arch_table(arch_entries)
    return files


def write_candidates(root, outdir, only):
    """Every candidate schedule of the selected classes as sources plus manifest.json for the
    tuner (crates/linus-gpu/examples/eri3c_tune.rs)."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    man = []
    for cls in classes():
        for kind in KINDS:
            key = class_key(cls, kind)
            if only and key not in only and key[1:] not in only:
                continue
            ci = ClassInfo(*cls, kind)
            cost = costs(cls, kind)
            for s in candidates(cls, kind):
                src, meta = emit(cls, kind, s)
                if src is None:
                    continue
                fname = f"{key}_{s.tag()}.cu"
                (outdir / fname).write_text(src)
                if meta["prelude"]:
                    (outdir / meta["prelude"]).write_text(rfun_source(ci.lt, ci.lk, s.r_rule))
                man.append({"key": key, "kind": kind, "la": cls[0], "lb": cls[1], "lk": cls[2],
                            "schedule": s.tag(), "entry": kernel_name(cls, kind, s), "file": fname,
                            "rec": ci.rec, "nrows": len(ci.rows), "hash": ci.pattern_hash(), **meta, **cost})
    (outdir / OPERATOR_FILE).write_text(operator_header())
    (outdir / HESSIAN_FILE).write_text(hessian_header())
    (outdir / "manifest.json").write_text(json.dumps(man, indent=1))
    for m in man:
        m["prelude"] = m["prelude"] or "-"
    cols = ["key", "schedule", "entry", "file", "mapping", "threads", "smem", "ktile", "rec", "nrows", "hash",
            "prelude"]
    (outdir / "manifest.tsv").write_text("\n".join(["\t".join(cols)] + ["\t".join(str(m[c]) for c in cols) for m in man]) + "\n")
    print(f"{len(man)} candidate kernels in {outdir}")


def merge_record(root, paths, arch=None):
    """Merges tuner results (JSON lines of eri3c_tune) into the schedule record (`arch`: into
    that architecture's own record): per class the last result wins."""
    p = pathlib.Path(root) / SCHEDULE_FILE
    data = json.loads(p.read_text()) if p.exists() else {}
    if arch:
        arch_number(arch)
        sch = data.setdefault("arch", {}).setdefault(arch, {}).setdefault("schedules", {})
    else:
        sch = data.setdefault("schedules", {})
    n = 0
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                sch[r["key"]] = {k: v for k, v in r.items() if k != "key"}
                n += 1
    data["about"] = ("Schedules of the generated three-center kernels (tools/intgen/cuda_eri3c.py), "
                     "chosen per class by crates/linus-gpu/examples/eri3c_tune.rs: the fastest "
                     "candidate on the workload, with every candidate's time in ms, registers and "
                     "local bytes. Classes without an entry use cuda_eri3c.default_schedule.")
    if "arch" in data:
        data["about"] += (" \"arch\": the records of architectures tuned on their own (sm_XY); where one "
                          "chooses another schedule than the default's, its variant is generated beside "
                          "the default and used on devices of that architecture.")
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"merged {n} results into {p}")


def add_cli(sub):
    p = sub.add_parser("cuda", help="the linus-gpu three-center kernels")
    p.add_argument("--candidates", metavar="DIR", help="write every candidate schedule (for the tuner) instead")
    p.add_argument("--only", nargs="*", help="class keys (e.g. e222 or 222) for --candidates")
    p.add_argument("--record", nargs="+", metavar="JSONL", help="merge tuner results into the schedule record, then regenerate")
    p.add_argument("--arch", metavar="sm_XY", help="with --record: the results are of this architecture (its own record)")


def run_cli(args, root):
    if args.candidates:
        write_candidates(root, args.candidates, set(args.only or []))
        return None
    if args.record:
        merge_record(root, args.record, args.arch)
    return production(root)
