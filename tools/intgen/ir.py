"""Layer (b) of the integral generator: a small expression IR and the schedules.

- `Graph`: an expression DAG over FP64 values (inputs, constants, mul, add, sub, neg, fma) with
  hash-consing, so building the same expression twice yields one node (common-subexpression
  elimination by construction). Optional algebraic folding (multiplications by +-1 and 0, FMAs with
  a zero addend) for targets that do not need to reproduce a fixed operation sequence. Sums are
  built as *ordered* FMA chains (`chain`): the order fixes the rounding, so a schedule never
  reassociates them.
- `Graph.order` / `live_values` / `op_count`: dead-code elimination from the outputs, a topological
  emission order, FP64 operation counts (what the roofline is measured against) and the peak number
  of live values of an order (register pressure).
- Algorithm builders shared by the emitters: `r_graph` (the Hermite R recursion of mdmath as a
  DAG), and the sparse contractions of the three-center kernels (see cuda_eri3c.py).
- Schedules: `RustSchedule` (the CPU kernels in crates/linus-ints/src/generated) and
  `CudaSchedule` (fusion, where values live, thread mapping, precision, unrolling of a CUDA
  kernel). The same graph printed under different schedules gives the variants the auto-tuner
  compares.
"""
from dataclasses import dataclass, field, replace

from mdmath import e_pattern, herm_idx, herm_tuv, nfunc, nherm, r_recursion


class Node:
    __slots__ = ("id", "op", "args", "value")

    def __init__(self, nid, op, args, value):
        self.id = nid
        self.op = op
        self.args = args
        self.value = value

    def __repr__(self):
        return f"Node({self.id}, {self.op}, {[a.id for a in self.args]}, {self.value!r})"


class Graph:
    """Hash-consed expression DAG. `fold` enables algebraic simplification (CUDA); the Rust emitters
    build without it, so their graphs keep the exact operation forms of the printed source."""

    def __init__(self, fold=False):
        self.nodes = []
        self.memo = {}
        self.fold = fold

    def _mk(self, op, args=(), value=None):
        key = (op, tuple(a.id for a in args), value)
        n = self.memo.get(key)
        if n is None:
            n = Node(len(self.nodes), op, tuple(args), value)
            self.nodes.append(n)
            self.memo[key] = n
        return n

    # leaves
    def inp(self, name):
        """A named value supplied by the emitter (parameter, register, load)."""
        return self._mk("in", (), name)

    def const(self, v):
        return self._mk("const", (), float(v))

    @staticmethod
    def is_const(n, v=None):
        return n.op == "const" and (v is None or n.value == v)

    # arithmetic
    def mul(self, a, b):
        if self.fold:
            for x, y in ((a, b), (b, a)):
                if self.is_const(x, 1.0):
                    return y
                if self.is_const(x, -1.0):
                    return self.neg(y)
                if self.is_const(x, 0.0):
                    return x
        return self._mk("mul", (a, b))

    def add(self, a, b):
        if self.fold:
            if self.is_const(a, 0.0):
                return b
            if self.is_const(b, 0.0):
                return a
            if b.op == "neg":
                return self.sub(a, b.args[0])
        return self._mk("add", (a, b))

    def sub(self, a, b):
        return self._mk("sub", (a, b))

    def neg(self, a):
        if a.op == "neg":
            return a.args[0]
        return self._mk("neg", (a,))

    def fma(self, a, b, c):
        """a * b + c."""
        if self.fold:
            if self.is_const(c, 0.0):
                return self.mul(a, b)
            for x, y in ((a, b), (b, a)):
                if self.is_const(x, 1.0):
                    return self.add(c, y) if not self.is_const(c, 0.0) else y
                if self.is_const(x, -1.0):
                    return self.sub(c, y)
                if self.is_const(x, 0.0):
                    return c
        return self._mk("fma", (a, b, c))

    def chain(self, terms, init=None):
        """The ordered sum init + a1 b1 + a2 b2 + ... as nested FMAs (a1 b1 first), or the first
        product when there is no initial value."""
        acc = init
        for a, b in terms:
            acc = self.mul(a, b) if acc is None else self.fma(a, b, acc)
        return acc

    # analysis
    def reachable(self, outputs):
        seen = set()
        stack = list(outputs)
        while stack:
            n = stack.pop()
            if n.id in seen:
                continue
            seen.add(n.id)
            stack.extend(n.args)
        return seen

    def order(self, outputs):
        """Nodes needed by `outputs`, in creation order (a topological order: arguments are always
        created before their users)."""
        need = self.reachable(outputs)
        return [n for n in self.nodes if n.id in need]

    def op_count(self, outputs):
        """FP64 operations needed by `outputs`: {'fma', 'mul', 'add'} counts (sub counts as add; neg
        is folded into its users by the hardware) and the flop total with an FMA as two."""
        c = {"fma": 0, "mul": 0, "add": 0}
        for n in self.order(outputs):
            if n.op == "fma":
                c["fma"] += 1
            elif n.op == "mul":
                c["mul"] += 1
            elif n.op in ("add", "sub"):
                c["add"] += 1
        c["flops"] = 2 * c["fma"] + c["mul"] + c["add"]
        return c

    def live_values(self, seq, outputs):
        """Peak number of simultaneously live computed values when the nodes of `seq` are evaluated
        in that order and `outputs` stay live to the end (inputs and constants not counted)."""
        last = {}
        for i, n in enumerate(seq):
            for a in n.args:
                last[a.id] = i
        for o in outputs:
            last[o.id] = len(seq)
        live = peak = 0
        ends = {}
        for i, n in enumerate(seq):
            if n.op not in ("in", "const"):
                live += 1
                ends.setdefault(last.get(n.id, i), []).append(n.id)
            peak = max(peak, live)
            live -= len(ends.pop(i, []))
        return peak


# ---------------------------------------------------------------- algorithm builders


def chunks(lst, n):
    """Split a reduction's terms into chunks of at most n, balanced (no tiny tail chunk)."""
    if not lst:
        return []
    k = -(-len(lst) // n)
    size = -(-len(lst) // k)
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def r_graph(g, L, rule, rn, xyz, targets=None, scale_const=None):
    """The Hermite R recursion of mdmath.r_recursion as nodes of `g`.

    rn(n): the node of R^(n)_000; xyz: the three nodes of the distance vector; targets: level-0
    indices needed (default all t + u + v <= L). Returns (need, nodes) with nodes[(n, t, u, v)] the
    node of R^(n)_tuv, built level by level from n = L down, in Hermite order within a level.

    Terms: R = X_d R1 (one term), X_d.fma(R1, R2) when i - 1 = 1, else X_d.fma(R1, R2 * (i - 1)),
    the forms of rtensor.rs. `scale_const(c)` overrides the constant node (default g.const)."""
    need, rules = r_recursion(L, rule, targets)
    cst = scale_const or g.const
    nodes = {}
    for n in range(L, -1, -1):
        for (t, u, v) in sorted(need.get(n, ()), key=lambda x: herm_idx(*x)):
            if (t, u, v) == (0, 0, 0):
                nodes[(n, 0, 0, 0)] = rn(n)
                continue
            d, i, terms = rules[(n, t, u, v)]
            a = nodes[(n + 1,) + terms[0]]
            if len(terms) == 1:
                nodes[(n, t, u, v)] = g.mul(xyz[d], a)
            else:
                b = nodes[(n + 1,) + terms[1]]
                if i - 1 == 1:
                    nodes[(n, t, u, v)] = g.fma(xyz[d], a, b)
                else:
                    nodes[(n, t, u, v)] = g.fma(xyz[d], a, g.mul(b, cst(float(i - 1))))
    return need, nodes


def shark_t_terms(cb, ck):
    """SHARK's T step for bra class cb and ket class ck: T[r][cd] += sum_s R[r + s] E'[s][cd] over
    the ket E nonzeros (s, cd). Returns [(t_out, [(ri, k), ...])] with t_out = r * ncd + cd in
    output order (cd outer, bra Hermite index r inner) and each term (index of R_{r+s}, ket
    nonzero k) in summation order."""
    (la, lb), (lc, ld) = cb, ck
    nhb = nherm(la + lb)
    ncd = nfunc(lc) * nfunc(ld)
    nzk = e_pattern(lc, ld)
    tuv_b, tuv_k = herm_tuv(la + lb), herm_tuv(lc + ld)
    by_cd = {}
    for k, (s, cd) in enumerate(nzk):
        by_cd.setdefault(cd, []).append((s, k))
    out = []
    for cd in range(ncd):
        terms = by_cd.get(cd, [])
        if not terms:
            continue
        for rr in range(nhb):
            tr = tuv_b[rr]
            out.append((rr * ncd + cd, [(herm_idx(tr[0] + tuv_k[s][0], tr[1] + tuv_k[s][1], tr[2] + tuv_k[s][2]), k)
                                        for s, k in terms]))
    return out


def shark_i_terms(cb, ck):
    """SHARK's I step: I[ab][cd] += sum_r E[r][ab] T[r][cd] over the bra E nonzeros (r, ab).
    Returns [(ab * ncd + cd, [(bra nonzero k, index r * ncd + cd of T), ...])] in output order."""
    (la, lb), (lc, ld) = cb, ck
    nab = nfunc(la) * nfunc(lb)
    ncd = nfunc(lc) * nfunc(ld)
    by_ab = {}
    for k, (rr, ab) in enumerate(e_pattern(la, lb)):
        by_ab.setdefault(ab, []).append((rr, k))
    out = []
    for ab in range(nab):
        terms = by_ab.get(ab, [])
        if not terms:
            continue
        for cd in range(ncd):
            out.append((ab * ncd + cd, [(k, rr * ncd + cd) for rr, k in terms]))
    return out


# ---------------------------------------------------------------- schedules


@dataclass(frozen=True)
class RustSchedule:
    """Schedule of the CPU kernels in crates/linus-ints/src/generated (all generic over `V: Real`,
    f64 or 8 lanes of f64).

    - R tensors: straight-line up to `r_max_l`, recursion direction `r_rule`, every value bound to a
      `let` (registers).
    - Block kernels (classes.rs): shells up to `kernel_max_l`, class pairs with at most
      `kernel_max_terms` FMA terms or an (ss) ket; the bra E values bound to `let`s up to
      `bra_lets_max` nonzeros, splat at their uses above; ket pairs in `lanes` lanes with
      `lane_head` header vectors per lane record.
    - Tiled kernels (tiled.rs): pair classes up to `tiled_max_l`; R in the cube layout with strides
      `cube_strides` up to `r_cube_max_l`; T and I steps in chunks of `chunk_t` / `chunk_i` terms
      with register-resident E values and `partial_sums` interleaved accumulators.
    """
    r_max_l: int = 12
    r_rule: str = "min"
    pattern_max_l: int = 4
    kernel_max_l: int = 4
    kernel_max_terms: int = 700
    bra_lets_max: int = 1000
    lanes: int = 8
    lane_head: int = 5
    tiled_max_l: int = 4
    tiled_total_l: int = 16
    r_cube_max_l: int = 12
    cube_strides: tuple = (293, 52)
    chunk_t: int = 16
    chunk_i: int = 16
    partial_sums: int = 2


RUST = RustSchedule()


@dataclass(frozen=True)
class CudaSchedule:
    """Schedule of a fused three-center kernel (Boys, R recursion and contractions in one kernel).

    mapping
      "thread": one thread per (orbital pair, auxiliary shell); straight-line code with R and H
                in registers; the thread is the single writer of its block of outputs.
      "block":  one block per (orbital pair, tile of `ktile` auxiliary shells). Phase A: one
                thread per job (pair primitive, auxiliary shell) of a stage of `ipchunk` pair
                primitives computes the contracted R tensor into shared memory (straight-line).
                Phase A2: all threads form H[k][t'] in shared memory. Phase B: one thread per
                (auxiliary shell, output row), table-driven over the row's E nonzeros, with nk
                accumulators in registers until the single store.
    order     "thread" mapping: "H" (H[k][t'] = sum_c S_k[c] R[c+t'] first, then the E contraction;
              fewer flops for l_aux >= 2) or "G" (per row, g_c = sum_t' R[c+t'] E[mn][t'] first,
              then S; fewer live values).
    r_rule    direction rule of the R recursion (mdmath.r_direction).
    r_code    "block" mapping, phase A: "inline" (straight-line R with dead values removed, in
              registers) or "loop" (the universal recursion program of linus-ints' eri3c over
              per-thread arrays: small code for high degrees).
    acc       "thread" mapping: "reg" (accumulate in registers, one store) or "global" (read-
              modify-write of the output per pair primitive, the spike's choice).
    block     "thread": threads per block; "block": the target threads per block (ktile times the
              row threads, at most 1024).
    ktile     "block" mapping: auxiliary shells per block (consecutive threads take consecutive
              shells, so a warp shares its output row when ktile is a multiple of the warp size).
    ipchunk   "block" mapping: pair primitives per shared-memory stage.
    minblocks second argument of __launch_bounds__ (0: none).
    precision "f64" (the only one; reduced precision only per kernel after an accuracy study).
    """
    mapping: str = "thread"
    order: str = "H"
    r_rule: str = "first"
    r_code: str = "inline"
    acc: str = "reg"
    block: int = 128
    ktile: int = 32
    ipchunk: int = 1
    minblocks: int = 0
    precision: str = "f64"

    def tag(self):
        if self.mapping == "thread":
            t = f"t{self.order}{self.r_rule[0]}{self.acc[0]}{self.block}"
        else:
            t = f"b{self.ktile}c{self.ipchunk}t{self.block}{self.r_rule[0]}{self.r_code[0]}"
        return t + (f"-l{self.minblocks}" if self.minblocks else "")

    @staticmethod
    def parse(tag):
        """Inverse of `tag`."""
        mb = 0
        if "-l" in tag:
            tag, mbs = tag.split("-l")
            mb = int(mbs)
        rule = {"f": "first", "m": "min"}
        if tag[0] == "t":
            return CudaSchedule(mapping="thread", order=tag[1], r_rule=rule[tag[2]],
                                acc={"r": "reg", "g": "global"}[tag[3]], block=int(tag[4:]), minblocks=mb)
        if tag[0] == "b":
            kt, rest = tag[1:].split("c")
            ipc, rest = rest.split("t")
            return CudaSchedule(mapping="block", ktile=int(kt), ipchunk=int(ipc), block=int(rest[:-2]),
                                r_rule=rule[rest[-2]], r_code={"i": "inline", "l": "loop"}[rest[-1]], minblocks=mb)
        raise ValueError(tag)

    def with_(self, **kw):
        return replace(self, **kw)
