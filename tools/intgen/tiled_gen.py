"""Rust emitter (layer c), part 3: crates/linus-ints/src/generated/tiled.rs, generated pieces of the
tiled McMurchie-Davidson kernels (crates/linus-ints/src/eri4c/batch.rs,
`block_eri_tiled`), for spherical pair classes up to (gg):

- r_cube_L (L = 0..16): R_tuv from R^(n)_000, written to the linear "cube" layout
  t * CUBE_S1 + u * CUBE_S2 + v (so R_{r+s} = cube[base(r) + base(s)]);
- tstep_<la><lb>: T step with this class as the ket. For each column cd and chunk of up to
  CHUNK_T nonzeros (s, cd) of E'[s][cd], the ket E values stay in registers while the bra
  Hermite index r streams:  T[cd][r] += sum_s cube[base(r) + base(s)] E'[s][cd];
- istep_<la><lb>: I step with this class as the bra. For each row ab and chunk of up to
  CHUNK_I nonzeros (r, ab) of E[r][ab], the bra E values stay in registers while cd streams:
  out[ab][cd] += sum_r E[r][ab] T[cd][r].

The cube offsets of s and the Hermite indices r are compile-time constants, so the loads use
immediate displacements and no registers hold offsets (the run-time-plan version needs one
register per offset, which caps its chunks at 8 and costs ~3 instructions per FMA).

The chunk sizes, the two interleaved partial sums and the cube strides are the schedule
(ir.RustSchedule); the sums are ordered FMA chains of an ir.Graph.
"""
from ir import RUST, Graph, chunks
from mdmath import e_pattern, nfunc, nherm, herm_tuv
from rtensor_gen import HEADER, RustPrinter, r_lets

# Schedule (ir.RUST): pair classes (la, lb) with la, lb <= tiled_max_l; R cube up to total L
# tiled_total_l; generated r_cube_L only up to r_cube_max_l: every `let` opens a nested debuginfo
# scope, and LLVM's DWARF emission recursed out of rustc's stack for L = 16 (4845 lets); above, the
# kernel copies the Hermite-order R into the cube (R is a negligible part of those classes).


def cube(t, u, v, sched=RUST):
    s1, s2 = sched.cube_strides
    return t * s1 + u * s2 + v


def pair_classes(sched=RUST):
    return [(la, lb) for la in range(sched.tiled_max_l + 1) for lb in range(la + 1)]


def class_index(la, lb):
    return la * (la + 1) // 2 + lb


def gen_r_cube(L, sched=RUST):
    """Straight-line R_tuv (t+u+v <= L) into the cube layout (same recursion as rtensor.rs)."""
    body, pr, nodes = r_lets(L, sched.r_rule)
    for (t, u, v) in herm_tuv(L):
        body.append(f"    *out.add({cube(t, u, v, sched)}) = {pr.ref(nodes[(0, t, u, v)])};")
    lines = [f"/// R_tuv for t+u+v <= {L} from rn[n] = R^(n)_000 into the cube layout.",
             "/// # Safety\n/// `out` must hold CUBE_N vectors.",
             "#[inline(always)]",
             "#[allow(unused_variables)]",
             f"pub unsafe fn r_cube_{L}<V: Real>(rn: &[V], x: V, y: V, z: V, out: *mut V) {{",
             f"    let rn = &rn[..{L + 1}];"]
    lines += body
    lines.append("}")
    return "\n".join(lines)


def split_sum(terms, init, nsplit, indent):
    """An ordered sum init + sum_j a_j b_j as `nsplit` interleaved partial sums a0, a1, ... (term j
    into a_{j mod nsplit}, a partial sum without an initial value starting from a product), each step
    a `let` rebinding. Returns the statements and the expression of the total (a0 + a1 + ...).
    terms: [(name of a_j, name of b_j)]; init: name of the initial value."""
    g = Graph()
    pr = RustPrinter()
    accs = [g.inp(init)] + [None] * (nsplit - 1)
    stmts = []
    for j, (a, b) in enumerate(terms):
        s = j % nsplit
        an, bn = g.inp(a), g.inp(b)
        node = g.mul(an, bn) if accs[s] is None else g.fma(an, bn, accs[s])
        stmts.append(f"{indent}let a{s} = {pr.defn(node)};")
        pr.bind(node, f"a{s}")
        accs[s] = node
    total = None
    for x in accs:
        if x is not None:
            total = x if total is None else g.add(total, x)
    return stmts, pr.ref(total)


def gen_tstep(la, lb, sched=RUST):
    nz = e_pattern(la, lb)
    tuv = herm_tuv(la + lb)
    ncd = nfunc(la) * nfunc(lb)
    bycol = {}
    for k, (s, cd) in enumerate(nz):
        bycol.setdefault(cd, []).append((cube(*tuv[s], sched=sched), k))
    o = [f"/// T step, ket class ({la}{lb}): {len(nz)} nonzeros over {ncd} columns.",
         "/// With FIRST, T is written instead of accumulated: the first chunk of every column starts",
         "/// from zero (the same sums as accumulating into zeroed T), empty columns are zeroed.",
         "/// # Safety\n/// `cube` holds R up to la + lb + bra L; `e` the lane E nonzeros of one ket primitive;",
         "/// `t` ncd x nhb vectors (column-major); `base` the cube offsets of the bra Hermite indices.",
         "#[inline(always)]",
         f"pub unsafe fn tstep_{la}{lb}<V: Lanes, const FIRST: bool>(cube: *const V, base: &[u32], e: *const f64, t: *mut V) {{",
         "    let nhb = base.len();"]
    for cd in range(ncd):
        if not bycol.get(cd):
            o.append("    if FIRST {")
            o.append(f"        let col = t.add({cd} * nhb);")
            o.append("        for rr in 0..nhb {")
            o.append("            *col.add(rr) = V::zero();")
            o.append("        }")
            o.append("    }")
        for ci, ch in enumerate(chunks(bycol.get(cd, []), sched.chunk_t)):
            o.append("    {")
            for j, (_, k) in enumerate(ch):
                o.append(f"        let e{j} = V::load8(&*(e.add({k} * 8) as *const [f64; 8]));")
            o.append(f"        let col = t.add({cd} * nhb);")
            o.append("        for (rr, &b) in base.iter().enumerate() {")
            o.append("            let rp = cube.add(b as usize);")
            o.append("            let x = col.add(rr);")
            if ci == 0:
                o.append("            let x0 = if FIRST { V::zero() } else { *x };")
            stmts, total = split_sum([(f"*rp.add({off})", f"e{j}") for j, (off, _) in enumerate(ch)],
                                     "x0" if ci == 0 else "*x", sched.partial_sums, "            ")
            o += stmts
            o.append(f"            *x = {total};")
            o.append("        }")
            o.append("    }")
    o.append("}")
    return "\n".join(o)


def gen_istep(la, lb, sched=RUST):
    nz = e_pattern(la, lb)
    nh = nherm(la + lb)
    nab = nfunc(la) * nfunc(lb)
    byrow = {}
    for k, (r, ab) in enumerate(nz):
        byrow.setdefault(ab, []).append((r, k))
    o = [f"/// I step, bra class ({la}{lb}): {len(nz)} nonzeros over {nab} rows, nhb = {nh}.",
         "/// With FIRST, `out` is written instead of accumulated (first chunk of every row from zero,",
         "/// empty rows zeroed).",
         "/// # Safety\n/// `t` ncd x nhb vectors (column-major), `eb` the bra E nonzeros of one primitive,",
         "/// `out` nab x ncd vectors; columns cd0..cd1 are updated.",
         "#[inline(always)]",
         f"pub unsafe fn istep_{la}{lb}<V: Lanes, const FIRST: bool>(t: *const V, eb: &[f64], out: *mut V, ncd: usize, cd0: usize, cd1: usize) {{",
         f"    let eb = &eb[..{len(nz)}];"]
    for ab in range(nab):
        if not byrow.get(ab):
            o.append("    if FIRST {")
            o.append(f"        let orow = out.add({ab} * ncd);")
            o.append("        for cd in cd0..cd1 {")
            o.append("            *orow.add(cd) = V::zero();")
            o.append("        }")
            o.append("    }")
        for ci, ch in enumerate(chunks(byrow.get(ab, []), sched.chunk_i)):
            o.append("    {")
            for j, (_, k) in enumerate(ch):
                o.append(f"        let b{j} = V::splat(eb[{k}]);")
            o.append(f"        let orow = out.add({ab} * ncd);")
            o.append("        for cd in cd0..cd1 {")
            o.append(f"            let tp = t.add(cd * {nh});")
            o.append("            let x = orow.add(cd);")
            if ci == 0:
                o.append("            let x0 = if FIRST { V::zero() } else { *x };")
            stmts, total = split_sum([(f"b{j}", f"*tp.add({r})") for j, (r, _) in enumerate(ch)],
                                     "x0" if ci == 0 else "*x", sched.partial_sums, "            ")
            o += stmts
            o.append(f"            *x = {total};")
            o.append("        }")
            o.append("    }")
    o.append("}")
    return "\n".join(o)


def generate(sched=RUST):
    out = [HEADER,
           "//! Tiled McMurchie-Davidson kernel pieces (see tools/intgen/tiled_gen.py): R in the cube",
           "//! layout, and per spherical pair class the T step (class as ket) and I step (class as bra)",
           "//! with compile-time offsets.", "",
           "use linus_num::simd::{Lanes, Real};", "",
           f"pub const TILED_GEN_MAX_L: usize = {sched.tiled_max_l};",
           f"pub const R_CUBE_MAX_L: usize = {sched.r_cube_max_l};",
           f"pub const CUBE_S1: usize = {sched.cube_strides[0]};",
           f"pub const CUBE_S2: usize = {sched.cube_strides[1]};", ""]
    for L in range(sched.r_cube_max_l + 1):
        out.append(gen_r_cube(L, sched))
        out.append("")
    cls = pair_classes(sched)
    for la, lb in cls:
        out.append(gen_tstep(la, lb, sched))
        out.append("")
        out.append(gen_istep(la, lb, sched))
        out.append("")
    out.append("/// Index of pair class (la, lb), la >= lb, in the tables of `tiled_isa_fns!`.")
    out.append("#[inline]")
    out.append("pub fn class_index(la: usize, lb: usize) -> usize {")
    out.append("    la * (la + 1) / 2 + lb")
    out.append("}")
    out.append("")
    out.append("/// `#[target_feature]` wrappers for one lane type: tables R_CUBE_FNS (index L), TSTEP_FNS,")
    out.append("/// ISTEP_FNS and their FIRST variants TSTEP_FIRST_FNS, ISTEP_FIRST_FNS (index `class_index`).")
    out.append("#[macro_export]")
    out.append("macro_rules! tiled_isa_fns {")
    out.append("    ($V:ty $(, $feat:literal)?) => {")
    for L in range(sched.r_cube_max_l + 1):
        out.append(f"        $(#[target_feature(enable = $feat)])? pub unsafe fn r_cube_{L}(rn: &[$V], x: $V, y: $V, z: $V, out: *mut $V) {{ $crate::generated::tiled::r_cube_{L}(rn, x, y, z, out) }}")
    for la, lb in cls:
        for suf, first in (("", "false"), ("_first", "true")):
            out.append(f"        $(#[target_feature(enable = $feat)])? pub unsafe fn tstep_{la}{lb}{suf}(cube: *const $V, base: &[u32], e: *const f64, t: *mut $V) {{ $crate::generated::tiled::tstep_{la}{lb}::<$V, {first}>(cube, base, e, t) }}")
            out.append(f"        $(#[target_feature(enable = $feat)])? pub unsafe fn istep_{la}{lb}{suf}(t: *const $V, eb: &[f64], out: *mut $V, ncd: usize, cd0: usize, cd1: usize) {{ $crate::generated::tiled::istep_{la}{lb}::<$V, {first}>(t, eb, out, ncd, cd0, cd1) }}")
    names = ", ".join(f"r_cube_{L}" for L in range(sched.r_cube_max_l + 1))
    out.append(f"        pub static R_CUBE_FNS: [unsafe fn(&[$V], $V, $V, $V, *mut $V); {sched.r_cube_max_l + 1}] = [{names}];")
    out.append(f"        pub static TSTEP_FNS: [unsafe fn(*const $V, &[u32], *const f64, *mut $V); {len(cls)}] = [" + ", ".join(f"tstep_{a}{b}" for a, b in cls) + "];")
    out.append(f"        pub static TSTEP_FIRST_FNS: [unsafe fn(*const $V, &[u32], *const f64, *mut $V); {len(cls)}] = [" + ", ".join(f"tstep_{a}{b}_first" for a, b in cls) + "];")
    out.append(f"        pub static ISTEP_FNS: [unsafe fn(*const $V, &[f64], *mut $V, usize, usize, usize); {len(cls)}] = [" + ", ".join(f"istep_{a}{b}" for a, b in cls) + "];")
    out.append(f"        pub static ISTEP_FIRST_FNS: [unsafe fn(*const $V, &[f64], *mut $V, usize, usize, usize); {len(cls)}] = [" + ", ".join(f"istep_{a}{b}_first" for a, b in cls) + "];")
    out.append("    };")
    out.append("}")
    return "\n".join(out) + "\n"
