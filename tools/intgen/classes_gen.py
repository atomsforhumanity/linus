"""Rust emitter (layer c), part 2: crates/linus-ints/src/generated/classes.rs, the structural
E-matrix patterns and the class-specialized batched block kernels.

Block kernel (bra class (la lb), ket class (lc ld), L = la+lb+lc+ld), SHARK Scheme 1 with the
ket pairs in SIMD lanes:

    for each bra primitive ip:                       (scalar p, P, E_ip broadcast)
        T = 0
        for each ket primitive kq (8 lanes):         (lane record: q, 1/q, Q, signed E nonzeros)
            R = R_tuv(alpha, P - Q) * 2 pi^2.5 / (p q sqrt(p+q))       (Boys + unrolled R)
            T[r][cd] += sum_{s: (s,cd) nonzero} R[r+s] E'[s][cd]
        I[ab][cd] += sum_{r: (r,ab) nonzero} E_ip[r][ab] T[r][cd]

Every loop over Hermite indices and functions is unrolled over the structural nonzeros, so
known zeros cost nothing (SHARK's "loop-free matrix multiplication routines"). The patterns come
from mdmath, the T and I sums are ordered FMA chains of an ir.Graph, and the schedule
(ir.RustSchedule) decides which class pairs get a block kernel and where the bra E values live.
"""
from ir import RUST, Graph, shark_i_terms, shark_t_terms
from mdmath import e_pattern, nfunc, nherm, herm_tuv, herm_idx
from rtensor_gen import HEADER, RustPrinter

# Straight-line block kernels: shells up to RUST.kernel_max_l, for class pairs with at most
# RUST.kernel_max_terms FMA terms or an (ss) ket. Everything else up to (gg|gg) runs on the tiled
# kernels (tiled_gen.py), which measured faster above ~700 terms: e.g. (dp|dp) 857 vs 1169 ns,
# (fp|ds) 496 vs 668 ns, (fd|ps) 481 vs 793 ns per primitive-quartet vector (AVX-512, one core),
# while (ds|ds) 118 vs 149 ns and (ff|ss) 386 vs 488 ns stay straight-line.
# Bra E values are bound to `let`s up to RUST.bra_lets_max nonzeros (see gen_kernel).


def pair_classes(lmax):
    return [(la, lb) for la in range(lmax + 1) for lb in range(la + 1)]


def class_order_key(c):
    return (c[0] + c[1], c[0])


def kernel_terms(cb, ck):
    nzb, nzk = e_pattern(*cb), e_pattern(*ck)
    nhb = nherm(cb[0] + cb[1])
    ncd = nfunc(ck[0]) * nfunc(ck[1])
    return len(nzk) * nhb + len(nzb) * ncd


def kernel_list(sched=RUST):
    cls = sorted(pair_classes(sched.kernel_max_l), key=class_order_key)
    out = []
    for i, cb in enumerate(cls):
        for ck in cls[:i + 1]:
            if kernel_terms(cb, ck) <= sched.kernel_max_terms or ck == (0, 0):
                out.append((cb, ck))
    return out


def kname(cb, ck):
    return f"block_{cb[0]}{cb[1]}_{ck[0]}{ck[1]}"


def gen_kernel(cb, ck, sched=RUST):
    (la, lb), (lc, ld) = cb, ck
    lab, lcd = la + lb, lc + ld
    L = lab + lcd
    nhb = nherm(lab)
    na, nb, nc, nd = nfunc(la), nfunc(lb), nfunc(lc), nfunc(ld)
    nab, ncd = na * nb, nc * nd
    nzb, nzk = e_pattern(la, lb), e_pattern(lc, ld)
    tuv_b, tuv_k = herm_tuv(lab), herm_tuv(lcd)
    nrec = sched.lane_head + len(nzk)
    nnzb = len(nzb)
    o = []
    o.append(f"/// ({la}{lb}|{lc}{ld}) for one bra pair and {sched.lanes} ket pairs (lanes): out[ab * {ncd} + cd].")
    o.append(f"/// p, pc: bra primitives; eb: bra E nonzeros ({nnzb} per primitive); lane: ket records")
    o.append(f"/// ({nrec} lane vectors per ket primitive). Fixed-size views keep the body free of bounds checks.")
    o.append("#[inline(always)]")
    o.append("#[allow(unused_variables, unused_mut)]")
    o.append(f"pub fn {kname(cb, ck)}<V: Lanes>(p: &[f64], pc: &[[f64; 3]], eb: &[f64], lane: &[[f64; 8]], nk: usize, op: &Operator, qpb: &[f64], qpk: &[f64], cut: f64, out: &mut [V]) {{")
    o.append("    let coulomb = op.is_coulomb();")
    o.append(f"    const NREC: usize = {nrec};")
    o.append(f"    const NNZB: usize = {nnzb};")
    o.append(f"    let mut acc = [V::zero(); {nab * ncd}];")
    o.append(f"    let mut fm = [V::zero(); {L + 1}];")
    o.append(f"    let mut rn = [V::zero(); {L + 1}];")
    o.append(f"    let mut r = [V::zero(); {nherm(L)}];")
    o.append(f"    let pc = &pc[..p.len()];")
    o.append(f"    let qpb = &qpb[..p.len()];")
    o.append(f"    let qpk = &qpk[..nk];")
    o.append(f"    let btab = boys_table();")
    o.append("    for ip in 0..p.len() {")
    o.append("        let (pp, [px, py, pz]) = (p[ip], pc[ip]);")
    o.append("        let vp = V::splat(pp);")
    o.append("        let pref0 = V::splat(ERI_PREF / pp);")
    o.append("        let qb = qpb[ip];")
    o.append(f"        let mut t = [V::zero(); {nhb * ncd}];")
    o.append("        for kq in 0..nk {")
    o.append("            if qb * qpk[kq] < cut {")
    o.append("                continue;")
    o.append("            }")
    o.append("            let kd: &[[f64; 8]; NREC] = lane[kq * NREC..(kq + 1) * NREC].try_into().unwrap();")
    o.append("            let q = V::load8(&kd[0]);")
    o.append("            let inv_q = V::load8(&kd[1]);")
    o.append("            let dx = V::splat(px) - V::load8(&kd[2]);")
    o.append("            let dy = V::splat(py) - V::load8(&kd[3]);")
    o.append("            let dz = V::splat(pz) - V::load8(&kd[4]);")
    o.append("            let inv_s = V::splat(1.0) / (vp + q);")
    o.append("            let alpha = vp * q * inv_s;")
    o.append("            let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));")
    o.append("            let pref = pref0 * inv_q * inv_s.sqrt();")
    o.append("            if coulomb {")
    o.append(f"                boys_lanes_tab(btab, {L}, alpha * r2, &mut fm);")
    o.append("                rn[0] = pref * fm[0];")
    if L > 0:
        o.append("                let m2a = alpha * V::splat(-2.0);")
        o.append("                let mut f = pref;")
        for n in range(1, L + 1):
            o.append(f"                f *= m2a;")
            o.append(f"                rn[{n}] = f * fm[{n}];")
    o.append("            } else {")
    o.append(f"                rn_operator({L}, alpha, r2, pref, op, &mut fm, &mut rn);")
    o.append("            }")
    o.append(f"            r_tensor_{L}(&rn, dx, dy, dz, &mut r);")
    # ket E loads
    for k in range(len(nzk)):
        o.append(f"            let e{k} = V::load8(&kd[{sched.lane_head + k}]);")
    # T step: ordered FMA chains over the ket E nonzeros, grouped by cd
    g = Graph()
    pr = RustPrinter()
    for t_out, terms in shark_t_terms(cb, ck):
        node = g.chain([(g.inp(f"r[{ri}]"), g.inp(f"e{k}")) for ri, k in terms], init=g.inp(f"t[{t_out}]"))
        o.append(f"            t[{t_out}] = {pr.defn(node)};")
    o.append("        }")
    # I step
    o.append(f"        let eb: &[f64; NNZB] = eb[ip * NNZB..(ip + 1) * NNZB].try_into().unwrap();")
    # Bra E values as `let`s at the top of the I step (the (ss)-ket kernels measured 9-17%
    # slower with the splats at their uses), except for large bra classes: thousands of `let`s
    # nest debuginfo scopes so deeply that LLVM's DWARF emission overflowed rustc's stack
    # ((gg|ss), 7649 lets); those splat at their uses.
    lets = len(nzb) <= sched.bra_lets_max
    if lets:
        for k in range(len(nzb)):
            o.append(f"        let b{k} = V::splat(eb[{k}]);")
    bra = (lambda k: f"b{k}") if lets else (lambda k: f"V::splat(eb[{k}])")
    for x, terms in shark_i_terms(cb, ck):
        node = g.chain([(g.inp(bra(k)), g.inp(f"t[{ti}]")) for k, ti in terms], init=g.inp(f"acc[{x}]"))
        o.append(f"        acc[{x}] = {pr.defn(node)};")
    o.append("    }")
    o.append(f"    out[..{nab * ncd}].copy_from_slice(&acc);")
    o.append("}")
    return "\n".join(o)


def generate(sched=RUST):
    out = [HEADER, "//! Structural nonzero patterns of the SHARK E-matrices E[r][ab] for spherical pair",
           "//! classes (p shells Cartesian x, y, z; l >= 2 real solid harmonics), as (r, ab) sorted",
           "//! by r, then ab (the batched pair data stores E values in exactly this order), and",
           "//! class-specialized batched block kernels (see tools/intgen/classes_gen.py).", "",
           "use crate::core::boys::{boys_lanes_tab, boys_table};", "use crate::eri4c::ERI_PREF;", "use crate::generated::rtensor::*;",
           "use crate::operator::{rn_operator, Operator};", "use linus_num::simd::Lanes;", ""]
    names = []
    for la, lb in pair_classes(sched.pattern_max_l):
        nz = e_pattern(la, lb)
        nm = f"E_NZ_{la}{lb}"
        names.append((la, lb, nm))
        body = ", ".join(f"({r}, {ab})" for r, ab in nz)
        out.append(f"/// ({la}{lb}): nherm {nherm(la + lb)}, {nfunc(la)}x{nfunc(lb)} functions, {len(nz)} nonzeros")
        out.append(f"pub const {nm}: &[(u16, u16)] = &[{body}];")
    out.append("")
    out.append(f"pub const PATTERN_MAX_L: usize = {sched.pattern_max_l};")
    out.append(f"pub const LANE_HEAD: usize = {sched.lane_head};")
    out.append("")
    out.append("/// Pattern of the spherical pair class (la, lb), la >= lb, if generated.")
    out.append("pub fn e_pattern(la: usize, lb: usize) -> Option<&'static [(u16, u16)]> {")
    out.append("    Some(match (la, lb) {")
    for la, lb, nm in names:
        out.append(f"        ({la}, {lb}) => {nm},")
    out.append("        _ => return None,")
    out.append("    })")
    out.append("}")
    out.append("")
    kl = kernel_list(sched)
    out.append(f"/// Number of generated block kernels.")
    out.append(f"pub const N_BLOCK_KERNELS: usize = {len(kl)};")
    out.append("")
    out.append("/// Index of the generated block kernel for (la lb | lc ld), if any.")
    out.append("pub fn block_kernel_index(la: usize, lb: usize, lc: usize, ld: usize) -> Option<usize> {")
    out.append("    Some(match (la, lb, lc, ld) {")
    for i, (cb, ck) in enumerate(kl):
        out.append(f"        ({cb[0]}, {cb[1]}, {ck[0]}, {ck[1]}) => {i},")
    out.append("        _ => return None,")
    out.append("    })")
    out.append("}")
    out.append("")
    for cb, ck in kl:
        out.append(gen_kernel(cb, ck, sched))
        out.append("")
    # per-ISA wrappers
    out.append("/// `#[target_feature]` wrappers of the block kernels for one lane type, and the table")
    out.append("/// `BLOCK_FNS` indexed by `block_kernel_index`.")
    out.append("#[macro_export]")
    out.append("macro_rules! block_isa_fns {")
    out.append("    ($V:ty $(, $feat:literal)?) => {")
    for cb, ck in kl:
        n = kname(cb, ck)
        out.append(f"        $(#[target_feature(enable = $feat)])? pub unsafe fn {n}(p: &[f64], pc: &[[f64; 3]], eb: &[f64], lane: &[[f64; 8]], nk: usize, op: &$crate::operator::Operator, qpb: &[f64], qpk: &[f64], cut: f64, out: &mut [$V]) {{ $crate::generated::classes::{n}(p, pc, eb, lane, nk, op, qpb, qpk, cut, out) }}")
    out.append(f"        pub static BLOCK_FNS: [unsafe fn(&[f64], &[[f64; 3]], &[f64], &[[f64; 8]], usize, &$crate::operator::Operator, &[f64], &[f64], f64, &mut [$V]); {len(kl)}] = [")
    for cb, ck in kl:
        out.append(f"            {kname(cb, ck)},")
    out.append("        ];")
    out.append("    };")
    out.append("}")
    import sys
    print("block kernels:", ", ".join(f"({a}{b}|{c}{d}):{kernel_terms((a, b), (c, d))}" for (a, b), (c, d) in kl), file=sys.stderr)
    return "\n".join(out) + "\n"
