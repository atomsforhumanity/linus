//! Batched four-center integrals for integral-direct Fock builds.
//!
//! Shell pairs are stored once, sorted by angular-momentum class, number of retained
//! primitive pairs and Schwarz factor, and cut into blocks of [`LANES`] consecutive pairs of
//! the same class and primitive count. A kernel call evaluates one bra pair against one ket
//! block: every SIMD lane holds a different ket pair, so the Boys function, the R tensor and
//! the contractions run at full vector width even for uncontracted shells, without gathers.
//!
//! Per primitive pair the SHARK E-matrix E\[r\]\[ab\] (contraction coefficients and spherical
//! transform folded in, see `hermite::build_pair`) is stored over its structural nonzeros
//! only (`PairClass::nz`). The ket copy is lane-interleaved and carries the Hermite sign
//! (-1)^{t+u+v}, so (ab|cd) = sum_{ip,kq} sum_{r,s} E_ip\[r\]\[ab\] R_{r+s} E'_kq\[s\]\[cd\].
//!
//! Primitive pairs whose primitive Schwarz factor cannot matter at the requested accuracy are
//! dropped (SHARK's primitive-pair prescreening), and so are shell pairs.
//!
//! Kernels per (bra class, ket class): straight-line generated kernels for small class pairs
//! (`generated::classes`), the tiled kernel ([`block_eri_tiled`]) for everything else up to
//! (gg|gg), and generic loops ([`block_eri_generic`]) above.

use crate::basis::{cart_components, ncart, BasisSet};
use crate::core::boys::boys_lanes_tab;
use crate::core::hermite::{herm_idx, herm_tuv, nherm, sum_table};
use crate::eri4c::ERI_PREF;
use crate::generated::{classes as gclasses, rtensor, tiled as gtiled};
use crate::operator::{rn_operator, Operator};
use crate::pairs::PairSet;
use linus_num::simd::{Lanes, Real, LANES};

/// One angular-momentum class of shell pairs (la >= lb).
#[derive(Clone, Debug)]
pub struct PairClass {
    pub la: usize,
    pub lb: usize,
    pub na: usize,
    pub nb: usize,
    pub lab: usize,
    pub nh: usize,
    /// Structural nonzeros (r, ab) of E\[r\]\[ab\], sorted by (r, ab).
    pub nz: Vec<(u16, u16)>,
    /// True when `nz` is the generated pattern of this class (generated kernels apply).
    pub generated: bool,
    /// Position in the class order (by la + lb, then la).
    pub order: usize,
    /// Tiled-kernel plan of this class as the ket: per cd, the nonzeros (s, cd) in chunks
    /// (cube offset of s, nonzero index).
    pub tplan: Vec<Chunk>,
    /// Tiled-kernel plan of this class as the bra: per ab, the nonzeros (r, ab) in chunks
    /// (Hermite index r, nonzero index).
    pub iplan: Vec<Chunk>,
}

/// Up to [`CHUNK`] structural nonzeros of one column (cd of a ket E-matrix, or ab of a bra
/// E-matrix) that the tiled kernels keep in registers together.
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub col: u16,
    pub n: u8,
    /// T plan: cube offset of the ket Hermite index s; I plan: bra Hermite index r
    pub off: [u16; CHUNK],
    /// index of the nonzero in the class's `nz` list
    pub k: [u16; CHUNK],
}

/// Nonzeros per tiled-kernel chunk.
pub const CHUNK: usize = 8;

/// Linear ("cube") layout of the R tensor in the tiled kernels: R_tuv at
/// t * CUBE_S1 + u * CUBE_S2 + v, so that R_{r+s} = cube[base(r) + base(s)] with base additive
/// in (t, u, v). The strides are injective on t + u + v <= [`TILED_MAX_L`] and spread the
/// entries of every L evenly over the 64 sets of a 48 KB 12-way L1 (each entry is one SIMD
/// vector, one cache line for AVX-512): with strides (289, 17, 1) up to 25 of the 455 lines of
/// L = 12 shared a set and 19% of the kernel's loads missed L1; (293, 52, 1) puts at most
/// ceil(n / 64) + 2 lines in a set for every L <= 16. The holes cost address space only.
pub const CUBE_S2: usize = gtiled::CUBE_S2;
pub const CUBE_S1: usize = gtiled::CUBE_S1;
pub const CUBE_N: usize = TILED_MAX_L * CUBE_S1 + 1;
/// Bytes of T per column block of the tiled I step (tunable for benchmarks).
pub static TILED_IBLOCK_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(24 * 1024);

/// Largest total angular momentum (la + lb + lc + ld) of the tiled kernels: (gg|gg).
pub const TILED_MAX_L: usize = 16;

/// Cube offset of every Hermite index up to TILED_MAX_L.
pub fn cube_pos() -> &'static [u32] {
    static T: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let v: Vec<u32> = herm_tuv()[..nherm(TILED_MAX_L)]
            .iter()
            .map(|&[t, u, v]| (t as usize * CUBE_S1 + u as usize * CUBE_S2 + v as usize) as u32)
            .collect();
        let mut seen = vec![false; CUBE_N];
        for &x in &v {
            assert!(
                (x as usize) < CUBE_N && !seen[x as usize],
                "cube layout not injective"
            );
            seen[x as usize] = true;
        }
        v
    })
}

/// Group nonzeros (row, col) by column into chunks of 8, 4, 2 and 1 (no padding); `off` maps
/// the row index to the stored offset.
fn make_plan(
    nz: &[(u16, u16)],
    ncol: usize,
    by_col_of: impl Fn(u16, u16) -> (u16, u16),
    off: impl Fn(u16) -> u16,
) -> Vec<Chunk> {
    let mut cols: Vec<Vec<(u16, u16)>> = vec![Vec::new(); ncol];
    for (k, &(a, b)) in nz.iter().enumerate() {
        let (row, col) = by_col_of(a, b);
        cols[col as usize].push((off(row), k as u16));
    }
    let mut plan = Vec::new();
    for (col, list) in cols.iter().enumerate() {
        let mut i = 0;
        while i < list.len() {
            let rest = list.len() - i;
            let n = if rest >= 8 {
                8
            } else if rest >= 4 {
                4
            } else if rest >= 2 {
                2
            } else {
                1
            };
            let mut c = Chunk {
                col: col as u16,
                n: n as u8,
                off: [0; CHUNK],
                k: [0; CHUNK],
            };
            for j in 0..n {
                c.off[j] = list[i + j].0;
                c.k[j] = list[i + j].1;
            }
            plan.push(c);
            i += n;
        }
    }
    plan
}

impl PairClass {
    #[inline]
    pub fn nab(&self) -> usize {
        self.na * self.nb
    }
}

/// A retained shell pair (shell `sa` has the higher angular momentum).
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub class: u32,
    pub sa: u32,
    pub sb: u32,
    pub nprim: u32,
    /// first primitive in the scalar primitive arrays
    pub prim: u32,
    /// offset of the E values (nprim * nz) in `BatchedPairs::e`
    pub e: u32,
    /// Schwarz factor sqrt(max |(ab|ab)|)
    pub q: f64,
}

/// LANES consecutive pairs of one group; lanes >= nlanes are padding (zero E).
#[derive(Clone, Debug)]
pub struct Block {
    pub class: u32,
    pub first: u32,
    pub nlanes: u32,
    pub nprim: u32,
    /// offset of the lane-interleaved primitive records in `BatchedPairs::lane`
    pub data: usize,
    pub qmax: f64,
    pub sa: [u32; LANES],
    pub sb: [u32; LANES],
    pub q: [f64; LANES],
    /// offset in `BatchedPairs::blk_qpk` of the per-primitive maxima (over lanes) of the
    /// primitive Schwarz factors
    pub kq0: usize,
}

/// Pairs of one class with one primitive count (consecutive in `pairs`, blocks too).
#[derive(Clone, Debug)]
pub struct Group {
    pub class: u32,
    pub nprim: u32,
    pub pairs: std::ops::Range<usize>,
    pub blocks: std::ops::Range<usize>,
}

/// Lane record per ket primitive: q, 1/q, Qx, Qy, Qz, then the signed E nonzeros.
pub const LANE_HEAD: usize = 5;

pub struct BatchedPairs {
    pub nshell: usize,
    pub classes: Vec<PairClass>,
    pub pairs: Vec<Pair>,
    pub groups: Vec<Group>,
    pub blocks: Vec<Block>,
    /// scalar primitive data (bra side)
    pub p: Vec<f64>,
    /// primitive Schwarz factors sqrt(max |(ab|ab)|) of each retained primitive pair
    pub qp: Vec<f64>,
    /// per block and ket primitive: largest primitive Schwarz factor over the lanes
    pub blk_qpk: Vec<f64>,
    pub pc: Vec<[f64; 3]>,
    pub e: Vec<f64>,
    /// lane-interleaved primitive data (ket side), see `LANE_HEAD`
    pub lane: Vec<f64>,
    pub qmax: f64,
    /// generated block kernel per (bra class, ket class) (index into the per-ISA tables), or -1
    pub gen_kernel: Vec<i32>,
    /// per (bra class, ket class) without a generated kernel: use the tiled kernel
    /// (`block_eri_tiled`) instead of the generic loops
    pub tiled: Vec<bool>,
    /// statistics
    pub n_pairs_total: usize,
    pub n_prims_total: usize,
    pub n_prims_kept: usize,
}

/// Structural nonzeros of E\[r\]\[ab\] from the shell transforms (nfunc x ncart, row-major).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn e_pattern(la: usize, lb: usize, ta: &[f64], tb: &[f64]) -> Vec<(u16, u16)> {
    let (ca, cb) = (cart_components(la), cart_components(lb));
    let (nca, ncb) = (ncart(la), ncart(lb));
    let (na, nb) = (ta.len() / nca, tb.len() / ncb);
    let tuv = herm_tuv();
    let mut nz = Vec::new();
    for r in 0..nherm(la + lb) {
        let [t, u, v] = tuv[r];
        for mu in 0..na {
            for nu in 0..nb {
                let hit = (0..nca).any(|ia| {
                    ta[mu * nca + ia] != 0.0
                        && (0..ncb).any(|ib| {
                            tb[nu * ncb + ib] != 0.0
                                && t as usize <= ca[ia][0] + cb[ib][0]
                                && u as usize <= ca[ia][1] + cb[ib][1]
                                && v as usize <= ca[ia][2] + cb[ib][2]
                        })
                });
                if hit {
                    nz.push((r as u16, (mu * nb + nu) as u16));
                }
            }
        }
    }
    nz
}

impl BatchedPairs {
    /// The batched layout of the pairs `set` of `basis`, each pair oriented with the shell of
    /// higher angular momentum first (equal l: the set's order, higher index first), as the
    /// kernels need: a pair stored the other way round has its expansion coefficients
    /// transposed in (a, b) while they are copied (the Hermite expansion of phi_a phi_b is that
    /// of phi_b phi_a). Shell pairs with Q * Qmax < `pair_cut` and primitive pairs with
    /// Qprim * Qmax < `prim_cut` are dropped.
    pub fn new(basis: &BasisSet, set: &PairSet, pair_cut: f64, prim_cut: f64) -> Self {
        let sh = &basis.shells;
        let ns = sh.len();
        // classes
        let lmax = basis.max_l();
        let mut class_of = vec![usize::MAX; (lmax + 1) * (lmax + 1)];
        let mut classes = Vec::new();
        let mut keys: Vec<(usize, usize)> = (0..=lmax)
            .flat_map(|la| (0..=la).map(move |lb| (la, lb)))
            .collect();
        keys.sort_by_key(|&(la, lb)| (la + lb, la));
        for (order, &(la, lb)) in keys.iter().enumerate() {
            // representative shells for the transforms
            let ra = sh.iter().find(|s| s.l == la);
            let rb = sh.iter().find(|s| s.l == lb);
            let (Some(ra), Some(rb)) = (ra, rb) else {
                continue;
            };
            let ta = BasisSet::shell_transform(ra);
            let tb = BasisSet::shell_transform(rb);
            let nz = e_pattern(la, lb, &ta, &tb);
            let generated =
                basis.pure && gclasses::e_pattern(la, lb).is_some_and(|g| g == nz.as_slice());
            class_of[la * (lmax + 1) + lb] = classes.len();
            let (na, nb) = (ra.nfunc(), rb.nfunc());
            let (tplan, iplan) = if la + lb <= TILED_MAX_L {
                let cp = cube_pos();
                (
                    make_plan(&nz, na * nb, |s, cd| (s, cd), |s| cp[s as usize] as u16),
                    make_plan(&nz, na * nb, |r, ab| (r, ab), |r| r),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            classes.push(PairClass {
                la,
                lb,
                na,
                nb,
                lab: la + lb,
                nh: nherm(la + lb),
                nz,
                generated,
                order,
                tplan,
                iplan,
            });
        }
        // every shell pair (oriented la >= lb, equal l: higher shell index first), with its
        // Schwarz factor and primitive Schwarz factors
        let n_pairs_total = set.len();
        let qv = set.schwarz();
        let qmax = set.schwarz_max();
        let n_prims_total: usize = set.pairs().iter().map(|p| p.nprim).sum();
        // retained pairs and primitives
        struct Kept {
            class: usize,
            sa: usize,
            sb: usize,
            q: f64,
            prims: Vec<usize>,
            idx: usize,
            /// the set's pair has the lower angular momentum first: transposed here
            swap: bool,
        }
        let mut kept: Vec<Kept> = Vec::new();
        for (idx, (pr, q)) in set.pairs().iter().zip(qv).enumerate() {
            let qp = set.prim_schwarz(idx);
            if pr.nprim == 0 || q * qmax < pair_cut {
                continue;
            }
            let prims: Vec<usize> = (0..pr.nprim)
                .filter(|&ip| qp[ip] * qmax >= prim_cut)
                .collect();
            if prims.is_empty() {
                continue;
            }
            let swap = pr.la < pr.lb;
            let (sa, sb, la, lb) = if swap {
                (pr.sb, pr.sa, pr.lb, pr.la)
            } else {
                (pr.sa, pr.sb, pr.la, pr.lb)
            };
            kept.push(Kept {
                class: class_of[la * (lmax + 1) + lb],
                sa,
                sb,
                q: *q,
                prims,
                idx,
                swap,
            });
        }
        kept.sort_by(|x, y| {
            (classes[x.class].order, x.prims.len())
                .cmp(&(classes[y.class].order, y.prims.len()))
                .then(y.q.partial_cmp(&x.q).unwrap())
                .then((x.sa, x.sb).cmp(&(y.sa, y.sb)))
        });
        let n_prims_kept = kept.iter().map(|k| k.prims.len()).sum();
        // scalar data
        let mut pairs = Vec::with_capacity(kept.len());
        let (mut p, mut pc, mut e, mut qpv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for k in &kept {
            let pr = set.pair(k.idx);
            let qp = set.prim_schwarz(k.idx);
            let cl = &classes[k.class];
            let sz = cl.nh * cl.nab();
            pairs.push(Pair {
                class: k.class as u32,
                sa: k.sa as u32,
                sb: k.sb as u32,
                nprim: k.prims.len() as u32,
                prim: p.len() as u32,
                e: e.len() as u32,
                q: k.q,
            });
            for &ip in &k.prims {
                qpv.push(qp[ip]);
                p.push(pr.p[ip]);
                pc.push(pr.pc[ip]);
                let ed = &pr.e[ip * sz..(ip + 1) * sz];
                // (a, b) of the class (l_a >= l_b) in the set's pair: (b, a) when swapped
                let at = |ab: usize| {
                    if k.swap {
                        let (ia, ib) = (ab / cl.nb, ab % cl.nb);
                        ib * cl.na + ia
                    } else {
                        ab
                    }
                };
                e.extend(
                    cl.nz
                        .iter()
                        .map(|&(r, ab)| ed[r as usize * cl.nab() + at(ab as usize)]),
                );
            }
        }
        // groups and blocks with lane-interleaved ket data
        let tuv = herm_tuv();
        let mut groups = Vec::new();
        let mut blocks = Vec::new();
        let mut lane = Vec::new();
        let mut blk_qpk: Vec<f64> = Vec::new();
        let mut i = 0;
        while i < pairs.len() {
            let (c, np) = (pairs[i].class, pairs[i].nprim);
            let mut j = i;
            while j < pairs.len() && pairs[j].class == c && pairs[j].nprim == np {
                j += 1;
            }
            let b0 = blocks.len();
            let cl = &classes[c as usize];
            let nnz = cl.nz.len();
            let sign: Vec<f64> = cl
                .nz
                .iter()
                .map(|&(r, _)| {
                    let [t, u, v] = tuv[r as usize];
                    if (t + u + v) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            let mut f = i;
            while f < j {
                let nl = (j - f).min(LANES);
                let mut blk = Block {
                    class: c,
                    first: f as u32,
                    nlanes: nl as u32,
                    nprim: np,
                    data: lane.len(),
                    qmax: 0.0,
                    sa: [pairs[f].sa; LANES],
                    sb: [pairs[f].sb; LANES],
                    q: [0.0; LANES],
                    kq0: blk_qpk.len(),
                };
                for kq in 0..np as usize {
                    let m = (0..nl)
                        .map(|l| qpv[pairs[f + l].prim as usize + kq])
                        .fold(0.0, f64::max);
                    blk_qpk.push(m);
                }
                for l in 0..nl {
                    blk.sa[l] = pairs[f + l].sa;
                    blk.sb[l] = pairs[f + l].sb;
                    blk.q[l] = pairs[f + l].q;
                    blk.qmax = blk.qmax.max(pairs[f + l].q);
                }
                let rec = (LANE_HEAD + nnz) * LANES;
                let base = lane.len();
                lane.resize(base + np as usize * rec, 0.0);
                for kq in 0..np as usize {
                    let r0 = base + kq * rec;
                    for l in 0..LANES {
                        if l < nl {
                            let pr = &pairs[f + l];
                            let ip = pr.prim as usize + kq;
                            lane[r0 + l] = p[ip];
                            lane[r0 + LANES + l] = 1.0 / p[ip];
                            lane[r0 + 2 * LANES + l] = pc[ip][0];
                            lane[r0 + 3 * LANES + l] = pc[ip][1];
                            lane[r0 + 4 * LANES + l] = pc[ip][2];
                            let eo = pr.e as usize + kq * nnz;
                            for k in 0..nnz {
                                lane[r0 + (LANE_HEAD + k) * LANES + l] = sign[k] * e[eo + k];
                            }
                        } else {
                            lane[r0 + l] = 1.0;
                            lane[r0 + LANES + l] = 1.0;
                        }
                    }
                }
                blocks.push(blk);
                f += nl;
            }
            groups.push(Group {
                class: c,
                nprim: np,
                pairs: i..j,
                blocks: b0..blocks.len(),
            });
            i = j;
        }
        let ncl = classes.len();
        let mut gen_kernel = vec![-1i32; ncl * ncl];
        // the study switches LINUS_KERNEL ("loop": generated, else generic loops; "tiled": tiled
        // wherever it applies; default: generated, else tiled, else generic loops) and
        // LINUS_NO_GEN (crate::core::switches; development builds only)
        use crate::core::switches::{kernel_choice, no_generated_kernels, KernelChoice};
        let kind = kernel_choice();
        let mut tiled = vec![false; ncl * ncl];
        for (ib, cb) in classes.iter().enumerate() {
            for (ik, ck) in classes.iter().enumerate() {
                tiled[ib * ncl + ik] = kind != KernelChoice::Loop && cb.lab + ck.lab <= TILED_MAX_L;
            }
        }
        if !no_generated_kernels() && kind != KernelChoice::Tiled {
            for (ib, cb) in classes.iter().enumerate() {
                for (ik, ck) in classes.iter().enumerate() {
                    if cb.generated && ck.generated {
                        if let Some(k) = gclasses::block_kernel_index(cb.la, cb.lb, ck.la, ck.lb) {
                            gen_kernel[ib * ncl + ik] = k as i32;
                        }
                    }
                }
            }
        }
        BatchedPairs {
            nshell: ns,
            gen_kernel,
            tiled,
            classes,
            pairs,
            groups,
            blocks,
            p,
            qp: qpv,
            blk_qpk,
            pc,
            e,
            lane,
            qmax,
            n_pairs_total,
            n_prims_total,
            n_prims_kept,
        }
    }

    #[inline]
    pub fn class_of_pair(&self, i: usize) -> &PairClass {
        &self.classes[self.pairs[i].class as usize]
    }
}

/// Scratch space of the batched kernels.
pub struct BlockWork<V> {
    pub fm: Vec<V>,
    pub rn: Vec<V>,
    pub r: Vec<V>,
    pub rwork: Vec<V>,
    pub t: Vec<V>,
    /// R in cube layout (tiled kernels), allocated on first use
    pub cube: Vec<V>,
    /// the Boys function table (`boys::boys_table`)
    pub btab: &'static [f64],
}

impl<V: Real> BlockWork<V> {
    /// Scratch for shells up to angular momentum `lmax`.
    pub fn new(lmax: usize) -> Self {
        let l = 4 * lmax;
        let nc = ncart(lmax);
        BlockWork {
            fm: vec![V::zero(); l + 1],
            rn: vec![V::zero(); l + 1],
            r: vec![V::zero(); nherm(l)],
            rwork: vec![V::zero(); 2 * nherm(l)],
            t: vec![V::zero(); nherm(2 * lmax) * nc * nc],
            cube: Vec::new(),
            btab: crate::core::boys::boys_table(),
        }
    }
}

/// R_tuv (t+u+v <= l) from rn\[n\] = R^(n)_000 by the loop recursion (any l).
#[inline(always)]
pub fn r_tensor_loop<V: Real>(l: usize, rn: &[V], x: V, y: V, z: V, out: &mut [V], work: &mut [V]) {
    if l == 0 {
        out[0] = rn[0];
        return;
    }
    let tuv = herm_tuv();
    let nh = nherm(l);
    let (a, b) = work[..2 * nh].split_at_mut(nh);
    let (mut prev, mut cur): (&mut [V], &mut [V]) = (a, b);
    prev[0] = rn[l];
    for n in (0..l).rev() {
        let target: &mut [V] = if n == 0 { &mut *out } else { &mut *cur };
        target[0] = rn[n];
        for i in 1..nherm(l - n) {
            let [t, u, v] = tuv[i];
            let (t, u, v) = (t as usize, u as usize, v as usize);
            target[i] = if t > 0 {
                let mut r = x * prev[herm_idx(t - 1, u, v)];
                if t > 1 {
                    r += V::splat((t - 1) as f64) * prev[herm_idx(t - 2, u, v)];
                }
                r
            } else if u > 0 {
                let mut r = y * prev[herm_idx(t, u - 1, v)];
                if u > 1 {
                    r += V::splat((u - 1) as f64) * prev[herm_idx(t, u - 2, v)];
                }
                r
            } else {
                let mut r = z * prev[herm_idx(t, u, v - 1)];
                if v > 1 {
                    r += V::splat((v - 1) as f64) * prev[herm_idx(t, u, v - 2)];
                }
                r
            };
        }
        if n > 0 {
            std::mem::swap(&mut prev, &mut cur);
        }
    }
}

/// Per primitive quartet (bra primitive scalar, LANES ket primitives): R_tuv scaled by the
/// ERI prefactor, t+u+v <= l, into `w.r`.
#[inline(always)]
pub fn prim_r<V: BatchKernels>(
    l: usize,
    p: f64,
    pcen: [f64; 3],
    kd: &[f64],
    op: &Operator,
    w: &mut BlockWork<V>,
) {
    let (dx, dy, dz) = prim_rn(l, p, pcen, kd, op, w);
    let BlockWork { rn, r, rwork, .. } = w;
    if l <= rtensor::R_MAX_L {
        unsafe { V::r_tensor(l, rn, dx, dy, dz, r) }
    } else {
        r_tensor_loop(l, rn, dx, dy, dz, r, rwork);
    }
}

/// The Boys stage of [`prim_r`]: R^(n)_000 (n <= l, prefactor included) into `w.rn`; returns
/// P - Q.
#[inline(always)]
pub fn prim_rn<V: BatchKernels>(
    l: usize,
    p: f64,
    pcen: [f64; 3],
    kd: &[f64],
    op: &Operator,
    w: &mut BlockWork<V>,
) -> (V, V, V) {
    let q = V::load(&kd[0..]);
    let inv_q = V::load(&kd[LANES..]);
    let dx = V::splat(pcen[0]) - V::load(&kd[2 * LANES..]);
    let dy = V::splat(pcen[1]) - V::load(&kd[3 * LANES..]);
    let dz = V::splat(pcen[2]) - V::load(&kd[4 * LANES..]);
    let vp = V::splat(p);
    let inv_s = V::splat(1.0) / (vp + q);
    let alpha = vp * q * inv_s;
    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
    // 2 pi^{5/2} / (p q sqrt(p+q))
    let pref = V::splat(ERI_PREF / p) * inv_q * inv_s.sqrt();
    if op.is_coulomb() {
        boys_lanes_tab(w.btab, l, alpha * r2, &mut w.fm);
        let m2a = alpha * V::splat(-2.0);
        let mut f = pref;
        w.rn[0] = pref * w.fm[0];
        for n in 1..=l {
            f *= m2a;
            w.rn[n] = f * w.fm[n];
        }
    } else {
        rn_operator(l, alpha, r2, pref, op, &mut w.fm, &mut w.rn);
    }
    (dx, dy, dz)
}

/// Integrals of bra pair `bra` with the LANES ket pairs of block `blk`, generic loops over the
/// structural nonzeros: out[ab * ncd + cd] (one vector per integral, lane = ket pair).
#[inline(always)]
/// Primitive quartets with Qp_ip * max_lane Qp_kq < `cut` are skipped.
pub fn block_eri_generic<V: BatchKernels>(
    bp: &BatchedPairs,
    bra: usize,
    blk: usize,
    op: &Operator,
    cut: f64,
    out: &mut [V],
    w: &mut BlockWork<V>,
) {
    let pr = &bp.pairs[bra];
    let b = &bp.blocks[blk];
    let cb = &bp.classes[pr.class as usize];
    let ck = &bp.classes[b.class as usize];
    let (nhb, nhk) = (cb.nh, ck.nh);
    let (nab, ncd) = (cb.nab(), ck.nab());
    let l = cb.lab + ck.lab;
    let st = sum_table(cb.lab, ck.lab);
    let nnzb = cb.nz.len();
    let nnzk = ck.nz.len();
    let rec = (LANE_HEAD + nnzk) * LANES;
    let out = &mut out[..nab * ncd];
    out.fill(V::zero());
    for ip in 0..pr.nprim as usize {
        let gi = pr.prim as usize + ip;
        let (p, pcen) = (bp.p[gi], bp.pc[gi]);
        let tl = &mut w.t[..nhb * ncd];
        tl.fill(V::zero());
        let qb = bp.qp[gi];
        for kq in 0..b.nprim as usize {
            if qb * bp.blk_qpk[b.kq0 + kq] < cut {
                continue;
            }
            let kd = &bp.lane[b.data + kq * rec..b.data + (kq + 1) * rec];
            prim_r(l, p, pcen, kd, op, w);
            let (r, t) = (&w.r, &mut w.t[..nhb * ncd]);
            for (k, &(s, cd)) in ck.nz.iter().enumerate() {
                let e = V::load(&kd[(LANE_HEAD + k) * LANES..]);
                let (s, cd) = (s as usize, cd as usize);
                for rr in 0..nhb {
                    let ri = st.idx[rr * nhk + s] as usize;
                    t[rr * ncd + cd] = r[ri].mul_add(e, t[rr * ncd + cd]);
                }
            }
        }
        let eb = &bp.e[pr.e as usize + ip * nnzb..pr.e as usize + (ip + 1) * nnzb];
        let t = &w.t[..nhb * ncd];
        for (k, &(rr, ab)) in cb.nz.iter().enumerate() {
            let e = V::splat(eb[k]);
            let (rr, ab) = (rr as usize, ab as usize);
            let orow = &mut out[ab * ncd..(ab + 1) * ncd];
            let trow = &t[rr * ncd..(rr + 1) * ncd];
            for (o, &tv) in orow.iter_mut().zip(trow) {
                *o = e.mul_add(tv, *o);
            }
        }
    }
}

/// T step of the tiled kernel for one ket primitive: for every chunk (column cd, nonzeros
/// (s_j, k_j)), T[cd][r] += sum_j R[base(r) + base(s_j)] E'_kq[s_j][cd] for r < nhb.
/// `t` is column-major (ncd x nhb), `e` points to the lane E nonzeros of the ket primitive.
/// Two partial sums per row keep the FMA chains short.
#[inline(always)]
unsafe fn tiled_tstep<V: Lanes>(
    plan: &[Chunk],
    cube: *const V,
    base: &[u32],
    e: *const f64,
    t: *mut V,
    nhb: usize,
) {
    let base = &base[..nhb];
    let ld = |k: u16| V::load8(&*(e.add(k as usize * LANES) as *const [f64; LANES]));
    for ch in plan {
        let col = t.add(ch.col as usize * nhb);
        let o = ch.off.map(|x| x as usize);
        match ch.n {
            8 => {
                let (e0, e1, e2, e3, e4, e5, e6, e7) = (
                    ld(ch.k[0]),
                    ld(ch.k[1]),
                    ld(ch.k[2]),
                    ld(ch.k[3]),
                    ld(ch.k[4]),
                    ld(ch.k[5]),
                    ld(ch.k[6]),
                    ld(ch.k[7]),
                );
                for (rr, &b) in base.iter().enumerate() {
                    let rp = cube.add(b as usize);
                    let x = col.add(rr);
                    let a0 = (*rp.add(o[0])).mul_add(e0, *x);
                    let a1 = *rp.add(o[1]) * e1;
                    let a0 = (*rp.add(o[2])).mul_add(e2, a0);
                    let a1 = (*rp.add(o[3])).mul_add(e3, a1);
                    let a0 = (*rp.add(o[4])).mul_add(e4, a0);
                    let a1 = (*rp.add(o[5])).mul_add(e5, a1);
                    let a0 = (*rp.add(o[6])).mul_add(e6, a0);
                    let a1 = (*rp.add(o[7])).mul_add(e7, a1);
                    *x = a0 + a1;
                }
            }
            4 => {
                let (e0, e1, e2, e3) = (ld(ch.k[0]), ld(ch.k[1]), ld(ch.k[2]), ld(ch.k[3]));
                for (rr, &b) in base.iter().enumerate() {
                    let rp = cube.add(b as usize);
                    let x = col.add(rr);
                    let a0 = (*rp.add(o[0])).mul_add(e0, *x);
                    let a1 = *rp.add(o[1]) * e1;
                    let a0 = (*rp.add(o[2])).mul_add(e2, a0);
                    let a1 = (*rp.add(o[3])).mul_add(e3, a1);
                    *x = a0 + a1;
                }
            }
            2 => {
                let (e0, e1) = (ld(ch.k[0]), ld(ch.k[1]));
                for (rr, &b) in base.iter().enumerate() {
                    let rp = cube.add(b as usize);
                    let x = col.add(rr);
                    *x = (*rp.add(o[1])).mul_add(e1, (*rp.add(o[0])).mul_add(e0, *x));
                }
            }
            _ => {
                let e0 = ld(ch.k[0]);
                for (rr, &b) in base.iter().enumerate() {
                    let x = col.add(rr);
                    *x = (*cube.add(b as usize + o[0])).mul_add(e0, *x);
                }
            }
        }
    }
}

/// I step of the tiled kernel for one bra primitive and the columns cd in `cds`: for every
/// chunk (row ab, nonzeros (r_j, k_j)), out[ab][cd] += sum_j E_ip[r_j][ab] T[cd][r_j].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_istep<V: Lanes>(
    plan: &[Chunk],
    t: *const V,
    eb: &[f64],
    out: *mut V,
    ncd: usize,
    nhb: usize,
    cds: std::ops::Range<usize>,
) {
    for ch in plan {
        let orow = out.add(ch.col as usize * ncd);
        let o = ch.off.map(|x| x as usize);
        let b = |j: usize| V::splat(*eb.get_unchecked(ch.k[j] as usize));
        match ch.n {
            8 => {
                let (b0, b1, b2, b3, b4, b5, b6, b7) =
                    (b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7));
                for cd in cds.clone() {
                    let tp = t.add(cd * nhb);
                    let x = orow.add(cd);
                    let a0 = b0.mul_add(*tp.add(o[0]), *x);
                    let a1 = b1 * *tp.add(o[1]);
                    let a0 = b2.mul_add(*tp.add(o[2]), a0);
                    let a1 = b3.mul_add(*tp.add(o[3]), a1);
                    let a0 = b4.mul_add(*tp.add(o[4]), a0);
                    let a1 = b5.mul_add(*tp.add(o[5]), a1);
                    let a0 = b6.mul_add(*tp.add(o[6]), a0);
                    let a1 = b7.mul_add(*tp.add(o[7]), a1);
                    *x = a0 + a1;
                }
            }
            4 => {
                let (b0, b1, b2, b3) = (b(0), b(1), b(2), b(3));
                for cd in cds.clone() {
                    let tp = t.add(cd * nhb);
                    let x = orow.add(cd);
                    let a0 = b0.mul_add(*tp.add(o[0]), *x);
                    let a1 = b1 * *tp.add(o[1]);
                    let a0 = b2.mul_add(*tp.add(o[2]), a0);
                    let a1 = b3.mul_add(*tp.add(o[3]), a1);
                    *x = a0 + a1;
                }
            }
            2 => {
                let (b0, b1) = (b(0), b(1));
                for cd in cds.clone() {
                    let tp = t.add(cd * nhb);
                    let x = orow.add(cd);
                    *x = b1.mul_add(*tp.add(o[1]), b0.mul_add(*tp.add(o[0]), *x));
                }
            }
            _ => {
                let b0 = b(0);
                for cd in cds.clone() {
                    let x = orow.add(cd);
                    *x = b0.mul_add(*t.add(cd * nhb + o[0]), *x);
                }
            }
        }
    }
}

/// Tiled McMurchie-Davidson kernel for any class pair with la + lb + lc + ld <= TILED_MAX_L
/// (same contraction as [`block_eri_generic`], output out[ab * ncd + cd]):
///
///   for each bra primitive ip:
///     for each ket primitive kq: R (Boys + recursion) into the cube layout, then the T step
///         T\[cd\]\[r\] += sum_s R[r+s] E'_kq\[s\]\[cd\] over the ket nonzeros, with the up to 8
///         nonzeros of a chunk (one column cd) in registers while r streams; R offsets are
///         additive in the cube (r+s -> base(r) + base(s)), so the loads need no index table;
///     I step out\[ab\]\[cd\] += sum_r E_ip\[r\]\[ab\] T\[cd\]\[r\], in column blocks of T that fit L1.
///
/// Loops over r and cd replace the straight-line code of the generated kernels, so the code
/// size is independent of the class, and every class up to (gg|gg) gets the same register
/// blocking from run-time chunk plans (`PairClass::tplan`, `iplan`).
#[inline(always)]
pub fn block_eri_tiled<V: BatchKernels>(
    bp: &BatchedPairs,
    bra: usize,
    blk: usize,
    op: &Operator,
    cut: f64,
    out: &mut [V],
    w: &mut BlockWork<V>,
) {
    let pr = &bp.pairs[bra];
    let b = &bp.blocks[blk];
    let cb = &bp.classes[pr.class as usize];
    let ck = &bp.classes[b.class as usize];
    let nhb = cb.nh;
    let (nab, ncd) = (cb.nab(), ck.nab());
    let l = cb.lab + ck.lab;
    assert!(l <= TILED_MAX_L);
    let nnzb = cb.nz.len();
    let nnzk = ck.nz.len();
    let rec = (LANE_HEAD + nnzk) * LANES;
    let out = &mut out[..nab * ncd];
    // compile-time-offset pieces for spherical classes up to (gg), else the run-time plans
    let tg = if cb.generated && ck.generated {
        V::tiled_gen()
    } else {
        None
    };
    // The generated pieces have FIRST variants that write T (first ket primitive of a bra
    // primitive) and the output (first contributing bra primitive) instead of accumulating
    // into zeroed arrays: the same sums (the first term is added to +0 either way) without
    // zeroing T per bra primitive and the output per call. The run-time plans accumulate.
    if tg.is_none() {
        out.fill(V::zero());
    }
    let mut out_set = false;
    if w.cube.len() < CUBE_N {
        w.cube = vec![V::zero(); CUBE_N];
    }
    let cp = &cube_pos()[..nherm(l)];
    let base = &cube_pos()[..nhb];
    assert!(w.t.len() >= nhb * ncd);
    let (ib, ik) = (
        gtiled::class_index(cb.la, cb.lb),
        gtiled::class_index(ck.la, ck.lb),
    );
    // columns of T per I-step block (L1 reuse of T vs. amortizing the chunk setup)
    let cblk = (TILED_IBLOCK_BYTES.load(std::sync::atomic::Ordering::Relaxed)
        / (nhb * std::mem::size_of::<V>()))
    .max(1);
    for ip in 0..pr.nprim as usize {
        let gi = pr.prim as usize + ip;
        let (p, pcen) = (bp.p[gi], bp.pc[gi]);
        let qb = bp.qp[gi];
        let mut any = false;
        for kq in 0..b.nprim as usize {
            if qb * bp.blk_qpk[b.kq0 + kq] < cut {
                continue;
            }
            let first = !any;
            if first && tg.is_none() {
                w.t[..nhb * ncd].fill(V::zero());
            }
            any = true;
            let kd = &bp.lane[b.data + kq * rec..b.data + (kq + 1) * rec];
            let e = unsafe { kd.as_ptr().add(LANE_HEAD * LANES) };
            match &tg {
                Some(g) if l < g.r_cube.len() => {
                    let (dx, dy, dz) = prim_rn(l, p, pcen, kd, op, w);
                    unsafe { (g.r_cube[l])(&w.rn, dx, dy, dz, w.cube.as_mut_ptr()) };
                }
                _ => {
                    prim_r(l, p, pcen, kd, op, w);
                    let (r, cube) = (&w.r[..cp.len()], &mut w.cube[..CUBE_N]);
                    for (h, &pos) in cp.iter().enumerate() {
                        cube[pos as usize] = r[h];
                    }
                }
            }
            match &tg {
                Some(g) => unsafe {
                    (if first {
                        g.tstep_first[ik]
                    } else {
                        g.tstep[ik]
                    })(w.cube.as_ptr(), base, e, w.t.as_mut_ptr())
                },
                None => unsafe {
                    tiled_tstep(&ck.tplan, w.cube.as_ptr(), base, e, w.t.as_mut_ptr(), nhb)
                },
            }
        }
        if !any {
            continue;
        }
        let eb = &bp.e[pr.e as usize + ip * nnzb..pr.e as usize + (ip + 1) * nnzb];
        let ofirst = !out_set;
        out_set = true;
        let mut c0 = 0;
        while c0 < ncd {
            let c1 = (c0 + cblk).min(ncd);
            match &tg {
                Some(g) => unsafe {
                    (if ofirst {
                        g.istep_first[ib]
                    } else {
                        g.istep[ib]
                    })(w.t.as_ptr(), eb, out.as_mut_ptr(), ncd, c0, c1)
                },
                None => unsafe {
                    tiled_istep(
                        &cb.iplan,
                        w.t.as_ptr(),
                        eb,
                        out.as_mut_ptr(),
                        ncd,
                        nhb,
                        c0..c1,
                    )
                },
            }
            c0 = c1;
        }
    }
    if tg.is_some() && !out_set {
        out.fill(V::zero());
    }
}

/// Hermite-space Coulomb ("J-engine", SHARK eq. 38-41) for one bra pair and a ket block,
/// without forming integrals. With the Hermite densities
///   db\[ip\]\[r\] = sum_ab E_ip\[r\]\[ab\] D_ab          (bra, scalar),
///   dk\[kq\]\[s\] = sum_cd E'_kq\[s\]\[cd\] D_cd          (ket, per lane, sign (-1)^|s| included),
/// it accumulates jb\[ip\]\[r\] += sum_kq sum_s R_{r+s} dk\[kq\]\[s\] (lanes summed later) and sets
/// jk\[kq\]\[s\] = sum_ip sum_r R_{r+s} db\[ip\]\[r\]; then J_ab = sum E_ip\[r\]\[ab\] jb\[ip\]\[r\] and
/// J_cd = sum E'_kq\[s\]\[cd\] jk\[kq\]\[s\]. Cost per primitive quartet: R plus 2 nherm_b nherm_k
/// FMAs, instead of the full integral block.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn block_jengine_generic<V: BatchKernels>(
    bp: &BatchedPairs,
    bra: usize,
    blk: usize,
    cut: f64,
    db: &[f64],
    dk: &[V],
    jb: &mut [V],
    jk: &mut [V],
    w: &mut BlockWork<V>,
) {
    let pr = &bp.pairs[bra];
    let b = &bp.blocks[blk];
    let cb = &bp.classes[pr.class as usize];
    let ck = &bp.classes[b.class as usize];
    let (nhb, nhk) = (cb.nh, ck.nh);
    let l = cb.lab + ck.lab;
    let st = sum_table(cb.lab, ck.lab);
    let nk = b.nprim as usize;
    let rec = (LANE_HEAD + ck.nz.len()) * LANES;
    let (db, dk) = (&db[..pr.nprim as usize * nhb], &dk[..nk * nhk]);
    let (jb, jk) = (&mut jb[..pr.nprim as usize * nhb], &mut jk[..nk * nhk]);
    for x in jk.iter_mut() {
        *x = V::zero();
    }
    for ip in 0..pr.nprim as usize {
        let gi = pr.prim as usize + ip;
        let (p, pcen, qb) = (bp.p[gi], bp.pc[gi], bp.qp[gi]);
        for kq in 0..nk {
            if qb * bp.blk_qpk[b.kq0 + kq] < cut {
                continue;
            }
            let kd = &bp.lane[b.data + kq * rec..b.data + (kq + 1) * rec];
            prim_r(l, p, pcen, kd, &Operator::Coulomb, w);
            let r = &w.r;
            let dkq = &dk[kq * nhk..(kq + 1) * nhk];
            let jkq = &mut jk[kq * nhk..(kq + 1) * nhk];
            for rr in 0..nhb {
                let idx = &st.idx[rr * nhk..(rr + 1) * nhk];
                let mut acc = jb[ip * nhb + rr];
                let dbr = V::splat(db[ip * nhb + rr]);
                for s in 0..nhk {
                    let x = r[idx[s] as usize];
                    acc = x.mul_add(dkq[s], acc);
                    jkq[s] = x.mul_add(dbr, jkq[s]);
                }
                jb[ip * nhb + rr] = acc;
            }
        }
    }
}

/// Generated pieces of the tiled kernel for one lane type (see `generated::tiled`).
pub struct TiledGen<V: 'static> {
    /// R in the cube layout, index L (L <= R_CUBE_MAX_L)
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub r_cube: &'static [unsafe fn(&[V], V, V, V, *mut V)],
    /// T step per ket class, index `generated::tiled::class_index`
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub tstep: &'static [unsafe fn(*const V, &[u32], *const f64, *mut V)],
    /// T step writing T (first ket primitive of a bra primitive; no zeroed T needed)
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub tstep_first: &'static [unsafe fn(*const V, &[u32], *const f64, *mut V)],
    /// I step per bra class
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub istep: &'static [unsafe fn(*const V, &[f64], *mut V, usize, usize, usize)],
    /// I step writing the output (first contributing bra primitive; no zeroed output needed)
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub istep_first: &'static [unsafe fn(*const V, &[f64], *mut V, usize, usize, usize)],
}

/// Per-ISA entry points of the batched kernels. Each implementation forwards to functions
/// compiled with that ISA's target features (so they are not inlined into callers compiled
/// without them).
pub trait BatchKernels: Lanes + 'static {
    /// Generated tiled-kernel pieces, if compiled for this lane type.
    fn tiled_gen() -> Option<TiledGen<Self>>;
    /// R_tuv for l <= R_MAX_L (unrolled).
    ///
    /// # Safety
    /// The CPU runs the lane type's instruction set (checked at run time, `simd::Isa::detect`).
    unsafe fn r_tensor(l: usize, rn: &[Self], x: Self, y: Self, z: Self, out: &mut [Self]);
    /// Integrals of a bra pair with a ket block (see [`block_eri_generic`]).
    /// `cut`: primitive quartets whose Schwarz bound is below it are skipped (0: none).
    ///
    /// # Safety
    /// The CPU runs the lane type's instruction set (checked at run time, `simd::Isa::detect`).
    unsafe fn block_eri(
        bp: &BatchedPairs,
        bra: usize,
        blk: usize,
        op: &Operator,
        cut: f64,
        out: &mut [Self],
        w: &mut BlockWork<Self>,
    );
    /// Hermite-space Coulomb (see [`block_jengine_generic`]).
    ///
    /// # Safety
    /// The CPU runs the lane type's instruction set (checked at run time, `simd::Isa::detect`).
    #[allow(clippy::too_many_arguments)]
    unsafe fn block_jengine(
        bp: &BatchedPairs,
        bra: usize,
        blk: usize,
        cut: f64,
        db: &[f64],
        dk: &[Self],
        jb: &mut [Self],
        jk: &mut [Self],
        w: &mut BlockWork<Self>,
    );
}

macro_rules! batch_kernels {
    // with the generated class kernels
    ($V:ty, $m:ident, gen $(, $feat:literal)?) => {
        // the generated class kernels' entry points take every array of a quartet block
        #[allow(clippy::too_many_arguments)]
        mod $m {
            #[allow(unused_imports)]
            use super::*;
            crate::r_tensor_isa_fns!($V $(, $feat)?);
            crate::block_isa_fns!($V $(, $feat)?);
            crate::tiled_isa_fns!($V $(, $feat)?);
            $(#[target_feature(enable = $feat)])?
            pub unsafe fn block_eri(bp: &BatchedPairs, bra: usize, blk: usize, op: &Operator, cut: f64, out: &mut [$V], w: &mut BlockWork<$V>) {
                block_eri_generic::<$V>(bp, bra, blk, op, cut, out, w)
            }
            $(#[target_feature(enable = $feat)])?
            pub unsafe fn block_eri_tiled(bp: &BatchedPairs, bra: usize, blk: usize, op: &Operator, cut: f64, out: &mut [$V], w: &mut BlockWork<$V>) {
                super::block_eri_tiled::<$V>(bp, bra, blk, op, cut, out, w)
            }
            #[allow(clippy::too_many_arguments)]
            $(#[target_feature(enable = $feat)])?
            pub unsafe fn block_jengine(bp: &BatchedPairs, bra: usize, blk: usize, cut: f64, db: &[f64], dk: &[$V], jb: &mut [$V], jk: &mut [$V], w: &mut BlockWork<$V>) {
                block_jengine_generic::<$V>(bp, bra, blk, cut, db, dk, jb, jk, w)
            }
        }
        impl BatchKernels for $V {
            #[inline(always)]
            fn tiled_gen() -> Option<TiledGen<Self>> {
                Some(TiledGen { r_cube: &$m::R_CUBE_FNS, tstep: &$m::TSTEP_FNS, tstep_first: &$m::TSTEP_FIRST_FNS, istep: &$m::ISTEP_FNS, istep_first: &$m::ISTEP_FIRST_FNS })
            }
            #[inline(always)]
            unsafe fn block_jengine(bp: &BatchedPairs, bra: usize, blk: usize, cut: f64, db: &[f64], dk: &[Self], jb: &mut [Self], jk: &mut [Self], w: &mut BlockWork<Self>) {
                $m::block_jengine(bp, bra, blk, cut, db, dk, jb, jk, w)
            }
            #[inline(always)]
            unsafe fn r_tensor(l: usize, rn: &[Self], x: Self, y: Self, z: Self, out: &mut [Self]) {
                $m::R_TENSOR_FNS[l](rn, x, y, z, out)
            }
            #[inline(always)]
            unsafe fn block_eri(bp: &BatchedPairs, bra: usize, blk: usize, op: &Operator, cut: f64, out: &mut [Self], w: &mut BlockWork<Self>) {
                let pr = &bp.pairs[bra];
                let b = &bp.blocks[blk];
                let k = bp.gen_kernel[pr.class as usize * bp.classes.len() + b.class as usize];
                if k >= 0 {
                    let (p0, np) = (pr.prim as usize, pr.nprim as usize);
                    let nnzb = bp.classes[pr.class as usize].nz.len();
                    let nrec = LANE_HEAD + bp.classes[b.class as usize].nz.len();
                    let e0 = pr.e as usize;
                    let d0 = b.data / LANES;
                    $m::BLOCK_FNS[k as usize](
                        &bp.p[p0..p0 + np],
                        &bp.pc[p0..p0 + np],
                        &bp.e[e0..e0 + np * nnzb],
                        &bp.lane.as_chunks::<LANES>().0[d0..d0 + b.nprim as usize * nrec],
                        b.nprim as usize,
                        op,
                        &bp.qp[p0..p0 + np],
                        &bp.blk_qpk[b.kq0..b.kq0 + b.nprim as usize],
                        cut,
                        out,
                    )
                } else if bp.tiled[pr.class as usize * bp.classes.len() + b.class as usize] {
                    $m::block_eri_tiled(bp, bra, blk, op, cut, out, w)
                } else {
                    $m::block_eri(bp, bra, blk, op, cut, out, w)
                }
            }
        }
    };
    // generic-loop kernels only
    ($V:ty, $m:ident, nogen $(, $feat:literal)?) => {
        // the generated class kernels' entry points take every array of a quartet block
        #[allow(clippy::too_many_arguments)]
        mod $m {
            #[allow(unused_imports)]
            use super::*;
            crate::r_tensor_isa_fns!($V $(, $feat)?);
            $(#[target_feature(enable = $feat)])?
            pub unsafe fn block_eri(bp: &BatchedPairs, bra: usize, blk: usize, op: &Operator, cut: f64, out: &mut [$V], w: &mut BlockWork<$V>) {
                block_eri_generic::<$V>(bp, bra, blk, op, cut, out, w)
            }
            $(#[target_feature(enable = $feat)])?
            pub unsafe fn block_eri_tiled(bp: &BatchedPairs, bra: usize, blk: usize, op: &Operator, cut: f64, out: &mut [$V], w: &mut BlockWork<$V>) {
                super::block_eri_tiled::<$V>(bp, bra, blk, op, cut, out, w)
            }
            #[allow(clippy::too_many_arguments)]
            $(#[target_feature(enable = $feat)])?
            pub unsafe fn block_jengine(bp: &BatchedPairs, bra: usize, blk: usize, cut: f64, db: &[f64], dk: &[$V], jb: &mut [$V], jk: &mut [$V], w: &mut BlockWork<$V>) {
                block_jengine_generic::<$V>(bp, bra, blk, cut, db, dk, jb, jk, w)
            }
        }
        impl BatchKernels for $V {
            #[inline(always)]
            fn tiled_gen() -> Option<TiledGen<Self>> {
                None
            }
            #[inline(always)]
            unsafe fn block_jengine(bp: &BatchedPairs, bra: usize, blk: usize, cut: f64, db: &[f64], dk: &[Self], jb: &mut [Self], jk: &mut [Self], w: &mut BlockWork<Self>) {
                $m::block_jengine(bp, bra, blk, cut, db, dk, jb, jk, w)
            }
            #[inline(always)]
            unsafe fn r_tensor(l: usize, rn: &[Self], x: Self, y: Self, z: Self, out: &mut [Self]) {
                $m::R_TENSOR_FNS[l](rn, x, y, z, out)
            }
            #[inline(always)]
            unsafe fn block_eri(bp: &BatchedPairs, bra: usize, blk: usize, op: &Operator, cut: f64, out: &mut [Self], w: &mut BlockWork<Self>) {
                let (pr, b) = (&bp.pairs[bra], &bp.blocks[blk]);
                if bp.tiled[pr.class as usize * bp.classes.len() + b.class as usize] {
                    $m::block_eri_tiled(bp, bra, blk, op, cut, out, w)
                } else {
                    $m::block_eri(bp, bra, blk, op, cut, out, w)
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
batch_kernels!(
    linus_num::simd::Avx512,
    k_avx512,
    gen,
    "avx512f,avx512dq,avx2,fma"
);
#[cfg(target_arch = "x86_64")]
batch_kernels!(linus_num::simd::Avx2, k_avx2, gen, "avx2,fma");
// baseline fallback (no AVX2): generic-loop kernels only, to keep build times down
batch_kernels!(linus_num::simd::Portable<false>, k_generic, nogen);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::core::hermite::{build_pair, ShellPair, ShellView};
    use crate::eri4c::EriWork;
    use crate::pairs::PairSet;
    use linus_num::simd::Portable;

    fn water_like(basis: &str, pure: bool) -> BasisSet {
        let def = BasisDef::load(basis, None).unwrap();
        // O, H, H + a coincident-center ghost-free extra H on O's position offset
        let z = [8u32, 1, 1, 6];
        let positions = [
            [0.0, 0.0, 0.1],
            [1.4, 0.2, -0.9],
            [-1.5, -0.1, -0.8],
            [0.3, 2.9, 0.7],
        ];
        BasisSet::new(&def, &z, &positions, pure).unwrap()
    }

    #[test]
    fn generated_patterns_match_runtime() {
        for la in 0..=gclasses::PATTERN_MAX_L {
            for lb in 0..=la {
                let mk = |l: usize| {
                    if l <= 1 {
                        crate::basis::cart_norm_matrix(l)
                    } else {
                        crate::core::solid_harmonics::cart2sph(l)
                    }
                };
                let nz = e_pattern(la, lb, &mk(la), &mk(lb));
                assert_eq!(
                    gclasses::e_pattern(la, lb).unwrap(),
                    nz.as_slice(),
                    "class ({la}{lb})"
                );
            }
        }
    }

    fn check_blocks<V: BatchKernels>(basis: &BasisSet) {
        for op in [
            Operator::Coulomb,
            Operator::LongRange { omega: 0.4 },
            Operator::from_libxc(0.157706, 0.842294, 0.3),
        ] {
            for path in [Path::Default, Path::Tiled, Path::Loop] {
                check_blocks_op::<V>(basis, &op, path);
            }
        }
    }

    /// Kernel selection of a test run: the default (generated, else tiled, else generic loops),
    /// tiled wherever it applies, or generic loops only.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Path {
        Default,
        Tiled,
        Loop,
    }

    /// Every bra pair against every ket block (all lanes, also kets beyond the bra), compared
    /// with the generic reference `eri_quartet_op`. Returns the class pairs exercised and how
    /// many of them ran a generated and a tiled kernel.
    fn check_blocks_op<V: BatchKernels>(
        basis: &BasisSet,
        op: &Operator,
        path: Path,
    ) -> (usize, usize, usize) {
        let set = PairSet::build(basis, crate::pairs::PairOptions::default());
        let mut bp = BatchedPairs::new(basis, &set, 0.0, 0.0);
        match path {
            Path::Default => {}
            Path::Tiled => {
                let ncl = bp.classes.len();
                for ib in 0..ncl {
                    for ik in 0..ncl {
                        let l = bp.classes[ib].lab + bp.classes[ik].lab;
                        bp.tiled[ib * ncl + ik] = l <= TILED_MAX_L;
                        if l <= TILED_MAX_L {
                            bp.gen_kernel[ib * ncl + ik] = -1;
                        }
                    }
                }
            }
            Path::Loop => {
                bp.gen_kernel.fill(-1);
                bp.tiled.fill(false);
            }
        }
        let nf = basis.shells.iter().map(|s| s.nfunc()).max().unwrap();
        let mut w = BlockWork::<V>::new(basis.max_l());
        let mut ew = EriWork::new();
        let mut out = vec![V::zero(); nf.pow(4)];
        let mut refb = vec![0.0; nf.pow(4)];
        let refs: Vec<ShellPair> = bp
            .pairs
            .iter()
            .map(|pr| {
                let (sa, sb) = (pr.sa as usize, pr.sb as usize);
                build_pair(
                    &ShellView::from_shell(&basis.shells[sa]),
                    &ShellView::from_shell(&basis.shells[sb]),
                    sa,
                    sb,
                )
            })
            .collect();
        let ncl = bp.classes.len();
        let mut seen = vec![false; ncl * ncl];
        let (mut worst, mut worst_q): (f64, f64) = (0.0, 0.0);
        for bra in 0..bp.pairs.len() {
            for (bi, blk) in bp.blocks.iter().enumerate() {
                unsafe { V::block_eri(&bp, bra, bi, op, 0.0, &mut out, &mut w) };
                seen[bp.pairs[bra].class as usize * ncl + blk.class as usize] = true;
                let ck = &bp.classes[blk.class as usize];
                let (nab, ncd) = (bp.class_of_pair(bra).nab(), ck.nab());
                for l in 0..blk.nlanes as usize {
                    crate::eri4c::eri_quartet(
                        &refs[bra],
                        &refs[blk.first as usize + l],
                        op,
                        &mut refb,
                        &mut ew,
                    );
                    let bmax = refb[..nab * ncd].iter().fold(0.0f64, |m, x| m.max(x.abs()));
                    let mut dmax: f64 = 0.0;
                    for x in 0..nab * ncd {
                        dmax = dmax.max((out[x].to_array()[l] - refb[x]).abs());
                    }
                    // Two metrics: relative to the block's largest integral, for blocks that are
                    // not cancellation-dominated (max |I| >= 1e-2 Q_ab Q_cd); and relative to the
                    // Schwarz bound Q_ab Q_cd for all blocks (symmetry-zero blocks are pure noise).
                    let qq = bp.pairs[bra].q * blk.q[l] * op.schwarz_factor();
                    if bmax >= 1e-2 * qq && bmax > 0.0 {
                        worst = worst.max(dmax / bmax);
                    }
                    if qq > 0.0 {
                        worst_q = worst_q.max(dmax / qq);
                    }
                }
            }
        }
        let nseen = seen.iter().filter(|&&x| x).count();
        let ngen = (0..ncl * ncl)
            .filter(|&k| seen[k] && bp.gen_kernel[k] >= 0)
            .count();
        let ntiled = (0..ncl * ncl)
            .filter(|&k| seen[k] && bp.gen_kernel[k] < 0 && bp.tiled[k])
            .count();
        eprintln!("{op:?} {path:?}: {nseen} class pairs ({ngen} generated, {ntiled} tiled): worst relative {worst:e}, worst relative to Schwarz bound {worst_q:e}");
        assert!(
            worst_q < 1e-14,
            "{path:?}: worst relative to Schwarz bound {worst_q:e}"
        );
        assert!(worst < 1e-13, "{path:?}: worst {worst:e}");
        (nseen, ngen, ntiled)
    }

    /// s..g shells (l <= lmax) on two centers, each contracted (2 primitives) and
    /// uncontracted, so every class pair occurs with one-center (coincident) and two-center
    /// pairs, P = Q quartets and general geometry.
    fn synthetic(pure: bool, lmax: usize) -> BasisSet {
        use crate::basis::{normalize_shell, Shell};
        let centers = [[0.0, 0.0, 0.0], [0.9, -0.4, 1.3]];
        let mut shells = Vec::new();
        let mut atom_shells = Vec::new();
        let mut ao = 0;
        for (atom, c) in centers.iter().enumerate() {
            let start = shells.len();
            for l in 0..=lmax {
                for contracted in [true, false] {
                    let f = 1.0 + 0.13 * l as f64 + 0.07 * atom as f64;
                    let (exps, coefs) = if contracted {
                        (vec![3.3 * f, 0.61 * f], vec![0.45, 0.7])
                    } else {
                        (vec![0.9 * f], vec![1.0])
                    };
                    let sh = Shell {
                        l,
                        pure: pure && l >= 2,
                        atom,
                        center: *c,
                        coefs: normalize_shell(l, &exps, &coefs),
                        exps,
                        ao_start: ao,
                    };
                    ao += sh.nfunc();
                    shells.push(sh);
                }
            }
            atom_shells.push(start..shells.len());
        }
        BasisSet {
            name: "synthetic".into(),
            shells,
            nbf: ao,
            atom_shells,
            pure,
        }
    }

    fn check_exhaustive<V: BatchKernels>() {
        for pure in [true, false] {
            let b = synthetic(pure, 4);
            for op in [Operator::Coulomb, Operator::from_libxc(0.19, 0.46, 0.33)] {
                let (nseen, ngen, ntiled) = check_blocks_op::<V>(&b, &op, Path::Default);
                assert_eq!(nseen, 15 * 15, "all class pairs up to l = 4");
                if pure {
                    assert_eq!(
                        ngen,
                        crate::generated::classes::N_BLOCK_KERNELS,
                        "every generated kernel exercised"
                    );
                }
                assert_eq!(
                    ngen + ntiled,
                    15 * 15,
                    "no class pair up to (gg|gg) on the generic loops"
                );
                // the generated classes also on the tiled kernels (the generic loops, now only
                // used above (gg|gg), are checked against every class in `blocks_match_*`)
                let (_, _, ntiled) = check_blocks_op::<V>(&b, &op, Path::Tiled);
                assert_eq!(ntiled, 15 * 15, "tiled kernels for every class pair");
            }
        }
    }

    #[test]
    fn exhaustive_classes_portable() {
        check_exhaustive::<Portable<false>>();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exhaustive_classes_avx2() {
        if linus_num::simd::Isa::detect() != linus_num::simd::Isa::Generic {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn run() {
                check_exhaustive::<linus_num::simd::Avx2>();
            }
            unsafe { run() }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exhaustive_classes_avx512() {
        if linus_num::simd::Isa::detect() == linus_num::simd::Isa::Avx512 {
            #[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
            unsafe fn run() {
                check_exhaustive::<linus_num::simd::Avx512>();
            }
            unsafe { run() }
        }
    }

    #[test]
    fn blocks_match_generic_portable() {
        check_blocks::<Portable<false>>(&water_like("def2-svp", true));
        check_blocks::<Portable<false>>(&water_like("6-31g*", false));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn blocks_match_generic_avx2() {
        if linus_num::simd::Isa::detect() != linus_num::simd::Isa::Generic {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn run(b: &BasisSet) {
                check_blocks::<linus_num::simd::Avx2>(b);
            }
            unsafe { run(&water_like("def2-svp", true)) }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn blocks_match_generic_avx512() {
        if linus_num::simd::Isa::detect() == linus_num::simd::Isa::Avx512 {
            #[target_feature(enable = "avx512f,avx512dq,fma")]
            unsafe fn run(b: &BasisSet) {
                check_blocks::<linus_num::simd::Avx512>(b);
            }
            unsafe { run(&water_like("def2-tzvp", true)) }
        }
    }
}
