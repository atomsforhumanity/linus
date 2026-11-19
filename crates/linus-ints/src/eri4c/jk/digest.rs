//! The J/K digester of the batched four-center engine ([`crate::engine::block`]): lane
//! screening per ket block, the integral kernels or the Hermite J-engine, and the digestion
//! of a block into J and K (the module docs of [`super`] describe the scheme).
//!
//! Determinism: J and K are added into 2^-64 fixed-point accumulators
//! (`linus_num::det::Fixed128`, exact and associative), one set per thread, so the digester
//! runs under the dynamic schedule ([`crate::engine::schedule::exact`]) and the result has
//! the same bits for any thread count.

use super::{BlockMeta, J_WEIGHT};
use crate::basis::{ncart, BasisSet};
use crate::core::hermite::nherm;
use crate::engine::block::{walk_task, BlockDigester, BraTask, KetBlock};
use crate::engine::schedule::Tasks;
use crate::eri4c::batch::{BatchKernels, BatchedPairs, Block, BlockWork, LANE_HEAD};
use crate::operator::Operator;
use linus_num::det::{Fixed128, Fixed128x8};
use linus_num::simd::{Lanes, Portable, LANES};
use std::mem::MaybeUninit;

/// The J/K digester: the read-only data of one build, shared by the threads.
pub(super) struct Ctx<'a> {
    pub(super) bp: &'a BatchedPairs,
    pub(super) basis: &'a BasisSet,
    pub(super) ns: usize,
    pub(super) nbf: usize,
    pub(super) nd: usize,
    pub(super) want_j: bool,
    pub(super) want_k: bool,
    pub(super) thresh: f64,
    /// densities, shell-blocked: block (x, y) row-major nx x ny at x0 * nbf + nx * y0 (AO
    /// offsets x0, y0), so a lane's block (a, c) is contiguous and its offset is linear in c0
    pub(super) dblk: &'a [Vec<f64>],
    /// max |D| per shell pair (over densities), and global max
    pub(super) dmax: &'a [f64],
    pub(super) dmax_all: f64,
    /// dmax of the pairs of every ket block, per lane (padding lanes 0)
    pub(super) dpb: &'a [[f64; LANES]],
    /// per density: D_cd of every ket block, lane-interleaved (element x of block bi, lane l
    /// at (joff[bi] + x) * LANES + l; padding lanes 0), so J digestion loads instead of
    /// gathering
    pub(super) dcdb: &'a [Vec<f64>],
    pub(super) joff: &'a [u32],
    pub(super) pair_loc: &'a [(u32, u32)],
    pub(super) meta: &'a [BlockMeta],
    pub(super) tmask: &'a [(u32, u64)],
    pub(super) blk_off: &'a [usize],
    pub(super) profile: bool,
    pub(super) xop: Operator,
    /// K integrals need their own pass (operator is not plain Coulomb)
    pub(super) k_separate: bool,
    /// Schwarz factor of the K operator
    pub(super) k_fac: f64,
    /// form integral blocks for J even where the J-engine would do (testing)
    pub(super) no_jengine: bool,
    pub(super) prim_factor: f64,
}

/// Fixed-point J accumulators of one integral element of a ket block, per lane (linus-num's
/// 2^-64 format). J lives only on retained shell pairs, so it is kept per ket block and lane:
/// J_cd of a block is added with whole vectors (no scatter), J_ab of a task at its pair's
/// (block, lane).
pub(super) type JAcc = Fixed128x8;

#[inline(always)]
pub(super) fn tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

impl Ctx<'_> {
    /// Accumulator position of shell block (x, y): element (u, v) is at base + u*su + v*sv.
    #[inline(always)]
    fn acc_block(&self, sx: usize, sy: usize) -> (usize, usize, usize) {
        let sh = &self.basis.shells;
        if sx >= sy {
            (self.blk_off[sx * self.ns + sy], sh[sy].nfunc(), 1)
        } else {
            (self.blk_off[sy * self.ns + sx], 1, sh[sx].nfunc())
        }
    }
}

/// Per-thread digestion scratch of the run-time-shaped digestion ([`digest`]); the
/// shape-specialized one ([`digest_k`]) keeps its arrays on the stack.
struct Scratch<V> {
    dcd: Vec<V>,
    dac: Vec<V>,
    dad: Vec<V>,
    dbc: Vec<V>,
    dbd: Vec<V>,
    kac: Vec<V>,
    kad: Vec<V>,
    kbc: Vec<V>,
    kbd: Vec<V>,
    jket: Vec<V>,
}

/// 64-byte aligned storage of the exchange panels (whole SIMD vectors per lane run).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([f64; LANES]);

/// State of the current task (one bra pair).
struct Task<V> {
    sa: usize,
    sb: usize,
    /// AO offsets of shells a and b
    a0: usize,
    b0: usize,
    /// max |D_ab| over the densities, and the bra pair's symmetry weight (0.5 for a == b)
    d_ab: f64,
    wab: f64,
    /// per shell y: max(|D_ay|, |D_by|) (exchange screening of the task)
    dk: Vec<f64>,
    /// per density: the block D_ab (na x nb)
    dab: Vec<f64>,
    /// first row of shell b in the row buffers (aliases a's rows for a diagonal pair)
    brow0: usize,
    /// per density: J_ab accumulators (lanes summed at the end of the task)
    jbra: Vec<V>,
    /// per density: exchange contributions K_ay, K_by of the task, column-major with `nr` rows
    /// per AO column: element (row r, AO column y) at y * nr + r, rows a then b (nrow = na + nb;
    /// na for a diagonal pair, whose b rows are the a rows), rows nrow..nr zero. nr (see
    /// [`panel_rows`]) makes one lane's rows of one column one unmasked add of 1, 2, 4 or
    /// whole LANES-wide vectors.
    panel: Vec<Vec<Line>>,
    /// per density: the density rows of a and b in the panel layout (element (row r, AO
    /// column y) = D[row r][y] at y * nr + r), filled per ket shell when a block first needs
    /// it (`dbuilt`), so exchange digestion loads each lane's rows of a column as one short
    /// vector and transposes instead of gathering element by element
    dpanel: Vec<Vec<Line>>,
    /// shells whose `dpanel` columns are valid for the current task (bitmap)
    dbuilt: Vec<u64>,
    nrow: usize,
    nr: usize,
    /// shells whose panel columns may be nonzero (bitmap over shells, set per block from
    /// [`BlockMeta::tmask`])
    tbits: Vec<u64>,
}

/// Rows of the panel for a bra pair: nrow = na + nb (na when diagonal), and the padded row
/// count nr per AO column: 1, 2 or 4 for nrow up to 4 (one narrow add per lane and column),
/// else nrow rounded up to LANES (whole vectors).
#[inline(always)]
fn panel_rows(na: usize, nb: usize, diag: bool) -> (usize, usize) {
    let nrow = if diag { na } else { na + nb };
    let nr = match nrow {
        0..=2 => nrow.max(1),
        3..=4 => 4,
        _ => nrow.div_ceil(LANES) * LANES,
    };
    (nrow, nr)
}

/// Add one block's exchange contributions for one set of ket columns to the task's panel.
/// Lane l's columns are AO columns cols[l] + y (y < ncol); column y holds rows a (from
/// ka[x * ncol + y]) then b (kb[x * ncol + y]), or ka + kb for a diagonal bra pair. Per column
/// and chunk of LANES rows the lanes are transposed and each lane adds one aligned vector.
/// Lanes with zero weight add exact zeros (their density factors are scaled by the weight, and
/// no partial sum is -0), so every lane adds unconditionally. Additions to one panel element
/// happen in the same order as in the masked per-lane version: by column, then lane.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn panel_scatter<V: Lanes>(
    panel: *mut f64,
    cols: &[usize; LANES],
    ka: &[V],
    kb: &[V],
    na: usize,
    nb: usize,
    ncol: usize,
    diag: bool,
) {
    if diag {
        panel_scatter_impl::<V, true>(panel, cols, ka, kb, na, nb, ncol)
    } else {
        panel_scatter_impl::<V, false>(panel, cols, ka, kb, na, nb, ncol)
    }
}

#[inline(always)]
unsafe fn panel_scatter_impl<V: Lanes, const DIAG: bool>(
    panel: *mut f64,
    cols: &[usize; LANES],
    ka: &[V],
    kb: &[V],
    na: usize,
    nb: usize,
    ncol: usize,
) {
    let (nrow, nr) = panel_rows(na, nb, DIAG);
    for y in 0..ncol {
        let mut r0 = 0;
        while r0 < nrow {
            let mut r = [V::zero(); LANES];
            for (j, rj) in r.iter_mut().enumerate() {
                let x = r0 + j;
                if DIAG {
                    if x < na {
                        *rj = *ka.get_unchecked(x * ncol + y) + *kb.get_unchecked(x * ncol + y);
                    }
                } else if x < na {
                    *rj = *ka.get_unchecked(x * ncol + y);
                } else if x < na + nb {
                    *rj = *kb.get_unchecked((x - na) * ncol + y);
                }
            }
            let t = V::transpose8(r);
            // rows past nrow are padding (zero); a last chunk of up to 4 rows adds 4
            let w = if nr < LANES {
                nr
            } else if nrow - r0 <= 4 {
                4
            } else {
                LANES
            };
            for (l, tl) in t.iter().enumerate() {
                let dst = panel.add((cols[l] + y) * nr + r0);
                match w {
                    1 => tl.add_to_low::<1>(dst),
                    2 => tl.add_to_low::<2>(dst),
                    4 => tl.add_to_low::<4>(dst),
                    _ => tl.add_to8(dst),
                }
            }
            r0 += LANES;
        }
    }
}

/// Fill the task's density row panels (all densities) for the shells in bit set `bits` of
/// bitmap word `w`: column y0 + v of shell y holds D[a0 + r][y0 + v] (r < na), then
/// D[b0 + r][y0 + v] (r < nb, not for a diagonal pair), from the shell-blocked density.
#[inline(never)]
fn dpanel_fill<V>(
    ctx: &Ctx,
    task: &mut Task<V>,
    w: usize,
    mut bits: u64,
    na: usize,
    nb: usize,
    diag: bool,
) {
    let sh = &ctx.basis.shells;
    let (nbf, nr) = (ctx.nbf, task.nr);
    let (a0, b0) = (task.a0, task.b0);
    while bits != 0 {
        let y = w * 64 + bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let (y0, ny) = (sh[y].ao_start, sh[y].nfunc());
        for di in 0..ctx.nd {
            let db = &ctx.dblk[di];
            let blk_a = &db[a0 * nbf + na * y0..a0 * nbf + na * (y0 + ny)];
            let blk_b = &db[b0 * nbf + nb * y0..b0 * nbf + nb * (y0 + ny)];
            let pl = &mut task.dpanel[di];
            let pv = unsafe {
                std::slice::from_raw_parts_mut(pl.as_mut_ptr() as *mut f64, pl.len() * LANES)
            };
            for v in 0..ny {
                let col = &mut pv[(y0 + v) * nr..(y0 + v) * nr + nr];
                for r in 0..na {
                    col[r] = blk_a[r * ny + v];
                }
                if !diag {
                    for r in 0..nb {
                        col[na + r] = blk_b[r * ny + v];
                    }
                }
            }
        }
    }
}

/// Densities of one block's exchange digestion for one set of ket columns, from the task's
/// density row panel: da[x * ncol + y] = D[a0 + x][cols[l] + y] * w_l (lane l) and db the same
/// for the rows of b (a's rows for a diagonal pair), i.e. exactly the values the per-element
/// gathers of the shell-blocked density gave. Per column, each lane's rows are read as one 4-
/// or 8-value run and transposed.
/// # Safety
/// The panel columns of every lane must be filled (`dpanel_fill`) or hold finite values with
/// zero lane weight; `da`, `db` must be valid for na x ncol, nb x ncol writes (every one of
/// those elements is written).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dens_cols<V: Lanes>(
    p: *const f64,
    nr: usize,
    cols: &[usize; LANES],
    ncol: usize,
    na: usize,
    nb: usize,
    diag: bool,
    vw: V,
    da: *mut V,
    db: *mut V,
) {
    let nrow = if diag { na } else { na + nb };
    for y in 0..ncol {
        let mut offs = [0usize; LANES];
        for l in 0..LANES {
            offs[l] = (cols[l] + y) * nr;
        }
        if nrow <= 4 {
            let t = V::load_rows4(p, &offs);
            for (x, &v) in t.iter().enumerate().take(nrow) {
                dens_put(da, db, x, y, ncol, na, diag, v * vw);
            }
        } else {
            let mut r0 = 0;
            while r0 < nrow {
                if nrow - r0 <= 4 {
                    let t = V::load_rows4(p.add(r0), &offs);
                    for (j, &v) in t.iter().enumerate().take(nrow - r0) {
                        dens_put(da, db, r0 + j, y, ncol, na, diag, v * vw);
                    }
                } else {
                    let t = V::load_rows8(p.add(r0), &offs);
                    for (j, &v) in t.iter().enumerate().take(nrow - r0) {
                        dens_put(da, db, r0 + j, y, ncol, na, diag, v * vw);
                    }
                }
                r0 += LANES;
            }
        }
    }
}

/// Row x (a rows, then b rows) of column y into da / db (see [`dens_cols`]).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dens_put<V: Lanes>(
    da: *mut V,
    db: *mut V,
    x: usize,
    y: usize,
    ncol: usize,
    na: usize,
    diag: bool,
    v: V,
) {
    if x < na {
        da.add(x * ncol + y).write(v);
        if diag {
            db.add(x * ncol + y).write(v);
        }
    } else {
        db.add((x - na) * ncol + y).write(v);
    }
}

/// J_cd of one block (jket * lane weights) to the block's fixed-point J accumulators
/// (`acc`: the block's ncd elements), one lane-wise 128-bit vector add per element. Lanes
/// with zero weight add zero.
#[inline(always)]
unsafe fn scatter_j<V: Lanes>(acc: &mut [JAcc], jket: &[V], vw2: V, ncd: usize) {
    let acc = &mut acc[..ncd];
    for (x, a) in acc.iter_mut().enumerate() {
        (*jket.get_unchecked(x) * vw2).fixed_add(&mut a.hi, &mut a.lo);
    }
}

/// D_cd of a block (lane-interleaved, `Ctx::dcdb`) times the lane weights.
#[inline(always)]
unsafe fn load_dcd<V: Lanes>(dcdb: &[f64], jo: usize, x: usize, vw2: V) -> V {
    V::load8(&*(dcdb.as_ptr().add((jo + x) * LANES) as *const [f64; LANES])) * vw2
}

/// Exchange FMAs of one block: K_ac += I D_bd (over b, d), K_bc += I D_ad, K_ad += I D_bc,
/// K_bd += I D_ac, in this loop order (the order of every partial sum is part of the result).
/// # Safety
/// Every slice must hold its na/nb x nc/nd extent, `ints` nab x ncd.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn k_fma<V: Lanes>(
    ints: &[V],
    dac: &[V],
    dad: &[V],
    dbc: &[V],
    dbd: &[V],
    kac: &mut [V],
    kad: &mut [V],
    kbc: &mut [V],
    kbd: &mut [V],
    na: usize,
    nb: usize,
    nc: usize,
    nd: usize,
) {
    let ncd = nc * nd;
    for a in 0..na {
        for b in 0..nb {
            let row = ints.get_unchecked((a * nb + b) * ncd..(a * nb + b + 1) * ncd);
            for c in 0..nc {
                let mut sac = *kac.get_unchecked(a * nc + c);
                let mut sbc = *kbc.get_unchecked(b * nc + c);
                let dbc_ = *dbc.get_unchecked(b * nc + c);
                let dac_ = *dac.get_unchecked(a * nc + c);
                for d in 0..nd {
                    let v = *row.get_unchecked(c * nd + d);
                    sac = v.mul_add(*dbd.get_unchecked(b * nd + d), sac);
                    sbc = v.mul_add(*dad.get_unchecked(a * nd + d), sbc);
                    let x = kad.get_unchecked_mut(a * nd + d);
                    *x = v.mul_add(dbc_, *x);
                    let x = kbd.get_unchecked_mut(b * nd + d);
                    *x = v.mul_add(dac_, *x);
                }
                *kac.get_unchecked_mut(a * nc + c) = sac;
                *kbc.get_unchecked_mut(b * nc + c) = sbc;
            }
        }
    }
}

/// J FMAs of one block: J_ab (register accumulators of the task, lanes kept apart) += I D_cd
/// and jket_cd = sum_ab I D_ab.
#[inline(always)]
unsafe fn j_fma<V: Lanes>(
    ints: &[V],
    dcd: &[V],
    dab: &[f64],
    jb: &mut [V],
    jket: &mut [V],
    nab: usize,
    ncd: usize,
) {
    for ab in 0..nab {
        let row = ints.get_unchecked(ab * ncd..(ab + 1) * ncd);
        let mut acc = *jb.get_unchecked(ab);
        let vd = V::splat(*dab.get_unchecked(ab));
        for cd in 0..ncd {
            acc = row.get_unchecked(cd).mul_add(*dcd.get_unchecked(cd), acc);
            let x = jket.get_unchecked_mut(cd);
            *x = row.get_unchecked(cd).mul_add(vd, *x);
        }
        *jb.get_unchecked_mut(ab) = acc;
    }
}

/// The first n entries of `b`, set to zero. With run-time n (M = 49: the ket-specialized
/// path) the zeros are written one vector at a time through volatile stores, which LLVM does
/// not turn into a memset call per array.
#[inline(always)]
fn zeroed_prefix<V: Lanes, const M: usize>(b: &mut [MaybeUninit<V>; M], n: usize) -> &mut [V] {
    let b = &mut b[..n];
    for x in b.iter_mut() {
        if M > 25 {
            unsafe { std::ptr::write_volatile(x.as_mut_ptr(), V::zero()) };
        } else {
            x.write(V::zero());
        }
    }
    // SAFETY: every element of the prefix is initialized; MaybeUninit<V> has V's layout
    unsafe { &mut *(b as *mut [MaybeUninit<V>] as *mut [V]) }
}

/// The first n entries of `b` as initialized values.
/// # Safety
/// They must all have been written.
#[inline(always)]
unsafe fn init_prefix<V, const M: usize>(b: &[MaybeUninit<V>; M], n: usize) -> &[V] {
    &*(&b[..n] as *const [MaybeUninit<V>] as *const [V])
}

/// Digest one block of integrals (bra pair of the task, LANES ket pairs) into J and K for a
/// compile-time ket shape (NC x ND functions): every scratch array is a local fixed-size
/// array of M vectors (M >= every na/nb x nc/nd product), so the unrolled ket loops keep
/// partial sums in registers (heap scratch slices may alias in LLVM's view, which forced
/// every partial sum through memory). `digest_dispatch` passes literal na, nb for the small
/// spherical shapes, which unrolls the bra loops as well.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn digest_k<V: Lanes, const NC: usize, const ND: usize, const M: usize>(
    ctx: &Ctx,
    task: &mut Task<V>,
    acc_j: &mut [Vec<JAcc>],
    jo: usize,
    meta: &BlockMeta,
    wl: &[f64; LANES],
    ints_j: Option<&[V]>,
    ints_k: Option<&[V]>,
    na: usize,
    nb: usize,
) {
    assert!(na * NC.max(ND) <= M && nb * NC.max(ND) <= M && NC * ND <= M);
    let nbf = ctx.nbf;
    let (nab, ncd) = (na * nb, NC * ND);
    let vw = V::from_array(*wl);
    let vw2 = vw * V::splat(2.0);
    // every gathered element (blocks (c, d), (a|b, c|d) of the shell-blocked D) lies inside
    // the nbf x nbf density
    debug_assert!((0..LANES).all(|l| meta.c0[l] + NC <= nbf && meta.d0[l] + ND <= nbf));
    for di in 0..ctx.nd {
        if let Some(ints) = ints_j {
            let ints = &ints[..nab * ncd];
            let dcdb = &ctx.dcdb[di][..(jo + ncd) * LANES];
            let mut dcd_ = [MaybeUninit::<V>::uninit(); M];
            for (x, v) in dcd_.iter_mut().enumerate().take(ncd) {
                v.write(unsafe { load_dcd(dcdb, jo, x, vw2) });
            }
            // SAFETY: the first ncd elements were written just above
            let dcd = unsafe { init_prefix(&dcd_, ncd) };
            let mut jket_ = [MaybeUninit::<V>::uninit(); M];
            let jket = zeroed_prefix(&mut jket_, ncd);
            unsafe {
                j_fma(
                    ints,
                    dcd,
                    &task.dab[di * nab..(di + 1) * nab],
                    &mut task.jbra[di * nab..(di + 1) * nab],
                    jket,
                    nab,
                    ncd,
                );
                scatter_j(&mut acc_j[di][jo..], jket, vw2, ncd);
            }
        }
        if let Some(ints) = ints_k {
            let ints = &ints[..nab * ncd];
            let diag = task.brow0 == 0;
            debug_assert_eq!(task.nr, panel_rows(na, nb, diag).1);
            // only the used prefixes are initialized (zeroing all M vectors of the ten arrays
            // was a 31 KB memset per block for the f-bra shapes)
            let (mut dac_, mut dad_, mut dbc_, mut dbd_) = (
                [MaybeUninit::<V>::uninit(); M],
                [MaybeUninit::<V>::uninit(); M],
                [MaybeUninit::<V>::uninit(); M],
                [MaybeUninit::<V>::uninit(); M],
            );
            // SAFETY: dens_cols writes all na x NC (nb x NC, ...) elements of each prefix
            let (dac, dad, dbc, dbd) = unsafe {
                let dp = task.dpanel[di].as_ptr() as *const f64;
                dens_cols(
                    dp,
                    task.nr,
                    &meta.c0,
                    NC,
                    na,
                    nb,
                    diag,
                    vw,
                    dac_.as_mut_ptr() as *mut V,
                    dbc_.as_mut_ptr() as *mut V,
                );
                dens_cols(
                    dp,
                    task.nr,
                    &meta.d0,
                    ND,
                    na,
                    nb,
                    diag,
                    vw,
                    dad_.as_mut_ptr() as *mut V,
                    dbd_.as_mut_ptr() as *mut V,
                );
                (
                    init_prefix(&dac_, na * NC),
                    init_prefix(&dad_, na * ND),
                    init_prefix(&dbc_, nb * NC),
                    init_prefix(&dbd_, nb * ND),
                )
            };
            let (mut kac_, mut kad_, mut kbc_, mut kbd_) = (
                [MaybeUninit::<V>::uninit(); M],
                [MaybeUninit::<V>::uninit(); M],
                [MaybeUninit::<V>::uninit(); M],
                [MaybeUninit::<V>::uninit(); M],
            );
            let (kac, kad) = (
                zeroed_prefix(&mut kac_, na * NC),
                zeroed_prefix(&mut kad_, na * ND),
            );
            let (kbc, kbd) = (
                zeroed_prefix(&mut kbc_, nb * NC),
                zeroed_prefix(&mut kbd_, nb * ND),
            );
            unsafe {
                k_fma(ints, dac, dad, dbc, dbd, kac, kad, kbc, kbd, na, nb, NC, ND);
                let pp = task.panel[di].as_mut_ptr() as *mut f64;
                panel_scatter(pp, &meta.c0, kac, kbc, na, nb, NC, diag);
                panel_scatter(pp, &meta.d0, kad, kbd, na, nb, ND, diag);
            }
        }
    }
}

/// Digest one block of integrals into J and K for run-time shapes (g shells and above,
/// Cartesian shells): same operations as [`digest_k`], with heap scratch.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn digest<V: Lanes>(
    ctx: &Ctx,
    task: &mut Task<V>,
    s: &mut Scratch<V>,
    acc_j: &mut [Vec<JAcc>],
    jo: usize,
    meta: &BlockMeta,
    wl: &[f64; LANES],
    ints_j: Option<&[V]>,
    ints_k: Option<&[V]>,
    na: usize,
    nb: usize,
    nc: usize,
    nd: usize,
) {
    let nbf = ctx.nbf;
    let (nab, ncd) = (na * nb, nc * nd);
    let vw = V::from_array(*wl);
    let vw2 = vw * V::splat(2.0);
    debug_assert!((0..LANES).all(|l| meta.c0[l] + nc <= nbf && meta.d0[l] + nd <= nbf));
    for di in 0..ctx.nd {
        if let Some(ints) = ints_j {
            let ints = &ints[..nab * ncd];
            let dcdb = &ctx.dcdb[di][..(jo + ncd) * LANES];
            let dcd = &mut s.dcd[..ncd];
            for (x, v) in dcd.iter_mut().enumerate() {
                *v = unsafe { load_dcd(dcdb, jo, x, vw2) };
            }
            let jket = &mut s.jket[..ncd];
            jket.fill(V::zero());
            unsafe {
                j_fma(
                    ints,
                    dcd,
                    &task.dab[di * nab..(di + 1) * nab],
                    &mut task.jbra[di * nab..(di + 1) * nab],
                    jket,
                    nab,
                    ncd,
                );
                scatter_j(&mut acc_j[di][jo..], jket, vw2, ncd);
            }
        }
        if let Some(ints) = ints_k {
            let ints = &ints[..nab * ncd];
            let dac = &mut s.dac[..na * nc];
            let dad = &mut s.dad[..na * nd];
            let dbc = &mut s.dbc[..nb * nc];
            let dbd = &mut s.dbd[..nb * nd];
            let kac = &mut s.kac[..na * nc];
            let kad = &mut s.kad[..na * nd];
            let kbc = &mut s.kbc[..nb * nc];
            let kbd = &mut s.kbd[..nb * nd];
            let diag = task.brow0 == 0;
            debug_assert_eq!(task.nr, panel_rows(na, nb, diag).1);
            unsafe {
                let dp = task.dpanel[di].as_ptr() as *const f64;
                dens_cols(
                    dp,
                    task.nr,
                    &meta.c0,
                    nc,
                    na,
                    nb,
                    diag,
                    vw,
                    dac.as_mut_ptr(),
                    dbc.as_mut_ptr(),
                );
                dens_cols(
                    dp,
                    task.nr,
                    &meta.d0,
                    nd,
                    na,
                    nb,
                    diag,
                    vw,
                    dad.as_mut_ptr(),
                    dbd.as_mut_ptr(),
                );
            }
            for k in [&mut *kac, &mut *kad, &mut *kbc, &mut *kbd] {
                k.fill(V::zero());
            }
            unsafe {
                k_fma(ints, dac, dad, dbc, dbd, kac, kad, kbc, kbd, na, nb, nc, nd);
                let pp = task.panel[di].as_mut_ptr() as *mut f64;
                panel_scatter(pp, &meta.c0, kac, kbc, na, nb, nc, diag);
                panel_scatter(pp, &meta.d0, kad, kbd, na, nb, nd, diag);
            }
        }
    }
}

/// Shape-specialized digestion: spherical shells up to d (1, 3, 5 functions) get fully
/// unrolled [`digest_k`] copies, bra shells up to 7 functions with a ket of spherical shells
/// up to f ket-specialized ones; anything else uses the run-time-sized [`digest`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn digest_dispatch<V: Lanes>(
    ctx: &Ctx,
    task: &mut Task<V>,
    s: &mut Scratch<V>,
    acc_j: &mut [Vec<JAcc>],
    jo: usize,
    meta: &BlockMeta,
    wl: &[f64; LANES],
    ints_j: Option<&[V]>,
    ints_k: Option<&[V]>,
    na: usize,
    nb: usize,
    nc: usize,
    nd: usize,
) {
    macro_rules! shapes {
        ($(($a:literal, $b:literal, $c:literal, $d:literal)),* $(,)?) => {
            match (na, nb, nc, nd) {
                $(($a, $b, $c, $d) => digest_k::<V, $c, $d, 25>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, $a, $b),)*
                _ if na <= 7 && nb <= 7 => match (nc, nd) {
                    (1, 1) => digest_k::<V, 1, 1, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (3, 1) => digest_k::<V, 3, 1, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (3, 3) => digest_k::<V, 3, 3, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (5, 1) => digest_k::<V, 5, 1, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (5, 3) => digest_k::<V, 5, 3, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (5, 5) => digest_k::<V, 5, 5, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (7, 1) => digest_k::<V, 7, 1, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (7, 3) => digest_k::<V, 7, 3, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (7, 5) => digest_k::<V, 7, 5, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    (7, 7) => digest_k::<V, 7, 7, 49>(ctx, task, acc_j, jo, meta, wl, ints_j, ints_k, na, nb),
                    _ => digest(ctx, task, s, acc_j, jo, meta, wl, ints_j, ints_k, na, nb, nc, nd),
                },
                _ => digest(ctx, task, s, acc_j, jo, meta, wl, ints_j, ints_k, na, nb, nc, nd),
            }
        };
    }
    // bra class >= ket class in the (l_a + l_b, l_a) order
    shapes!(
        (1, 1, 1, 1),
        (3, 1, 1, 1),
        (3, 1, 3, 1),
        (3, 3, 1, 1),
        (3, 3, 3, 1),
        (3, 3, 3, 3),
        (3, 3, 5, 1),
        (5, 1, 1, 1),
        (5, 1, 3, 1),
        (5, 1, 3, 3),
        (5, 1, 5, 1),
        (5, 3, 1, 1),
        (5, 3, 3, 1),
        (5, 3, 3, 3),
        (5, 3, 5, 1),
        (5, 3, 5, 3),
        (5, 5, 1, 1),
        (5, 5, 3, 1),
        (5, 5, 3, 3),
        (5, 5, 5, 1),
        (5, 5, 5, 3),
        (5, 5, 5, 5),
    )
}

/// Hermite J-engine state of a thread (per density: bra Hermite densities and J of the
/// current task; per block: ket Hermite densities and J).
struct JEngine<V> {
    maxp: usize,
    nhmax: usize,
    /// [density][bra primitive][r]
    db: Vec<f64>,
    db_ready: bool,
    /// [density][bra primitive][r], lanes summed at the end of the task
    jb: Vec<V>,
    /// [ket primitive][s]
    dk: Vec<V>,
    jk: Vec<V>,
    used: bool,
}

/// J of one block through the Hermite J-engine: ket Hermite densities from the lane E data
/// and the gathered D_cd, J_ab accumulated in Hermite space (folded in at the end of the
/// task), J_cd transformed back and scattered like the integral path does.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn jengine_block<V: BatchKernels>(
    ctx: &Ctx,
    task: &mut Task<V>,
    s: &mut Scratch<V>,
    je: &mut JEngine<V>,
    acc_j: &mut [Vec<JAcc>],
    blk: &Block,
    bi: usize,
    i: usize,
    wl: &[f64; LANES],
    cut: f64,
    work: &mut BlockWork<V>,
) {
    let bp = ctx.bp;
    let pr = &bp.pairs[i];
    let cb = &bp.classes[pr.class as usize];
    let ck = &bp.classes[blk.class as usize];
    let (nhb, nhk) = (cb.nh, ck.nh);
    let (nab, ncd) = (cb.nab(), ck.nab());
    let nk = blk.nprim as usize;
    let nrec = LANE_HEAD + ck.nz.len();
    let lanes = &bp.lane[blk.data..blk.data + nk * nrec * LANES];
    if !je.db_ready {
        // bra Hermite densities db[ip][r] = sum_ab E_ip[r][ab] D_ab
        let nnzb = cb.nz.len();
        for di in 0..ctx.nd {
            let dab = &task.dab[di * nab..(di + 1) * nab];
            for ip in 0..pr.nprim as usize {
                let eb = &bp.e[pr.e as usize + ip * nnzb..pr.e as usize + (ip + 1) * nnzb];
                let o = di * je.maxp * je.nhmax + ip * nhb;
                let db = &mut je.db[o..o + nhb];
                db.fill(0.0);
                for (k, &(r, ab)) in cb.nz.iter().enumerate() {
                    db[r as usize] += eb[k] * dab[ab as usize];
                }
            }
        }
        je.db_ready = true;
    }
    je.used = true;
    let vw2 = V::from_array(*wl) * V::splat(2.0);
    let jo = ctx.joff[bi] as usize;
    for di in 0..ctx.nd {
        let dcdb = &ctx.dcdb[di][..(jo + ncd) * LANES];
        let dcd = &mut s.dcd[..ncd];
        for (x, v) in dcd.iter_mut().enumerate() {
            *v = unsafe { load_dcd(dcdb, jo, x, vw2) };
        }
        // ket Hermite densities (lane weights folded in)
        let dk = &mut je.dk[..nk * nhk];
        for x in dk.iter_mut() {
            *x = V::zero();
        }
        for kq in 0..nk {
            let rec = &lanes[kq * nrec * LANES..(kq + 1) * nrec * LANES];
            for (k, &(sidx, cd)) in ck.nz.iter().enumerate() {
                let e = V::load(&rec[(LANE_HEAD + k) * LANES..]);
                dk[kq * nhk + sidx as usize] =
                    e.mul_add(dcd[cd as usize], dk[kq * nhk + sidx as usize]);
            }
        }
        // per density: dense [ip * nhb + r] views of the task's bra arrays
        let o = di * je.maxp * je.nhmax;
        let np = pr.nprim as usize * nhb;
        unsafe {
            V::block_jengine(
                bp,
                i,
                bi,
                cut,
                &je.db[o..o + np],
                &je.dk[..nk * nhk],
                &mut je.jb[o..o + np],
                &mut je.jk[..nk * nhk],
                work,
            )
        };
        // J_cd = sum_kq sum_s E'_kq[s][cd] jk[kq][s], times the lane weights
        let jket = &mut s.jket[..ncd];
        for x in jket.iter_mut() {
            *x = V::zero();
        }
        for kq in 0..nk {
            let rec = &lanes[kq * nrec * LANES..(kq + 1) * nrec * LANES];
            for (k, &(sidx, cd)) in ck.nz.iter().enumerate() {
                let e = V::load(&rec[(LANE_HEAD + k) * LANES..]);
                jket[cd as usize] = e.mul_add(je.jk[kq * nhk + sidx as usize], jket[cd as usize]);
            }
        }
        // scatter J_cd (exact fixed point)
        unsafe { scatter_j(&mut acc_j[di][jo..], jket, vw2, ncd) };
    }
}

/// The per-thread statistics of a build.
#[derive(Clone, Debug, Default)]
pub(super) struct ThreadStats {
    pub(super) blocks: u64,
    pub(super) blocks_j_only: u64,
    pub(super) prim_quartets_j_only: u64,
    pub(super) lanes: u64,
    pub(super) prim_quartets: u64,
    /// per (bra class, ket class): blocks, lanes, prim lane quartets, cycles, integral
    /// cycles, digestion cycles, evaluated primitive quartet vectors
    pub(super) prof: Vec<[u64; 7]>,
    /// cycles in the worker, and in task flushes
    pub(super) cyc_total: u64,
    pub(super) cyc_flush: u64,
}

/// The sizes of a thread's state (from the plan and the request).
pub(super) struct WorkerSizes {
    pub(super) lmax: usize,
    pub(super) nbf: usize,
    pub(super) ns: usize,
    pub(super) nd: usize,
    pub(super) want_j: bool,
    pub(super) want_k: bool,
    pub(super) k_separate: bool,
    /// J accumulator elements (per density), K accumulator elements (per density)
    pub(super) jlen: usize,
    pub(super) acc_len: usize,
    /// the largest primitive count of a retained pair
    pub(super) maxp: usize,
    /// class pairs of the profile (0 without profiling)
    pub(super) nprof: usize,
}

/// One thread's part of a J/K workspace: its fixed-point accumulators, statistics and
/// scratch, kept between builds (the accumulators are zeroed when the thread takes its first
/// task of a build).
pub(super) struct Worker<V> {
    pub(super) acc_j: Vec<Vec<JAcc>>,
    pub(super) acc_k: Vec<Vec<Fixed128>>,
    /// whether the thread took a task in this build (its accumulators hold its part)
    pub(super) active: bool,
    pub(super) stats: ThreadStats,
    work: BlockWork<V>,
    ints: Vec<V>,
    ints_k: Vec<V>,
    s: Scratch<V>,
    task: Task<V>,
    je: JEngine<V>,
}

impl<V: Lanes> Worker<V> {
    pub(super) fn new(z: &WorkerSizes) -> Self {
        let lmax = z.lmax;
        let nf = 2 * lmax + 1;
        let nfc = (lmax + 1) * (lmax + 2) / 2; // Cartesian d+ shells
        let nfmax = nf.max(nfc);
        let (ns, nbf, nd) = (z.ns, z.nbf, z.nd);
        let nf2 = nfmax * nfmax;
        let nhmax = nherm(2 * lmax);
        let maxp = z.maxp;
        Worker {
            acc_j: if z.want_j {
                (0..nd).map(|_| vec![JAcc::ZERO; z.jlen]).collect()
            } else {
                Vec::new()
            },
            acc_k: if z.want_k {
                (0..nd).map(|_| vec![Fixed128::ZERO; z.acc_len]).collect()
            } else {
                Vec::new()
            },
            active: false,
            stats: ThreadStats {
                prof: vec![[0; 7]; z.nprof],
                ..Default::default()
            },
            work: BlockWork::<V>::new(lmax),
            ints: vec![V::zero(); nfmax.pow(4)],
            ints_k: vec![V::zero(); if z.k_separate { nfmax.pow(4) } else { 0 }],
            s: Scratch {
                dcd: vec![V::zero(); nf2],
                dac: vec![V::zero(); nf2],
                dad: vec![V::zero(); nf2],
                dbc: vec![V::zero(); nf2],
                dbd: vec![V::zero(); nf2],
                kac: vec![V::zero(); nf2],
                kad: vec![V::zero(); nf2],
                kbc: vec![V::zero(); nf2],
                kbd: vec![V::zero(); nf2],
                jket: vec![V::zero(); nf2],
            },
            task: Task {
                sa: 0,
                sb: 0,
                a0: 0,
                b0: 0,
                d_ab: 0.0,
                wab: 0.0,
                dk: vec![0.0; ns],
                dab: vec![0.0; nd * nf2],
                brow0: 0,
                jbra: vec![V::zero(); nd * nf2],
                panel: (0..nd)
                    .map(|_| vec![Line([0.0; LANES]); (2 * nfmax).div_ceil(LANES) * nbf])
                    .collect(),
                // one extra line: short panels (nr < 8) are read 4 or 8 rows at a time
                dpanel: (0..nd)
                    .map(|_| vec![Line([0.0; LANES]); (2 * nfmax).div_ceil(LANES) * nbf + 1])
                    .collect(),
                dbuilt: vec![0; ns.div_ceil(64)],
                nrow: 0,
                nr: 0,
                tbits: vec![0; ns.div_ceil(64)],
            },
            je: JEngine {
                maxp,
                nhmax,
                db: vec![0.0; nd * maxp * nhmax],
                db_ready: false,
                jb: vec![V::zero(); nd * maxp * nhmax],
                dk: vec![V::zero(); maxp * nhmax],
                jk: vec![V::zero(); maxp * nhmax],
                used: false,
            },
        }
    }

    /// A state without buffers (it holds a thread's place while the thread works on its own
    /// state as a local, see [`work`]).
    fn vacant() -> Self {
        Worker {
            acc_j: Vec::new(),
            acc_k: Vec::new(),
            active: false,
            stats: ThreadStats::default(),
            work: BlockWork {
                fm: Vec::new(),
                rn: Vec::new(),
                r: Vec::new(),
                rwork: Vec::new(),
                t: Vec::new(),
                cube: Vec::new(),
                btab: &[],
            },
            ints: Vec::new(),
            ints_k: Vec::new(),
            s: Scratch {
                dcd: Vec::new(),
                dac: Vec::new(),
                dad: Vec::new(),
                dbc: Vec::new(),
                dbd: Vec::new(),
                kac: Vec::new(),
                kad: Vec::new(),
                kbc: Vec::new(),
                kbd: Vec::new(),
                jket: Vec::new(),
            },
            task: Task {
                sa: 0,
                sb: 0,
                a0: 0,
                b0: 0,
                d_ab: 0.0,
                wab: 0.0,
                dk: Vec::new(),
                dab: Vec::new(),
                brow0: 0,
                jbra: Vec::new(),
                panel: Vec::new(),
                dpanel: Vec::new(),
                dbuilt: Vec::new(),
                nrow: 0,
                nr: 0,
                tbits: Vec::new(),
            },
            je: JEngine {
                maxp: 0,
                nhmax: 0,
                db: Vec::new(),
                db_ready: false,
                jb: Vec::new(),
                dk: Vec::new(),
                jk: Vec::new(),
                used: false,
            },
        }
    }

    /// The thread takes its first task of a build: zero its accumulators.
    #[inline(never)]
    fn activate(&mut self) {
        for v in self.acc_j.iter_mut() {
            v.fill(JAcc::ZERO);
        }
        for v in self.acc_k.iter_mut() {
            v.fill(Fixed128::ZERO);
        }
        self.active = true;
    }
}

/// The bytes of one thread's state (a [`Worker`]) of these sizes: its accumulators, the
/// integral blocks, the digestion scratch and panels, the J-engine's Hermite buffers and the
/// kernels' scratch (the tiled kernels' R cube included).
pub(super) fn worker_bytes(z: &WorkerSizes) -> usize {
    let lmax = z.lmax;
    let nfmax = (2 * lmax + 1).max(ncart(lmax));
    let (nf2, nhmax) = (nfmax * nfmax, nherm(2 * lmax));
    // every lane type is LANES doubles
    let v = std::mem::size_of::<[f64; LANES]>();
    let lines = (2 * nfmax).div_ceil(LANES) * z.nbf;
    let acc = z.nd
        * (z.want_j as usize * z.jlen * std::mem::size_of::<JAcc>()
            + z.want_k as usize * z.acc_len * std::mem::size_of::<Fixed128>());
    let ints = (1 + z.k_separate as usize) * nfmax.pow(4) * v;
    let scratch = 10 * nf2 * v;
    let task = z.nd * (2 * lines + 1) * std::mem::size_of::<Line>()
        + z.nd * nf2 * (8 + v)
        + 8 * z.ns
        + 16 * z.ns.div_ceil(64);
    let je = z.nd * z.maxp * nhmax * (8 + v) + 2 * z.maxp * nhmax * v;
    let l = 4 * lmax;
    let block_work = (2 * (l + 1) + 3 * nherm(l) + nherm(2 * lmax) * ncart(lmax).pow(2)) * v
        + crate::eri4c::batch::CUBE_N * v;
    acc + ints + scratch + task + je + block_work
}

impl<V: BatchKernels> BlockDigester<V> for Ctx<'_> {
    type Thread = Worker<V>;

    #[inline(always)]
    fn begin_task(&self, th: &mut Worker<V>, t: &BraTask<'_>) -> f64 {
        let ctx = self;
        let sh = &ctx.basis.shells;
        let (ns, nbf, nd) = (ctx.ns, ctx.nbf, ctx.nd);
        let Worker { task, je, .. } = th;
        let (pr, cb) = (t.pair, t.class);
        let (sa, sbb) = (pr.sa as usize, pr.sb as usize);
        let (na, nb) = (cb.na, cb.nb);
        let nab = na * nb;
        let same_ab = sa == sbb;
        task.sa = sa;
        task.sb = sbb;
        task.a0 = sh[sa].ao_start;
        task.b0 = sh[sbb].ao_start;
        task.brow0 = if same_ab { 0 } else { na };
        (task.nrow, task.nr) = panel_rows(na, nb, same_ab);
        task.d_ab = ctx.dmax[sa * ns + sbb];
        if ctx.want_k {
            let (ra, rb) = (
                &ctx.dmax[sa * ns..(sa + 1) * ns],
                &ctx.dmax[sbb * ns..(sbb + 1) * ns],
            );
            for y in 0..ns {
                task.dk[y] = ctx.k_fac * ra[y].max(rb[y]);
            }
        }
        for di in 0..nd {
            for a in 0..na {
                for b in 0..nb {
                    task.dab[di * nab + a * nb + b] =
                        ctx.dblk[di][task.a0 * nbf + na * task.b0 + a * nb + b];
                }
            }
        }
        for x in task.jbra[..nd * nab].iter_mut() {
            *x = V::zero();
        }
        task.wab = if same_ab { 0.5 } else { 1.0 };
        je.db_ready = false;
        // largest possible density weight of a lane (J blocks count 4x, see below)
        ctx.dmax_all * (J_WEIGHT * ctx.want_j as u8 as f64).max(ctx.k_fac * ctx.want_k as u8 as f64)
    }

    #[inline(always)]
    fn block(&self, th: &mut Worker<V>, t: &BraTask<'_>, ket: &KetBlock<'_>) {
        let ctx = self;
        let bp = ctx.bp;
        let ncl = bp.classes.len();
        let Worker {
            task,
            s,
            je,
            acc_j,
            work,
            ints,
            ints_k,
            stats: out,
            ..
        } = th;
        let (i, pr, cb) = (t.i, t.pair, t.class);
        let (bi, blk, ck) = (ket.index, ket.block, ket.class);
        let (na, nb) = (cb.na, cb.nb);
        let (nc, ndd) = (ck.na, ck.nb);
        let same_ab = task.sa == task.sb;
        let (qi, d_ab, wab) = (pr.q, task.d_ab, task.wab);
        // lane weights (0 = skip): symmetry-unique (lanes up to pair i), valid, and passing
        // the bound; all lanes at once (bit masks)
        let meta = &ctx.meta[bi];
        let first = blk.first as usize;
        let nl = (blk.nlanes as usize).min(i + 1 - first);
        let valid = if nl >= LANES {
            0xff
        } else {
            ((1u32 << nl) - 1) as u8
        };
        let thr = V::splat(ctx.thresh);
        let qq = V::splat(qi) * V::load8(&blk.q);
        let fj = if ctx.want_j {
            V::splat(J_WEIGHT) * V::splat(d_ab).max(V::load8(&ctx.dpb[bi]))
        } else {
            V::zero()
        };
        let fk = if ctx.want_k {
            // SAFETY: sai and sbi hold shell indices (< ns; padding lanes 0, set by
            // BlockMeta::build), and task.dk has ns entries
            unsafe {
                V::gather(&task.dk, V::make_idx(meta.sai), 0).max(V::gather(
                    &task.dk,
                    V::make_idx(meta.sbi),
                    0,
                ))
            }
        } else {
            V::zero()
        };
        // x >= thresh as !(x < thresh) (no NaNs)
        let pass_j = !(qq * fj).lt(thr);
        let pass_k = !(qq * fk).lt(thr);
        let live = valid & !qq.lt(thr) & (pass_j | pass_k);
        if live == 0 {
            return;
        }
        let (need_j, need_k) = (live & pass_j != 0, live & pass_k != 0);
        // largest density factor of the live lanes (for primitive screening)
        let dfac = V::select(live, fj.max(fk), V::zero())
            .to_array()
            .iter()
            .fold(0.0f64, |m, &x| m.max(x));
        // the diagonal lane (pair i itself) counts half
        let half = if i - first < LANES {
            V::select(1 << (i - first), V::splat(0.5), V::splat(1.0))
        } else {
            V::splat(1.0)
        };
        let wl = V::select(live, V::splat(wab) * V::load8(&meta.wcd) * half, V::zero()).to_array();
        let c0 = if ctx.profile { tsc() } else { 0 };
        // Paths: when some lane needs exchange, the integral block is formed (with the
        // Coulomb operator it also gives J); J without a Coulomb integral block (J-only
        // blocks, J-only requests, or a non-Coulomb K operator) uses the Hermite J-engine.
        let do_k = ctx.want_k && need_k;
        let ints_coulomb = do_k && !ctx.k_separate;
        let do_j = ctx.want_j && (ints_coulomb || need_j);
        let jeng = do_j && !ints_coulomb && !ctx.no_jengine;
        // primitive quartets below prim_factor * thresh (with this block's density) are skipped
        let cut = ctx.prim_factor * ctx.thresh / dfac.max(1e-300);
        let mut npq_eval = 0u64;
        if ctx.profile {
            for ip in 0..pr.nprim as usize {
                let qb = bp.qp[pr.prim as usize + ip];
                for kq in 0..blk.nprim as usize {
                    npq_eval += (qb * bp.blk_qpk[blk.kq0 + kq] >= cut) as u64;
                }
            }
        }
        if ints_coulomb || (do_j && !jeng) {
            unsafe { V::block_eri(bp, i, bi, &Operator::Coulomb, cut, ints, work) };
        }
        if ctx.k_separate && do_k {
            unsafe {
                V::block_eri(
                    bp,
                    i,
                    bi,
                    &ctx.xop,
                    cut / ctx.k_fac.max(1e-300),
                    ints_k,
                    work,
                )
            };
        }
        if jeng {
            jengine_block::<V>(ctx, task, s, je, acc_j, blk, bi, i, &wl, cut, work);
        }
        let c1 = if ctx.profile { tsc() } else { 0 };
        let nlive = live.count_ones() as u64;
        out.blocks += 1;
        out.lanes += nlive;
        out.prim_quartets += pr.nprim as u64 * blk.nprim as u64 * nlive;
        if !need_k {
            out.blocks_j_only += 1;
            out.prim_quartets_j_only += pr.nprim as u64 * blk.nprim as u64 * nlive;
        }
        let ints_j_ref = if do_j && !jeng { Some(&ints[..]) } else { None };
        let ints_k_ref = if do_k {
            Some(if ctx.k_separate {
                &ints_k[..]
            } else {
                &ints[..]
            })
        } else {
            None
        };
        if do_k {
            // ket shells of this block (all valid lanes): density rows of a and b for
            // their columns, and the panel columns the flush visits (dead lanes add
            // zeros, which the flush skips)
            let tm = ctx.meta[bi].tm;
            for &(w, bits) in &ctx.tmask[tm[0] as usize..tm[1] as usize] {
                task.tbits[w as usize] |= bits;
                let need = bits & !task.dbuilt[w as usize];
                if need != 0 {
                    task.dbuilt[w as usize] |= need;
                    dpanel_fill(ctx, task, w as usize, need, na, nb, same_ab);
                }
            }
        }
        digest_dispatch(
            ctx,
            task,
            s,
            acc_j,
            ctx.joff[bi] as usize,
            &ctx.meta[bi],
            &wl,
            ints_j_ref,
            ints_k_ref,
            na,
            nb,
            nc,
            ndd,
        );
        if ctx.profile {
            let c2 = tsc();
            let e = &mut out.prof[cb.order * ncl + ck.order];
            e[0] += 1;
            e[1] += nlive;
            e[2] += pr.nprim as u64 * blk.nprim as u64 * nlive;
            e[3] += c2 - c0;
            e[4] += c1 - c0;
            e[5] += c2 - c1;
            e[6] += npq_eval;
        }
    }

    #[inline(always)]
    #[allow(
        clippy::needless_range_loop,
        reason = "the task's J and the accumulators in step"
    )]
    fn end_task(&self, th: &mut Worker<V>, t: &BraTask<'_>) {
        let ctx = self;
        let bp = ctx.bp;
        let sh = &ctx.basis.shells;
        let nd = ctx.nd;
        let Worker {
            task,
            je,
            acc_j,
            acc_k,
            stats: out,
            ..
        } = th;
        let (i, pr, cb) = (t.i, t.pair, t.class);
        let (sa, sbb) = (task.sa, task.sb);
        let (na, nb) = (cb.na, cb.nb);
        let nab = na * nb;
        let same_ab = sa == sbb;
        let cf0 = if ctx.profile { tsc() } else { 0 };
        if je.used {
            // J_ab += sum_ip sum_r E_ip[r][ab] jb[ip][r]
            let nhb = cb.nh;
            let nnzb = cb.nz.len();
            for di in 0..nd {
                for ip in 0..pr.nprim as usize {
                    let eb = &bp.e[pr.e as usize + ip * nnzb..pr.e as usize + (ip + 1) * nnzb];
                    let jb = &je.jb[di * je.maxp * je.nhmax + ip * nhb..];
                    for (k, &(r, ab)) in cb.nz.iter().enumerate() {
                        let x = &mut task.jbra[di * nab + ab as usize];
                        *x = V::splat(eb[k]).mul_add(jb[r as usize], *x);
                    }
                }
                let o = di * je.maxp * je.nhmax;
                for x in je.jb[o..o + pr.nprim as usize * nhb].iter_mut() {
                    *x = V::zero();
                }
            }
            je.used = false;
        }
        for di in 0..nd {
            if ctx.want_j {
                // J_ab at this pair's own (block, lane): element (a, b) of its block
                let (pb, pl) = ctx.pair_loc[i];
                let accj = &mut acc_j[di][ctx.joff[pb as usize] as usize..];
                for ab in 0..nab {
                    let v = task.jbra[di * nab + ab].hsum();
                    accj[ab].add(pl as usize, Fixed128::from_f64(v));
                }
            }
            if ctx.want_k {
                let acck = &mut acc_k[di];
                let pl = &mut task.panel[di];
                // the panel as f64 (Line is repr(C) [f64; LANES])
                let pv = unsafe {
                    std::slice::from_raw_parts_mut(pl.as_mut_ptr() as *mut f64, pl.len() * LANES)
                };
                let pr_ = task.nr;
                let nrow_sets: &[(usize, usize, usize)] = if same_ab {
                    &[(sa, 0, na)]
                } else {
                    &[(sa, 0, na), (sbb, na, nb)]
                };
                for (wi, &word) in task.tbits.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let y = wi * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let (y0, ny) = (sh[y].ao_start, sh[y].nfunc());
                        for &(sx, r0, nr) in nrow_sets {
                            let (base, su, sv) = ctx.acc_block(sx, y);
                            for v in 0..ny {
                                let col = &mut pv[(y0 + v) * pr_ + r0..(y0 + v) * pr_ + r0 + nr];
                                for (u, x) in col.iter_mut().enumerate() {
                                    if *x != 0.0 {
                                        acck[base + u * su + v * sv] += Fixed128::from_f64(*x);
                                        *x = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        task.tbits.fill(0);
        task.dbuilt.fill(0);
        if ctx.profile {
            out.cyc_flush += tsc() - cf0;
        }
    }
}

/// One thread's share of a build: the tasks the scheduler hands it, each walked by the block
/// engine with this digester.
#[inline(always)]
fn work<V: BatchKernels>(ctx: &Ctx, slot: &mut Worker<V>, tasks: Tasks<'_>) {
    // the state is a local of the build (moved out of the workspace and back, its buffers
    // untouched), so LLVM keeps its counters and buffer addresses in registers across the
    // blocks, as in a stack frame of its own (behind the slot's reference they were reloaded
    // after every store: 4% of the cycles of an SCF's J/K at one thread)
    let mut th = std::mem::replace(slot, Worker::vacant());
    let cw0 = if ctx.profile { tsc() } else { 0 };
    for i in tasks {
        if !th.active {
            th.activate();
        }
        walk_task::<V, Ctx>(ctx.bp, ctx.thresh, ctx, &mut th, i as usize);
    }
    if ctx.profile {
        th.stats.cyc_total = tsc() - cw0;
    }
    *slot = th;
}

/// A thread's state for the kernels of one instruction set.
pub(super) enum Slot {
    #[cfg(target_arch = "x86_64")]
    Avx512(Box<Worker<linus_num::simd::Avx512>>),
    #[cfg(target_arch = "x86_64")]
    Avx2(Box<Worker<linus_num::simd::Avx2>>),
    Generic(Box<Worker<Portable<false>>>),
}

impl Slot {
    /// The state for the kernels of `isa` (built inside code compiled for it).
    pub(super) fn new(isa: linus_num::simd::Isa, z: &WorkerSizes) -> Slot {
        use linus_num::simd::Isa;
        #[cfg(target_arch = "x86_64")]
        {
            #[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
            unsafe fn new512(z: &WorkerSizes) -> Slot {
                Slot::Avx512(Box::new(Worker::new(z)))
            }
            #[target_feature(enable = "avx2,fma")]
            unsafe fn new256(z: &WorkerSizes) -> Slot {
                Slot::Avx2(Box::new(Worker::new(z)))
            }
            match isa {
                // SAFETY: the plan's instruction set is at most the CPU's (Isa::clamped)
                Isa::Avx512 => return unsafe { new512(z) },
                Isa::Avx2 => return unsafe { new256(z) },
                Isa::Generic => {}
            }
        }
        let _ = isa;
        Slot::Generic(Box::new(Worker::new(z)))
    }

    /// The accumulators, activity and statistics of the slot's thread.
    pub(super) fn parts(&self) -> (&[Vec<JAcc>], &[Vec<Fixed128>], bool, &ThreadStats) {
        match self {
            #[cfg(target_arch = "x86_64")]
            Slot::Avx512(w) => (&w.acc_j, &w.acc_k, w.active, &w.stats),
            #[cfg(target_arch = "x86_64")]
            Slot::Avx2(w) => (&w.acc_j, &w.acc_k, w.active, &w.stats),
            Slot::Generic(w) => (&w.acc_j, &w.acc_k, w.active, &w.stats),
        }
    }

    /// Start a build: no tasks taken yet, statistics zeroed.
    pub(super) fn reset(&mut self) {
        fn r<V>(w: &mut Worker<V>) {
            w.active = false;
            let nprof = w.stats.prof.len();
            w.stats = ThreadStats {
                prof: vec![[0; 7]; nprof],
                ..Default::default()
            };
        }
        match self {
            #[cfg(target_arch = "x86_64")]
            Slot::Avx512(w) => r(w),
            #[cfg(target_arch = "x86_64")]
            Slot::Avx2(w) => r(w),
            Slot::Generic(w) => r(w),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
unsafe fn work_avx512(ctx: &Ctx, th: &mut Worker<linus_num::simd::Avx512>, tasks: Tasks<'_>) {
    work::<linus_num::simd::Avx512>(ctx, th, tasks)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn work_avx2(ctx: &Ctx, th: &mut Worker<linus_num::simd::Avx2>, tasks: Tasks<'_>) {
    work::<linus_num::simd::Avx2>(ctx, th, tasks)
}

/// Run a thread's share of a build with the kernels its slot was made for.
pub(super) fn run(ctx: &Ctx, slot: &mut Slot, tasks: Tasks<'_>) {
    match slot {
        // SAFETY: a slot of an instruction set is made only when the CPU has it (Slot::new)
        #[cfg(target_arch = "x86_64")]
        Slot::Avx512(w) => unsafe { work_avx512(ctx, w, tasks) },
        #[cfg(target_arch = "x86_64")]
        Slot::Avx2(w) => unsafe { work_avx2(ctx, w, tasks) },
        Slot::Generic(w) => work::<Portable<false>>(ctx, w, tasks),
    }
}
