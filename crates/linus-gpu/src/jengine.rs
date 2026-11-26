//! Density-fitted Coulomb terms in Hermite space: the J-engine contractions shared by the J-only
//! DF gradient (`grad::df_j::DfJHermite`) and the memory-lean DF-J (`jdirect_hermite::HermiteJ`).
//!
//! For the density rows w of a layout (`df_density_rows`: D_mn + D_nm off the diagonal shell
//! blocks), [`HermiteDf::fit`] forms, with no three-center integral block:
//! 1. per orbital pair primitive its Hermite density h_t = sum_mn E^mn_t w_mn (the eri3c integral
//!    builder's records; `dfj_pair_density`);
//! 2. gamma_P from the auxiliary bras (degree exactly l) against the pair kets h (`jengb`);
//! 3. c = V^-1 gamma (the metric's Cholesky factor, two triangular solves);
//! 4. per auxiliary primitive of degree l (exponent a, contraction coefficient c) its density
//!    g_u = c (2a)^-l sum_m c_m S_m\[u\] over the Hermite indices u of degree exactly l (a solid
//!    harmonic Gaussian is (2a)^-l sum_u S\[u\] Lambda_u; `dfj_aux_density`).
//!
//! The users add their passes on these tables ([`Passes`]): the gradient's pair bras of degree
//! la + lb + 1 against the fit's kets (`jengk`) and auxiliary bras of degree l + 1 against the
//! pairs (`jengb`) and the fit (`jengbk`); the memory-lean J's pair bras of degree la + lb against
//! the fit's kets (`jengk`) and J's rows (`dfj_pair_rows`). The kernels are the generated families
//! of [`crate::kernels::jengine_gen`], one module per (family, bra degree) with every ket degree of
//! that row (a fixed source: one cache entry serves every molecule), and the support kernels of
//! device/jengine/support.cu ([`MODULE`]).
//!
//! Nothing is screened (every primitive pair the records hold against every auxiliary
//! primitive). Every output element has one writing thread and every sum a fixed order (ket
//! degrees and classes in order, chunks of kets summed in order by `pcm_reduce`), so the results
//! are bitwise reproducible run to run.
//!
//! Coverage: orbital l <= 3 and auxiliary l <= 4 ([`MAX_PAIR`], [`MAX_AUX`]): [`HermiteDf::new`]
//! declines (None) beyond.

use crate::blas::{MatMut, MatRef, Op};
use crate::cpu::{cart2sph, cart_components, herm_tuv, ncart, nherm};
use crate::eri3c::{Eri3c, Eri3cHost};
use crate::error::GpuError;
use crate::kernels::boys::boys_table;
use crate::kernels::eri3c_class::{nfunc, ClassShape, Eri3cKind};
use crate::kernels::jengine_gen::{Variant, JENGB, JENGBK, JENGK, MAX_AUX, MAX_PAIR, THREADS};
use crate::kernels::registry::KernelModule;
use crate::linalg::Gemm;
use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
use std::borrow::Cow;

/// The support kernels: the fixed-point accumulator, then device/jengine/support.cu.
pub(crate) const MODULE: KernelModule = KernelModule::new(
    "jengine_support",
    || {
        Cow::Owned(format!(
            "{}\n{}",
            crate::detsum::DEVICE_SRC,
            include_str!("../device/jengine/support.cu")
        ))
    },
    &[
        "dfj_prims",
        "dfj_pair_density",
        "dfj_pair_grad",
        "dfj_pair_rows",
        "dfj_aux_density",
        "dfj_aux_gamma",
        "dfj_aux_grad",
    ],
);

/// Thread blocks per SM a launch of the generated kernels aims at (ket chunks when the bras are
/// few).
const BLOCKS_PER_SM: usize = 8;

/// (kets per chunk, chunks) for `nb` blocks of bras over `nk` kets in tiles of `tile`: enough
/// chunks for `target` blocks in all.
fn chunking(nb: usize, nk: usize, tile: usize, target: usize) -> (usize, usize) {
    let tiles = nk.div_ceil(tile).max(1);
    let want = target.div_ceil(nb.max(1)).clamp(1, tiles);
    let chunk = tiles.div_ceil(want) * tile;
    (chunk, nk.div_ceil(chunk).max(1))
}

/// The Hermite indices of degree exactly l, in Hermite order.
fn degree_set(l: usize) -> Vec<[usize; 3]> {
    herm_tuv()[nherm(l) - ncart(l)..nherm(l)]
        .iter()
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect()
}

/// CSR of `entries` (key, position, other) by key < `nkey`: starts, positions, others.
fn csr(mut entries: Vec<(usize, usize, usize)>, nkey: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    entries.sort();
    let mut starts = vec![0i32; nkey + 1];
    let (mut pos, mut other) = (Vec::new(), Vec::new());
    for &(k, j, o) in &entries {
        starts[k + 1] += 1;
        pos.push(j as i32);
        other.push(o as i32);
    }
    for k in 0..nkey {
        starts[k + 1] += starts[k];
    }
    (starts, pos, other)
}

/// The position of `t` in `set`, a Hermite index set of [`degree_set`].
fn position(set: &[[usize; 3]], t: [usize; 3]) -> Result<usize, GpuError> {
    set.iter()
        .position(|&x| x == t)
        .ok_or_else(|| GpuError::Contract("Hermite DF-J: a Hermite index outside its set".into()))
}

/// The passes a user runs on the shared tables: they decide the extra tables and the kernels.
#[derive(Clone, Copy)]
pub(crate) enum Passes<'a> {
    /// The J-only gradient (`grad::df_j`), with the derivative builder of the same pairs.
    Gradient(&'a Eri3c),
    /// J's layout rows (`jdirect_hermite`).
    Rows,
}

/// An orbital pair class of the eri3c builders (l1 >= l2 in kernel order).
pub(crate) struct PairClass {
    /// The class's index in the builders.
    pub(crate) ci: usize,
    /// Hermite degree l1 + l2, the second shell's functions, primitives.
    pub(crate) l: usize,
    pub(crate) n2: usize,
    pub(crate) nprim: usize,
    /// [p, P] per primitive in the integral records' order (the kets h).
    pub(crate) prims: DeviceBuf<f64>,
    pub(crate) prim_pair: DeviceBuf<i32>,
    pub(crate) kidx: DeviceBuf<i32>,
    /// Record length of the integral builder.
    rs: usize,
    /// The value set's pattern by Hermite index: CSR starts, positions in the set, mn.
    hp: DeviceBuf<i32>,
    he: DeviceBuf<i32>,
    hm: DeviceBuf<i32>,
    /// Hermite densities (nprim nherm(l)).
    pub(crate) h: DeviceBuf<f64>,
    pub(crate) grad: Option<PairGrad>,
    pub(crate) rows: Option<PairRows>,
}

/// A pair class's tables of the gradient's pair-centre pass.
pub(crate) struct PairGrad {
    /// Record length of the derivative builder.
    pub(crate) rsd: usize,
    /// [p, P] per primitive in the derivative records' order (the pair-centre bras): a swapped
    /// pair's primitives are transposed between the integral records (from the layout's
    /// `Pair3c`, (sa, sb) order) and the derivative records (built in kernel order).
    pub(crate) dprims: DeviceBuf<f64>,
    /// The six derivative sets: starts (7), Hermite index and mn per nonzero.
    pub(crate) gs: DeviceBuf<i32>,
    pub(crate) gr: DeviceBuf<i32>,
    pub(crate) gm: DeviceBuf<i32>,
    /// The accumulators of degree l + 1 (nprim nherm(l + 1)).
    pub(crate) a1: DeviceBuf<f64>,
}

/// A pair class's tables of J's rows.
pub(crate) struct PairRows {
    /// Pairs and functions per pair, primitives per pair (starts).
    pub(crate) npair: usize,
    pub(crate) nab: usize,
    pub(crate) p0: DeviceBuf<i32>,
    /// The value set's pattern by mn: CSR starts, positions, Hermite index.
    pub(crate) mp: DeviceBuf<i32>,
    pub(crate) me: DeviceBuf<i32>,
    pub(crate) mr: DeviceBuf<i32>,
    /// The accumulators of degree l against the fit (nprim nherm(l)).
    pub(crate) a: DeviceBuf<f64>,
}

/// The auxiliary primitives of one angular momentum.
pub(crate) struct AuxDeg {
    pub(crate) l: usize,
    pub(crate) nprim: usize,
    nshell: usize,
    pub(crate) prims: DeviceBuf<f64>,
    w: DeviceBuf<f64>,
    pub(crate) pshell: DeviceBuf<i32>,
    p0: DeviceBuf<i32>,
    col: DeviceBuf<i32>,
    /// S\[m nc + u\], the solid harmonics over the Hermite indices of degree l.
    s: DeviceBuf<f64>,
    pub(crate) kidx: DeviceBuf<i32>,
    /// Densities g (nprim nc) and gamma's accumulators (nprim nc).
    pub(crate) g: DeviceBuf<f64>,
    acc: DeviceBuf<f64>,
    pub(crate) grad: Option<AuxGrad>,
}

/// An auxiliary degree's tables of the gradient's auxiliary-centre pass.
pub(crate) struct AuxGrad {
    pub(crate) atom: DeviceBuf<i32>,
    /// The position of u + e_k among the Hermite indices of degree l + 1 (3 nc).
    pub(crate) up: DeviceBuf<i32>,
    /// The accumulators of degree l + 1 against the pairs and against the fit (nprim nc1).
    pub(crate) a1: DeviceBuf<f64>,
    pub(crate) a1m: DeviceBuf<f64>,
}

struct Kernels {
    prims: Kernel,
    pair_density: Kernel,
    aux_density: Kernel,
    aux_gamma: Kernel,
    reduce: Kernel,
}

/// The Hermite tables of one eri3c integral builder and its auxiliary basis (module docs), the
/// generated kernels its passes take, and the chunk partials.
pub(crate) struct HermiteDf {
    pub(crate) pairs: Vec<PairClass>,
    pub(crate) aux: Vec<AuxDeg>,
    kernels: Kernels,
    /// `(family, b, k)` -> kernel.
    jk: Vec<((&'static str, usize, usize), Kernel)>,
    /// Zeros for the bounds (no screening: threshold 0) and the chunk partials.
    zeros: DeviceBuf<f64>,
    part: DeviceBuf<f64>,
    btab: DeviceBuf<f64>,
    target: usize,
    /// Device bytes held (tables, densities, accumulators, partials).
    bytes: usize,
}

impl HermiteDf {
    /// Whether the generated kernels cover every class of `host` (orbital pairs of degree
    /// <= MAX_PAIR, auxiliary l <= MAX_AUX).
    pub(crate) fn covers(host: &Eri3cHost) -> bool {
        host.classes
            .iter()
            .all(|c| c.pairs.is_empty() || c.la + c.lb <= MAX_PAIR)
            && host.aux.iter().all(|a| a.nshell() == 0 || a.l <= MAX_AUX)
    }

    /// Sets up over the integral builder `ints` (its records) for `passes`; None when a class
    /// is beyond the generated kernels (or the derivative builder's pairs or sets differ from
    /// the integral builder's). Its tables, densities, accumulators and partials are held from
    /// the setup ([`HermiteDf::device_bytes`]).
    pub(crate) fn new(
        gpu: &Gpu,
        ints: &Eri3c,
        passes: Passes<'_>,
    ) -> Result<Option<Self>, GpuError> {
        let hi = ints.host();
        if hi.kind != Eri3cKind::Integrals || !Self::covers(hi) {
            return Ok(None);
        }
        let ders = match passes {
            Passes::Gradient(d) => Some(d),
            Passes::Rows => None,
        };
        if let Some(d) = ders {
            let hd = d.host();
            if hi.classes.len() != hd.classes.len() || hi.aux.len() != hd.aux.len() {
                return Ok(None);
            }
            for (c, dc) in hi.classes.iter().zip(&hd.classes) {
                if !c.pairs.is_empty() && (c.pairs != dc.pairs || c.p0 != dc.p0) {
                    return Ok(None);
                }
            }
        }
        let src = MODULE.source();
        let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
        let psrc = crate::pcm::MODULE.source();
        let kernels = Kernels {
            prims: k("dfj_prims")?,
            pair_density: k("dfj_pair_density")?,
            aux_density: k("dfj_aux_density")?,
            aux_gamma: k("dfj_aux_gamma")?,
            reduce: gpu.kernel(crate::pcm::MODULE.name, &psrc, "pcm_reduce")?,
        };
        let target = BLOCKS_PER_SM * gpu.info().sm_count.max(1) as usize;
        let t = THREADS as usize;
        let mut bytes = 0;
        let mut pairs = Vec::new();
        let mut maxn = 1;
        for (ci, c) in hi.classes.iter().enumerate() {
            let nprim = *c.p0.last().unwrap_or(&0) as usize;
            if c.pairs.is_empty() || nprim == 0 {
                continue;
            }
            let l = c.la + c.lb;
            let nh = nherm(l);
            let nab = nfunc(c.la) * nfunc(c.lb);
            // the value set by Hermite index (and by mn for J's rows)
            let shape = ClassShape::new(c.la, c.lb, 0, Eri3cKind::Integrals);
            let set: Vec<(usize, usize, usize)> = shape.sets[0]
                .iter()
                .enumerate()
                .map(|(j, &(r, mn))| (r, j, mn))
                .collect();
            let (hp, he, hm) = csr(set.clone(), nh);
            let mut prim_pair = Vec::with_capacity(nprim);
            for p in 0..c.pairs.len() {
                for _ in c.p0[p]..c.p0[p + 1] {
                    prim_pair.push(p as i32);
                }
            }
            let gather = |b: &Eri3c, rs: usize| -> Result<DeviceBuf<f64>, GpuError> {
                let (rec, _, _) = b
                    .class_records(ci)
                    .ok_or_else(|| GpuError::Contract("Hermite DF-J: class records".into()))?;
                let mut pr = gpu.alloc_zeros::<f64>(4 * nprim)?;
                // SAFETY: matches dfj_prims; nprim records of rs values, 4 nprim outputs.
                unsafe {
                    gpu.launch(
                        &kernels.prims,
                        Launch::linear(4 * nprim, 128),
                        &[
                            Arg::I64(nprim as i64),
                            Arg::I32(rs as i32),
                            Arg::buf(rec),
                            Arg::buf_mut(&mut pr),
                        ],
                    )
                }?;
                Ok(pr)
            };
            let grad = match ders {
                Some(d) => {
                    // the derivative sets back to back from the record's first nonzero
                    let dshape = ClassShape::new(c.la, c.lb, 0, Eri3cKind::Derivatives);
                    let (mut gs, mut gr, mut gm) = (vec![0i32], Vec::new(), Vec::new());
                    for (s, dset) in dshape.sets.iter().enumerate() {
                        if dshape.soff[s] != gr.len() {
                            return Ok(None);
                        }
                        for &(r, mn) in dset {
                            gr.push(r as i32);
                            gm.push(mn as i32);
                        }
                        gs.push(gr.len() as i32);
                    }
                    let rsd = d.host().classes[ci].rec;
                    let g = PairGrad {
                        rsd,
                        dprims: gather(d, rsd)?,
                        gs: gpu.upload(&gs)?,
                        gr: gpu.upload(&gr)?,
                        gm: gpu.upload(&gm)?,
                        a1: gpu.alloc_zeros(nprim * nherm(l + 1))?,
                    };
                    bytes += g.dprims.bytes()
                        + g.gs.bytes()
                        + g.gr.bytes()
                        + g.gm.bytes()
                        + g.a1.bytes();
                    Some(g)
                }
                None => None,
            };
            let rows = match ders {
                Some(_) => None,
                None => {
                    let (mp, me, mr) = csr(set.iter().map(|&(r, j, mn)| (mn, j, r)).collect(), nab);
                    let r = PairRows {
                        npair: c.pairs.len(),
                        nab,
                        p0: gpu.upload(&c.p0)?,
                        mp: gpu.upload(&mp)?,
                        me: gpu.upload(&me)?,
                        mr: gpu.upload(&mr)?,
                        a: gpu.alloc_zeros(nprim * nh)?,
                    };
                    bytes +=
                        r.p0.bytes() + r.mp.bytes() + r.me.bytes() + r.mr.bytes() + r.a.bytes();
                    Some(r)
                }
            };
            let kidx: Vec<i32> = (0..nprim as i32).collect();
            maxn = maxn.max(nprim);
            let pc = PairClass {
                ci,
                l,
                n2: nfunc(c.lb),
                nprim,
                prims: gather(ints, c.rec)?,
                prim_pair: gpu.upload(&prim_pair)?,
                kidx: gpu.upload(&kidx)?,
                rs: c.rec,
                hp: gpu.upload(&hp)?,
                he: gpu.upload(&he)?,
                hm: gpu.upload(&hm)?,
                h: gpu.alloc_zeros(nprim * nh)?,
                grad,
                rows,
            };
            bytes += pc.prims.bytes()
                + pc.prim_pair.bytes()
                + pc.kidx.bytes()
                + pc.hp.bytes()
                + pc.he.bytes()
                + pc.hm.bytes()
                + pc.h.bytes();
            pairs.push(pc);
        }
        let mut aux = Vec::new();
        for a in &hi.aux {
            let nshell = a.nshell();
            let nprim = *a.p0.last().unwrap_or(&0) as usize;
            if nshell == 0 || nprim == 0 {
                continue;
            }
            let l = a.l;
            let nc = ncart(l);
            let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
            // per primitive: a, c (2a)^-l (-1)^l / a, A, q -> [a, A] and w = c (2a)^-l
            let mut prims = Vec::with_capacity(4 * nprim);
            let mut w = Vec::with_capacity(nprim);
            let mut pshell = Vec::with_capacity(nprim);
            for s in 0..nshell {
                for i in a.p0[s] as usize..a.p0[s + 1] as usize {
                    let d = &a.d[6 * i..6 * i + 6];
                    prims.extend_from_slice(&[d[0], d[2], d[3], d[4]]);
                    w.push(d[1] * d[0] * sign);
                    pshell.push(s as i32);
                }
            }
            // S[m][u]: Cartesian x, y, z for p; linus-ints' solid harmonics from l = 2 on
            let nf = nfunc(l);
            let set = degree_set(l);
            let mut smat = vec![0.0; nf * nc];
            let carts = cart_components(l);
            if l <= 1 {
                for (m, c) in carts.iter().enumerate() {
                    smat[m * nc + position(&set, *c)?] = 1.0;
                }
            } else {
                let tr = cart2sph(l);
                for m in 0..nf {
                    for (ci, c) in carts.iter().enumerate() {
                        smat[m * nc + position(&set, *c)?] = tr[m * carts.len() + ci];
                    }
                }
            }
            let grad = match ders {
                Some(_) => {
                    let set1 = degree_set(l + 1);
                    let mut up = vec![0i32; 3 * nc];
                    for k in 0..3 {
                        for (u, t) in set.iter().enumerate() {
                            let mut v = *t;
                            v[k] += 1;
                            up[k * nc + u] = position(&set1, v)? as i32;
                        }
                    }
                    let nc1 = ncart(l + 1);
                    let g = AuxGrad {
                        atom: gpu.upload(&a.atom)?,
                        up: gpu.upload(&up)?,
                        a1: gpu.alloc_zeros(nprim * nc1)?,
                        a1m: gpu.alloc_zeros(nprim * nc1)?,
                    };
                    bytes += g.atom.bytes() + g.up.bytes() + g.a1.bytes() + g.a1m.bytes();
                    Some(g)
                }
                None => None,
            };
            let kidx: Vec<i32> = (0..nprim as i32).collect();
            maxn = maxn.max(nprim);
            let ad = AuxDeg {
                l,
                nprim,
                nshell,
                prims: gpu.upload(&prims)?,
                w: gpu.upload(&w)?,
                pshell: gpu.upload(&pshell)?,
                p0: gpu.upload(&a.p0)?,
                col: gpu.upload(&a.col)?,
                s: gpu.upload(&smat)?,
                kidx: gpu.upload(&kidx)?,
                g: gpu.alloc_zeros(nprim * nc)?,
                acc: gpu.alloc_zeros(nprim * nc)?,
                grad,
            };
            bytes += ad.prims.bytes()
                + ad.w.bytes()
                + ad.pshell.bytes()
                + ad.p0.bytes()
                + ad.col.bytes()
                + ad.s.bytes()
                + ad.kidx.bytes()
                + ad.g.bytes()
                + ad.acc.bytes();
            aux.push(ad);
        }
        // the (family, bra, ket) degree triples the passes take, and the largest chunk partials
        let grad = ders.is_some();
        let mut wanted: Vec<(&'static Variant, usize, usize)> = Vec::new();
        let mut part = 1;
        let mut need = |v: &'static Variant, b: usize, kd: usize, nb: usize, nk: usize, nh| {
            if wanted
                .iter()
                .all(|e| (e.0.name, e.1, e.2) != (v.name, b, kd))
            {
                wanted.push((v, b, kd));
            }
            let nch = chunking(nb.div_ceil(t), nk, v.tile(kd), target).1;
            part = part.max(nch * nb * nh);
        };
        for a in &aux {
            for p in &pairs {
                // gamma (both passes)
                need(&JENGB, a.l, p.l, a.nprim, p.nprim, ncart(a.l));
                if grad {
                    need(&JENGB, a.l + 1, p.l, a.nprim, p.nprim, ncart(a.l + 1));
                }
            }
            if grad {
                for a2 in &aux {
                    need(&JENGBK, a.l + 1, a2.l, a.nprim, a2.nprim, ncart(a.l + 1));
                }
            }
        }
        for p in &pairs {
            // the pair bras against the fit: degree l + 1 (gradient) or l (J's rows)
            let b = if grad { p.l + 1 } else { p.l };
            for a in &aux {
                need(&JENGK, b, a.l, p.nprim, a.nprim, nherm(b));
            }
        }
        // one module per (family, bra degree): every ket degree of that row, loaded once with
        // the entries wanted
        wanted.sort_by_key(|e| (e.0.name, e.1, e.2));
        let mut jk: Vec<((&'static str, usize, usize), Kernel)> = Vec::new();
        let mut i = 0;
        while i < wanted.len() {
            let (v, b) = (wanted[i].0, wanted[i].1);
            let mut e = i;
            while e < wanted.len() && wanted[e].0.name == v.name && wanted[e].1 == b {
                e += 1;
            }
            let src: String = (0..=v.kmax).map(|kd| v.source(b, kd)).collect();
            let names: Vec<String> = wanted[i..e]
                .iter()
                .map(|w| format!("{}_{b}_{}", v.name, w.2))
                .collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let ks = gpu.kernels(&format!("{}_{b}", v.name), &src, &refs)?;
            for (w, k) in wanted[i..e].iter().zip(ks) {
                jk.push(((v.name, b, w.2), k));
            }
            i = e;
        }
        let zeros = gpu.alloc_zeros(maxn)?;
        let part = gpu.alloc_zeros(part)?;
        let btab = gpu.upload(&boys_table())?;
        bytes += zeros.bytes() + part.bytes() + btab.bytes();
        Ok(Some(HermiteDf {
            pairs,
            aux,
            kernels,
            jk,
            zeros,
            part,
            btab,
            target,
            bytes,
        }))
    }

    /// Device bytes the tables, densities, accumulators and partials hold.
    pub(crate) fn device_bytes(&self) -> usize {
        self.bytes
    }
}

impl PairClass {
    /// Record length of the integral builder.
    pub(crate) fn rs(&self) -> usize {
        self.rs
    }
}

impl HermiteDf {
    /// acc (nbra nhb) = (first: 0, else acc) + sign sum over the chunks of the accumulators of
    /// the family's kernel of bra degree b against ket degree kd: the bras (nbra primitives
    /// [p, P]) against the kets (nket primitives [q, Q] with their densities, read through kidx).
    #[allow(
        clippy::too_many_arguments,
        reason = "the generated kernels' operands, grouped by side"
    )]
    pub(crate) fn accumulate(
        &self,
        gpu: &Gpu,
        (family, b): (&'static Variant, usize),
        (nb, bp): (usize, &DeviceBuf<f64>),
        (kd, nk, kp, kh, kidx): (
            usize,
            usize,
            &DeviceBuf<f64>,
            &DeviceBuf<f64>,
            &DeviceBuf<i32>,
        ),
        nhb: usize,
        acc: &DeviceBuf<f64>,
        first: bool,
        sign: f64,
    ) -> Result<(), GpuError> {
        if nb == 0 || nk == 0 {
            return Ok(());
        }
        let t = THREADS as usize;
        let (chunk, nch) = chunking(nb.div_ceil(t), nk, family.tile(kd), self.target);
        if self.part.len() < nch * nb * nhb || acc.len() < nb * nhb {
            return Err(GpuError::Contract("Hermite DF-J: accumulator sizes".into()));
        }
        let k = &self
            .jk
            .iter()
            .find(|e| e.0 == (family.name, b, kd))
            .ok_or_else(|| GpuError::Contract("Hermite DF-J: kernel not compiled".into()))?
            .1;
        let cfg = Launch {
            grid: (nb.div_ceil(t) as u32, nch as u32, 1),
            block: (THREADS, 1, 1),
            shared_bytes: 0,
        };
        // SAFETY: matches the generated jeng* kernels; nb bra primitives of 4 values with zero
        // bounds, nk kets (4 values, their densities, zero bounds; kidx a permutation) in chunks
        // of `chunk`, nch nb nhb partials (checked above).
        unsafe {
            gpu.launch(
                k,
                cfg,
                &[
                    Arg::I64(nb as i64),
                    Arg::buf(bp),
                    Arg::buf(&self.zeros),
                    Arg::I64(nk as i64),
                    Arg::I64(chunk as i64),
                    Arg::buf(kidx),
                    Arg::buf(kp),
                    Arg::buf(kh),
                    Arg::buf(&self.zeros),
                    Arg::buf(&self.btab),
                    Arg::F64(0.0),
                    Arg::buf(&self.part),
                ],
            )
        }?;
        let len = nb * nhb;
        // SAFETY: matches pcm_reduce; nch len partials, len outputs; the base is the output
        // itself (read and written by the same thread) after the first contribution.
        unsafe {
            gpu.launch(
                &self.kernels.reduce,
                Launch::linear(len, 128),
                &[
                    Arg::I64(len as i64),
                    Arg::I64(nch as i64),
                    Arg::buf(&self.part),
                    if first { Arg::Ptr(0) } else { Arg::buf(acc) },
                    Arg::F64(sign),
                    Arg::buf(acc),
                ],
            )
        }
    }

    /// Steps 1-4 of the module docs for the density rows `w` (the layout rows of `ints`, this
    /// engine's builder) and the metric's Cholesky factor `chol` (naux x naux, lower): the
    /// pairs' Hermite densities h, gamma, c = V^-1 gamma and the auxiliary densities g.
    pub(crate) fn fit(
        &mut self,
        gpu: &Gpu,
        ints: &Eri3c,
        w: &DeviceBuf<f64>,
        (chol, gemm, naux): (&DeviceBuf<f64>, &Gemm, usize),
    ) -> Result<(), GpuError> {
        // 1. the pairs' Hermite densities
        for c in &mut self.pairs {
            let (rec, out, _) = ints
                .class_records(c.ci)
                .ok_or_else(|| GpuError::Contract("Hermite DF-J: class records".into()))?;
            let nh = nherm(c.l);
            // SAFETY: matches dfj_pair_density; nprim records of rs values and their pairs'
            // rows, the value set's tables, the density rows of the layout; nprim nh outputs.
            unsafe {
                gpu.launch(
                    &self.kernels.pair_density,
                    Launch::linear(c.nprim * nh, 128),
                    &[
                        Arg::I64(c.nprim as i64),
                        Arg::I32(nh as i32),
                        Arg::I32(c.rs as i32),
                        Arg::I32(c.n2 as i32),
                        Arg::buf(rec),
                        Arg::buf(&c.prim_pair),
                        Arg::buf(out),
                        Arg::buf(&c.hp),
                        Arg::buf(&c.he),
                        Arg::buf(&c.hm),
                        Arg::buf(w),
                        Arg::buf_mut(&mut c.h),
                    ],
                )
            }?;
        }
        // 2. gamma: the auxiliary bras (degree l) against the pair kets, classes in order
        let mut gamma = gpu.alloc_zeros::<f64>(naux.max(1))?;
        for a in &self.aux {
            let nc = ncart(a.l);
            for (i, p) in self.pairs.iter().enumerate() {
                self.accumulate(
                    gpu,
                    (&JENGB, a.l),
                    (a.nprim, &a.prims),
                    (p.l, p.nprim, &p.prims, &p.h, &p.kidx),
                    nc,
                    &a.acc,
                    i == 0,
                    1.0,
                )?;
            }
            // SAFETY: matches dfj_aux_gamma; nshell nf threads reading their shells'
            // primitives' accumulators (nc each), writing naux values at the shells' columns.
            unsafe {
                gpu.launch(
                    &self.kernels.aux_gamma,
                    Launch::linear(a.nshell * nfunc(a.l), 128),
                    &[
                        Arg::I64(a.nshell as i64),
                        Arg::I32(nc as i32),
                        Arg::I32(nfunc(a.l) as i32),
                        Arg::buf(&a.p0),
                        Arg::buf(&a.col),
                        Arg::buf(&a.w),
                        Arg::buf(&a.s),
                        Arg::buf(&a.acc),
                        Arg::buf_mut(&mut gamma),
                    ],
                )
            }?;
        }
        // 3. c = V^-1 gamma
        for op in [Op::N, Op::T] {
            gemm.trsm_lower_left(
                op,
                (naux, 1),
                MatRef::new(chol, 0, naux),
                MatMut::new(&mut gamma, 0, naux),
            )?;
        }
        let c = gamma;
        // 4. the auxiliary densities of c
        for a in &mut self.aux {
            let nc = ncart(a.l);
            // SAFETY: matches dfj_aux_density; nprim nc outputs, the coefficients at the shells'
            // columns (below naux).
            unsafe {
                gpu.launch(
                    &self.kernels.aux_density,
                    Launch::linear(a.nprim * nc, 128),
                    &[
                        Arg::I64(a.nprim as i64),
                        Arg::I32(nc as i32),
                        Arg::I32(nfunc(a.l) as i32),
                        Arg::buf(&a.pshell),
                        Arg::buf(&a.col),
                        Arg::buf(&a.w),
                        Arg::buf(&a.s),
                        Arg::buf(&c),
                        Arg::buf_mut(&mut a.g),
                    ],
                )
            }?;
        }
        Ok(())
    }
}
