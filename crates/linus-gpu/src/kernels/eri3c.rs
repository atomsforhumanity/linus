//! Three-center Coulomb integrals (k|mn) of one shell class (l_a l_b | l_k), McMurchie-Davidson
//! in Hermite space exactly as linus-ints' CPU kernel (`eri3c`): for each orbital primitive
//! pair (exponent p, centre P, Hermite matrix E\[mn\]\[t'\]) and auxiliary primitive (exponent a,
//! centre A, coefficient c (2a)^-l (-1)^l),
//!
//!   (k|mn) += pref sum_c S_k\[c\] sum_t' R_{c+t'}(alpha, P - A) E\[mn\]\[t'\],
//!
//! with the Boys function from the same table and Taylor/downward recursion, and the same R
//! recursion order. The generator writes the whole primitive-triple body as straight-line code
//! for one class; one thread computes every (k|mn) of one (pair, auxiliary shell), looping over
//! their primitives, so each output has one writer (deterministic).
//!
//! Two contraction orders ([`Variant`]): `H` is the CPU's (H\[k\]\[t'\] = sum_c S_k\[c\] R_{c+t'},
//! then H E), `G` contracts E first per mn (G\[c\] = sum_t' R_{c+t'} E\[mn\]\[t'\], then S): a few
//! more flops but far fewer live registers.
//!
//! Output: out[(row(pair) + mn) * ld + col(aux shell) + k], row(pair) = pair index * nab,
//! the layout of linus-ints' `pair_block` stacked over pairs.

use super::boys::{BOYS_COLS, MAX_LT};
use super::eri3c_class::nfunc;
use super::{lit, KernelStats};
use crate::cpu::{
    build_pair, cart2sph, cart_components, herm_idx, herm_tuv, ncart, nherm, Aux3c, BasisSet,
    Pair3c, ShellView, ERI_PREF,
};
use std::fmt::Write;

/// A shell class (l_a l_b | l_k); shells with l >= 2 spherical, s and p Cartesian (linus's
/// convention in a spherical basis).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Class {
    pub la: usize,
    pub lb: usize,
    pub lk: usize,
}

impl Class {
    pub fn nab(&self) -> usize {
        nfunc(self.la) * nfunc(self.lb)
    }
    pub fn nk(&self) -> usize {
        nfunc(self.lk)
    }
    pub fn lt(&self) -> usize {
        self.la + self.lb + self.lk
    }
    pub fn tag(&self) -> String {
        const L: &[u8] = b"spdfghi";
        format!(
            "({}{}|{})",
            L[self.la] as char, L[self.lb] as char, L[self.lk] as char
        )
    }
}

/// Contraction order (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    H,
    G,
}

pub fn kernel_name(c: Class, v: Variant) -> String {
    format!(
        "eri3c_{}{}{}_{}",
        c.la,
        c.lb,
        c.lk,
        if v == Variant::H { "h" } else { "g" }
    )
}

/// CUDA C++ of the kernel for `class` and its statistics (flops per primitive triple).
#[allow(clippy::needless_range_loop)]
pub fn generate(class: Class, variant: Variant) -> (String, KernelStats) {
    let lt = class.lt();
    assert!(lt <= MAX_LT);
    let lab = class.la + class.lb;
    let nh = nherm(lab);
    let nab = class.nab();
    let nk = class.nk();
    let lk = class.lk;
    let nck = ncart(lk);
    let carts = cart_components(lk);
    let sk = cart2sph(lk);
    let tuv = herm_tuv();
    let name = kernel_name(class, variant);
    let mut s = String::new();
    let mut fl = 0usize;
    let w = &mut s;
    let _ = writeln!(
        w,
        "// linus-gpu three-center integrals {} ({:?} contraction; generated)",
        class.tag(),
        variant
    );
    let _ = writeln!(
        w,
        "extern \"C\" __global__ void __launch_bounds__(128) {name}(long long npair, int nauxsh,
    const int *__restrict__ pr_p0, const double *__restrict__ pr_p, const double *__restrict__ pr_pc, const double *__restrict__ pr_et, const long long *__restrict__ pr_row,
    const int *__restrict__ ax_p0, const double *__restrict__ ax_exp, const double *__restrict__ ax_coef, const double *__restrict__ ax_c, const int *__restrict__ ax_col,
    const double *__restrict__ btab, long long ld, double *__restrict__ out)
{{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npair * nauxsh) return;
    const long long ip = tid / nauxsh;
    const int ks = (int)(tid - ip * nauxsh);
    const double cx = ax_c[3 * ks], cy = ax_c[3 * ks + 1], cz = ax_c[3 * ks + 2];
    double *o = out + pr_row[ip] * ld + ax_col[ks];
    bool first = true;
    for (int a = pr_p0[ip]; a < pr_p0[ip + 1]; ++a) {{
        const double p = pr_p[a];
        const double px = pr_pc[3 * a] - cx, py = pr_pc[3 * a + 1] - cy, pz = pr_pc[3 * a + 2] - cz;
        const double rr = px * px + py * py + pz * pz;
        const double *et = pr_et + (long long)a * {};
        for (int b = ax_p0[ks]; b < ax_p0[ks + 1]; ++b) {{
            const double ae = ax_exp[b];
            const double sp = p + ae;
            const double al = p * ae / sp;
            const double t = al * rr;
            const double pref = {} / (p * ae * sqrt(sp)) * ax_coef[b];",
        nab * nh,
        lit(ERI_PREF)
    );
    fl += 9 + 6;
    // Boys F_0..F_lt
    let fv: Vec<String> = (0..=lt).map(|m| format!("f{m}")).collect();
    let _ = writeln!(w, "            double {};", fv.join(", "));
    let _ = writeln!(w, "            if (t < 117.0) {{");
    let _ = writeln!(w, "                const int i = (int)(t * 10.0 + 0.5);");
    let _ = writeln!(w, "                const double dt = i * 0.1 - t;");
    let _ = writeln!(
        w,
        "                const double *row = btab + i * {BOYS_COLS};"
    );
    let _ = writeln!(w, "                double acc = row[{}];", lt + 7);
    let inv_int = [0.0, 1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0, 1.0 / 7.0];
    for k in (0..7).rev() {
        let _ = writeln!(
            w,
            "                acc = row[{}] + acc * dt * {};",
            lt + k,
            lit(inv_int[k + 1])
        );
        fl += 3;
    }
    let _ = writeln!(w, "                f{lt} = acc;");
    let fact = [
        1.0,
        1.0,
        0.5,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
        1.0 / 40320.0,
    ];
    let mut ser = lit(fact[8]);
    for k in (0..8).rev() {
        ser = format!("{} + dt * ({ser})", lit(fact[k]));
    }
    fl += 17;
    let _ = writeln!(
        w,
        "                const double e = row[{}] * ({ser});",
        BOYS_COLS - 1
    );
    for m in (0..lt).rev() {
        let _ = writeln!(
            w,
            "                f{m} = (2.0 * t * f{} + e) * {};",
            m + 1,
            lit(1.0 / (2 * m + 1) as f64)
        );
        fl += 4;
    }
    let _ = writeln!(w, "            }} else {{");
    let _ = writeln!(w, "                const double ti = 1.0 / t;");
    let _ = writeln!(
        w,
        "                f0 = 0.5 * sqrt({} * ti);",
        lit(std::f64::consts::PI)
    );
    for m in 1..=lt {
        let _ = writeln!(
            w,
            "                f{m} = f{} * ({} * ti);",
            m - 1,
            lit((2 * m - 1) as f64 * 0.5)
        );
    }
    let _ = writeln!(w, "            }}");
    // scaled R^(n)_000
    let _ = writeln!(w, "            const double m2a = -2.0 * al;");
    let _ = writeln!(w, "            double sc = pref;");
    for n in 0..=lt {
        let _ = writeln!(w, "            const double q{n} = sc * f{n};");
        fl += 1;
        if n < lt {
            let _ = writeln!(w, "            sc *= m2a;");
            fl += 1;
        }
    }
    // R recursion, levels n = lt-1 .. 0 (r{n}_{h}); level lt is q{lt}
    let rv = |n: usize, h: usize| -> String {
        if h == 0 {
            format!("q{n}")
        } else {
            format!("r{n}_{h}")
        }
    };
    for n in (0..lt).rev() {
        for h in 1..nherm(lt - n) {
            let [t, u, v] = tuv[h];
            let (t, u, v) = (t as usize, u as usize, v as usize);
            let (axis, j1, j2, c) = if t > 0 {
                (
                    "px",
                    herm_idx(t - 1, u, v),
                    if t > 1 { herm_idx(t - 2, u, v) } else { 0 },
                    t as f64 - 1.0,
                )
            } else if u > 0 {
                (
                    "py",
                    herm_idx(t, u - 1, v),
                    if u > 1 { herm_idx(t, u - 2, v) } else { 0 },
                    u as f64 - 1.0,
                )
            } else {
                (
                    "pz",
                    herm_idx(t, u, v - 1),
                    if v > 1 { herm_idx(t, u, v - 2) } else { 0 },
                    v as f64 - 1.0,
                )
            };
            let rhs = if c > 0.0 {
                fl += 3;
                format!(
                    "{axis} * {} + {} * {}",
                    rv(n + 1, j1),
                    lit(c),
                    rv(n + 1, j2)
                )
            } else {
                fl += 1;
                format!("{axis} * {}", rv(n + 1, j1))
            };
            let _ = writeln!(w, "            const double {} = {rhs};", rv(n, h));
        }
    }
    // S_k[c] terms: for function k, the nonzero (c, coefficient)
    let sterms: Vec<Vec<(usize, f64)>> = (0..nk)
        .map(|k| {
            (0..nck)
                .filter(|&c| sk[k * nck + c] != 0.0)
                .map(|c| (c, sk[k * nck + c]))
                .collect()
        })
        .collect();
    let ridx = |c: usize, tp: usize| -> String {
        let [a0, a1, a2] = tuv[tp];
        rv(
            0,
            herm_idx(
                carts[c][0] + a0 as usize,
                carts[c][1] + a1 as usize,
                carts[c][2] + a2 as usize,
            ),
        )
    };
    let store = |w: &mut String, idx: String, val: &str| {
        let _ = writeln!(
            w,
            "                    o[{idx}] = first ? {val} : o[{idx}] + {val};"
        );
    };
    match variant {
        Variant::H => {
            for (k, st) in sterms.iter().enumerate() {
                let _ = writeln!(w, "            {{");
                for tp in 0..nh {
                    let sum: Vec<String> = st
                        .iter()
                        .map(|&(c, coef)| {
                            if coef == 1.0 {
                                ridx(c, tp)
                            } else {
                                format!("{} * {}", lit(coef), ridx(c, tp))
                            }
                        })
                        .collect();
                    fl += 2 * st.len() - 1;
                    let _ = writeln!(
                        w,
                        "                const double h{tp} = {};",
                        sum.join(" + ")
                    );
                }
                let _ = writeln!(w, "                #pragma unroll 1");
                let _ = writeln!(w, "                for (int mn = 0; mn < {nab}; ++mn) {{");
                let _ = writeln!(w, "                    const double *e = et + mn * {nh};");
                let dot: Vec<String> = (0..nh).map(|tp| format!("h{tp} * e[{tp}]")).collect();
                fl += nab * (2 * nh - 1);
                let _ = writeln!(
                    w,
                    "                    const double v = {};",
                    dot.join(" + ")
                );
                store(w, format!("mn * ld + {k}"), "v");
                let _ = writeln!(w, "                }}");
                let _ = writeln!(w, "            }}");
            }
        }
        Variant::G => {
            let _ = writeln!(w, "            #pragma unroll 1");
            let _ = writeln!(w, "            for (int mn = 0; mn < {nab}; ++mn) {{");
            let _ = writeln!(w, "                const double *e = et + mn * {nh};");
            for tp in 0..nh {
                let _ = writeln!(w, "                const double e{tp} = e[{tp}];");
            }
            for c in 0..nck {
                let dot: Vec<String> = (0..nh)
                    .map(|tp| format!("{} * e{tp}", ridx(c, tp)))
                    .collect();
                fl += nab * (2 * nh - 1);
                let _ = writeln!(
                    w,
                    "                const double g{c} = {};",
                    dot.join(" + ")
                );
            }
            for (k, st) in sterms.iter().enumerate() {
                let sum: Vec<String> = st
                    .iter()
                    .map(|&(c, coef)| {
                        if coef == 1.0 {
                            format!("g{c}")
                        } else {
                            format!("{} * g{c}", lit(coef))
                        }
                    })
                    .collect();
                fl += nab * (2 * st.len() - 1);
                let _ = writeln!(
                    w,
                    "                {{ const double v = {};",
                    sum.join(" + ")
                );
                store(w, format!("mn * ld + {k}"), "v");
                let _ = writeln!(w, "                }}");
            }
            let _ = writeln!(w, "            }}");
        }
    }
    fl += nab * nk; // the accumulation
    let _ = writeln!(w, "            first = false;");
    let _ = writeln!(w, "        }}");
    let _ = writeln!(w, "    }}");
    let _ = writeln!(w, "}}");
    let lines = s.lines().count();
    (s, KernelStats { flops: fl, lines })
}

/// Kernel inputs of a [`ClassWork`]. Pairs: primitive offsets `pr_p0` (pairs + 1), exponent
/// sums `pr_p`, centres `pr_pc` (3 per primitive), Hermite matrices `pr_et` (nab x nherm per
/// primitive, \[mn\]\[t'\]), output row of mn = 0 `pr_row`. Auxiliary shells: primitive offsets
/// `ax_p0`, exponents, coefficients c (2a)^-l (-1)^l, centres (3 per shell), first column.
#[derive(Clone, Debug, Default)]
pub struct Flat {
    pub pr_p0: Vec<i32>,
    pub pr_p: Vec<f64>,
    pub pr_pc: Vec<f64>,
    pub pr_et: Vec<f64>,
    pub pr_row: Vec<i64>,
    pub ax_p0: Vec<i32>,
    pub ax_exp: Vec<f64>,
    pub ax_coef: Vec<f64>,
    pub ax_c: Vec<f64>,
    pub ax_col: Vec<i32>,
}

/// The work of one class in a calculation: orbital shell pairs and the auxiliary shells of
/// l_k (renumbered from 0).
pub struct ClassWork {
    pub class: Class,
    pub pairs: Vec<Pair3c>,
    pub aux: BasisSet,
}

impl ClassWork {
    /// Every pair of shells (a, b) with l_a = la, l_b = lb (a >= b when la == lb) that keeps
    /// at least one primitive pair, and every auxiliary shell with l = lk.
    pub fn new(basis: &BasisSet, aux: &BasisSet, class: Class) -> ClassWork {
        let sa: Vec<usize> = (0..basis.shells.len())
            .filter(|&i| basis.shells[i].l == class.la)
            .collect();
        let sb: Vec<usize> = (0..basis.shells.len())
            .filter(|&i| basis.shells[i].l == class.lb)
            .collect();
        let mut pairs = Vec::new();
        for &a in &sa {
            for &b in &sb {
                if class.la == class.lb && b > a {
                    continue;
                }
                let sp = build_pair(
                    &ShellView::from_shell(&basis.shells[a]),
                    &ShellView::from_shell(&basis.shells[b]),
                    a,
                    b,
                );
                if sp.nprim > 0 {
                    pairs.push(Pair3c::from_shell_pair(&sp));
                }
            }
        }
        let mut sub = aux.clone();
        sub.shells.retain(|s| s.l == class.lk);
        let mut ao = 0;
        for s in &mut sub.shells {
            s.ao_start = ao;
            ao += s.nfunc();
        }
        sub.nbf = ao;
        sub.atom_shells.clear();
        ClassWork {
            class,
            pairs,
            aux: sub,
        }
    }

    /// Output rows (pairs x nab) and columns (auxiliary functions of the class).
    pub fn shape(&self) -> (usize, usize) {
        (self.pairs.len() * self.class.nab(), self.aux.nbf)
    }

    /// Primitive triples (pair primitive x auxiliary primitive).
    pub fn prim_triples(&self) -> usize {
        let np: usize = self.pairs.iter().map(|p| p.nprim()).sum();
        let na: usize = self.aux.shells.iter().map(|s| s.nprim()).sum();
        np * na
    }

    /// The kernel inputs as flat arrays (see [`Flat`]).
    pub fn flatten(&self) -> Flat {
        let nab = self.class.nab();
        let mut f = Flat {
            pr_p0: vec![0],
            ax_p0: vec![0],
            ..Default::default()
        };
        for (i, pr) in self.pairs.iter().enumerate() {
            assert_eq!(pr.nab(), nab);
            f.pr_p.extend_from_slice(&pr.p);
            for c in &pr.pc {
                f.pr_pc.extend_from_slice(c);
            }
            f.pr_et.extend_from_slice(&pr.et);
            f.pr_p0.push(f.pr_p.len() as i32);
            f.pr_row.push((i * nab) as i64);
        }
        let l = self.class.lk as i32;
        for s in &self.aux.shells {
            for (&a, &c) in s.exps.iter().zip(&s.coefs) {
                f.ax_exp.push(a);
                f.ax_coef
                    .push(c * (2.0 * a).powi(-l) * if l % 2 == 0 { 1.0 } else { -1.0 });
            }
            f.ax_p0.push(f.ax_exp.len() as i32);
            f.ax_c.extend_from_slice(&s.center);
            f.ax_col.push(s.ao_start as i32);
        }
        f
    }

    /// The CPU result (linus-ints' `pair_block`, no screening), in the kernel's layout, using
    /// the current rayon pool.
    pub fn cpu(&self) -> Vec<f64> {
        use rayon::prelude::*;
        let aux3 = Aux3c::new(&self.aux);
        let chunks = aux3.chunks(0..self.aux.shells.len());
        let (rows, ld) = self.shape();
        let nab = self.class.nab();
        let mut out = vec![0.0; rows * ld];
        out.par_chunks_mut(nab * ld.max(1))
            .zip(&self.pairs)
            .for_each_init(crate::cpu::Work3c::new, |w, (blk, pair)| {
                crate::cpu::pair_block(
                    pair,
                    &chunks,
                    blk,
                    ld,
                    0.0,
                    &linus_ints::operator::Operator::Coulomb,
                    w,
                );
            });
        out
    }
}

#[cfg(feature = "gpu")]
pub use device::GpuEri3c;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::error::GpuError;
    use crate::kernels::boys::boys_table;
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};

    /// One class's data on the device, ready to launch.
    pub struct GpuEri3c {
        pub class: Class,
        pub variant: Variant,
        npair: usize,
        nauxsh: usize,
        rows: usize,
        ld: usize,
        kernel: Kernel,
        pr_p0: DeviceBuf<i32>,
        pr_p: DeviceBuf<f64>,
        pr_pc: DeviceBuf<f64>,
        pr_et: DeviceBuf<f64>,
        pr_row: DeviceBuf<i64>,
        ax_p0: DeviceBuf<i32>,
        ax_exp: DeviceBuf<f64>,
        ax_coef: DeviceBuf<f64>,
        ax_c: DeviceBuf<f64>,
        ax_col: DeviceBuf<i32>,
        btab: DeviceBuf<f64>,
    }

    impl GpuEri3c {
        pub fn new(gpu: &Gpu, work: &ClassWork, variant: Variant) -> Result<Self, GpuError> {
            let class = work.class;
            let (src, _) = generate(class, variant);
            let kernel = gpu.kernel(
                &kernel_name(class, variant),
                &src,
                &kernel_name(class, variant),
            )?;
            let f = work.flatten();
            let (rows, ld) = work.shape();
            Ok(GpuEri3c {
                class,
                variant,
                npair: work.pairs.len(),
                nauxsh: work.aux.shells.len(),
                rows,
                ld,
                kernel,
                pr_p0: gpu.upload(&f.pr_p0)?,
                pr_p: gpu.upload(&f.pr_p)?,
                pr_pc: gpu.upload(&f.pr_pc)?,
                pr_et: gpu.upload(&f.pr_et)?,
                pr_row: gpu.upload(&f.pr_row)?,
                ax_p0: gpu.upload(&f.ax_p0)?,
                ax_exp: gpu.upload(&f.ax_exp)?,
                ax_coef: gpu.upload(&f.ax_coef)?,
                ax_c: gpu.upload(&f.ax_c)?,
                ax_col: gpu.upload(&f.ax_col)?,
                btab: gpu.upload(&boys_table())?,
            })
        }

        /// Elements of the output (rows x ld).
        pub fn out_len(&self) -> usize {
            self.rows * self.ld
        }

        /// Computes every (k|mn) of the class into `out` (overwritten).
        pub fn run(&self, gpu: &Gpu, out: &mut DeviceBuf<f64>) -> Result<(), GpuError> {
            assert!(out.len() >= self.out_len());
            let n = self.npair * self.nauxsh;
            let args = [
                Arg::I64(self.npair as i64),
                Arg::I32(self.nauxsh as i32),
                Arg::buf(&self.pr_p0),
                Arg::buf(&self.pr_p),
                Arg::buf(&self.pr_pc),
                Arg::buf(&self.pr_et),
                Arg::buf(&self.pr_row),
                Arg::buf(&self.ax_p0),
                Arg::buf(&self.ax_exp),
                Arg::buf(&self.ax_coef),
                Arg::buf(&self.ax_c),
                Arg::buf(&self.ax_col),
                Arg::buf(&self.btab),
                Arg::I64(self.ld as i64),
                Arg::buf_mut(out),
            ];
            // SAFETY: the arguments match the generated signature; each thread writes the
            // nab x nk block of its (pair, auxiliary shell) inside rows x ld (asserted above).
            unsafe { gpu.launch(&self.kernel, Launch::linear(n, 128), &args) }
        }

        pub fn kernel(&self) -> &Kernel {
            &self.kernel
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_dd_d_both_variants() {
        for v in [Variant::H, Variant::G] {
            let c = Class {
                la: 2,
                lb: 2,
                lk: 2,
            };
            let (src, st) = generate(c, v);
            assert!(src.contains(&kernel_name(c, v)));
            assert!(st.flops > 8000, "{v:?} {}", st.flops);
        }
    }
}
