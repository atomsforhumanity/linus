//! Collocation: basis functions and their Cartesian derivatives on grid points.
//!
//! Each output (function f, derivative component d) of a shell is a polynomial in the
//! displacement from the centre times radial sums, phi_fd = sum_k P_fdk(x, y, z) R_k(r^2) with
//! R_k = sum_p c_p (-2 a_p)^k exp(-a_p r^2) (as linus-ints' `colloc`). The generator writes one
//! kernel per (l, spherical or Cartesian) with the polynomials as straight-line code, spherical
//! transform and derivative factors folded into the constants. One thread computes every
//! output of one shell at one point; a launch covers a list of shells of one l (grid y) and
//! a block of points (grid x), so writes are coalesced along the points.
//!
//! Output: function f of a launched shell, component d (`ao::deriv_indices` order: value,
//! then x, y, z), point g at `out[(ao + f) * sf + d * sc + g]`, with the shell's first
//! function `ao` from the launch's list and the strides `sf`, `sc` kernel arguments:
//! - component-major as linus-ints' `ao::eval_ao` (`out[(d * nao + ao) * ld + g]`):
//!   sf = ld, sc = nao ld (`GpuCollocator`);
//! - function-major as linus-ints' `Collocator` (`out[(ao * ncomp + d) * ld + g]`):
//!   sf = ncomp ld, sc = ld (the XC engine, with block-local function numbers).

use super::{lit, KernelStats};
use crate::cpu::{cart2sph, cart_components, deriv_indices, ncart, BasisSet};
use std::fmt::Write;

/// Highest angular momentum and derivative order the generator supports.
pub const MAX_L: usize = 6;
pub const MAX_ORDER: usize = 3;

/// Terms of d^p/dx^p [x^l exp(-a x^2)] = sum coef x^m (-2a)^k exp(-a x^2), as (coef, m, k).
fn deriv_terms(l: usize, p: usize) -> Vec<(f64, usize, usize)> {
    let mut t: Vec<(f64, i64, usize)> = vec![(1.0, l as i64, 0)];
    for _ in 0..p {
        let mut n: Vec<(f64, i64, usize)> = Vec::new();
        for &(c, m, k) in &t {
            if m > 0 {
                n.push((c * m as f64, m - 1, k));
            }
            n.push((c, m + 1, k + 1));
        }
        n.sort_by_key(|&(_, m, k)| (m, k));
        let mut merged: Vec<(f64, i64, usize)> = Vec::new();
        for x in n {
            match merged.last_mut() {
                Some(last) if last.1 == x.1 && last.2 == x.2 => last.0 += x.0,
                _ => merged.push(x),
            }
        }
        t = merged;
    }
    t.into_iter().map(|(c, m, k)| (c, m as usize, k)).collect()
}

/// Polynomials of one (l, pure, order): for output o = d * nf + f, the terms
/// (coefficient, monomial exponents, radial order k), equal (monomial, k) merged.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub l: usize,
    pub pure: bool,
    pub order: usize,
    pub nf: usize,
    pub ncomp: usize,
    pub terms: Vec<Vec<(f64, [usize; 3], usize)>>,
}

impl Recipe {
    pub fn new(l: usize, pure: bool, order: usize) -> Recipe {
        assert!(l <= MAX_L && order <= MAX_ORDER);
        let carts = cart_components(l);
        let ncar = ncart(l);
        let t: Vec<f64> = if pure {
            cart2sph(l)
        } else {
            let mut id = vec![0.0; ncar * ncar];
            for i in 0..ncar {
                id[i * ncar + i] = 1.0;
            }
            id
        };
        let nf = if pure { 2 * l + 1 } else { ncar };
        let comps = deriv_indices(order);
        let mut terms = Vec::with_capacity(comps.len() * nf);
        for d in &comps {
            for f in 0..nf {
                let mut acc: Vec<(f64, [usize; 3], usize)> = Vec::new();
                for (ic, c) in carts.iter().enumerate() {
                    let tv = t[f * ncar + ic];
                    if tv == 0.0 {
                        continue;
                    }
                    for &(cx, mx, kx) in &deriv_terms(c[0], d[0]) {
                        for &(cy, my, ky) in &deriv_terms(c[1], d[1]) {
                            for &(cz, mz, kz) in &deriv_terms(c[2], d[2]) {
                                let (m, k, v) = ([mx, my, mz], kx + ky + kz, tv * cx * cy * cz);
                                match acc.iter_mut().find(|e| e.1 == m && e.2 == k) {
                                    Some(e) => e.0 += v,
                                    None => acc.push((v, m, k)),
                                }
                            }
                        }
                    }
                }
                acc.retain(|e| e.0 != 0.0);
                terms.push(acc);
            }
        }
        Recipe {
            l,
            pure,
            order,
            nf,
            ncomp: comps.len(),
            terms,
        }
    }

    /// Largest power of x, y or z in any term.
    pub fn max_power(&self) -> usize {
        self.terms
            .iter()
            .flatten()
            .map(|t| t.1.iter().copied().max().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }
}

/// Kernel name for (l, pure, order).
pub fn kernel_name(l: usize, pure: bool, order: usize) -> String {
    format!("colloc_l{l}{}_d{order}", if pure { "s" } else { "c" })
}

/// CUDA C++ of the kernel for (l, pure, order), and its source statistics (flops per shell and
/// point, excluding the exponentials: each primitive adds one exp and 2 + 2 order flops).
#[allow(clippy::needless_range_loop)]
pub fn generate(l: usize, pure: bool, order: usize) -> (String, KernelStats) {
    let r = Recipe::new(l, pure, order);
    let name = kernel_name(l, pure, order);
    let mut s = String::new();
    let mut flops = 0;
    let _ = writeln!(
        s,
        "// linus-gpu collocation: l = {l}, {}, derivatives to order {order} (generated)",
        if pure { "spherical" } else { "Cartesian" }
    );
    let _ = writeln!(
        s,
        "extern \"C\" __global__ void __launch_bounds__(128) {name}(const double *__restrict__ gx, const double *__restrict__ gy, const double *__restrict__ gz, int npts, long long sf, long long sc,
    const int *__restrict__ sh_list, const int *__restrict__ sh_ao, const int *__restrict__ sh_p0, const double *__restrict__ sh_c, const double *__restrict__ pexp, const double *__restrict__ pcoef, double *__restrict__ out)
{{
    const int s = sh_list[blockIdx.y];
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= npts) return;
    const double x = gx[g] - sh_c[3 * s], y = gy[g] - sh_c[3 * s + 1], z = gz[g] - sh_c[3 * s + 2];
    const double rr = x * x + y * y + z * z;"
    );
    flops += 8;
    let nk = order + 1;
    let rk: Vec<String> = (0..nk).map(|k| format!("r{k}")).collect();
    let _ = writeln!(
        s,
        "    double {} ;",
        rk.iter()
            .map(|v| format!("{v} = 0.0"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(s, "    for (int p = sh_p0[s]; p < sh_p0[s + 1]; ++p) {{");
    let _ = writeln!(s, "        const double a = pexp[p];");
    let _ = writeln!(s, "        double e = pcoef[p] * exp(-a * rr);");
    if order > 0 {
        let _ = writeln!(s, "        const double m2a = -2.0 * a;");
    }
    for k in 0..nk {
        let _ = writeln!(s, "        r{k} += e;");
        if k + 1 < nk {
            let _ = writeln!(s, "        e *= m2a;");
        }
    }
    let _ = writeln!(s, "    }}");
    // powers
    let pmax = r.max_power();
    for (v, name) in [("x", "x"), ("y", "y"), ("z", "z")] {
        for m in 2..=pmax {
            let prev = if m == 2 {
                v.to_string()
            } else {
                format!("{name}{}", m - 1)
            };
            let _ = writeln!(s, "    const double {name}{m} = {prev} * {v};");
            flops += 1;
        }
    }
    let pw = |v: &str, m: usize| -> Option<String> {
        match m {
            0 => None,
            1 => Some(v.to_string()),
            _ => Some(format!("{v}{m}")),
        }
    };
    let _ = writeln!(
        s,
        "    double *o = out + (long long)sh_ao[blockIdx.y] * sf + g;"
    );
    for d in 0..r.ncomp {
        for f in 0..r.nf {
            let terms = &r.terms[d * r.nf + f];
            let mut parts = Vec::new();
            for k in 0..nk {
                let mut poly = Vec::new();
                for &(c, m, kk) in terms {
                    if kk != k {
                        continue;
                    }
                    let mut fac: Vec<String> = [("x", m[0]), ("y", m[1]), ("z", m[2])]
                        .iter()
                        .filter_map(|&(v, e)| pw(v, e))
                        .collect();
                    if c != 1.0 || fac.is_empty() {
                        fac.insert(0, lit(c));
                    }
                    flops += fac.len(); // fac.len() - 1 multiplies and one add
                    poly.push(fac.join(" * "));
                }
                if !poly.is_empty() {
                    parts.push(format!("({}) * {}", poly.join(" + "), rk[k]));
                    flops += 1;
                }
            }
            let rhs = if parts.is_empty() {
                "0.0".to_string()
            } else {
                parts.join(" + ")
            };
            let _ = writeln!(s, "    o[{d} * sc + {f} * sf] = {rhs};");
        }
    }
    let _ = writeln!(s, "}}");
    let lines = s.lines().count();
    (s, KernelStats { flops, lines })
}

/// One module with the kernels of every (l, pure) in `basis`, for derivative order `order`.
pub fn module_source(basis: &BasisSet, order: usize) -> String {
    let mut keys: Vec<(usize, bool)> = basis.shells.iter().map(|s| (s.l, s.pure)).collect();
    keys.sort();
    keys.dedup();
    module_source_for(&keys, order)
}

/// One module with the kernels of the (l, pure) `keys` (sorted, distinct), for derivative order
/// `order`.
pub fn module_source_for(keys: &[(usize, bool)], order: usize) -> String {
    keys.iter()
        .map(|&(l, p)| generate(l, p, order).0)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Shells of one (l, pure) in the layout the kernels read.
#[derive(Clone, Debug, Default)]
pub struct ShellGroup {
    pub l: usize,
    pub pure: bool,
    pub ao: Vec<i32>,
    /// primitive offsets, len = shells + 1
    pub p0: Vec<i32>,
    pub center: Vec<f64>,
    pub exp: Vec<f64>,
    pub coef: Vec<f64>,
}

/// The basis split into [`ShellGroup`]s.
pub fn groups(basis: &BasisSet) -> Vec<ShellGroup> {
    let mut out: Vec<ShellGroup> = Vec::new();
    for sh in &basis.shells {
        let g = match out
            .iter_mut()
            .position(|g| g.l == sh.l && g.pure == sh.pure)
        {
            Some(i) => &mut out[i],
            None => {
                out.push(ShellGroup {
                    l: sh.l,
                    pure: sh.pure,
                    p0: vec![0],
                    ..Default::default()
                });
                out.last_mut().unwrap()
            }
        };
        g.ao.push(sh.ao_start as i32);
        g.center.extend_from_slice(&sh.center);
        g.exp.extend_from_slice(&sh.exps);
        g.coef.extend_from_slice(&sh.coefs);
        g.p0.push(g.exp.len() as i32);
    }
    out.sort_by_key(|g| (g.l, g.pure));
    out
}

#[cfg(feature = "gpu")]
pub use device::{GpuCollocator, ShellTables};

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::error::GpuError;
    use crate::{Arg, DevPtr, DeviceBuf, Gpu, Kernel, Launch};

    /// Every shell of a basis on the device, in basis order: primitive offsets (shells + 1),
    /// centres (3 per shell), exponents and contraction coefficients (per primitive). The
    /// kernels index them with the global shell numbers of a launch's shell list.
    pub struct ShellTables {
        pub p0: DeviceBuf<i32>,
        pub center: DeviceBuf<f64>,
        pub exp: DeviceBuf<f64>,
        pub coef: DeviceBuf<f64>,
    }

    impl ShellTables {
        pub fn new(gpu: &Gpu, basis: &BasisSet) -> Result<Self, GpuError> {
            let mut p0 = vec![0i32];
            let (mut center, mut exp, mut coef) = (vec![], vec![], vec![]);
            for sh in &basis.shells {
                center.extend_from_slice(&sh.center);
                exp.extend_from_slice(&sh.exps);
                coef.extend_from_slice(&sh.coefs);
                p0.push(exp.len() as i32);
            }
            Ok(ShellTables {
                p0: gpu.upload(&p0)?,
                center: gpu.upload(&center)?,
                exp: gpu.upload(&exp)?,
                coef: gpu.upload(&coef)?,
            })
        }

        /// Queues `kernel` (a collocation kernel of [`generate`]) for the `nsh` shells whose
        /// global numbers are at `list` and first output functions at `ao` (device addresses
        /// of i32 arrays), at the points x, y, z (device addresses of `npts` values each),
        /// writing function f, component d, point g of list entry s at
        /// out\[(ao\[s\] + f) sf + d sc + g\].
        ///
        /// # Safety
        /// The addresses must hold what is described; `kernel` must be the kernel of the
        /// shells' (l, pure) with the output's derivative order; every write must lie inside
        /// the output buffer.
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn launch(
            &self,
            gpu: &Gpu,
            kernel: &Kernel,
            nsh: usize,
            list: DevPtr<i32>,
            ao: DevPtr<i32>,
            xyz: [DevPtr<f64>; 3],
            npts: usize,
            sf: usize,
            sc: usize,
            out: DevPtr<f64>,
        ) -> Result<(), GpuError> {
            if nsh == 0 || npts == 0 {
                return Ok(());
            }
            let cfg = Launch {
                grid: (npts.div_ceil(128) as u32, nsh as u32, 1),
                block: (128, 1, 1),
                shared_bytes: 0,
            };
            let args = [
                xyz[0].into(),
                xyz[1].into(),
                xyz[2].into(),
                Arg::I32(npts as i32),
                Arg::I64(sf as i64),
                Arg::I64(sc as i64),
                list.into(),
                ao.into(),
                Arg::buf(&self.p0),
                Arg::buf(&self.center),
                Arg::buf(&self.exp),
                Arg::buf(&self.coef),
                out.into(),
            ];
            // SAFETY: the caller's contract; the arguments match the generated signature.
            unsafe { gpu.launch(kernel, cfg, &args) }
        }
    }

    struct DevGroup {
        nsh: usize,
        kernel: Kernel,
        list: DeviceBuf<i32>,
        ao: DeviceBuf<i32>,
    }

    /// Collocation of one basis on the device (derivatives up to `order`), every shell at
    /// every point, component-major output.
    pub struct GpuCollocator {
        pub order: usize,
        pub nao: usize,
        tables: ShellTables,
        groups: Vec<DevGroup>,
    }

    impl GpuCollocator {
        pub fn new(gpu: &Gpu, basis: &BasisSet, order: usize) -> Result<Self, GpuError> {
            let src = module_source(basis, order);
            let mut keys: Vec<(usize, bool)> = basis.shells.iter().map(|s| (s.l, s.pure)).collect();
            keys.sort();
            keys.dedup();
            let mut out = Vec::new();
            for (l, pure) in keys {
                let sel: Vec<usize> = (0..basis.shells.len())
                    .filter(|&i| basis.shells[i].l == l && basis.shells[i].pure == pure)
                    .collect();
                let list: Vec<i32> = sel.iter().map(|&i| i as i32).collect();
                let ao: Vec<i32> = sel
                    .iter()
                    .map(|&i| basis.shells[i].ao_start as i32)
                    .collect();
                out.push(DevGroup {
                    nsh: sel.len(),
                    kernel: gpu.kernel("colloc", &src, &kernel_name(l, pure, order))?,
                    list: gpu.upload(&list)?,
                    ao: gpu.upload(&ao)?,
                });
            }
            Ok(GpuCollocator {
                order,
                nao: basis.nbf,
                tables: ShellTables::new(gpu, basis)?,
                groups: out,
            })
        }

        /// Values and derivatives at points (x, y, z)[0..npts] into
        /// out[(d * nao + ao) * ld + point] (out needs ncomp * nao * ld elements).
        #[allow(clippy::too_many_arguments)]
        pub fn eval(
            &self,
            gpu: &Gpu,
            x: &DeviceBuf<f64>,
            y: &DeviceBuf<f64>,
            z: &DeviceBuf<f64>,
            npts: usize,
            ld: usize,
            out: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let nc = crate::cpu::ncomp(self.order);
            assert!(
                ld >= npts
                    && out.len() >= nc * self.nao * ld
                    && x.len() >= npts
                    && y.len() >= npts
                    && z.len() >= npts
            );
            let xyz = [x.ptr(), y.ptr(), z.ptr()];
            for g in &self.groups {
                // SAFETY: the group's kernel, shell numbers and first functions; it reads npts
                // points and writes rows ao..ao+nf of each component, within out (asserted
                // above).
                unsafe {
                    self.tables.launch(
                        gpu,
                        &g.kernel,
                        g.nsh,
                        g.list.ptr(),
                        g.ao.ptr(),
                        xyz,
                        npts,
                        ld,
                        self.nao * ld,
                        out.ptr(),
                    )
                }?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_reproduces_eval_ao_at_a_point() {
        // evaluate the polynomials on the host and compare with linus-ints' eval_ao
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2","0.4"],"coefficients":[["0.6","0.5"]]},
          {"function_type":"gto","angular_momentum":[4],"exponents":["0.9"],"coefficients":[["1.0"]]}]}}}"#;
        let def = crate::cpu::BasisDef::from_bse_json(json).unwrap();
        let b = BasisSet::new(&def, &[8], &[[0.1, -0.2, 0.3]], true).unwrap();
        let (px, py, pz) = (0.7, -0.4, 1.1);
        // every order the generator supports (the XC Hessian's third derivatives included)
        for order in 0..=MAX_ORDER {
            let nc = crate::cpu::ncomp(order);
            let mut out = vec![0.0; nc * b.nbf];
            crate::cpu::eval_ao(&b, &[0, 1], &[px], &[py], &[pz], order, &mut out);
            for sh in &b.shells {
                let r = Recipe::new(sh.l, sh.pure, order);
                let (x, y, z) = (px - sh.center[0], py - sh.center[1], pz - sh.center[2]);
                let rr = x * x + y * y + z * z;
                let mut rk = [0.0; MAX_ORDER + 1];
                for (&a, &c) in sh.exps.iter().zip(&sh.coefs) {
                    let mut e = c * (-a * rr).exp();
                    for v in rk.iter_mut() {
                        *v += e;
                        e *= -2.0 * a;
                    }
                }
                for d in 0..nc {
                    for f in 0..r.nf {
                        let v: f64 = r.terms[d * r.nf + f]
                            .iter()
                            .map(|&(c, m, k)| {
                                c * x.powi(m[0] as i32)
                                    * y.powi(m[1] as i32)
                                    * z.powi(m[2] as i32)
                                    * rk[k]
                            })
                            .sum();
                        let refv = out[d * b.nbf + sh.ao_start + f];
                        assert!(
                            (v - refv).abs() < 1e-13 * (1.0 + refv.abs()),
                            "order {order} l {} d {d} f {f}: {v} vs {refv}",
                            sh.l
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn generates_every_class() {
        for l in 0..=4 {
            for order in 0..=MAX_ORDER {
                let (src, st) = generate(l, l >= 2, order);
                assert!(src.contains(&kernel_name(l, l >= 2, order)));
                assert!(st.flops > 0);
            }
        }
    }
}
