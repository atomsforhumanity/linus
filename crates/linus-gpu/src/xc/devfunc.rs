//! The functional on the device (feature `gpu`): libxc's own Maple-generated code of a
//! functional's leaves (linus-xc's [`DevicePlan`]) compiled by NVRTC into one kernel that does per
//! point what libxc's evaluation does on the host, so the XC engines' grid densities never leave
//! the device.
//!
//! The kernel repeats libxc 7.0.0's evaluation step for step: per leaf the input thresholds of
//! `work_{lda,gga,mgga}_inc.c` (densities below the leaf's threshold give zero, the others are
//! clamped, sigma bounded by the spin densities' and, where the leaf asks, by 8 rho tau) and its
//! Maple function into zeroed outputs; per libxc mixture the node's own term, then each auxiliary
//! node's outputs times its coefficient added in order (`mix_func.c`); and per linus
//! [`linus_xc::Functional`] the first component (times its weight unless 1), then weight x each
//! further component. The module compiles without floating-point contraction, as linus's libxc
//! build does, so the results differ from the host's only where the device's math library (exp,
//! log, pow, cbrt, atan, erf) rounds differently from glibc's: within a few units in the last
//! place per point. The generated code holds every leaf's parameters, thresholds and dimensions
//! as exact literals.
//!
//! [`DevFunc::new`] compiles the module and checks it against libxc on a set of points spread
//! over the densities, gradients, kinetic energy densities and spin polarizations a molecule has;
//! a functional whose device results disagree (a leaf whose libxc state the plan does not carry)
//! is refused with the reason, and the engines evaluate it with libxc on the host.

use crate::cpu::{DeviceLeaf, DeviceNode, DevicePlan, Family};
use crate::error::GpuError;
use crate::xc::functional::HostFunctional;
use crate::xc::XcKind;
use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
use std::fmt::Write;

const PRELUDE: &str = include_str!("../../device/xc/devfunc.cuh");

/// A C++ literal of `x`'s exact value (hexadecimal floating point), or None for an infinity or
/// NaN.
fn hexf(x: f64) -> Option<String> {
    if !x.is_finite() {
        return None;
    }
    if x == 0.0 {
        return Some(if x.is_sign_negative() { "-0.0" } else { "0.0" }.into());
    }
    let bits = x.to_bits();
    let sign = if bits >> 63 == 1 { "-" } else { "" };
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let man = bits & ((1u64 << 52) - 1);
    Some(if exp == 0 {
        format!("{sign}0x0.{man:013x}p-1022")
    } else {
        format!("{sign}0x1.{man:013x}p{}", exp - 1023)
    })
}

/// The generator's state: the leaves' namespaces (numbered in plan order) and the kernel body.
struct Gen {
    nsp: usize,
    leaves: String,
    body: String,
    nleaf: usize,
    ntmp: usize,
}

/// The output arrays of an accumulator named `p` (per point, in libxc's layout).
fn decl_outputs(p: &str) -> String {
    format!(
        "double {p}_zk[1] = {{0.0}}; double {p}_vrho[2] = {{0.0, 0.0}}; \
         double {p}_vsigma[3] = {{0.0, 0.0, 0.0}}; double {p}_vlapl[2] = {{0.0, 0.0}}; \
         double {p}_vtau[2] = {{0.0, 0.0}};\n"
    )
}

/// The outputs (the energy per particle and the first derivatives) and their largest
/// dimensions (polarized).
const FIRST: [(&str, usize); 5] = [
    ("zk", 1),
    ("vrho", 2),
    ("vsigma", 3),
    ("vlapl", 2),
    ("vtau", 2),
];

/// `dst += coef * src` for every output (`coef` a C++ literal).
fn add_outputs(dst: &str, src: &str, coef: &str) -> String {
    let mut s = String::new();
    for (n, d) in FIRST {
        for i in 0..d {
            let _ = writeln!(s, "{dst}_{n}[{i}] += {coef} * {src}_{n}[{i}];");
        }
    }
    s
}

impl Gen {
    /// A leaf's namespace (its structs, parameters and Maple code) and the code that adds its
    /// thresholded evaluation to the zeroed accumulator `acc`.
    fn leaf(&mut self, l: &DeviceLeaf, acc: &str) -> Result<String, String> {
        let i = self.nleaf;
        self.nleaf += 1;
        let h = |x: f64| hexf(x).ok_or_else(|| format!("libxc functional {}: {x}", l.number));
        let ns = format!("xcl{i}");
        let _ = writeln!(self.leaves, "namespace {ns} {{\n{}", l.source.typedefs);
        if !l.params.is_empty() {
            let vals: Result<Vec<String>, String> = l.params.iter().map(|&x| h(x)).collect();
            let _ = writeln!(
                self.leaves,
                "__device__ const double xcl_params[{}] = {{{}}};",
                l.params.len(),
                vals?.join(", ")
            );
        }
        let _ = writeln!(self.leaves, "{}\n}} // namespace {ns}", l.source.maple);
        let nsp = self.nsp;
        if l.nspin as usize != nsp {
            return Err(format!("libxc functional {}: nspin {}", l.number, l.nspin));
        }
        let (gga, mgga) = (l.family >= Family::Gga, l.family == Family::Mgga);
        let fam = match l.family {
            Family::Lda => "lda",
            Family::Gga => "gga",
            _ => "mgga",
        };
        let d = &l.dims;
        let mut s = String::new();
        let _ = writeln!(s, "{{ // leaf {i}: libxc {} ({})", l.number, l.source.file);
        let _ = writeln!(
            s,
            "const xc_func_info_type info = {{{}, {}}};\nxc_func_type p;\np.info = &info; \
             p.nspin = {};\np.cam_omega = {}; p.cam_alpha = {}; p.cam_beta = {};",
            l.number,
            l.flags,
            l.nspin,
            h(l.cam_omega)?,
            h(l.cam_alpha)?,
            h(l.cam_beta)?
        );
        let names = [
            "rho",
            "sigma",
            "lapl",
            "tau",
            "zk",
            "vrho",
            "vsigma",
            "vlapl",
            "vtau",
            "v2rho2",
            "v2rhosigma",
            "v2rholapl",
            "v2rhotau",
            "v2sigma2",
            "v2sigmalapl",
            "v2sigmatau",
            "v2lapl2",
            "v2lapltau",
            "v2tau2",
        ];
        for (n, v) in names.iter().zip(d) {
            let _ = write!(s, "p.dim.{n} = {v}; ");
        }
        let _ = writeln!(
            s,
            "\np.params = {};\np.dens_threshold = {}; p.zeta_threshold = {}; \
             p.sigma_threshold = {}; p.tau_threshold = {};",
            if l.params.is_empty() {
                "0".to_string()
            } else {
                format!("{ns}::xcl_params")
            },
            h(l.dens_threshold)?,
            h(l.zeta_threshold)?,
            h(l.sigma_threshold)?,
            h(l.tau_threshold)?
        );
        let _ = writeln!(s, "xc_{fam}_out_params out = {{}};");
        let mut outs = vec!["zk", "vrho"];
        if gga {
            outs.push("vsigma");
        }
        if mgga {
            outs.extend(["vlapl", "vtau"]);
        }
        for o in &outs {
            let _ = write!(s, "out.{o} = {acc}_{o}; ");
        }
        // work_{lda,gga,mgga}_inc.c
        let _ = writeln!(
            s,
            "\nconst double dens = {};\nif (!(dens < p.dens_threshold)) {{\n\
             double my_rho[2] = {{0.0, 0.0}}, my_sigma[3] = {{0.0, 0.0, 0.0}}, \
             my_tau[2] = {{0.0, 0.0}};\n\
             my_rho[0] = m_max(p.dens_threshold, in_rho[0]);",
            if nsp == 2 {
                "in_rho[0] + in_rho[1]"
            } else {
                "in_rho[0]"
            }
        );
        if gga {
            s += "my_sigma[0] = m_max(p.sigma_threshold * p.sigma_threshold, in_sigma[0]);\n";
        }
        let tau_checks = |s: &mut String, k: usize, sg: usize| {
            let _ = writeln!(
                s,
                "if (p.info->flags & XC_FLAGS_NEEDS_TAU) {{ my_tau[{k}] = m_max(p.tau_threshold, \
                 in_tau[{k}]); if (p.info->flags & XC_FLAGS_ENFORCE_FHC) my_sigma[{sg}] = \
                 m_min(my_sigma[{sg}], 8.0 * my_rho[{k}] * my_tau[{k}]); }}"
            );
        };
        if mgga {
            tau_checks(&mut s, 0, 0);
        }
        if nsp == 2 {
            s += "my_rho[1] = m_max(p.dens_threshold, in_rho[1]);\n";
            if gga {
                s += "my_sigma[2] = m_max(p.sigma_threshold * p.sigma_threshold, in_sigma[2]);\n";
            }
            if mgga {
                tau_checks(&mut s, 1, 2);
            }
            if gga {
                s += "{ my_sigma[1] = in_sigma[1]; const double s_ave = 0.5 * (my_sigma[0] + \
                      my_sigma[2]); my_sigma[1] = (my_sigma[1] >= -s_ave ? my_sigma[1] : -s_ave); \
                      my_sigma[1] = (my_sigma[1] <= +s_ave ? my_sigma[1] : +s_ave); }\n";
            }
        }
        let spin = if nsp == 2 { "pol" } else { "unpol" };
        let args = match l.family {
            Family::Lda => "my_rho",
            Family::Gga => "my_rho, my_sigma",
            _ => "my_rho, my_sigma, in_lapl, my_tau",
        };
        let _ = writeln!(s, "{ns}::func_vxc_{spin}(&p, 0, {args}, &out);\n}}\n}}");
        Ok(s)
    }

    /// The code that evaluates node `n` into the zeroed accumulator `acc`.
    fn node(&mut self, n: &DeviceNode, acc: &str) -> Result<String, String> {
        let mut s = String::new();
        if let Some(l) = &n.own {
            s += &self.leaf(l, acc)?;
        }
        for (coef, c) in &n.mix {
            let t = format!("t{}", self.ntmp);
            self.ntmp += 1;
            let _ = write!(s, "{{\n{}", decl_outputs(&t));
            s += &self.node(c, &t)?;
            s += &add_outputs(acc, &t, &hexf(*coef).ok_or("a mixing coefficient")?);
            s += "}\n";
        }
        Ok(s)
    }
}

/// The kernel's name: the energy per particle and the first derivatives.
pub(crate) const ENTRY: &str = "xc_devfunc_vxc";

/// The module's source for `plan` (the energy per particle and the first derivatives: the
/// engines' V_xc and gradient) and the engine's ingredients `kind`: one thread per point, inputs
/// and outputs in libxc's layout (zk, vrho, vsigma, vtau; the arrays of ingredients the engine
/// does not have are not touched).
pub(crate) fn source(plan: &DevicePlan, kind: XcKind) -> Result<String, String> {
    let nsp = if plan.polarized { 2 } else { 1 };
    let nsig = if nsp == 2 { 3 } else { 1 };
    let mut g = Gen {
        nsp,
        leaves: String::new(),
        body: String::new(),
        nleaf: 0,
        ntmp: 0,
    };
    let mut body = String::new();
    let _ = write!(body, "{}", decl_outputs("o"));
    for (k, (w, n)) in plan.components.iter().enumerate() {
        let c = format!("c{k}");
        let _ = write!(body, "{{\n{}", decl_outputs(&c));
        body += &g.node(n, &c)?;
        // linus_xc::Functional::eval_into: the first component as is (times its weight unless
        // 1), the others added as weight x component
        if k == 0 {
            for (nm, d) in FIRST {
                for i in 0..d {
                    if *w == 1.0 {
                        let _ = writeln!(body, "o_{nm}[{i}] = {c}_{nm}[{i}];");
                    } else {
                        let _ = writeln!(
                            body,
                            "o_{nm}[{i}] = {c}_{nm}[{i}] * {};",
                            hexf(*w).ok_or("a component weight")?
                        );
                    }
                }
            }
        } else {
            body += &add_outputs("o", &c, &hexf(*w).ok_or("a component weight")?);
        }
        body += "}\n";
    }
    g.body = body;
    let mut src = String::new();
    // the Maple code's second derivatives (fxc) are not compiled: no engine evaluates them here
    src += "#ifndef NULL\n#define NULL 0\n#endif\n";
    src += "#define XC_DONT_COMPILE_FXC\n";
    src += PRELUDE;
    src += &g.leaves;
    let (gga, mgga) = (kind.gga, kind.mgga);
    let _ = write!(
        src,
        "extern \"C\" __global__ void {}(long long np, const double *__restrict__ rho, \
         const double *__restrict__ sigma, const double *__restrict__ tau, double *zk, \
         double *vrho, double *vsigma, double *vtau)\n{{\n\
         const long long ip = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         if (ip >= np) return;\n\
         double in_rho[2] = {{0.0, 0.0}}, in_sigma[3] = {{0.0, 0.0, 0.0}}, \
         in_tau[2] = {{0.0, 0.0}}, in_lapl[2] = {{0.0, 0.0}};\n",
        ENTRY
    );
    for s in 0..nsp {
        let _ = writeln!(src, "in_rho[{s}] = rho[ip * {nsp} + {s}];");
        if mgga {
            let _ = writeln!(src, "in_tau[{s}] = tau[ip * {nsp} + {s}];");
        }
    }
    if gga {
        for k in 0..nsig {
            let _ = writeln!(src, "in_sigma[{k}] = sigma[ip * {nsig} + {k}];");
        }
    }
    src += "(void)in_lapl; (void)in_tau; (void)in_sigma;\n";
    src += &g.body;
    let _ = writeln!(src, "zk[ip] = o_zk[0];");
    for s in 0..nsp {
        let _ = writeln!(src, "vrho[ip * {nsp} + {s}] = o_vrho[{s}];");
        if mgga {
            let _ = writeln!(src, "vtau[ip * {nsp} + {s}] = o_vtau[{s}];");
        }
    }
    if gga {
        for k in 0..nsig {
            let _ = writeln!(src, "vsigma[ip * {nsig} + {k}] = o_vsigma[{k}];");
        }
    }
    src += "}\n";
    Ok(src)
}

/// The device module of functional `name` (a linus-xc name) for `nsp` spin channels, as the
/// engines compile it (before the device layer's prelude), and its kernel's name; Err: why it
/// has none (tests/nvrtc.rs compiles the scoreboard's functionals without a device).
pub fn module_source_for(name: &str, nsp: usize) -> Result<(String, &'static str), String> {
    let f = HostFunctional::new(name, None).map_err(|e| e.to_string())?;
    let plan = f.device_plan(nsp)?;
    Ok((source(&plan, f.kind())?, ENTRY))
}

/// A functional compiled for the device, for one spin treatment, checked against libxc.
pub(crate) struct DevFunc {
    kernel: Kernel,
    nsp: usize,
    kind: XcKind,
}

/// The check's tolerance relative to the largest output of a kind ([`DevFunc::check`]).
const CHECK_TOL: f64 = 1e-10;

/// Points per block of the functional's kernel.
const THREADS: u32 = 128;

impl DevFunc {
    /// The device kernel of `host`'s functional for `nsp` spin channels, compiled and
    /// checked against libxc (see the module docs); Err: why it is not used (the engines then
    /// evaluate on the host).
    pub(crate) fn new(gpu: &Gpu, host: &HostFunctional, nsp: usize) -> Result<DevFunc, String> {
        let kind = host.kind();
        if kind.none {
            return Err("no semilocal part".into());
        }
        let plan = host.device_plan(nsp)?;
        let src = source(&plan, kind)?;
        // the generated module and its options are the cache's key
        let name = "xc_devfunc";
        let kernel = gpu
            .kernel_with(name, &src, ENTRY, &["--fmad=false"])
            .map_err(|e| format!("compiling the functional: {e}"))?;
        let f = DevFunc { kernel, nsp, kind };
        f.check(gpu, host)
            .map_err(|e| format!("check against libxc: {e}"))?;
        Ok(f)
    }

    /// zk, vrho, vsigma and vtau (libxc's layout; a one-element placeholder for an ingredient the
    /// functional does not have) at the `npts` points of `rho`, `sigma` and `tau`.
    pub(crate) fn eval(
        &self,
        gpu: &Gpu,
        npts: usize,
        rho: &DeviceBuf<f64>,
        sigma: &DeviceBuf<f64>,
        tau: &DeviceBuf<f64>,
    ) -> Result<[DeviceBuf<f64>; 4], GpuError> {
        let (nsp, gga, mgga) = (self.nsp, self.kind.gga, self.kind.mgga);
        let nsig = if nsp == 2 { 3 } else { 1 };
        if rho.len() < npts * nsp
            || (gga && sigma.len() < npts * nsig)
            || (mgga && tau.len() < npts * nsp)
        {
            return Err(GpuError::Contract(
                "device functional: input too small".into(),
            ));
        }
        let zk = gpu.alloc_zeros::<f64>(npts.max(1))?;
        let vrho = gpu.alloc_zeros::<f64>((npts * nsp).max(1))?;
        let vsigma = gpu.alloc_zeros::<f64>(if gga { (npts * nsig).max(1) } else { 1 })?;
        let vtau = gpu.alloc_zeros::<f64>(if mgga { (npts * nsp).max(1) } else { 1 })?;
        if npts > 0 {
            // SAFETY: matches the generated kernel: npts points of rho (nsp each), sigma (nsig,
            // GGA and meta-GGA only), tau (nsp, meta-GGA only) in; zk, vrho and those of vsigma
            // and vtau out (sizes checked and allocated above).
            unsafe {
                gpu.launch(
                    &self.kernel,
                    Launch::linear(npts, THREADS),
                    &[
                        Arg::I64(npts as i64),
                        Arg::buf(rho),
                        Arg::buf(sigma),
                        Arg::buf(tau),
                        Arg::buf(&zk),
                        Arg::buf(&vrho),
                        Arg::buf(&vsigma),
                        Arg::buf(&vtau),
                    ],
                )
            }?;
        }
        Ok([zk, vrho, vsigma, vtau])
    }

    /// The device's results against libxc's on [`check_points`]: every output within
    /// [`CHECK_TOL`] of libxc's, relative to the largest of its kind. The device math library's
    /// last-place differences give at most 1.4e-12 there (SCAN's open-shell vrho, the worst of
    /// the functionals tested); a wrong parameter, mixing coefficient or threshold gives 1e-6 and
    /// more.
    fn check(&self, gpu: &Gpu, host: &HostFunctional) -> Result<(), String> {
        let nsp = self.nsp;
        let (rho, sigma, tau) = check_points(nsp);
        let npts = rho.len() / nsp;
        let e = |e: GpuError| e.to_string();
        let up = |v: &[f64]| gpu.upload(if v.is_empty() { &[0.0] } else { v }).map_err(e);
        let d = self
            .eval(gpu, npts, &up(&rho)?, &up(&sigma)?, &up(&tau)?)
            .map_err(e)?;
        let (gga, mgga) = (self.kind.gga, self.kind.mgga);
        let r = host
            .first(
                nsp,
                npts,
                &rho,
                if gga { &sigma } else { &[] },
                if mgga { &tau } else { &[] },
            )
            .map_err(e)?;
        let pairs: [(&str, &[f64], &DeviceBuf<f64>, bool); 4] = [
            ("zk", &r.zk, &d[0], true),
            ("vrho", &r.vrho, &d[1], true),
            ("vsigma", &r.vsigma, &d[2], gga),
            ("vtau", &r.vtau, &d[3], mgga),
        ];
        for (name, h, dv, on) in pairs {
            if !on {
                continue;
            }
            let dh = gpu.download(dv).map_err(e)?;
            let scale = h.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
            for (i, (a, b)) in h.iter().zip(&dh).enumerate() {
                let ok = (a - b).abs() <= CHECK_TOL * scale || (a.is_nan() && b.is_nan());
                if !ok {
                    return Err(format!(
                        "{name}[{i}]: device {b:e}, libxc {a:e} (largest {scale:e})"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Points spread over what a molecule's grid holds, in libxc's layout for `nsp` spins: densities
/// from 1e-14 to 1e3 (below and above the leaves' thresholds), reduced gradients from 0 to about
/// 20 (with the spin cross term inside its bounds), tau a little above the von Weizsaecker
/// bound and up (not tau = 0 where the gradient vanishes: wB97M-V's vtau is ill-conditioned there,
/// 1e-4 relative between libxc 7.0.0 and 7.1.2), and
/// spin polarizations across (-1, 1) for open shells. Not fully polarized points: there the
/// empty channel's potential is ill-conditioned (wB97X-V's vrho of an empty channel differs by
/// 4e-7 relative between libxc 7.0.0 and 7.1.2 on the CPU), so it tests nothing.
pub(crate) fn check_points(nsp: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut rho, mut sigma, mut tau) = (Vec::new(), Vec::new(), Vec::new());
    let zetas: &[f64] = if nsp == 2 {
        &[-0.999, -0.6, -0.05, 0.0, 0.3, 0.9, 0.999]
    } else {
        &[0.0]
    };
    for i in 0..18i32 {
        let n = 10f64.powi(i - 14);
        for &s in &[0.0, 0.05, 0.4, 1.3, 4.0, 20.0] {
            // |grad n| = 2 (3 pi^2)^(1/3) n^(4/3) s
            let g = 2.0 * (3.0 * std::f64::consts::PI.powi(2)).cbrt() * n.powf(4.0 / 3.0) * s;
            for &a in &[1.02, 1.7, 6.0] {
                let tw = g * g / (8.0 * n);
                let tu =
                    0.3 * (3.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0) * n.powf(5.0 / 3.0);
                let t = tw + a * tu - tu;
                for &z in zetas {
                    if nsp == 1 {
                        rho.push(n);
                        sigma.push(g * g);
                        tau.push(t.max(tw));
                    } else {
                        let (na, nb) = (0.5 * n * (1.0 + z), 0.5 * n * (1.0 - z));
                        rho.extend([na, nb]);
                        let (ga, gb) = (g * 0.5 * (1.0 + z), g * 0.5 * (1.0 - z));
                        // the spins' gradients at an angle: cos 0.3
                        sigma.extend([ga * ga, 0.3 * ga * gb, gb * gb]);
                        let (ta, tb) = (t.max(tw) * 0.5 * (1.0 + z), t.max(tw) * 0.5 * (1.0 - z));
                        tau.extend([ta, tb]);
                    }
                }
            }
        }
    }
    (rho, sigma, tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_are_exact() {
        for x in [
            1.0,
            -2.5,
            1e-300,
            4.9e-324,
            1e-15,
            0.0042,
            1.0 / 3.0,
            f64::MAX,
        ] {
            let s = hexf(x).unwrap();
            let (neg, rest) = match s.strip_prefix('-') {
                Some(r) => (true, r),
                None => (false, s.as_str()),
            };
            let (m, e) = rest.trim_start_matches("0x").split_once('p').unwrap();
            let (i, f) = m.split_once('.').unwrap();
            let mant = i64::from_str_radix(i, 16).unwrap() as f64
                + u64::from_str_radix(f, 16).unwrap() as f64 / 2f64.powi(52);
            let e: i32 = e.parse().unwrap();
            // 2^e in two exact steps (2^-1022 alone for the subnormals)
            let v = mant * 2f64.powi(e.max(-1000)) * 2f64.powi(e - e.max(-1000));
            assert_eq!(if neg { -v } else { v }.to_bits(), x.to_bits(), "{s}");
        }
        assert_eq!(hexf(0.0).unwrap(), "0.0");
        assert!(hexf(f64::INFINITY).is_none());
    }

    /// The scoreboard's functionals (and a few more) compile and pass their check against
    /// libxc, closed and open shell; times the compile.
    #[test]
    fn device_functionals_agree_with_libxc() {
        let Some(gpu) = crate::harness::gpu() else {
            return;
        };
        for name in [
            "b3lyp",
            "pbe",
            "r2scan",
            "wb97x-d3bj",
            "wb97m-v",
            "pbe0",
            "tpss",
            "svwn",
            "b97m-v",
            "scan",
        ] {
            let f = HostFunctional::new(name, None).unwrap();
            for nsp in [1, 2] {
                let t = std::time::Instant::now();
                let d = DevFunc::new(&gpu, &f, nsp);
                eprintln!(
                    "{name} nsp {nsp}: {:.2} s, {}",
                    t.elapsed().as_secs_f64(),
                    d.as_ref().map_or_else(|e| e.clone(), |_| "ok".into())
                );
                d.unwrap_or_else(|e| panic!("{name} nsp {nsp}: {e}"));
            }
        }
    }

    #[test]
    fn b3lyp_source_has_its_four_leaves() {
        let f = HostFunctional::new("b3lyp", None).unwrap();
        let plan = f.device_plan(1).unwrap();
        let src = source(&plan, f.kind()).unwrap();
        for i in 0..4 {
            assert!(src.contains(&format!("namespace xcl{i} ")));
        }
        assert!(!src.contains("namespace xcl4 "));
        assert!(src.contains("xcl1::func_vxc_unpol(&p, 0, my_rho, my_sigma, &out)"));
    }
}
