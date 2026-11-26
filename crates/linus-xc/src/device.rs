//! What a device needs to evaluate a functional itself (feature `device-sources`): libxc's own
//! Maple-generated code of the leaf functionals (extracted from the libxc 7.0.0 tarball at build
//! time, up to second derivatives) and, per [`Functional`], its libxc tree with every leaf's
//! initialized state (parameters, thresholds, dimensions) as libxc holds it.
//!
//! A GPU engine compiles the leaves' code (CUDA C++, by NVRTC) into one kernel that repeats, per
//! point, what libxc's evaluation does: each leaf's input thresholds (`work_{lda,gga,mgga}_inc.c`),
//! its Maple function, the mixing of a libxc mixture (`mix_func.c`: out += coef x per auxiliary
//! functional, after the node's own term) and the [`Functional`]'s weighted sum of components.
//! Functionals outside the extracted set ([`DeviceSource`]s) have no plan
//! ([`Functional::device_plan`] says why), and the engines evaluate them with libxc on the host.

use crate::ffi;
use crate::functional::Functional;
use crate::kernel::Family;
use crate::libxc::LibxcFunctional;

/// One libxc source file of device-evaluable functionals (from build.rs).
#[derive(Debug)]
pub struct DeviceSource {
    /// The file's name in libxc's src/ (e.g. `gga_x_b88`).
    pub file: &'static str,
    /// `lda`, `gga` or `mgga`: the Maple code's family.
    pub family: &'static str,
    /// The functional numbers the file defines (some may be mixtures, which never reach it).
    pub numbers: &'static [i32],
    /// The parameter struct the Maple code reads, and its size in doubles (None: not all
    /// doubles, so the parameters cannot be copied: no device evaluation).
    pub params_type: Option<&'static str>,
    pub params_doubles: Option<usize>,
    /// The file's `typedef struct` blocks.
    pub typedefs: &'static str,
    /// The Maple-generated functions `func_{exc,vxc,fxc}_{unpol,pol}`.
    pub maple: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/device_sources.rs"));

/// Why no functional can be evaluated on a device in this build (no libxc source tarball at
/// build time), or None.
pub fn unavailable() -> Option<&'static str> {
    UNAVAILABLE
}

/// The source of libxc functional `number`, if it was extracted.
pub fn source_of(number: i32) -> Option<&'static DeviceSource> {
    SOURCES.iter().find(|s| s.numbers.contains(&number))
}

/// A leaf of a functional's libxc tree: a functional with its own Maple code, as initialized.
#[derive(Clone, Debug)]
pub struct DeviceLeaf {
    pub number: i32,
    pub source: &'static DeviceSource,
    /// libxc's `info->flags` (what it has: exc, vxc, fxc; needs tau; enforces the Fermi hole
    /// curvature).
    pub flags: i32,
    /// LDA, GGA or meta-GGA (the Maple code's family).
    pub family: Family,
    /// 1 or 2.
    pub nspin: i32,
    /// libxc's `dim`: rho, sigma, lapl, tau, zk, then the derivatives to second order (vrho,
    /// vsigma, vlapl, vtau, v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2, v2sigmalapl,
    /// v2sigmatau, v2lapl2, v2lapltau, v2tau2).
    pub dims: [i32; 19],
    pub dens_threshold: f64,
    pub zeta_threshold: f64,
    pub sigma_threshold: f64,
    pub tau_threshold: f64,
    pub cam_omega: f64,
    pub cam_alpha: f64,
    pub cam_beta: f64,
    /// The parameter struct's doubles (empty when the code reads none).
    pub params: Vec<f64>,
}

/// A node of a functional's libxc tree: its own term (a leaf) and/or a mixture of
/// auxiliary nodes (libxc adds coef x aux to the node's own outputs, in order).
#[derive(Clone, Debug)]
pub struct DeviceNode {
    pub own: Option<DeviceLeaf>,
    pub mix: Vec<(f64, DeviceNode)>,
}

/// A [`Functional`] for a device: its components (weight, libxc tree) in evaluation order; the
/// first is taken as is (times its weight when that is not 1), the others added as weight x
/// component.
#[derive(Clone, Debug)]
pub struct DevicePlan {
    pub polarized: bool,
    pub components: Vec<(f64, DeviceNode)>,
}

impl DevicePlan {
    /// The leaves in evaluation order.
    pub fn leaves(&self) -> Vec<&DeviceLeaf> {
        fn walk<'a>(n: &'a DeviceNode, out: &mut Vec<&'a DeviceLeaf>) {
            if let Some(l) = &n.own {
                out.push(l);
            }
            for (_, c) in &n.mix {
                walk(c, out);
            }
        }
        let mut out = Vec::new();
        for (_, n) in &self.components {
            walk(n, &mut out);
        }
        out
    }
}

impl Functional {
    /// The plan of a device evaluation of this functional, or why there is none (a component
    /// that is not a libxc functional, a leaf whose code was not extracted or whose parameters are
    /// not plain doubles, a Laplacian-dependent meta-GGA).
    pub fn device_plan(&self) -> Result<DevicePlan, String> {
        if let Some(why) = unavailable() {
            return Err(why.to_string());
        }
        let mut components = Vec::new();
        for c in self.components() {
            let k = c
                .kernel
                .as_any()
                .and_then(|a| a.downcast_ref::<LibxcFunctional>())
                .ok_or_else(|| {
                    format!("component {} is not a libxc functional", c.kernel.name())
                })?;
            components.push((c.weight, k.device_node()?));
        }
        Ok(DevicePlan {
            polarized: self.polarized(),
            components,
        })
    }
}

impl LibxcFunctional {
    /// This functional's libxc tree for a device (see [`Functional::device_plan`]).
    pub fn device_node(&self) -> Result<DeviceNode, String> {
        // SAFETY: the pointer is an initialized xc_func_type for this object's lifetime; the
        // walk only reads it and its auxiliary functionals (initialized with it).
        unsafe { node(self.raw()) }
    }
}

/// The node of the initialized functional `p`.
///
/// # Safety
/// `p` must point to an initialized `xc_func_type`.
unsafe fn node(p: *const ffi::xc_func_type) -> Result<DeviceNode, String> {
    // SAFETY: the caller's contract; the info getters read the static info struct.
    let f = unsafe { &*p };
    let info = unsafe { ffi::xc_func_get_info(p) };
    let number = unsafe { ffi::xc_func_info_get_number(info) };
    let flags = unsafe { ffi::xc_func_info_get_flags(info) };
    let libxc_family = unsafe { ffi::xc_func_info_get_family(info) };
    if flags & ffi::XC_FLAGS_NEEDS_LAPLACIAN != 0 {
        return Err(format!("libxc functional {number} needs the Laplacian"));
    }
    let mut mix = Vec::new();
    for i in 0..f.n_func_aux.max(0) as usize {
        // SAFETY: n_func_aux initialized auxiliary functionals and coefficients.
        let (coef, aux) = unsafe { (*f.mix_coef.add(i), *f.func_aux.add(i)) };
        mix.push((coef, unsafe { node(aux) }?));
    }
    let own = match source_of(number) {
        Some(src) => {
            let family = match libxc_family {
                ffi::XC_FAMILY_LDA | ffi::XC_FAMILY_HYB_LDA => Family::Lda,
                ffi::XC_FAMILY_GGA | ffi::XC_FAMILY_HYB_GGA => Family::Gga,
                ffi::XC_FAMILY_MGGA | ffi::XC_FAMILY_HYB_MGGA => Family::Mgga,
                other => return Err(format!("libxc functional {number}: family {other}")),
            };
            let fam_name = match family {
                Family::Lda => "lda",
                Family::Gga => "gga",
                _ => "mgga",
            };
            if src.family != fam_name {
                return Err(format!(
                    "libxc functional {number}: family {fam_name}, its code {}",
                    src.family
                ));
            }
            let params = match (src.params_type, src.params_doubles) {
                (None, _) => Vec::new(),
                (Some(t), None) => {
                    return Err(format!("libxc functional {number}: {t} is not all doubles"))
                }
                (Some(t), Some(n)) => {
                    if f.params.is_null() {
                        return Err(format!("libxc functional {number}: no {t}"));
                    }
                    // SAFETY: libxc allocated the struct (n doubles, build.rs's count of its
                    // members) at initialization.
                    unsafe { std::slice::from_raw_parts(f.params as *const f64, n) }.to_vec()
                }
            };
            let d = &f.dim;
            Some(DeviceLeaf {
                number,
                source: src,
                flags,
                family,
                nspin: f.nspin,
                dims: [
                    d.rho,
                    d.sigma,
                    d.lapl,
                    d.tau,
                    d.zk,
                    d.vrho,
                    d.vsigma,
                    d.vlapl,
                    d.vtau,
                    d.v2rho2,
                    d.v2rhosigma,
                    d.v2rholapl,
                    d.v2rhotau,
                    d.v2sigma2,
                    d.v2sigmalapl,
                    d.v2sigmatau,
                    d.v2lapl2,
                    d.v2lapltau,
                    d.v2tau2,
                ],
                dens_threshold: f.dens_threshold,
                zeta_threshold: f.zeta_threshold,
                sigma_threshold: f.sigma_threshold,
                tau_threshold: f.tau_threshold,
                cam_omega: f.cam_omega,
                cam_alpha: f.cam_alpha,
                cam_beta: f.cam_beta,
                params,
            })
        }
        None if mix.is_empty() => {
            return Err(format!(
                "libxc functional {number} ({}) has no device code",
                crate::libxc_key(number).unwrap_or_default()
            ))
        }
        // a mixture without its own term (B3LYP, PBE0, ...)
        None => None,
    };
    Ok(DeviceNode { own, mix })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The extracted table covers the functionals of linus's common aliases, and their plans
    /// carry libxc's parameters.
    #[test]
    fn common_functionals_have_plans() {
        assert!(unavailable().is_none(), "{:?}", unavailable());
        for name in [
            "b3lyp",
            "pbe",
            "r2scan",
            "wb97x-d3bj",
            "wb97m-v",
            "pbe0",
            "tpss",
            "blyp",
        ] {
            for pol in [false, true] {
                let f = Functional::new(name, pol).unwrap();
                let p = f.device_plan().unwrap_or_else(|e| panic!("{name}: {e}"));
                assert!(!p.leaves().is_empty());
                for l in p.leaves() {
                    assert_eq!(l.nspin, if pol { 2 } else { 1 });
                    assert_eq!(Some(l.params.len()), l.source.params_doubles.or(Some(0)));
                }
            }
        }
        // B3LYP: a libxc mixture of four leaves; B88's beta
        let p = Functional::new("b3lyp", false)
            .unwrap()
            .device_plan()
            .unwrap();
        let files: Vec<&str> = p.leaves().iter().map(|l| l.source.file).collect();
        assert_eq!(files, ["lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp"]);
        assert_eq!(p.leaves()[1].params[0], 0.0042);
    }
}
