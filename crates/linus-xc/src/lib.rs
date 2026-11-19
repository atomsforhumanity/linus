//! Exchange-correlation functionals for linus.
//!
//! * [`Functional`]: what the SCF/Hessian code uses. Built from a name or a
//!   weighted mixture ([`spec`]), it reports what it needs (family, tau,
//!   Laplacian), its exact exchange ([`ExactExchange`], libxc convention) and
//!   VV10 parameters, and evaluates batches of grid points.
//! * [`XcKernel`]: the backend trait. [`LibxcFunctional`] implements it on top
//!   of a static libxc 7.0.0 (hand-written FFI in `ffi`); native Rust or GPU
//!   kernels can implement it later and replace libxc component by component.
//!
//! # Layout and conventions
//! Inputs and outputs are separate arrays per quantity (struct of arrays) in
//! libxc's layout: spin components interleaved per point, e.g. polarized
//! `rho = [a0, b0, a1, b1, ...]`, `sigma = [aa0, ab0, bb0, aa1, ...]`. See
//! [`XcInput`] and [`Out::dim`]. `zk` is the energy **per particle**
//! (`E = Σ w ρ zk`); derivatives are those of the energy per volume `ρ zk`,
//! as in libxc and PySCF's `eval_xc`.
//!
//! # Thread safety
//! [`Functional`] and [`LibxcFunctional`] are `Send + Sync`: libxc 7.0.0
//! evaluation reads the initialized functional only and keeps no global
//! mutable state, so one instance can serve all rayon workers; give each
//! worker its own [`XcOutput`]. Initialization is serialized internally
//! because a few libxc init routines write function-local statics.
//!
//! ```no_run
//! use linus_xc::{Functional, XcInput, XcOutput};
//! let f = Functional::new("b3lyp", false).unwrap();
//! assert_eq!(f.exx_fraction(), 0.2);
//! let (rho, sigma) = (vec![0.1, 0.2], vec![0.01, 0.02]);
//! let mut out = XcOutput::new();
//! f.eval(1, &XcInput::gga(&rho, &sigma), &mut out).unwrap();
//! println!("{:?} {:?} {:?}", out.zk(), out.vrho(), out.vsigma());
//! ```

// the libxc C API as declared (not all of it is used)
#[allow(dead_code)]
pub(crate) mod ffi;
mod functional;
mod kernel;
mod libxc;
pub mod spec;

pub use functional::{Component, Functional};
pub use kernel::{
    ExactExchange, Family, Out, RangeSeparation, Result, Vv10, XcError, XcInput, XcKernel,
    XcOutRefs, XcOutput, N_OUT,
};
pub use libxc::{libxc_available_ids, libxc_id, libxc_key, libxc_version, LibxcFunctional};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn send_sync() {
        assert_send_sync::<Functional>();
        assert_send_sync::<LibxcFunctional>();
    }

    #[test]
    fn hybrid_parameters() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        let f = Functional::new("b3lyp", false).unwrap();
        assert!(close(f.exx_fraction(), 0.2) && f.range_separation().is_none());
        assert_eq!(f.family(), Family::Gga);
        let f = Functional::new("pbe0", true).unwrap();
        assert!(close(f.exx_fraction(), 0.25));
        let f = Functional::new("hf", false).unwrap();
        assert_eq!(f.family(), Family::Hf);
        assert!(close(f.exx_fraction(), 1.0));
        let f = Functional::new("wb97x", false).unwrap();
        let (a, b, w) = f.range_separation().unwrap();
        assert!(
            close(a, 1.0) && close(a + b, 0.157706) && close(w, 0.3),
            "{a} {b} {w}"
        );
        assert_eq!(f.exact_exchange().kernel, RangeSeparation::Erf);
        let f = Functional::new("cam-b3lyp", false).unwrap();
        let (a, b, w) = f.range_separation().unwrap();
        assert!(
            close(a, 0.65) && close(a + b, 0.19) && close(w, 0.33),
            "{a} {b} {w}"
        );
        let f = Functional::new("wb97m-v", false).unwrap();
        assert!(f.needs_tau() && !f.needs_lapl() && f.family() == Family::Mgga);
        let v = f.vv10().unwrap();
        assert!(close(v.b, 6.0) && close(v.c, 0.01) && close(v.scale, 1.0));
        let f = Functional::new("wb97m-d3bj", false).unwrap();
        assert!(f.vv10().is_none() && f.dispersion() == Some("D3BJ"));
        let f = Functional::new(
            "0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP",
            false,
        )
        .unwrap();
        assert!(close(f.exx_fraction(), 0.2) && f.components().len() == 4);
        let f = Functional::new("0.5*b3lyp + 0.1*hf", false).unwrap();
        assert!(close(f.exx_fraction(), 0.2));
    }

    #[test]
    fn dims_match_libxc() {
        for (name, pol) in [
            ("lda_x", false),
            ("gga_x_b88", true),
            ("mgga_x_r2scan", true),
            ("mgga_x_br89", true),
        ] {
            let k = LibxcFunctional::from_name(name, pol).unwrap();
            let d = k.dims();
            let ours = |o: Out| o.dim(pol) as i32;
            let theirs = [
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
            ];
            for (o, t) in Out::ALL.iter().zip(theirs) {
                if o.produced_by(k.family(), true, true) {
                    assert_eq!(ours(*o), t, "{name} {}", o.name());
                }
            }
        }
    }

    #[test]
    fn errors_instead_of_libxc_exit() {
        let f = Functional::new("r2scan", false).unwrap();
        let rho = [0.1];
        let sigma = [0.01];
        let mut out = XcOutput::new();
        assert!(matches!(
            f.eval(1, &XcInput::gga(&rho, &sigma), &mut out),
            Err(XcError::MissingInput("tau"))
        ));
        assert!(matches!(
            f.eval(3, &XcInput::mgga(&rho, &sigma, &[0.1]), &mut out),
            Err(XcError::Order { .. })
        ));
        assert!(matches!(
            f.eval(1, &XcInput::mgga(&rho, &sigma, &[0.1, 0.2]), &mut out),
            Err(XcError::InputLength { name: "tau", .. })
        ));
    }

    #[test]
    fn single_kernel_and_mixture_agree() {
        // B3LYP from libxc's own mixture vs our explicit mixture.
        let rho: Vec<f64> = (1..=50).map(|i| 0.01 * i as f64).collect();
        let sigma: Vec<f64> = rho.iter().map(|r| 0.3 * r * r).collect();
        let a = Functional::new("b3lyp", false).unwrap();
        let b = Functional::new(
            "0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP",
            false,
        )
        .unwrap();
        let (mut oa, mut ob) = (XcOutput::new(), XcOutput::new());
        let inp = XcInput::gga(&rho, &sigma);
        a.eval(2, &inp, &mut oa).unwrap();
        b.eval(2, &inp, &mut ob).unwrap();
        for o in [
            Out::Zk,
            Out::Vrho,
            Out::Vsigma,
            Out::V2rho2,
            Out::V2rhosigma,
            Out::V2sigma2,
        ] {
            for (x, y) in oa.get(o).iter().zip(ob.get(o)) {
                assert!(
                    (x - y).abs() <= 1e-14 * x.abs().max(1e-10),
                    "{} {x} {y}",
                    o.name()
                );
            }
        }
    }
}
