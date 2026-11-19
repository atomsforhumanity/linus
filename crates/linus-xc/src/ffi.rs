//! Hand-written FFI for the part of the libxc 7.0.0 C API that linus uses.
//!
//! Written against `src/xc.h` of the libxc 7.0.0 tarball (sha256 in build.rs).
//! `xc_func_type` and `xc_dimensions` are mirrored field by field because we read
//! `dim` and the thresholds directly (libxc 7.0.0 has no getters for those);
//! `xc_func_info_type` stays opaque and is read through the `xc_func_info_get_*`
//! getters. The test `layouts_match_c_headers` checks every offset and size
//! against a C shim compiled with the same headers (csrc/layout_check.c).
//!
//! Evaluation uses the classic per-order entry points (`xc_gga_exc_vxc`, ...).
//! Pointers to outputs a functional does not produce may be null; libxc checks
//! flags before writing (e.g. `vlapl` only if `XC_FLAGS_NEEDS_LAPLACIAN`).
//! libxc calls `exit(1)` when a *required* output pointer is null or the order
//! is not compiled in, so the safe layer validates all of that first.

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    missing_docs
)]

use std::os::raw::{c_char, c_double, c_int, c_void};

pub const XC_UNPOLARIZED: c_int = 1;
pub const XC_POLARIZED: c_int = 2;

pub const XC_EXCHANGE: c_int = 0;
pub const XC_CORRELATION: c_int = 1;
pub const XC_EXCHANGE_CORRELATION: c_int = 2;
pub const XC_KINETIC: c_int = 3;

pub const XC_FAMILY_UNKNOWN: c_int = -1;
pub const XC_FAMILY_LDA: c_int = 1;
pub const XC_FAMILY_GGA: c_int = 2;
pub const XC_FAMILY_MGGA: c_int = 4;
pub const XC_FAMILY_LCA: c_int = 8;
pub const XC_FAMILY_OEP: c_int = 16;
pub const XC_FAMILY_HYB_GGA: c_int = 32;
pub const XC_FAMILY_HYB_MGGA: c_int = 64;
pub const XC_FAMILY_HYB_LDA: c_int = 128;

pub const XC_FLAGS_HAVE_EXC: c_int = 1 << 0;
pub const XC_FLAGS_HAVE_VXC: c_int = 1 << 1;
pub const XC_FLAGS_HAVE_FXC: c_int = 1 << 2;
pub const XC_FLAGS_HAVE_KXC: c_int = 1 << 3;
pub const XC_FLAGS_HAVE_LXC: c_int = 1 << 4;
pub const XC_FLAGS_1D: c_int = 1 << 5;
pub const XC_FLAGS_2D: c_int = 1 << 6;
pub const XC_FLAGS_3D: c_int = 1 << 7;
/// Range separation via the error function (the usual case).
pub const XC_FLAGS_HYB_CAM: c_int = 1 << 8;
/// Range separation via a Yukawa function (rare).
pub const XC_FLAGS_HYB_CAMY: c_int = 1 << 9;
pub const XC_FLAGS_VV10: c_int = 1 << 10;
/// Deprecated alias of `XC_FLAGS_HYB_CAM` that some functionals still set.
pub const XC_FLAGS_HYB_LC: c_int = 1 << 11;
/// Deprecated alias of `XC_FLAGS_HYB_CAMY`.
pub const XC_FLAGS_HYB_LCY: c_int = 1 << 12;
pub const XC_FLAGS_STABLE: c_int = 1 << 13;
pub const XC_FLAGS_DEVELOPMENT: c_int = 1 << 14;
pub const XC_FLAGS_NEEDS_LAPLACIAN: c_int = 1 << 15;
pub const XC_FLAGS_NEEDS_TAU: c_int = 1 << 16;
pub const XC_FLAGS_ENFORCE_FHC: c_int = 1 << 17;

/// Opaque `xc_func_info_type`; read it with the `xc_func_info_get_*` getters.
#[repr(C)]
pub struct xc_func_info_type {
    _private: [u8; 0],
}

/// `struct xc_dimensions`: spin dimensions of every input and output array.
/// The C struct lists `rho, sigma, lapl, tau, zk` and then
/// `MGGA_OUT_PARAMS_NO_EXC` (4 + 10 + 20 + 35 derivative entries).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct xc_dimensions {
    pub rho: c_int,
    pub sigma: c_int,
    pub lapl: c_int,
    pub tau: c_int,
    pub zk: c_int,
    pub vrho: c_int,
    pub vsigma: c_int,
    pub vlapl: c_int,
    pub vtau: c_int,
    pub v2rho2: c_int,
    pub v2rhosigma: c_int,
    pub v2rholapl: c_int,
    pub v2rhotau: c_int,
    pub v2sigma2: c_int,
    pub v2sigmalapl: c_int,
    pub v2sigmatau: c_int,
    pub v2lapl2: c_int,
    pub v2lapltau: c_int,
    pub v2tau2: c_int,
    /// Third derivatives (20 entries, `v3rho3` .. `v3tau3`).
    pub v3: [c_int; 20],
    /// Fourth derivatives (35 entries, `v4rho4` .. `v4tau4`).
    pub v4: [c_int; 35],
}

/// `struct xc_func_type` (libxc 7.0.0, unchanged in 7.1.x).
#[repr(C)]
pub struct xc_func_type {
    pub info: *mut xc_func_info_type,
    pub nspin: c_int,
    pub n_func_aux: c_int,
    pub func_aux: *mut *mut xc_func_type,
    pub mix_coef: *mut c_double,
    /// Range-separation parameter.
    pub cam_omega: c_double,
    /// Full-range exact-exchange fraction (= long-range fraction).
    pub cam_alpha: c_double,
    /// Short-range-only exact-exchange fraction (SR total is alpha + beta).
    pub cam_beta: c_double,
    pub nlc_b: c_double,
    pub nlc_C: c_double,
    pub dim: xc_dimensions,
    pub ext_params: *mut c_double,
    pub params: *mut c_void,
    pub dens_threshold: c_double,
    pub zeta_threshold: c_double,
    pub sigma_threshold: c_double,
    pub tau_threshold: c_double,
}

// The library itself is linked by build.rs (static `xc` plus `m`).
extern "C" {
    pub fn xc_version(major: *mut c_int, minor: *mut c_int, micro: *mut c_int);
    pub fn xc_version_string() -> *const c_char;

    pub fn xc_functional_get_number(name: *const c_char) -> c_int;
    /// Returns a `malloc`ed copy; release with `free`.
    pub fn xc_functional_get_name(number: c_int) -> *mut c_char;
    pub fn xc_family_from_id(id: c_int, family: *mut c_int, number: *mut c_int) -> c_int;
    pub fn xc_number_of_functionals() -> c_int;
    pub fn xc_available_functional_numbers(list: *mut c_int);

    pub fn xc_func_alloc() -> *mut xc_func_type;
    pub fn xc_func_init(p: *mut xc_func_type, functional: c_int, nspin: c_int) -> c_int;
    pub fn xc_func_end(p: *mut xc_func_type);
    pub fn xc_func_free(p: *mut xc_func_type);
    pub fn xc_func_get_info(p: *const xc_func_type) -> *const xc_func_info_type;

    pub fn xc_func_set_dens_threshold(p: *mut xc_func_type, t_dens: c_double);
    pub fn xc_func_set_zeta_threshold(p: *mut xc_func_type, t_zeta: c_double);
    pub fn xc_func_set_sigma_threshold(p: *mut xc_func_type, t_sigma: c_double);
    pub fn xc_func_set_tau_threshold(p: *mut xc_func_type, t_tau: c_double);

    pub fn xc_func_info_get_number(info: *const xc_func_info_type) -> c_int;
    pub fn xc_func_info_get_kind(info: *const xc_func_info_type) -> c_int;
    pub fn xc_func_info_get_name(info: *const xc_func_info_type) -> *const c_char;
    pub fn xc_func_info_get_family(info: *const xc_func_info_type) -> c_int;
    pub fn xc_func_info_get_flags(info: *const xc_func_info_type) -> c_int;

    pub fn xc_hyb_exx_coef(p: *const xc_func_type) -> c_double;
    pub fn xc_hyb_cam_coef(
        p: *const xc_func_type,
        omega: *mut c_double,
        alpha: *mut c_double,
        beta: *mut c_double,
    );
    pub fn xc_nlc_coef(p: *const xc_func_type, nlc_b: *mut c_double, nlc_C: *mut c_double);
    pub fn xc_num_aux_funcs(p: *const xc_func_type) -> c_int;

    // ---- LDA ----
    pub fn xc_lda_exc(p: *const xc_func_type, np: usize, rho: *const f64, zk: *mut f64);
    pub fn xc_lda_exc_vxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        zk: *mut f64,
        vrho: *mut f64,
    );
    pub fn xc_lda_exc_vxc_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        zk: *mut f64,
        vrho: *mut f64,
        v2rho2: *mut f64,
    );
    pub fn xc_lda_vxc(p: *const xc_func_type, np: usize, rho: *const f64, vrho: *mut f64);
    pub fn xc_lda_vxc_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        vrho: *mut f64,
        v2rho2: *mut f64,
    );
    pub fn xc_lda_fxc(p: *const xc_func_type, np: usize, rho: *const f64, v2rho2: *mut f64);

    // ---- GGA ----
    pub fn xc_gga_exc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        zk: *mut f64,
    );
    pub fn xc_gga_exc_vxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        zk: *mut f64,
        vrho: *mut f64,
        vsigma: *mut f64,
    );
    pub fn xc_gga_exc_vxc_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        zk: *mut f64,
        vrho: *mut f64,
        vsigma: *mut f64,
        v2rho2: *mut f64,
        v2rhosigma: *mut f64,
        v2sigma2: *mut f64,
    );
    pub fn xc_gga_vxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        vrho: *mut f64,
        vsigma: *mut f64,
    );
    pub fn xc_gga_vxc_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        vrho: *mut f64,
        vsigma: *mut f64,
        v2rho2: *mut f64,
        v2rhosigma: *mut f64,
        v2sigma2: *mut f64,
    );
    pub fn xc_gga_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        v2rho2: *mut f64,
        v2rhosigma: *mut f64,
        v2sigma2: *mut f64,
    );

    // ---- meta-GGA ----
    pub fn xc_mgga_exc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        lapl: *const f64,
        tau: *const f64,
        zk: *mut f64,
    );
    pub fn xc_mgga_exc_vxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        lapl: *const f64,
        tau: *const f64,
        zk: *mut f64,
        vrho: *mut f64,
        vsigma: *mut f64,
        vlapl: *mut f64,
        vtau: *mut f64,
    );
    pub fn xc_mgga_exc_vxc_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        lapl: *const f64,
        tau: *const f64,
        zk: *mut f64,
        vrho: *mut f64,
        vsigma: *mut f64,
        vlapl: *mut f64,
        vtau: *mut f64,
        v2rho2: *mut f64,
        v2rhosigma: *mut f64,
        v2rholapl: *mut f64,
        v2rhotau: *mut f64,
        v2sigma2: *mut f64,
        v2sigmalapl: *mut f64,
        v2sigmatau: *mut f64,
        v2lapl2: *mut f64,
        v2lapltau: *mut f64,
        v2tau2: *mut f64,
    );
    pub fn xc_mgga_vxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        lapl: *const f64,
        tau: *const f64,
        vrho: *mut f64,
        vsigma: *mut f64,
        vlapl: *mut f64,
        vtau: *mut f64,
    );
    pub fn xc_mgga_vxc_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        lapl: *const f64,
        tau: *const f64,
        vrho: *mut f64,
        vsigma: *mut f64,
        vlapl: *mut f64,
        vtau: *mut f64,
        v2rho2: *mut f64,
        v2rhosigma: *mut f64,
        v2rholapl: *mut f64,
        v2rhotau: *mut f64,
        v2sigma2: *mut f64,
        v2sigmalapl: *mut f64,
        v2sigmatau: *mut f64,
        v2lapl2: *mut f64,
        v2lapltau: *mut f64,
        v2tau2: *mut f64,
    );
    pub fn xc_mgga_fxc(
        p: *const xc_func_type,
        np: usize,
        rho: *const f64,
        sigma: *const f64,
        lapl: *const f64,
        tau: *const f64,
        v2rho2: *mut f64,
        v2rhosigma: *mut f64,
        v2rholapl: *mut f64,
        v2rhotau: *mut f64,
        v2sigma2: *mut f64,
        v2sigmalapl: *mut f64,
        v2sigmatau: *mut f64,
        v2lapl2: *mut f64,
        v2lapltau: *mut f64,
        v2tau2: *mut f64,
    );

    /// From the C library; used to release `xc_functional_get_name` strings
    /// (libxc's CPU `libxc_malloc` is plain `malloc`).
    pub fn free(p: *mut c_void);
}

#[cfg(test)]
extern "C" {
    fn linus_xc_layout(out: *mut *const std::os::raw::c_longlong) -> std::os::raw::c_long;
}

/// Same list, same order as `layout[]` in csrc/layout_check.c.
#[cfg(test)]
pub(crate) fn rust_layout() -> Vec<(&'static str, i64)> {
    use std::mem::{offset_of, size_of};
    macro_rules! off {
        ($t:ty, $f:ident) => {
            (
                concat!(stringify!($t), ".", stringify!($f)),
                offset_of!($t, $f) as i64,
            )
        };
    }
    vec![
        ("sizeof(xc_func_type)", size_of::<xc_func_type>() as i64),
        off!(xc_func_type, info),
        off!(xc_func_type, nspin),
        off!(xc_func_type, n_func_aux),
        off!(xc_func_type, func_aux),
        off!(xc_func_type, mix_coef),
        off!(xc_func_type, cam_omega),
        off!(xc_func_type, cam_alpha),
        off!(xc_func_type, cam_beta),
        off!(xc_func_type, nlc_b),
        off!(xc_func_type, nlc_C),
        off!(xc_func_type, dim),
        off!(xc_func_type, ext_params),
        off!(xc_func_type, params),
        off!(xc_func_type, dens_threshold),
        off!(xc_func_type, zeta_threshold),
        off!(xc_func_type, sigma_threshold),
        off!(xc_func_type, tau_threshold),
        ("sizeof(xc_dimensions)", size_of::<xc_dimensions>() as i64),
        off!(xc_dimensions, rho),
        off!(xc_dimensions, sigma),
        off!(xc_dimensions, lapl),
        off!(xc_dimensions, tau),
        off!(xc_dimensions, zk),
        off!(xc_dimensions, vrho),
        off!(xc_dimensions, vsigma),
        off!(xc_dimensions, vlapl),
        off!(xc_dimensions, vtau),
        off!(xc_dimensions, v2rho2),
        off!(xc_dimensions, v2rhosigma),
        off!(xc_dimensions, v2rholapl),
        off!(xc_dimensions, v2rhotau),
        off!(xc_dimensions, v2sigma2),
        off!(xc_dimensions, v2sigmalapl),
        off!(xc_dimensions, v2sigmatau),
        off!(xc_dimensions, v2lapl2),
        off!(xc_dimensions, v2lapltau),
        off!(xc_dimensions, v2tau2),
        ("xc_dimensions.v3rho3", offset_of!(xc_dimensions, v3) as i64),
        (
            "xc_dimensions.v4tau4",
            (offset_of!(xc_dimensions, v4) + 34 * size_of::<c_int>()) as i64,
        ),
        ("XC_UNPOLARIZED", XC_UNPOLARIZED as i64),
        ("XC_POLARIZED", XC_POLARIZED as i64),
        ("XC_FAMILY_LDA", XC_FAMILY_LDA as i64),
        ("XC_FAMILY_GGA", XC_FAMILY_GGA as i64),
        ("XC_FAMILY_MGGA", XC_FAMILY_MGGA as i64),
        ("XC_FAMILY_HYB_LDA", XC_FAMILY_HYB_LDA as i64),
        ("XC_FAMILY_HYB_GGA", XC_FAMILY_HYB_GGA as i64),
        ("XC_FAMILY_HYB_MGGA", XC_FAMILY_HYB_MGGA as i64),
        ("XC_FLAGS_HAVE_EXC", XC_FLAGS_HAVE_EXC as i64),
        ("XC_FLAGS_HAVE_VXC", XC_FLAGS_HAVE_VXC as i64),
        ("XC_FLAGS_HAVE_FXC", XC_FLAGS_HAVE_FXC as i64),
        ("XC_FLAGS_HYB_CAM", XC_FLAGS_HYB_CAM as i64),
        ("XC_FLAGS_HYB_CAMY", XC_FLAGS_HYB_CAMY as i64),
        ("XC_FLAGS_VV10", XC_FLAGS_VV10 as i64),
        ("XC_FLAGS_HYB_LC", XC_FLAGS_HYB_LC as i64),
        ("XC_FLAGS_HYB_LCY", XC_FLAGS_HYB_LCY as i64),
        ("XC_FLAGS_NEEDS_LAPLACIAN", XC_FLAGS_NEEDS_LAPLACIAN as i64),
        ("XC_FLAGS_NEEDS_TAU", XC_FLAGS_NEEDS_TAU as i64),
        ("XC_EXCHANGE", XC_EXCHANGE as i64),
        ("XC_CORRELATION", XC_CORRELATION as i64),
        ("XC_EXCHANGE_CORRELATION", XC_EXCHANGE_CORRELATION as i64),
        ("XC_KINETIC", XC_KINETIC as i64),
        ("XC_MAJOR_VERSION", 7),
        ("XC_MINOR_VERSION", 0),
        ("XC_MICRO_VERSION", 0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_c_headers() {
        let mut ptr = std::ptr::null();
        let n = unsafe { linus_xc_layout(&mut ptr) } as usize;
        let c = unsafe { std::slice::from_raw_parts(ptr, n) };
        let r = rust_layout();
        assert_eq!(r.len(), c.len(), "layout lists out of sync");
        let bad: Vec<_> = r
            .iter()
            .zip(c)
            .filter(|((_, rv), cv)| *rv != **cv)
            .map(|((name, rv), cv)| format!("{name}: rust {rv} vs C {cv}"))
            .collect();
        assert!(bad.is_empty(), "layout mismatches:\n{}", bad.join("\n"));
    }

    #[test]
    fn linked_version_is_7_0_0() {
        let (mut a, mut b, mut c) = (0, 0, 0);
        unsafe { xc_version(&mut a, &mut b, &mut c) };
        assert_eq!((a, b, c), (7, 0, 0));
    }
}
