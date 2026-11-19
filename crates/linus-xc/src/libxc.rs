//! Safe RAII wrapper around one initialized libxc functional.

use crate::ffi;
use crate::kernel::{
    ExactExchange, Family, Out, RangeSeparation, Result, Vv10, XcError, XcInput, XcKernel,
    XcOutRefs,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::ptr::NonNull;
use std::sync::Mutex;

/// Serializes `xc_func_init`/`xc_func_end`. Evaluation needs no lock, but some
/// libxc 7.0.0 init routines write function-local `static` arrays before
/// copying them (e.g. `hyb_gga_xc_hse.c` sets `funcs_id[0]` differently for
/// each HSE/HJS variant, `hyb_mgga_xc_kcis.c`, `hyb_gga_xc_b1wc.c`), so two
/// concurrent inits of related functionals could race.
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// One libxc functional (by id), initialized for a spin treatment.
///
/// # Thread safety
/// `Send + Sync`: after `xc_func_init`, libxc 7.0.0 evaluation only reads the
/// `xc_func_type` (all entry points take `const xc_func_type *`), keeps no
/// global mutable state (the only file-scope arrays are read-only Chebyshev
/// tables in `bessel.c`/`expint_e1.c`), and allocates per-call scratch with
/// `malloc` (`xc_mix_func`). So one instance can be evaluated from many threads
/// at once. Mutation (thresholds) needs `&mut self`.
#[derive(Debug)]
pub struct LibxcFunctional {
    ptr: NonNull<ffi::xc_func_type>,
    id: i32,
    key: String,
    description: String,
    polarized: bool,
    libxc_family: i32,
    flags: i32,
    kind: i32,
}

// SAFETY: see the type-level docs; evaluation is read-only on the C struct and
// libxc has no mutable global state on the evaluation path.
unsafe impl Send for LibxcFunctional {}
unsafe impl Sync for LibxcFunctional {}

impl LibxcFunctional {
    /// Initialize libxc functional `id` (e.g. 402 = HYB_GGA_XC_B3LYP).
    pub fn new(id: i32, polarized: bool) -> Result<Self> {
        let key = libxc_key(id).ok_or_else(|| XcError::UnknownFunctional(id.to_string()))?;
        let nspin = if polarized {
            ffi::XC_POLARIZED
        } else {
            ffi::XC_UNPOLARIZED
        };
        // SAFETY: xc_func_alloc returns malloc'ed storage or null.
        let raw = unsafe { ffi::xc_func_alloc() };
        let ptr = NonNull::new(raw).expect("xc_func_alloc returned null");
        let code = {
            let _g = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: ptr is valid, id was checked to exist above.
            unsafe { ffi::xc_func_init(ptr.as_ptr(), id, nspin) }
        };
        if code != 0 {
            // On failure info may be null: xc_func_end would dereference it.
            unsafe { ffi::xc_func_free(ptr.as_ptr()) };
            return Err(XcError::Init {
                id,
                name: key,
                code,
            });
        }
        // SAFETY: initialized above; info is set by xc_func_init on success.
        let (libxc_family, flags, kind, description) = unsafe {
            let info = ffi::xc_func_get_info(ptr.as_ptr());
            let name = ffi::xc_func_info_get_name(info);
            let description = if name.is_null() {
                String::new()
            } else {
                CStr::from_ptr(name).to_string_lossy().into_owned()
            };
            (
                ffi::xc_func_info_get_family(info),
                ffi::xc_func_info_get_flags(info),
                ffi::xc_func_info_get_kind(info),
                description,
            )
        };
        let f = LibxcFunctional {
            ptr,
            id,
            key,
            description,
            polarized,
            libxc_family,
            flags,
            kind,
        };
        // families linus evaluates (LDA, GGA, meta-GGA and their hybrids); others (e.g. LCA)
        // are an error here rather than a panic later (dropping `f` frees it)
        let known = [
            ffi::XC_FAMILY_LDA,
            ffi::XC_FAMILY_HYB_LDA,
            ffi::XC_FAMILY_GGA,
            ffi::XC_FAMILY_HYB_GGA,
            ffi::XC_FAMILY_MGGA,
            ffi::XC_FAMILY_HYB_MGGA,
        ];
        if !known.contains(&libxc_family) {
            return Err(XcError::Parse { spec: f.key.clone(), msg: format!("libxc family {libxc_family} is not supported (LDA, GGA and meta-GGA functionals are)") });
        }
        Ok(f)
    }

    /// Look up and initialize by libxc name (`"GGA_X_B88"`, `"xc_gga_x_b88"`,
    /// case-insensitive) or decimal id.
    pub fn from_name(name: &str, polarized: bool) -> Result<Self> {
        let id = libxc_id(name).ok_or_else(|| XcError::UnknownFunctional(name.to_string()))?;
        Self::new(id, polarized)
    }

    /// libxc id.
    pub fn id(&self) -> i32 {
        self.id
    }
    /// libxc's long description, e.g. "Becke 88".
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Raw libxc `XC_FLAGS_*` bits.
    pub fn flags(&self) -> i32 {
        self.flags
    }
    /// Raw libxc family (`XC_FAMILY_HYB_GGA`, ...).
    pub fn libxc_family(&self) -> i32 {
        self.libxc_family
    }
    /// `XC_EXCHANGE`, `XC_CORRELATION`, `XC_EXCHANGE_CORRELATION` or `XC_KINETIC`.
    pub fn kind(&self) -> i32 {
        self.kind
    }
    /// Number of internal libxc components (0 if not a libxc mixture).
    pub fn num_aux(&self) -> i32 {
        unsafe { ffi::xc_num_aux_funcs(self.ptr.as_ptr()) }
    }
    /// libxc's dimension table for this functional (used to cross-check ours).
    pub fn dims(&self) -> ffi::xc_dimensions {
        unsafe { (*self.ptr.as_ptr()).dim }
    }
    /// Global exact-exchange coefficient as returned by `xc_hyb_exx_coef`.
    pub fn exx_coef(&self) -> f64 {
        unsafe { ffi::xc_hyb_exx_coef(self.ptr.as_ptr()) }
    }

    fn has(&self, flag: i32) -> bool {
        self.flags & flag != 0
    }
}

impl Drop for LibxcFunctional {
    fn drop(&mut self) {
        let _g = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: initialized in `new`, dropped once.
        unsafe {
            ffi::xc_func_end(self.ptr.as_ptr());
            ffi::xc_func_free(self.ptr.as_ptr());
        }
    }
}

impl XcKernel for LibxcFunctional {
    fn name(&self) -> &str {
        &self.key
    }
    fn libxc_id(&self) -> Option<i32> {
        Some(self.id)
    }
    fn polarized(&self) -> bool {
        self.polarized
    }
    fn family(&self) -> Family {
        match self.libxc_family {
            ffi::XC_FAMILY_LDA | ffi::XC_FAMILY_HYB_LDA => Family::Lda,
            ffi::XC_FAMILY_GGA | ffi::XC_FAMILY_HYB_GGA => Family::Gga,
            ffi::XC_FAMILY_MGGA | ffi::XC_FAMILY_HYB_MGGA => Family::Mgga,
            f => unreachable!("libxc family {f} for functional {}", self.id),
        }
    }
    fn needs_tau(&self) -> bool {
        self.has(ffi::XC_FLAGS_NEEDS_TAU)
    }
    fn needs_lapl(&self) -> bool {
        self.has(ffi::XC_FLAGS_NEEDS_LAPLACIAN)
    }
    fn max_order(&self) -> usize {
        if self.has(ffi::XC_FLAGS_HAVE_FXC) {
            2
        } else if self.has(ffi::XC_FLAGS_HAVE_VXC) {
            1
        } else {
            0
        }
    }
    fn exact_exchange(&self) -> ExactExchange {
        let (mut omega, mut alpha, mut beta) = (0.0, 0.0, 0.0);
        unsafe { ffi::xc_hyb_cam_coef(self.ptr.as_ptr(), &mut omega, &mut alpha, &mut beta) };
        let kernel = if self.has(ffi::XC_FLAGS_HYB_CAMY) || self.has(ffi::XC_FLAGS_HYB_LCY) {
            RangeSeparation::Yukawa
        } else if self.has(ffi::XC_FLAGS_HYB_CAM) || self.has(ffi::XC_FLAGS_HYB_LC) {
            RangeSeparation::Erf
        } else {
            RangeSeparation::None
        };
        ExactExchange {
            alpha,
            beta,
            omega,
            kernel,
        }
    }
    fn vv10(&self) -> Option<Vv10> {
        if !self.has(ffi::XC_FLAGS_VV10) {
            return None;
        }
        let (mut b, mut c) = (0.0, 0.0);
        unsafe { ffi::xc_nlc_coef(self.ptr.as_ptr(), &mut b, &mut c) };
        Some(Vv10 { b, c, scale: 1.0 })
    }
    fn dens_threshold(&self) -> f64 {
        unsafe { (*self.ptr.as_ptr()).dens_threshold }
    }
    fn set_dens_threshold(&mut self, threshold: f64) {
        // libxc ignores non-positive values and propagates to its own components.
        unsafe { ffi::xc_func_set_dens_threshold(self.ptr.as_ptr(), threshold) }
    }

    fn eval(&self, order: usize, input: &XcInput<'_>, out: &mut XcOutRefs<'_>) -> Result<()> {
        let max = self.max_order();
        if order > max {
            return Err(XcError::Order {
                name: self.key.clone(),
                order,
                max,
            });
        }
        let family = self.family();
        let (tau_needed, lapl_needed) = (self.needs_tau(), self.needs_lapl());
        let np = input.validate(self.polarized, family, tau_needed, lapl_needed)?;
        out.validate(order, np, self.polarized, family, tau_needed, lapl_needed)?;
        if np == 0 {
            return Ok(());
        }
        let have_exc = self.has(ffi::XC_FLAGS_HAVE_EXC);
        if !have_exc && order == 0 {
            return Err(XcError::Unsupported(format!(
                "{} has no energy (potential-only functional)",
                self.key
            )));
        }

        let p = self.ptr.as_ptr() as *const ffi::xc_func_type;
        let rho = input.rho.as_ptr();
        let sigma = input.sigma.map_or(std::ptr::null(), |s| s.as_ptr());
        // Ingredients a meta-GGA does not use are never read, but libxc still
        // forms `&lapl[ip*dim]`; point them at `rho` (same length) not null.
        let tau = if tau_needed {
            input.tau.unwrap().as_ptr()
        } else {
            rho
        };
        let lapl = if lapl_needed {
            input.lapl.unwrap().as_ptr()
        } else {
            rho
        };

        let zk = if have_exc {
            out.ptr(Out::Zk)
        } else {
            std::ptr::null_mut()
        };
        let (vrho, vsigma, vlapl, vtau) = (
            out.ptr(Out::Vrho),
            out.ptr(Out::Vsigma),
            out.ptr(Out::Vlapl),
            out.ptr(Out::Vtau),
        );
        let (v2rho2, v2rhosigma, v2sigma2) = (
            out.ptr(Out::V2rho2),
            out.ptr(Out::V2rhosigma),
            out.ptr(Out::V2sigma2),
        );
        let (v2rholapl, v2rhotau, v2sigmalapl) = (
            out.ptr(Out::V2rholapl),
            out.ptr(Out::V2rhotau),
            out.ptr(Out::V2sigmalapl),
        );
        let (v2sigmatau, v2lapl2, v2lapltau, v2tau2) = (
            out.ptr(Out::V2sigmatau),
            out.ptr(Out::V2lapl2),
            out.ptr(Out::V2lapltau),
            out.ptr(Out::V2tau2),
        );

        // SAFETY: all required arrays were validated (presence and exact length)
        // above; optional ones are null or valid; libxc writes only np*dim
        // entries of arrays the functional produces.
        unsafe {
            match (family, order, have_exc) {
                (Family::Lda, 0, _) => ffi::xc_lda_exc(p, np, rho, zk),
                (Family::Lda, 1, true) => ffi::xc_lda_exc_vxc(p, np, rho, zk, vrho),
                (Family::Lda, 1, false) => ffi::xc_lda_vxc(p, np, rho, vrho),
                (Family::Lda, 2, true) => ffi::xc_lda_exc_vxc_fxc(p, np, rho, zk, vrho, v2rho2),
                (Family::Lda, 2, false) => ffi::xc_lda_vxc_fxc(p, np, rho, vrho, v2rho2),
                (Family::Gga, 0, _) => ffi::xc_gga_exc(p, np, rho, sigma, zk),
                (Family::Gga, 1, true) => ffi::xc_gga_exc_vxc(p, np, rho, sigma, zk, vrho, vsigma),
                (Family::Gga, 1, false) => ffi::xc_gga_vxc(p, np, rho, sigma, vrho, vsigma),
                (Family::Gga, 2, true) => ffi::xc_gga_exc_vxc_fxc(
                    p, np, rho, sigma, zk, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2,
                ),
                (Family::Gga, 2, false) => ffi::xc_gga_vxc_fxc(
                    p, np, rho, sigma, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2,
                ),
                (Family::Mgga, 0, _) => ffi::xc_mgga_exc(p, np, rho, sigma, lapl, tau, zk),
                (Family::Mgga, 1, true) => ffi::xc_mgga_exc_vxc(
                    p, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau,
                ),
                (Family::Mgga, 1, false) => {
                    ffi::xc_mgga_vxc(p, np, rho, sigma, lapl, tau, vrho, vsigma, vlapl, vtau)
                }
                (Family::Mgga, 2, true) => ffi::xc_mgga_exc_vxc_fxc(
                    p,
                    np,
                    rho,
                    sigma,
                    lapl,
                    tau,
                    zk,
                    vrho,
                    vsigma,
                    vlapl,
                    vtau,
                    v2rho2,
                    v2rhosigma,
                    v2rholapl,
                    v2rhotau,
                    v2sigma2,
                    v2sigmalapl,
                    v2sigmatau,
                    v2lapl2,
                    v2lapltau,
                    v2tau2,
                ),
                (Family::Mgga, 2, false) => ffi::xc_mgga_vxc_fxc(
                    p,
                    np,
                    rho,
                    sigma,
                    lapl,
                    tau,
                    vrho,
                    vsigma,
                    vlapl,
                    vtau,
                    v2rho2,
                    v2rhosigma,
                    v2rholapl,
                    v2rhotau,
                    v2sigma2,
                    v2sigmalapl,
                    v2sigmatau,
                    v2lapl2,
                    v2lapltau,
                    v2tau2,
                ),
                _ => unreachable!(),
            }
        }
        if !have_exc {
            // Make misuse of a potential-only functional's energy loud.
            if let Some(z) = out.get_mut(Out::Zk) {
                z.fill(f64::NAN);
            }
        }
        Ok(())
    }

    fn try_clone(&self) -> Result<Box<dyn XcKernel>> {
        let mut c = LibxcFunctional::new(self.id, self.polarized)?;
        c.set_dens_threshold(self.dens_threshold());
        Ok(Box::new(c))
    }
}

/// libxc id for a name (`"gga_x_b88"`, `"XC_GGA_X_B88"`, any case) or a
/// decimal id string. `None` if unknown.
pub fn libxc_id(name: &str) -> Option<i32> {
    let name = name.trim();
    if let Ok(id) = name.parse::<i32>() {
        return libxc_key(id).map(|_| id);
    }
    let c = CString::new(name).ok()?;
    let id = unsafe { ffi::xc_functional_get_number(c.as_ptr()) };
    (id >= 0).then_some(id)
}

/// libxc's key (lower-case name, e.g. `"hyb_gga_xc_b3lyp"`) for an id.
pub fn libxc_key(id: i32) -> Option<String> {
    // xc_family_from_id allocates nothing; check first.
    if unsafe { ffi::xc_family_from_id(id, std::ptr::null_mut(), std::ptr::null_mut()) }
        == ffi::XC_FAMILY_UNKNOWN
    {
        return None;
    }
    let p = unsafe { ffi::xc_functional_get_name(id) };
    if p.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { ffi::free(p.cast()) };
    Some(s)
}

/// Version of the linked libxc, e.g. `(7, 0, 0)`.
pub fn libxc_version() -> (i32, i32, i32) {
    let (mut a, mut b, mut c): (c_int, c_int, c_int) = (0, 0, 0);
    unsafe { ffi::xc_version(&mut a, &mut b, &mut c) };
    (a, b, c)
}

/// All functional ids known to the linked libxc.
pub fn libxc_available_ids() -> Vec<i32> {
    let n = unsafe { ffi::xc_number_of_functionals() } as usize;
    let mut v = vec![0; n];
    unsafe { ffi::xc_available_functional_numbers(v.as_mut_ptr()) };
    v
}
