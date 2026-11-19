/*
 * Layout oracle for the hand-written FFI in src/ffi.rs.
 *
 * Compiled by build.rs against the exact libxc headers we link, so the Rust
 * test `ffi::tests::layouts_match_c_headers` compares our #[repr(C)] mirrors with
 * what the C compiler says, field by field. Keep the order in sync with
 * `ffi::rust_layout()`.
 */
#include <stddef.h>
#include <xc.h>

#define OFF(T, f) ((long long)offsetof(T, f))

static const long long layout[] = {
  /* xc_func_type */
  (long long)sizeof(xc_func_type),
  OFF(xc_func_type, info),
  OFF(xc_func_type, nspin),
  OFF(xc_func_type, n_func_aux),
  OFF(xc_func_type, func_aux),
  OFF(xc_func_type, mix_coef),
  OFF(xc_func_type, cam_omega),
  OFF(xc_func_type, cam_alpha),
  OFF(xc_func_type, cam_beta),
  OFF(xc_func_type, nlc_b),
  OFF(xc_func_type, nlc_C),
  OFF(xc_func_type, dim),
  OFF(xc_func_type, ext_params),
  OFF(xc_func_type, params),
  OFF(xc_func_type, dens_threshold),
  OFF(xc_func_type, zeta_threshold),
  OFF(xc_func_type, sigma_threshold),
  OFF(xc_func_type, tau_threshold),
  /* xc_dimensions */
  (long long)sizeof(xc_dimensions),
  OFF(xc_dimensions, rho),
  OFF(xc_dimensions, sigma),
  OFF(xc_dimensions, lapl),
  OFF(xc_dimensions, tau),
  OFF(xc_dimensions, zk),
  OFF(xc_dimensions, vrho),
  OFF(xc_dimensions, vsigma),
  OFF(xc_dimensions, vlapl),
  OFF(xc_dimensions, vtau),
  OFF(xc_dimensions, v2rho2),
  OFF(xc_dimensions, v2rhosigma),
  OFF(xc_dimensions, v2rholapl),
  OFF(xc_dimensions, v2rhotau),
  OFF(xc_dimensions, v2sigma2),
  OFF(xc_dimensions, v2sigmalapl),
  OFF(xc_dimensions, v2sigmatau),
  OFF(xc_dimensions, v2lapl2),
  OFF(xc_dimensions, v2lapltau),
  OFF(xc_dimensions, v2tau2),
  OFF(xc_dimensions, v3rho3),
  OFF(xc_dimensions, v4tau4),
  /* public constants we hard-code in Rust */
  XC_UNPOLARIZED, XC_POLARIZED,
  XC_FAMILY_LDA, XC_FAMILY_GGA, XC_FAMILY_MGGA,
  XC_FAMILY_HYB_LDA, XC_FAMILY_HYB_GGA, XC_FAMILY_HYB_MGGA,
  XC_FLAGS_HAVE_EXC, XC_FLAGS_HAVE_VXC, XC_FLAGS_HAVE_FXC,
  XC_FLAGS_HYB_CAM, XC_FLAGS_HYB_CAMY, XC_FLAGS_VV10,
  XC_FLAGS_HYB_LC, XC_FLAGS_HYB_LCY,
  XC_FLAGS_NEEDS_LAPLACIAN, XC_FLAGS_NEEDS_TAU,
  XC_EXCHANGE, XC_CORRELATION, XC_EXCHANGE_CORRELATION, XC_KINETIC,
  XC_MAJOR_VERSION, XC_MINOR_VERSION, XC_MICRO_VERSION,
};

long linus_xc_layout(const long long **out)
{
  *out = layout;
  return (long)(sizeof(layout) / sizeof(layout[0]));
}
