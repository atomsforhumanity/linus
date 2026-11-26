// linus-gpu device functionals (src/xc/devfunc.rs): the environment libxc's Maple-generated code
// expects (libxc 7.0.0's util.h and xc.h, as far as that code uses them: its macros, the
// functional struct it reads and the output structs it writes), for the leaf functionals the
// generator places after this header, each in its own namespace. The structs carry only the
// members the code reads; the generator fills them per leaf with libxc's initialized values.

#define GPU_DEVICE_FUNCTION __device__
#define GPU_FUNCTION __device__
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC

#define XC_UNPOLARIZED 1
#define XC_POLARIZED 2
#define XC_FLAGS_HAVE_EXC (1 << 0)
#define XC_FLAGS_HAVE_VXC (1 << 1)
#define XC_FLAGS_HAVE_FXC (1 << 2)
#define XC_FLAGS_HAVE_KXC (1 << 3)
#define XC_FLAGS_HAVE_LXC (1 << 4)
#define XC_FLAGS_NEEDS_LAPLACIAN (1 << 15)
#define XC_FLAGS_NEEDS_TAU (1 << 16)
#define XC_FLAGS_ENFORCE_FHC (1 << 17)

// libxc's util.h (the double values of its long double constants)
#ifndef M_E
#define M_E 2.7182818284590452354
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#define POW_2(x) ((x) * (x))
#define POW_3(x) ((x) * (x) * (x))
#define POW_1_2(x) sqrt(x)
#define POW_1_4(x) sqrt(sqrt(x))
#define POW_3_2(x) ((x) * sqrt(x))
#define CBRT(x) cbrt(x)
#define POW_1_3(x) cbrt(x)
#define POW_2_3(x) (cbrt(x) * cbrt(x))
#define POW_4_3(x) ((x) * cbrt(x))
#define POW_5_3(x) ((x) * cbrt(x) * cbrt(x))
#define POW_7_3(x) ((x) * (x) * cbrt(x))
#define my_piecewise3(c, x1, x2) ((c) ? (x1) : (x2))
#define my_piecewise5(c1, x1, c2, x2, x3) ((c1) ? (x1) : ((c2) ? (x2) : (x3)))
#define M_SQRTPI 1.772453850905516027298167483341145182798
#define M_CBRTPI 1.464591887561523263020142527263790391739
#define M_SQRT3 1.732050807568877293527446341505872366943
#define M_CBRT2 1.259921049894873164767210607278228350570
#define M_CBRT3 1.442249570307408382321638310780109588392
#define M_CBRT4 1.587401051968199474751705639272308260391
#define M_CBRT5 1.709975946676696989353108872543860109868
#define M_CBRT6 1.817120592832139658891211756327260502428
#define M_CBRT7 1.912931182772389101199116839548760282862
#define M_CBRT9 2.080083823051904114530056824357885386338
#define m_min(x, y) (((x) < (y)) ? (x) : (y))
#define m_max(x, y) (((x) < (y)) ? (y) : (x))
#define assert(x) ((void)0)
#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-16
#endif
#ifndef DBL_MIN
#define DBL_MIN 2.2250738585072014e-308
#endif

struct xc_dimensions {
    int rho, sigma, lapl, tau, zk;
    int vrho, vsigma, vlapl, vtau;
    int v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2,
        v2lapltau, v2tau2;
};

struct xc_func_info_type {
    int number;
    int flags;
};

struct xc_func_type {
    const xc_func_info_type *info;
    int nspin;
    double cam_omega, cam_alpha, cam_beta;
    xc_dimensions dim;
    const void *params;
    double dens_threshold, zeta_threshold, sigma_threshold, tau_threshold;
};

struct xc_lda_out_params {
    double *zk;
    double *vrho;
    double *v2rho2;
};

struct xc_gga_out_params {
    double *zk;
    double *vrho, *vsigma;
    double *v2rho2, *v2rhosigma, *v2sigma2;
};

struct xc_mgga_out_params {
    double *zk;
    double *vrho, *vsigma, *vlapl, *vtau;
    double *v2rho2, *v2rhosigma, *v2rholapl, *v2rhotau, *v2sigma2;
    double *v2sigmalapl, *v2sigmatau, *v2lapl2, *v2lapltau, *v2tau2;
};
