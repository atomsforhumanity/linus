/*
 * Make libxc's cbrt() calls use glibc's cbrt, as they do in any C program
 * (and in PySCF, Psi4, ...).
 *
 * Why: Rust's compiler-builtins (linked last into every Rust binary) defines a
 * *weak* cbrt (a port of CORE-MATH's correctly rounded cbrt). libxc.a is
 * bundled into the linus-xc rlib, which the linker sees before
 * compiler-builtins, while -lm (glibc) comes after it. So libxc's undefined
 * cbrt gets bound to compiler-builtins' cbrt, not glibc's (pow/exp/log are not
 * in compiler-builtins and still come from glibc). The two cbrt differ in the
 * last bit for some arguments; libxc calls cbrt everywhere, and near-zero
 * spin densities amplify those ulps to 1e-5..1e-2 relative differences in
 * derivatives, so results stop matching every other libxc user.
 *
 * Fix: a strong cbrt, linked from this rlib before compiler-builtins, that
 * forwards to the *versioned* glibc symbol (which our own definition cannot
 * satisfy). It interposes cbrt for the whole executable, which is glibc's
 * behaviour anyway. Built only for linux-gnu (see build.rs).
 */
#if defined(__x86_64__)
#define LINUS_XC_CBRT_VERSION "GLIBC_2.2.5"
#elif defined(__aarch64__)
#define LINUS_XC_CBRT_VERSION "GLIBC_2.17"
#endif

#ifdef LINUS_XC_CBRT_VERSION
extern double linus_xc_glibc_cbrt(double);
__asm__(".symver linus_xc_glibc_cbrt, cbrt@" LINUS_XC_CBRT_VERSION);

/* Hidden: if this definition were exported, ld.so would let it satisfy the
   versioned reference above (unversioned definitions match any version) and
   cbrt would call itself forever. */
__attribute__((visibility("hidden"))) double cbrt(double x)
{
  return linus_xc_glibc_cbrt(x);
}

/* Lets tests check which cbrt libxc ends up with. */
__attribute__((visibility("hidden"))) int linus_xc_cbrt_is_glibc(void)
{
  return 1;
}
#endif
