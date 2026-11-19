//! Two-electron operators: full Coulomb, erf-attenuated (long-range) Coulomb, and linear
//! combinations c/r12 + l erf(omega r12)/r12 (which covers erfc = short range, and the CAM
//! functionals' exact-exchange operator). [`Operator`] is the one operator type of linus: the
//! four-center, three-center and grid-potential integrals, the J/K builders
//! (`JkBuilder::set_exchange_operator`) and the derivative codes all take it, coefficients
//! included (the 1/r coefficient is applied also without an erf part).
//!
//! In the McMurchie-Davidson scheme only the Boys stage depends on the operator. For
//! erf(omega r)/r the Hermite Coulomb integrals use alpha_w = alpha k with
//! k = omega^2 / (omega^2 + alpha) in the Boys argument and in the recursion, and the
//! prefactor gains sqrt(k) (Neese, SHARK, eq. 51-54). The recursion from R^(n)_000 to R_tuv is
//! linear and independent of alpha, so a combination only changes the starting values:
//!
//!   R^(n)_000 = pref [ c (-2 alpha)^n F_n(alpha R^2) + l sqrt(k) (-2 alpha k)^n F_n(alpha k R^2) ]
//!
//! with c the coefficient of 1/r12 and l that of erf(omega r12)/r12.

use crate::core::boys::boys_lanes;
use linus_num::simd::Lanes;

/// A two-electron operator. Its coefficients are named by what they multiply: `coulomb` the
/// full-range 1/r12, `long_range` erf(omega r12)/r12 (libxc's CAM `alpha`/`beta` multiply 1/r
/// and erfc instead; [`Operator::from_libxc`] is the one conversion).
///
/// Several values describe the same operator (`Attenuated { coulomb: 1, long_range: 0, .. }` is
/// [`Operator::Coulomb`]; an erf part with omega = 0 vanishes): equality compares the
/// coefficients ([`Operator::coulomb`], [`Operator::long_range`], [`Operator::omega`]), and
/// [`Operator::kernel`] resolves the one integral kernel an operator is evaluated with.
#[derive(Clone, Copy, Debug, Default)]
pub enum Operator {
    /// 1/r12.
    #[default]
    Coulomb,
    /// erf(omega r12)/r12, the long-range part of 1/r12.
    LongRange { omega: f64 },
    /// coulomb / r12 + long_range erf(omega r12) / r12: erfc(omega r12)/r12 is
    /// { coulomb: 1, long_range: -1 }, a global hybrid's exact exchange { coulomb: a_x,
    /// long_range: 0, omega: 0 }, a range-separated hybrid's [`Operator::from_libxc`].
    Attenuated {
        coulomb: f64,
        long_range: f64,
        omega: f64,
    },
}

impl Operator {
    /// The exact-exchange operator of a libxc range-separated hybrid (`cam_alpha`, `cam_beta`,
    /// `cam_omega`, linus-xc's `ExactExchange`): alpha / r + beta erfc(omega r) / r =
    /// (alpha + beta) / r - beta erf(omega r) / r. The one conversion from libxc's erfc form
    /// to the operator's coefficients; callers convert once, never back ((a + b) - b need not
    /// be a in floating point).
    ///
    /// It takes the three numbers rather than an `ExactExchange` because linus-ints does not
    /// depend on linus-xc: the integral crate stays free of the crate that builds libxc.
    pub fn from_libxc(alpha: f64, beta: f64, omega: f64) -> Self {
        Operator::Attenuated {
            coulomb: alpha + beta,
            long_range: -beta,
            omega,
        }
    }

    /// The coefficient of 1/r12.
    #[inline]
    pub fn coulomb(&self) -> f64 {
        match *self {
            Operator::Coulomb => 1.0,
            Operator::LongRange { .. } => 0.0,
            Operator::Attenuated { coulomb, .. } => coulomb,
        }
    }

    /// The coefficient of erf(omega r12)/r12.
    #[inline]
    pub fn long_range(&self) -> f64 {
        match *self {
            Operator::Coulomb => 0.0,
            Operator::LongRange { .. } => 1.0,
            Operator::Attenuated { long_range, .. } => long_range,
        }
    }

    /// The range-separation parameter omega (0 for 1/r12).
    #[inline]
    pub fn omega(&self) -> f64 {
        match *self {
            Operator::Coulomb => 0.0,
            Operator::LongRange { omega } | Operator::Attenuated { omega, .. } => omega,
        }
    }

    /// The operator has an erf (long-range) part: a nonzero long-range coefficient and omega.
    #[inline]
    pub fn has_erf(&self) -> bool {
        self.long_range() != 0.0 && self.omega() != 0.0
    }

    /// The fraction a of a global operator a/r (no erf part; HF exchange 1, a global hybrid's
    /// exact-exchange fraction); None for operators with an erf part. The exact-exchange code
    /// takes the cheaper plain-Coulomb path for global fractions (K of 1/r scaled by a).
    #[inline]
    pub fn global_fraction(&self) -> Option<f64> {
        (!self.has_erf()).then_some(self.coulomb())
    }

    #[inline]
    pub fn is_coulomb(&self) -> bool {
        self.coulomb() == 1.0 && (self.long_range() == 0.0 || self.omega() == 0.0)
    }

    /// Bound |(ab|cd)_op| <= factor * Q_ab Q_cd with Coulomb Schwarz factors Q
    /// (|(ab|cd)_erf| <= sqrt((ab|ab)_erf (cd|cd)_erf) <= Q_ab Q_cd since erf(wr)/r is a
    /// positive-definite kernel bounded by 1/r).
    #[inline]
    pub fn schwarz_factor(&self) -> f64 {
        if self.omega() == 0.0 {
            self.coulomb().abs()
        } else {
            self.coulomb().abs() + self.long_range().abs()
        }
    }

    /// The screening threshold on Coulomb bounds for integrals of this operator: `thresh`
    /// divided by [`Self::schwarz_factor`], so that a contribution bounded by Coulomb Schwarz
    /// factors is dropped only if its bound for this operator is below `thresh`.
    #[inline]
    pub fn screen_threshold(&self, thresh: f64) -> f64 {
        thresh / self.schwarz_factor().max(1e-300)
    }

    /// The operator as the integral kernels evaluate it, the one selection rule between the
    /// single-operator kernels ([`Operator::Coulomb`], [`Operator::LongRange`]: one Boys
    /// evaluation per primitive batch) and the combined one ([`Operator::Attenuated`]: two).
    /// An erf part with omega = 0 vanishes (erf(0 r)/r = 0).
    pub fn kernel(&self) -> Operator {
        let (coulomb, omega) = (self.coulomb(), self.omega());
        let long_range = if omega == 0.0 { 0.0 } else { self.long_range() };
        if long_range == 0.0 && coulomb == 1.0 {
            Operator::Coulomb
        } else if coulomb == 0.0 && long_range == 1.0 {
            Operator::LongRange { omega }
        } else {
            Operator::Attenuated {
                coulomb,
                long_range,
                omega,
            }
        }
    }
}

impl PartialEq for Operator {
    /// Equal coefficients: the same operator however it is spelled.
    fn eq(&self, other: &Self) -> bool {
        self.coulomb() == other.coulomb()
            && self.long_range() == other.long_range()
            && self.omega() == other.omega()
    }
}

/// Starting values rn\[n\] = R^(n)_000 (n = 0..=l) of the Hermite recursion for `op`, given
/// alpha = pq/(p+q), r2 = |P-Q|^2 and the Coulomb prefactor 2 pi^2.5 / (p q sqrt(p+q)).
/// `fm` is scratch of length l + 1.
#[inline(always)]
pub fn rn_operator<V: Lanes>(
    l: usize,
    alpha: V,
    r2: V,
    pref: V,
    op: &Operator,
    fm: &mut [V],
    rn: &mut [V],
) {
    let mut first = true;
    if op.coulomb() != 0.0 {
        boys_lanes(l, alpha * r2, fm);
        let m2a = alpha * V::splat(-2.0);
        let mut f = pref * V::splat(op.coulomb());
        for n in 0..=l {
            rn[n] = f * fm[n];
            f *= m2a;
        }
        first = false;
    }
    if op.long_range() != 0.0 && op.omega() != 0.0 {
        let w2 = V::splat(op.omega() * op.omega());
        let k = w2 / (w2 + alpha);
        let aw = alpha * k;
        boys_lanes(l, aw * r2, fm);
        let m2a = aw * V::splat(-2.0);
        let mut f = pref * k.sqrt() * V::splat(op.long_range());
        for n in 0..=l {
            rn[n] = if first {
                f * fm[n]
            } else {
                f.mul_add(fm[n], rn[n])
            };
            f *= m2a;
        }
        first = false;
    }
    if first {
        for x in rn[..=l].iter_mut() {
            *x = V::zero();
        }
    }
}

/// Boys values of `op` in the form the Coulomb R recursion consumes: with alpha = pq/(p+q)
/// unchanged, fm\[n\] = a F_n(alpha r2) + b sqrt(k) k^n F_n(alpha k r2), k = omega^2 / (omega^2 +
/// alpha), so that pref (-2 alpha)^n fm\[n\] is the operator's R^(n)_000 (see the module docs).
/// Code that builds R tensors from (alpha, F_n) needs no other change for erf operators;
/// for the Coulomb operator this is `boys(l, alpha r2, fm)`.
#[inline]
pub fn boys_operator(l: usize, alpha: f64, r2: f64, op: &Operator, fm: &mut [f64]) {
    if op.coulomb() == 1.0 && (op.long_range() == 0.0 || op.omega() == 0.0) {
        crate::core::boys::boys(l, alpha * r2, fm);
        return;
    }
    let mut tmp = [0.0f64; crate::core::boys::MAX_M + 1];
    for x in fm[..=l].iter_mut() {
        *x = 0.0;
    }
    if op.coulomb() != 0.0 {
        crate::core::boys::boys(l, alpha * r2, &mut tmp);
        for n in 0..=l {
            fm[n] = op.coulomb() * tmp[n];
        }
    }
    if op.long_range() != 0.0 && op.omega() != 0.0 {
        let w2 = op.omega() * op.omega();
        let k = w2 / (w2 + alpha);
        crate::core::boys::boys(l, alpha * k * r2, &mut tmp);
        let mut f = op.long_range() * k.sqrt();
        for n in 0..=l {
            fm[n] += f * tmp[n];
            f *= k;
        }
    }
}

/// Scalar version of [`rn_operator`] (reference path).
pub fn rn_operator_scalar(
    l: usize,
    alpha: f64,
    r2: f64,
    pref: f64,
    op: &Operator,
    fm: &mut [f64],
    rn: &mut [f64],
) {
    for x in rn[..=l].iter_mut() {
        *x = 0.0;
    }
    if op.coulomb() != 0.0 {
        crate::core::boys::boys(l, alpha * r2, fm);
        let mut f = pref * op.coulomb();
        for n in 0..=l {
            rn[n] += f * fm[n];
            f *= -2.0 * alpha;
        }
    }
    if op.long_range() != 0.0 && op.omega() != 0.0 {
        let w2 = op.omega() * op.omega();
        let k = w2 / (w2 + alpha);
        let aw = alpha * k;
        crate::core::boys::boys(l, aw * r2, fm);
        let mut f = pref * k.sqrt() * op.long_range();
        for n in 0..=l {
            rn[n] += f * fm[n];
            f *= -2.0 * aw;
        }
    }
}
