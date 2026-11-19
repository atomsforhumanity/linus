//! The response operator: the induced Fock matrix of a first-order density projected on the
//! occupied orbitals, dF_s\[dD\] C_os, for factored first-order densities
//! dD_s = A_s C_os^T + C_os A_s^T given by their factors `a[s][k]` (nbf x nocc_s; channel s,
//! perturbation k). Every CPHF/CPKS density has this form, and everything the response
//! equations, the Hessian and the polarizability need from dF is dF C_o, so no nbf x nbf
//! first-order matrix is formed.
//!
//! The operator is composed once per response ([`ResponseOperator`]: a Hessian's, a
//! polarizability's, or an IR run's electric-field response on its Hessian's J/K) from the
//! method families' linear responses, each linearized at the reference's occupied orbitals.
//! It applies them in a fixed order, whatever the order they were attached in:
//! 1. J/K ([`ResponseJk`]): (c_j J[sum_s dD_s] - c_k K\[dD_s\]) C_s, with c_j = 2 for one channel
//!    (a closed shell: dD is the alpha density) and 1 for two, and c_k the exact-exchange
//!    fraction (range-separated hybrids: 1, the operator fixed at the J/K's setup);
//! 2. the XC kernel ([`ResponseKernel`]: `linus::xc::engine::XcKernel`, the semilocal kernel and
//!    VV10's, summed inside it as one term);
//! 3. the environment ([`EnvironmentResponse`]: the solvent's reaction field).
//!
//! Each term's product is computed whole and then added to the J/K's. The order of these
//! additions is what the results' bits depend on, so it is the operator's, not its callers'.
//! Each term's time goes to the operator's timers (`cphf_jk`, `cphf_xc`, and the environment's
//! own name, `cphf_solvent`).
//!
//! Two-level responses (large Hessians with density fitting) run their iterations with a
//! cheaper J/K: [`ResponseOperator::begin_iterations`] frees the main J/K's stored data and then
//! builds the iteration J/K, so the two are never held together; the XC kernel and the
//! environment are the same at both levels. [`ResponseOperator::end_iterations`] drops the
//! iteration J/K and [`ResponseOperator::restore`] rebuilds the main one for the final
//! products.

use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::terms::EnvironmentResponse;
use linus_base::terms::ResponseJk;
use linus_base::terms::ResponseKernel;
use linus_core::timing::Timers;
use std::sync::Arc;

/// Which J/K serves the products of a [`ResponseOperator`].
enum Level {
    /// The main J/K.
    Main,
    /// A two-level response's iterations: their J/K (the main one has released its data).
    Iterations(Box<dyn ResponseJk>),
    /// The iterations ended and the main J/K has not been restored yet.
    Released,
}

/// The induced-Fock operator of a linear response (see the module docs): the J/K, the XC
/// kernel (Kohn-Sham) and the environment (implicit solvent), linearized at the occupied
/// orbitals `c_occ` of each spin channel and applied in that fixed order.
pub struct ResponseOperator<'a> {
    c_occ: Vec<MatRef<'a, f64>>,
    jk: Box<dyn ResponseJk>,
    level: Level,
    c_k: f64,
    kernel: Option<&'a dyn ResponseKernel>,
    environment: Option<Arc<dyn EnvironmentResponse>>,
    timers: Timers,
}

impl<'a> ResponseOperator<'a> {
    /// The operator of the J/K `jk` (set up for the occupied orbitals `c_occ`, one matrix per
    /// spin channel: one for a closed shell, two for an open shell) with the exact-exchange
    /// fraction `c_k`; the XC kernel and the environment are attached with
    /// [`Self::with_kernel`] and [`Self::with_environment`].
    pub fn new(c_occ: &[MatRef<'a, f64>], jk: Box<dyn ResponseJk>, c_k: f64) -> Self {
        assert!((1..=2).contains(&c_occ.len()), "one or two channels");
        ResponseOperator {
            c_occ: c_occ.to_vec(),
            jk,
            level: Level::Main,
            c_k,
            kernel: None,
            environment: None,
            timers: Timers::default(),
        }
    }

    /// The XC kernel (Kohn-Sham; None for Hartree-Fock), at the ground state of `c_occ`.
    pub fn with_kernel(mut self, kernel: Option<&'a dyn ResponseKernel>) -> Self {
        self.kernel = kernel;
        self
    }

    /// The environment's response (None in the gas phase).
    pub fn with_environment(mut self, environment: Option<Arc<dyn EnvironmentResponse>>) -> Self {
        self.environment = environment;
        self
    }

    /// The spin channels: 1 (closed shell) or 2.
    pub fn channels(&self) -> usize {
        self.c_occ.len()
    }

    /// dF_s\[dD\] C_s for the first-order densities with factors `a[s][k]` (nbf x nocc_s), as
    /// `[s][k]`: the J/K's products, then the XC kernel's and the environment's added (the
    /// module docs). A failed product (a device error) is the error.
    pub fn induced_fock_occ(
        &mut self,
        a: &[Vec<Mat<f64>>],
        par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let jk: &mut dyn ResponseJk = match &mut self.level {
            Level::Main => self.jk.as_mut(),
            Level::Iterations(j) => j.as_mut(),
            Level::Released => {
                panic!("ResponseOperator: restore the main J/K after the iterations")
            }
        };
        let t0 = std::time::Instant::now();
        // (c_j J[sum_s dD_s] - c_k K[dD_s]) C_s; one channel: dD is the alpha density, c_j = 2
        let c_j = if self.c_occ.len() == 1 { 2.0 } else { 1.0 };
        let mut fs = jk.fock_occ(a, c_j, self.c_k, par)?;
        self.timers.add("cphf_jk", t0.elapsed().as_secs_f64());
        if let Some(xc) = self.kernel {
            let t0 = std::time::Instant::now();
            // one channel: the kernel of the total first-order density 2 dD at 2 C_o C_o^T
            add_products(&mut fs, &xc.contract_occ(&self.c_occ, a)?);
            self.timers.add("cphf_xc", t0.elapsed().as_secs_f64());
        }
        if let Some(env) = &self.environment {
            let t0 = std::time::Instant::now();
            add_products(&mut fs, &env.induced_fock_occ(&self.c_occ, a, par)?);
            self.timers.add(env.timer(), t0.elapsed().as_secs_f64());
        }
        Ok(fs)
    }

    /// A two-level response's iterations: the main J/K frees its stored data, then `build`
    /// makes the J/K that serves the products until [`Self::end_iterations`].
    pub fn begin_iterations(&mut self, build: impl FnOnce() -> Box<dyn ResponseJk>) {
        assert!(
            matches!(self.level, Level::Main),
            "one iteration level at a time"
        );
        self.jk.release();
        self.level = Level::Iterations(build());
    }

    /// Whether a two-level response's iteration J/K serves the products.
    pub fn at_iteration_level(&self) -> bool {
        matches!(self.level, Level::Iterations(_))
    }

    /// Ends the iterations of [`Self::begin_iterations`]: their J/K is dropped and its timers
    /// returned. The main J/K serves products again after [`Self::restore`].
    pub fn end_iterations(&mut self) -> Option<Timers> {
        match std::mem::replace(&mut self.level, Level::Released) {
            Level::Iterations(j) => j.timers(),
            _ => panic!("ResponseOperator: no iterations to end"),
        }
    }

    /// Rebuilds the main J/K's stored data after [`Self::end_iterations`].
    pub fn restore(&mut self, par: Par) {
        assert!(
            matches!(self.level, Level::Released),
            "ResponseOperator: nothing to restore"
        );
        self.jk.restore(par);
        self.level = Level::Main;
    }

    /// The main J/K's own timers, if it keeps any (its integral and GEMM phases).
    pub fn jk_timers(&self) -> Option<Timers> {
        self.jk.timers()
    }

    /// The products' phases so far (`cphf_jk`, `cphf_xc`, the environment's; and those its
    /// solver adds).
    pub fn timers_mut(&mut self) -> &mut Timers {
        &mut self.timers
    }

    /// The main J/K (to serve another response of the same orbitals, [`ResponseJk::serve`])
    /// and the products' phases.
    pub fn into_parts(self) -> (Box<dyn ResponseJk>, Timers) {
        (self.jk, self.timers)
    }
}

/// fs[s][k] += v[s][k].
fn add_products(fs: &mut [Vec<Mat<f64>>], v: &[Vec<Mat<f64>>]) {
    for (fs_, vs) in fs.iter_mut().zip(v) {
        for (f, vk) in fs_.iter_mut().zip(vs) {
            *f += vk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Each test term scales the factors by its own odd factor (`scale`), so a product shows
    /// which terms entered it; `log` counts the J/K's events (products, releases, restores).
    struct Scaled {
        scale: f64,
        log: Arc<[AtomicUsize; 3]>,
    }

    fn scaled(a: &[Vec<Mat<f64>>], f: f64) -> Vec<Vec<Mat<f64>>> {
        a.iter()
            .map(|v| v.iter().map(|m| m * f).collect())
            .collect()
    }

    impl ResponseJk for Scaled {
        fn name(&self) -> String {
            "scaled".into()
        }
        fn fock_occ(
            &mut self,
            a: &[Vec<Mat<f64>>],
            c_j: f64,
            c_k: f64,
            _par: Par,
        ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
            self.log[0].fetch_add(1, Ordering::Relaxed);
            Ok(scaled(a, self.scale * c_j - c_k))
        }
        fn release(&mut self) {
            self.log[1].fetch_add(1, Ordering::Relaxed);
        }
        fn restore(&mut self, _par: Par) {
            self.log[2].fetch_add(1, Ordering::Relaxed);
        }
        fn timers(&self) -> Option<Timers> {
            let mut t = Timers::default();
            t.add("resp_scaled", self.scale);
            Some(t)
        }
    }

    impl ResponseKernel for Scaled {
        fn contract_occ(
            &self,
            _c_o: &[MatRef<'_, f64>],
            a: &[Vec<Mat<f64>>],
        ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
            Ok(scaled(a, self.scale))
        }
    }

    impl EnvironmentResponse for Scaled {
        fn timer(&self) -> &'static str {
            "cphf_test_env"
        }
        fn induced_fock_occ(
            &self,
            _c_occ: &[MatRef<'_, f64>],
            a: &[Vec<Mat<f64>>],
            _par: Par,
        ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
            Ok(scaled(a, self.scale))
        }
    }

    fn term(scale: f64) -> (Scaled, Arc<[AtomicUsize; 3]>) {
        let log = Arc::new([
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        ]);
        (
            Scaled {
                scale,
                log: log.clone(),
            },
            log,
        )
    }

    /// The operator adds the kernel's and then the environment's products to the J/K's, each
    /// whole, whichever is attached first; and its two-level J/K follows its states.
    #[test]
    fn the_terms_add_in_a_fixed_order_and_the_levels_switch() {
        let c = Mat::<f64>::from_fn(5, 2, |i, j| (i + 3 * j) as f64 * 0.1);
        let co = [c.as_ref(), c.as_ref()];
        let a: Vec<Vec<Mat<f64>>> = (0..2)
            .map(|s| {
                (0..3)
                    .map(|k| Mat::from_fn(5, 2, |i, j| 0.37 + (i + 2 * j + 5 * k + s) as f64))
                    .collect()
            })
            .collect();
        let (jk, log) = term(3.0);
        let (kernel, _) = term(5.0);
        let (env, _) = term(7.0);
        let env: Arc<dyn EnvironmentResponse> = Arc::new(env);
        // two channels: c_j = 1, c_k = 0.25
        let mut want = scaled(&a, 3.0 - 0.25);
        add_products(&mut want, &scaled(&a, 5.0));
        add_products(&mut want, &scaled(&a, 7.0));
        let mut op = ResponseOperator::new(&co, Box::new(jk), 0.25)
            .with_environment(Some(env.clone()))
            .with_kernel(Some(&kernel));
        let same = |x: &[Vec<Mat<f64>>], y: &[Vec<Mat<f64>>]| {
            x.iter().flatten().zip(y.iter().flatten()).all(|(p, q)| {
                (0..p.ncols())
                    .all(|j| (0..p.nrows()).all(|i| p[(i, j)].to_bits() == q[(i, j)].to_bits()))
            })
        };
        assert!(same(&op.induced_fock_occ(&a, Par::Seq).unwrap(), &want));
        assert!(op.timers.get("cphf_test_env").is_some());
        // two levels: the main J/K released, the iteration J/K's products, then the main's again
        let (it, it_log) = term(11.0);
        op.begin_iterations(|| Box::new(it));
        assert!(op.at_iteration_level());
        let mut want_it = scaled(&a, 11.0 - 0.25);
        add_products(&mut want_it, &scaled(&a, 5.0));
        add_products(&mut want_it, &scaled(&a, 7.0));
        assert!(same(&op.induced_fock_occ(&a, Par::Seq).unwrap(), &want_it));
        assert_eq!(op.end_iterations().unwrap().get("resp_scaled"), Some(11.0));
        op.restore(Par::Seq);
        assert!(same(&op.induced_fock_occ(&a, Par::Seq).unwrap(), &want));
        let n = |l: &Arc<[AtomicUsize; 3]>| l.each_ref().map(|x| x.load(Ordering::Relaxed));
        assert_eq!(n(&log), [2, 1, 1]);
        assert_eq!(n(&it_log), [1, 0, 0]);
        let (jk, timers) = op.into_parts();
        assert_eq!(jk.name(), "scaled");
        assert!(timers.get("cphf_jk").is_some() && timers.get("cphf_xc").is_some());
    }

    /// The main J/K serves no product between the end of the iterations and its restore.
    #[test]
    #[should_panic(expected = "restore the main J/K")]
    fn no_product_before_the_restore() {
        let c = Mat::<f64>::zeros(3, 1);
        let (jk, _) = term(1.0);
        let (it, _) = term(2.0);
        let mut op = ResponseOperator::new(&[c.as_ref()], Box::new(jk), 1.0);
        op.begin_iterations(|| Box::new(it));
        let _ = op.end_iterations();
        let a = vec![vec![Mat::<f64>::zeros(3, 1)]];
        let _ = op.induced_fock_occ(&a, Par::Seq);
    }
}
