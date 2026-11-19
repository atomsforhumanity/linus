//! The integral context of one geometry: the setups that depend only on the basis sets, the
//! method and the positions, built once per geometry on first use and shared by the SCF, the
//! gradient, the Hessian and the response properties of that geometry (Psi4's per-wavefunction
//! setups, but immutable per geometry rather than a mutable object graph).
//!
//! - **Shell pairs** ([`PairSet`], index-ordered): one set, at the highest derivative order a
//!   phase asked for. The Hermite expansions E are the same at every order (one primitive loop,
//!   `linus_ints::pairs`), so a gradient's or Hessian's derivative set serves every lower order
//!   after it (the Hessian's explicit terms, its response J/K, the field response, the PCM's
//!   terms). A phase that no longer needs the derivative expansions releases them
//!   ([`IntegralContext::release_pair_derivatives`]), as the Hessian does before its response;
//!   the SCF releases its set once its setups are built ([`IntegralContext::release_pairs`]), so
//!   it is not alive through the iterations.
//! - **The auxiliary basis sets** placed on the atoms, one per name (the SCF's density fitting,
//!   the gradient's, the Hessian's and the response's of the same purpose take one set).
//! - **The Coulomb metrics** of the auxiliary basis sets ([`CoulombMetric`]: the factor of
//!   (P|Q)), one per name, built once for the SCF's fitting, the gradient's, the Hessian's and
//!   the response's. A metric is held only while a later phase of the call will use it: the
//!   session releases it ([`IntegralContext::release_metrics`]) after the SCF's setup of an
//!   energy, at the end of a gradient, and after the setup of a Hessian's or a polarizability's
//!   response (the response's J/K keeps no metric); the fits hold their own reference while
//!   they run.
//! - **The overlap matrix** S.
//! - **The XC integrators** of the method, one per grid a phase asked for (with VV10 on its NLC
//!   grid), shared by the SCF, the gradient, the Hessian and the polarizability. An integrator's
//!   evaluations take it by shared reference and are pure functions of the densities, and its
//!   caches (the batches' shell lists and groups, the collocation engines) are pure functions of
//!   the geometry: a shared integrator gives the same bits as one built for the phase. VV10's
//!   statistics accumulate in it, so a phase reads them as differences.
//!
//! What the context holds does not change once built (cells behind locks, filled on first use
//! from whichever phase asks first); the session replaces the context when the atoms move.

use faer::Mat;
use linus_base::linalg::ao_view_mut;
use linus_dft::XcIntegrator;
use linus_grid::GridSpec;
use linus_ints::{BasisSet, DerivOrder, OneEPlan, PairOptions, PairSet};
use linus_jk::df::CoulombMetric;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// What an XC integrator is built on: its grid and VV10's NLC grid (None: no VV10 term).
type XcKey = (GridSpec, Option<GridSpec>);

/// The integral context of one geometry (see the module docs).
#[derive(Default)]
pub(crate) struct IntegralContext {
    /// The index-ordered shell pairs at the highest derivative order built so far.
    pairs: Mutex<Option<Arc<PairSet>>>,
    /// The placed auxiliary basis sets by name.
    aux: Mutex<Vec<(String, Arc<BasisSet>)>>,
    /// The Coulomb metrics of the auxiliary basis sets by name.
    metrics: Mutex<Vec<(String, Arc<CoulombMetric>)>>,
    /// S.
    overlap: OnceLock<Mat<f64>>,
    /// The XC integrators by grid.
    xc: Mutex<Vec<(XcKey, Arc<XcIntegrator>)>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl IntegralContext {
    /// The index-ordered shell pairs of `basis` (this context's geometry) with expansions up to
    /// at least derivative order `order`: the set built so far when its order suffices, else a
    /// new one at `order` (which replaces it).
    pub fn pairs(&self, basis: &BasisSet, order: DerivOrder) -> Arc<PairSet> {
        if let Some(p) = lock(&self.pairs)
            .as_ref()
            .filter(|p| p.deriv_order() >= order)
        {
            return p.clone();
        }
        // built outside the lock (the build runs on the pool)
        let p = Arc::new(PairSet::build(basis, PairOptions::with_order(order)));
        let mut slot = lock(&self.pairs);
        match slot.as_ref().filter(|q| q.deriv_order() >= order) {
            Some(q) => q.clone(),
            None => {
                *slot = Some(p.clone());
                p
            }
        }
    }

    /// Frees the shell pairs when no phase holds them any more (the SCF's, once its setups are
    /// built: they are not kept through its iterations).
    pub fn release_pairs(&self) {
        let mut slot = lock(&self.pairs);
        if slot.as_ref().is_some_and(|p| Arc::strong_count(p) == 1) {
            *slot = None;
        }
    }

    /// Frees the shell pairs' derivative expansions (the set becomes one of order zero) when no
    /// phase holds the set any more; a held set keeps them.
    pub fn release_pair_derivatives(&self) {
        if let Some(p) = lock(&self.pairs).as_mut().and_then(Arc::get_mut) {
            p.drop_derivs();
        }
    }

    /// The auxiliary basis set `name` placed on this context's atoms: the one an earlier phase
    /// built, else the one `build` makes (kept for the later phases).
    pub fn aux<E>(
        &self,
        name: &str,
        build: impl FnOnce() -> Result<BasisSet, E>,
    ) -> Result<Arc<BasisSet>, E> {
        let found = |v: &[(String, Arc<BasisSet>)]| {
            v.iter().find(|(n, _)| n == name).map(|(_, b)| b.clone())
        };
        if let Some(b) = found(&lock(&self.aux)) {
            return Ok(b);
        }
        let b = Arc::new(build()?);
        let mut v = lock(&self.aux);
        if let Some(c) = found(&v) {
            return Ok(c);
        }
        v.push((name.to_string(), b.clone()));
        Ok(b)
    }

    /// The Coulomb metric of the auxiliary basis set `name`: the one an earlier phase built and
    /// the context still holds, else the one `build` makes (held for the later phases until
    /// [`Self::release_metrics`]).
    pub fn metric<E>(
        &self,
        name: &str,
        build: impl FnOnce() -> Result<CoulombMetric, E>,
    ) -> Result<Arc<CoulombMetric>, E> {
        let found = |v: &[(String, Arc<CoulombMetric>)]| {
            v.iter().find(|(n, _)| n == name).map(|(_, m)| m.clone())
        };
        if let Some(m) = found(&lock(&self.metrics)) {
            return Ok(m);
        }
        // built outside the lock (the integrals and the factorization run on the pool)
        let m = Arc::new(build()?);
        let mut v = lock(&self.metrics);
        if let Some(c) = found(&v) {
            return Ok(c);
        }
        v.push((name.to_string(), m.clone()));
        Ok(m)
    }

    /// Drops the context's references to the metrics: a metric is freed once the phases
    /// holding it are done (an SCF's fitting holds its own through the iterations).
    pub fn release_metrics(&self) {
        lock(&self.metrics).clear();
    }

    /// The overlap matrix S of `basis` (this context's geometry).
    pub fn overlap(&self, basis: &BasisSet) -> &Mat<f64> {
        self.overlap.get_or_init(|| {
            let mut s = Mat::zeros(basis.nbf, basis.nbf);
            OneEPlan::new(basis, DerivOrder::Zero).overlap(&mut ao_view_mut(s.as_mut()));
            s
        })
    }

    /// The XC integrator on the grid `grid` with VV10 on `nlc_grid`: the one built for an
    /// earlier phase of this geometry, else the one `build` makes (kept for the later phases).
    pub fn xc<E>(
        &self,
        grid: &GridSpec,
        nlc_grid: Option<&GridSpec>,
        build: impl FnOnce() -> Result<XcIntegrator, E>,
    ) -> Result<Arc<XcIntegrator>, E> {
        let found = |v: &[(XcKey, Arc<XcIntegrator>)]| {
            v.iter()
                .find(|((g, n), _)| g == grid && n.as_ref() == nlc_grid)
                .map(|(_, x)| x.clone())
        };
        if let Some(x) = found(&lock(&self.xc)) {
            return Ok(x);
        }
        // built outside the lock (the grid build runs on the pool)
        let x = Arc::new(build()?);
        let mut v = lock(&self.xc);
        if let Some(y) = found(&v) {
            return Ok(y);
        }
        v.push(((grid.clone(), nlc_grid.cloned()), x.clone()));
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::driver::{CalcOptions, JkMethod, Session};
    use crate::Structure;
    use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};

    const WATER: &str = "3\n\nO 0.0 0.05 0.11\nH 0.0 0.76 -0.47\nH 0.1 -0.75 -0.46\n";

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// One pair set serves every order: E (and the Schwarz factors) are the same at every
    /// derivative order, and E^A the same at the first and second (def2-TZVP: d and f shells).
    #[test]
    fn pairs_identical_across_orders() {
        let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
        let def = CalcOptions::default().load_basis("def2-tzvp").unwrap();
        let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
        let sets: Vec<PairSet> = [DerivOrder::Zero, DerivOrder::One, DerivOrder::Two]
            .into_iter()
            .map(|o| PairSet::build(&basis, PairOptions::with_order(o)))
            .collect();
        for s in &sets[1..] {
            assert_eq!(s.len(), sets[0].len());
            for (a, b) in sets[0].pairs().iter().zip(s.pairs()) {
                assert_eq!(bits(&a.e), bits(&b.e));
                assert_eq!(a.emax.to_bits(), b.emax.to_bits());
            }
            assert_eq!(bits(sets[0].schwarz()), bits(s.schwarz()));
        }
        for (a, b) in sets[1].derivs().iter().zip(sets[2].derivs()) {
            assert_eq!(bits(&a.ea), bits(&b.ea));
            assert_eq!(bits(&a.ea_t), bits(&b.ea_t));
        }
    }

    /// The Coulomb metric's Cholesky factor has the same bits at any parallelism (the DF setups
    /// factor it sequentially below 1500 auxiliary functions and on up to 16 threads above, the
    /// derivatives on the session's pool): one factor can serve them all. 2000 functions, so
    /// the blocked factorization's parallel updates run.
    #[test]
    fn metric_cholesky_is_parallelism_independent() {
        use faer::{Mat, Par};
        let n = 2000;
        // a well-conditioned SPD matrix with a metric's decay: 1/(1 + |i - j|) + n I
        let a = Mat::from_fn(n, n, |i, j| {
            1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { n as f64 } else { 0.0 }
        });
        let seq = linus_jk::df::metric_cholesky(a.clone(), Par::Seq).unwrap();
        for t in [2, 7, 16] {
            let par = linus_jk::df::metric_cholesky(a.clone(), Par::rayon(t)).unwrap();
            let same =
                (0..n).all(|j| (0..n).all(|i| seq[(i, j)].to_bits() == par[(i, j)].to_bits()));
            assert!(same, "{t} threads");
        }
    }

    /// The one metric serves every site that built its own: the SCF's fitting and the response's
    /// took (P|Q) from the auxiliary basis set prepared for the 3-center kernels, the gradient and
    /// the Hessian from their derivative plans' (orders one and two), each factored on its
    /// parallelism; the shared [`CoulombMetric`] has the bits of each.
    #[test]
    fn one_metric_for_every_site() {
        use faer::{Mat, Par};
        use linus_ints::eri3c::{self, Aux3c};
        use linus_ints::operator::Operator;
        use linus_ints::{Eri3DerivOptions, Eri3DerivPlan};
        let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
        let place = |name: &str| {
            let def = CalcOptions::default().load_basis(name).unwrap();
            BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap()
        };
        let (basis, aux) = (place("def2-tzvp"), place("def2-universal-jkfit"));
        let naux = aux.nbf;
        let shared = super::CoulombMetric::new(&aux, Par::rayon(4)).unwrap();
        let l = shared.factor();
        let same = |v: Vec<f64>, par: Par, site: &str| {
            let own = linus_jk::df::metric_cholesky(
                Mat::from_fn(naux, naux, |i, j| v[i * naux + j]),
                par,
            )
            .unwrap();
            let bitwise =
                (0..naux).all(|j| (0..naux).all(|i| own[(i, j)].to_bits() == l[(i, j)].to_bits()));
            assert!(bitwise, "{site}");
        };
        // the SCF's and the response's
        same(
            eri3c::metric(&Aux3c::new(&aux), 0.0, &Operator::Coulomb),
            Par::Seq,
            "SCF",
        );
        // the gradient's and the Hessian's
        for (order, site) in [(DerivOrder::One, "gradient"), (DerivOrder::Two, "Hessian")] {
            let pairs = PairSet::build(&basis, PairOptions::with_order(order));
            let ap = PairSet::aux(&aux, PairOptions::with_order(order));
            let plan = Eri3DerivPlan::new(
                &basis,
                &pairs,
                &aux,
                &ap,
                order,
                Eri3DerivOptions::default(),
            )
            .unwrap();
            let v = eri3c::metric(plan.aux3c(), 0.0, &Operator::Coulomb);
            same(v, Par::rayon(2), site);
        }
    }

    /// The SCF's shared XC integrator gives its gradient the bits of one built for the gradient
    /// (a meta-GGA with VV10, so VV10's integrator is shared too), and the gradient builds no
    /// grid.
    #[test]
    fn shared_xc_integrator_is_bitwise_a_fresh_one() {
        let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
        let o = CalcOptions {
            method: Some("wb97m-v".into()),
            basis: Some("def2-svp".into()),
            jk: JkMethod::Df {
                aux: "def2-universal-jkfit".into(),
            }
            .into(),
            threads: Some(2),
            backend: serde_json::from_value(serde_json::json!("cpu")).unwrap(),
            ..Default::default()
        };
        let mut calc = Session::new(mol, o).unwrap();
        let g = calc.gradient().unwrap();
        let grid = |e: &&(String, f64, _)| e.0 == "grid" || e.0 == "nlc_grid";
        assert!(
            !g.timers.entries.iter().any(|e| grid(&e)),
            "the gradient built a grid"
        );
        // the gradient's densities: the reference's (crate::reference)
        let gs = calc.ground.clone().unwrap();
        let reference = gs.reference(&calc.cx.pool, calc.par());
        let (da, db) = (
            &reference.channels.alpha().density,
            &reference.channels.beta().density,
        );
        let fresh = calc
            .cx
            .pool
            .install(|| calc.build_xc(&mut Default::default()))
            .unwrap()
            .unwrap();
        let gr = calc.cx.plan.grid_response();
        let gx = calc
            .cx
            .pool
            .install(|| fresh.gradient(da, db, gs.scf.restricted, gr));
        let b = |v: &[[f64; 3]]| v.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(b(&g.parts.xc), b(&gx));
    }
}
