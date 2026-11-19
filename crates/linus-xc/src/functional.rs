//! [`Functional`]: a weighted sum of XC kernels plus exact exchange.

use crate::kernel::{
    ExactExchange, Family, Out, RangeSeparation, Result, Vv10, XcError, XcInput, XcKernel,
    XcOutRefs, XcOutput, N_OUT,
};
use crate::libxc::LibxcFunctional;
use crate::spec::{self, TermKind};
use std::cell::RefCell;

/// One semilocal component of a [`Functional`].
#[derive(Debug)]
pub struct Component {
    pub weight: f64,
    pub kernel: Box<dyn XcKernel>,
    /// Whether this component's VV10 parameters are reported by
    /// [`Functional::vv10`] (false for ORCA-style `-D3BJ` aliases).
    pub keep_vv10: bool,
}

/// An exchange-correlation functional: `Σ_k w_k f_k` (semilocal kernels) plus
/// exact exchange, built from a name or a mixture spec (see [`crate::spec`]).
///
/// `Functional` is `Send + Sync`: share one instance (`&Functional`) across
/// rayon workers and give each worker its own [`XcOutput`].
#[derive(Debug)]
pub struct Functional {
    spec: String,
    polarized: bool,
    components: Vec<Component>,
    dispersion: Option<&'static str>,
    family: Family,
    needs_tau: bool,
    needs_lapl: bool,
    max_order: usize,
    exx: ExactExchange,
}

impl Functional {
    /// Build from a name (`"b3lyp"`, `"wB97M-V"`, `"GGA_X_B88"`, `"402"`) or a
    /// mixture (`"0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + ..."`).
    pub fn new(spec: &str, polarized: bool) -> Result<Self> {
        let parsed = spec::parse(spec)?;
        let mut hf_weight = 0.0;
        let mut components = Vec::new();
        for t in parsed.terms {
            match t.kind {
                TermKind::Hf => hf_weight += t.weight,
                TermKind::Libxc(id) if t.weight != 0.0 => components.push(Component {
                    weight: t.weight,
                    kernel: Box::new(LibxcFunctional::new(id, polarized)?),
                    keep_vv10: t.keep_vv10,
                }),
                TermKind::Libxc(_) => {}
            }
        }
        Self::from_components(
            spec.to_string(),
            polarized,
            components,
            hf_weight,
            parsed.dispersion,
        )
    }

    /// Build from explicit kernels (e.g. a future native implementation).
    pub fn from_components(
        spec: String,
        polarized: bool,
        components: Vec<Component>,
        hf_weight: f64,
        dispersion: Option<&'static str>,
    ) -> Result<Self> {
        if let Some(c) = components
            .iter()
            .find(|c| c.kernel.polarized() != polarized)
        {
            return Err(XcError::Unsupported(format!(
                "component {} has the wrong spin treatment",
                c.kernel.name()
            )));
        }
        let family = components
            .iter()
            .map(|c| c.kernel.family())
            .max()
            .unwrap_or(Family::Hf);
        let needs_tau = components
            .iter()
            .any(|c| c.kernel.family() == Family::Mgga && c.kernel.needs_tau());
        let needs_lapl = components
            .iter()
            .any(|c| c.kernel.family() == Family::Mgga && c.kernel.needs_lapl());
        let max_order = components
            .iter()
            .map(|c| c.kernel.max_order())
            .min()
            .unwrap_or(2);
        let exx = combine_exx(&spec, &components, hf_weight)?;
        Ok(Functional {
            spec,
            polarized,
            components,
            dispersion,
            family,
            needs_tau,
            needs_lapl,
            max_order,
            exx,
        })
    }

    /// The spec this functional was built from.
    pub fn spec(&self) -> &str {
        &self.spec
    }
    pub fn polarized(&self) -> bool {
        self.polarized
    }
    pub fn nspin(&self) -> usize {
        if self.polarized {
            2
        } else {
            1
        }
    }
    /// Highest rung among the components (`Hf` if there is no semilocal part).
    pub fn family(&self) -> Family {
        self.family
    }
    /// `sigma` is needed (GGA and meta-GGA).
    pub fn needs_sigma(&self) -> bool {
        self.family >= Family::Gga
    }
    /// `tau` is needed.
    pub fn needs_tau(&self) -> bool {
        self.needs_tau
    }
    /// The density Laplacian is needed.
    pub fn needs_lapl(&self) -> bool {
        self.needs_lapl
    }
    /// Highest available derivative order (min over components; 2 with fxc).
    pub fn max_order(&self) -> usize {
        self.max_order
    }
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    /// All exact-exchange coefficients (libxc convention, see [`ExactExchange`]).
    pub fn exact_exchange(&self) -> ExactExchange {
        self.exx
    }
    /// Fraction of full-range exact exchange: the global hybrid fraction (0.2
    /// for B3LYP); for range-separated hybrids the long-range fraction `alpha`.
    pub fn exx_fraction(&self) -> f64 {
        self.exx.alpha
    }
    /// `(alpha, beta, omega)` if range-separated: K = alpha K + beta K_sr(omega).
    pub fn range_separation(&self) -> Option<(f64, f64, f64)> {
        self.exx
            .is_range_separated()
            .then_some((self.exx.alpha, self.exx.beta, self.exx.omega))
    }
    pub fn is_hybrid(&self) -> bool {
        self.exx.is_hybrid()
    }
    /// VV10 non-local correlation the host must add, if any.
    pub fn vv10(&self) -> Option<Vv10> {
        let mut found: Option<Vv10> = None;
        for c in self.components.iter().filter(|c| c.keep_vv10) {
            if let Some(v) = c.kernel.vv10() {
                // Only one VV10 component is meaningful; sum the weights if the
                // same parameters appear twice (e.g. "0.5*b97m-v + 0.5*b97m-v").
                match &mut found {
                    None => {
                        found = Some(Vv10 {
                            scale: c.weight * v.scale,
                            ..v
                        })
                    }
                    Some(f) if f.b == v.b && f.c == v.c => f.scale += c.weight * v.scale,
                    Some(_) => return found, // documented: first one wins
                }
            }
        }
        found
    }
    /// Dispersion correction the name implies but the kernel does not contain
    /// (`"D3BJ"` for ORCA-style `wB97X-D3BJ`), for the host to add.
    pub fn dispersion(&self) -> Option<&'static str> {
        self.dispersion
    }

    /// Smallest density threshold over components (points with total density
    /// below a component's threshold get zero from that component).
    pub fn dens_threshold(&self) -> f64 {
        self.components
            .iter()
            .map(|c| c.kernel.dens_threshold())
            .fold(f64::INFINITY, f64::min)
    }
    /// Set the density threshold of every component (non-positive values are ignored by libxc).
    pub fn set_dens_threshold(&mut self, threshold: f64) {
        for c in &mut self.components {
            c.kernel.set_dens_threshold(threshold);
        }
    }

    /// Number of grid points in `input`.
    pub fn np(&self, input: &XcInput<'_>) -> usize {
        input.np(self.polarized)
    }

    /// Evaluate on a batch into reusable owned buffers (resized as needed).
    /// `order` 0: `zk`; 1: plus first derivatives; 2: plus second derivatives.
    /// See [`XcOutput`] for the energy convention and array layout.
    pub fn eval(&self, order: usize, input: &XcInput<'_>, out: &mut XcOutput) -> Result<()> {
        let np = input.validate(
            self.polarized,
            self.family.max(Family::Lda),
            self.needs_tau,
            self.needs_lapl,
        )?;
        out.prepare(
            order,
            np,
            self.polarized,
            self.family.max(Family::Lda),
            self.needs_tau,
            self.needs_lapl,
        );
        self.eval_into(order, input, &mut out.refs_mut())
    }

    /// Evaluate into caller-provided arrays (see [`XcOutRefs`] for the contract).
    pub fn eval_into(
        &self,
        order: usize,
        input: &XcInput<'_>,
        out: &mut XcOutRefs<'_>,
    ) -> Result<()> {
        if order > self.max_order {
            return Err(XcError::Order {
                name: self.spec.clone(),
                order,
                max: self.max_order,
            });
        }
        // A pure-HF functional still fills zk/vrho/v2rho2 with zeros.
        let family = self.family.max(Family::Lda);
        let np = input.validate(self.polarized, family, self.needs_tau, self.needs_lapl)?;
        out.validate(
            order,
            np,
            self.polarized,
            family,
            self.needs_tau,
            self.needs_lapl,
        )?;
        let wanted =
            |o: Out| o.order() <= order && o.produced_by(family, self.needs_tau, self.needs_lapl);
        let made_by = |k: &dyn XcKernel, o: Out| {
            o.order() <= order && o.produced_by(k.family(), k.needs_tau(), k.needs_lapl())
        };

        let Some((first, rest)) = self.components.split_first() else {
            for o in Out::ALL.into_iter().filter(|&o| wanted(o)) {
                out.get_mut(o).unwrap().fill(0.0);
            }
            return Ok(());
        };

        // First component straight into the outputs: no copy for the common
        // single-kernel case (B3LYP, PBE0, wB97X-V, ...).
        first.kernel.eval(order, input, out)?;
        for o in Out::ALL.into_iter().filter(|&o| wanted(o)) {
            let buf = out.get_mut(o).unwrap();
            if !made_by(&*first.kernel, o) {
                buf.fill(0.0);
            } else if first.weight != 1.0 {
                buf.iter_mut().for_each(|x| *x *= first.weight);
            }
        }

        // Further components through per-thread scratch, then out += w * scratch.
        // The buffer is taken out of the thread-local so re-entrant kernels are fine.
        let mut scratch = SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()));
        for c in rest {
            let k = &*c.kernel;
            let sizes: [usize; N_OUT] = Out::ALL.map(|o| {
                if made_by(k, o) {
                    np * o.dim(self.polarized)
                } else {
                    0
                }
            });
            let total: usize = sizes.iter().sum();
            if scratch.len() < total {
                scratch.resize(total, 0.0);
            }
            let mut refs = XcOutRefs::new();
            let mut tail: &mut [f64] = &mut scratch[..total];
            for o in Out::ALL {
                let (head, t) = std::mem::take(&mut tail).split_at_mut(sizes[o as usize]);
                tail = t;
                if !head.is_empty() {
                    refs.set(o, head);
                }
            }
            k.eval(order, input, &mut refs)?;
            for o in Out::ALL.into_iter().filter(|&o| made_by(k, o)) {
                let src = refs.get(o).unwrap();
                let dst = out.get_mut(o).unwrap();
                let w = c.weight;
                dst.iter_mut().zip(src).for_each(|(d, s)| *d += w * s);
            }
        }
        SCRATCH.with(|s| *s.borrow_mut() = scratch);
        Ok(())
    }
}

thread_local! {
    static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

impl Clone for Functional {
    /// Re-initializes every component (same parameters and thresholds). Not
    /// needed for threading (`Functional` is `Sync`), but handy to change
    /// thresholds per copy.
    fn clone(&self) -> Self {
        let components = self
            .components
            .iter()
            .map(|c| Component {
                weight: c.weight,
                kernel: c
                    .kernel
                    .try_clone()
                    .expect("re-initializing a functional that initialized before"),
                keep_vv10: c.keep_vv10,
            })
            .collect();
        Functional {
            components,
            spec: self.spec.clone(),
            ..*self
        }
    }
}

/// Combine exact-exchange coefficients of all components and HF terms.
fn combine_exx(spec: &str, components: &[Component], hf_weight: f64) -> Result<ExactExchange> {
    let mut exx = ExactExchange {
        alpha: hf_weight,
        ..Default::default()
    };
    for c in components {
        let e = c.kernel.exact_exchange();
        exx.alpha += c.weight * e.alpha;
        if e.beta != 0.0 {
            if exx.beta != 0.0 && (exx.omega != e.omega || exx.kernel != e.kernel) {
                return Err(XcError::Unsupported(format!(
                    "'{spec}' mixes range-separated components with different omega or kernel"
                )));
            }
            exx.beta += c.weight * e.beta;
            exx.omega = e.omega;
            exx.kernel = e.kernel;
        }
    }
    if exx.beta == 0.0 {
        exx.omega = 0.0;
        exx.kernel = RangeSeparation::None;
    }
    Ok(exx)
}
