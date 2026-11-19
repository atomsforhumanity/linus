//! SCF loop strategies: the host loop ([`super::run_scf`] over a J/K builder and an XC term) and
//! the device loop (`linus::backend::gpu::scf_loop`, feature `gpu`). The driver runs a loop with
//! the environment term it opens for the SCF (the solvent's reaction field,
//! `solvent::ScfEnvironment`); the loop does not know where the term comes from.

use super::{run_scf, ScfConfig, ScfInput, ScfResult};
use faer::Par;
use linus_base::error::LinusError;
use linus_base::terms::JkBuilder;
use linus_base::terms::{CoulombHook, HookOutcome};
use linus_base::terms::{EnvironmentTerm, XcPotential};

/// An SCF loop with its J/K and XC terms; the environment term comes with each run.
pub trait ScfLoop {
    /// Install (Some) or remove (None) a Coulomb hook on the loop's density-fitted J build (the
    /// PCM's fit hand-off; `FellBack` with the reason when the loop has no such build or its setup
    /// did not fit).
    fn set_coulomb_hook(
        &mut self,
        hook: Option<Box<dyn CoulombHook>>,
    ) -> Result<HookOutcome, LinusError>;
    /// The SCF with the environment term `env` (None: gas phase).
    fn run(&mut self, env: Option<&mut dyn EnvironmentTerm>) -> Result<ScfResult, LinusError>;
}

/// The host loop over a J/K builder and an XC term.
pub struct HostLoop<'a, 'b> {
    pub inp: &'a ScfInput<'a>,
    pub jk: &'a mut dyn JkBuilder,
    pub xc: Option<&'b mut dyn XcPotential>,
    pub opts: &'a ScfConfig,
    pub par: Par,
}

impl ScfLoop for HostLoop<'_, '_> {
    fn set_coulomb_hook(
        &mut self,
        hook: Option<Box<dyn CoulombHook>>,
    ) -> Result<HookOutcome, LinusError> {
        self.jk.set_coulomb_hook(hook)
    }

    fn run(&mut self, env: Option<&mut dyn EnvironmentTerm>) -> Result<ScfResult, LinusError> {
        let xc = self.xc.as_deref_mut().map(|x| x as _);
        run_scf(self.inp, self.jk, xc, env, self.opts, self.par)
    }
}
