//! What each phase of a calculation holds in host memory at its peak, and the smallest
//! `memory_gib` that runs it: one function for the plan's `memory.required_gib` and for the
//! failure before any work (`host_memory` with `details.required_gib`), so the two agree.
//!
//! `memory_gib` limits the whole process (linus_base::memory's module docs). A phase's need is
//! the process's baseline ([`linus_base::memory::Process`], a device's host part), the
//! reservations of the SCF's large allocations (the PCM's), the parts its engines size from the
//! problem with their own functions (the SCF's DF tensors and exchange workspace,
//! [`linus_jk::df::DfJk::scf_storage_needs`]; the DF gradient's blocks,
//! [`linus_jk::df::gradient::df_gradient_workspace`]; a response's fixed arrays,
//! `CalcContext::response_memory`), the budgets that are a share of the limit (the Hessian's H-bar
//! blocks, the response's exchange slots, the XC kernel's accumulators) and [`RestModel`]: the
//! measured matrices, XC buffers, per-thread buffers and allocator slack of the phase. With
//! density-fitted exchange the SCF's tensors also keep their own rule (at most 75% of the limit
//! less the reservations), so its requirement is the larger of the two.

use crate::derivatives::capabilities::Request;
use crate::driver::backend::{Family, Placement};
use crate::driver::{CalcContext, JkMethod, LinusError, Purpose};
use crate::two_electron::ResponseUse;
use linus_base::error::MemoryNeed;
use linus_base::memory::{gib_holding, Part, PhaseNeed, DEVICE_HOST_BYTES};
use linus_ints::DerivOrder;

const GIB: f64 = 1073741824.0;

/// The measured part of a phase's need beside the parts sized from the problem: the matrices of
/// the basis (bytes per nbf^2 f64 matrix and spin channel), the buffers of each thread and a
/// fixed part (small arrays and the allocator's slack: glibc keeps freed blocks of up to 32 MiB
/// in its arenas, 0.02-0.15 GiB in the measurements). Fitted to the peaks of the MEMFIT spread
/// (bench/memfit/README.md) with a margin.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RestModel {
    pub per_matrix: f64,
    pub per_thread: usize,
    pub fixed: usize,
}

impl RestModel {
    /// Bytes for `nbf` basis functions, `nch` spin channels and `threads` threads.
    pub fn bytes(&self, nbf: usize, nch: usize, threads: usize) -> usize {
        let s = (nbf * nbf * 8) as f64;
        (self.per_matrix * s * nch as f64) as usize + self.per_thread * threads + self.fixed
    }
}

/// The SCF's: the SCF's matrices and accelerators (DIIS), the XC grid and its buffers.
pub(crate) const SCF_REST: RestModel = RestModel {
    per_matrix: 60.0,
    per_thread: 32 << 20,
    fixed: 96 << 20,
};

/// The gradient's: the derivative pair sets and the 3-center derivative layout (large with f
/// functions: ~0.3 GiB for caffeine/def2-TZVP), the reference's matrices, the XC gradient.
pub(crate) const GRADIENT_REST: RestModel = RestModel {
    per_matrix: 340.0,
    per_thread: 32 << 20,
    fixed: 192 << 20,
};

/// A response's (Hessian, polarizability): the response J/K's term-1 blocks, the explicit
/// terms' derivative blocks, the CPHF/CPKS subspace, the XC kernel.
pub(crate) const RESPONSE_REST: RestModel = RestModel {
    per_matrix: 120.0,
    per_thread: 40 << 20,
    fixed: 448 << 20,
};

/// A share of the limit a phase's engine takes, up to a cap (bytes): `cap` and `fraction`.
type Share = (usize, f64);

/// The smallest limit `m` with `fixed + sum min(cap, fraction m) <= m` (the shares grow with
/// the limit at slopes below one, so it exists and is unique).
fn holding(fixed: usize, shares: &[Share]) -> usize {
    let need = |m: f64| {
        fixed as f64
            + shares
                .iter()
                .map(|&(cap, f)| (cap as f64).min(f * m))
                .sum::<f64>()
    };
    let (mut lo, mut hi) = (
        fixed as f64,
        fixed as f64 + shares.iter().map(|s| s.0 as f64).sum::<f64>(),
    );
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if need(mid) <= mid {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi.ceil() as usize
}

impl CalcContext {
    /// The process's baseline: its kind's ([`linus_base::memory::Process`]) and a device's host part.
    pub(crate) fn baseline_part(&self) -> Part {
        let device = self.backend.device().is_some();
        Part {
            what: self.process.name(),
            bytes: self.process.baseline_bytes() + if device { DEVICE_HOST_BYTES } else { 0 },
        }
    }

    fn nch_occ(&self) -> Result<(usize, [usize; 2]), LinusError> {
        let (na, nb) = self.mol.nalpha_nbeta().map_err(LinusError::Input)?;
        let restricted = self.plan.restricted().unwrap_or(na == nb);
        Ok((if restricted { 1 } else { 2 }, [na, nb]))
    }

    /// The Coulomb metric's Cholesky factor of the J/K's auxiliary basis for `purpose` (kept for
    /// the geometry), when the J/K fits.
    fn metric_part(&self, purpose: Purpose) -> Result<Option<Part>, LinusError> {
        Ok(match self.plan.jk().aux() {
            Some(a) => {
                let naux = self.aux_set(a, purpose)?.nbf;
                Some(Part {
                    what: "DF metric",
                    bytes: naux * naux * 8,
                })
            }
            None => None,
        })
    }

    /// The SCF's need beside its DF tensors and exchange workspace: the baseline, the
    /// reservations `reserved`, the metric and [`SCF_REST`].
    pub(crate) fn scf_fixed_parts(
        &self,
        purpose: Purpose,
        reserved: &[linus_base::memory::Reservation],
    ) -> Result<Vec<Part>, LinusError> {
        let (nch, _) = self.nch_occ()?;
        let mut parts = vec![self.baseline_part()];
        parts.extend(reserved.iter().map(|r| Part {
            what: "reserved (pcm)",
            bytes: r.bytes,
        }));
        parts.extend(self.metric_part(purpose)?);
        parts.push(Part {
            what: "SCF matrices, XC, threads, allocator",
            bytes: SCF_REST.bytes(self.basis.nbf, nch, self.plan.threads()),
        });
        Ok(parts)
    }

    /// The SCF's need for `purpose` (see the module docs): with density-fitted exchange on the
    /// CPU, the storage the run takes at this limit (the one whose limit is smallest when none
    /// fits), its requirement the smallest over the storages allowed.
    fn scf_need(&self, purpose: Purpose) -> Result<PhaseNeed, LinusError> {
        let reserved = self.scf_memory_reservations()?;
        let r: usize = reserved.iter().map(|x| x.bytes).sum();
        let mut parts = self.scf_fixed_parts(purpose, &reserved)?;
        let fixed: usize = parts.iter().map(|p| p.bytes).sum();
        let df_k = matches!(self.plan.jk(), JkMethod::Df { .. })
            && self.method.exchange.is_some()
            && self.placement(Family::Jk) != Some(Placement::Device);
        if matches!(self.plan.jk(), JkMethod::Exact)
            && self.plan.exact_storage() == crate::driver::ExactStorage::Incore
        {
            // the in-core four-index integrals: within the tensors' budget and the whole process
            let t = linus_jk::IncoreJk::bytes_needed(self.basis.nbf, self.method.omega().is_some());
            parts.push(Part {
                what: "in-core integrals",
                bytes: t,
            });
            let mut need = PhaseNeed::new("scf", parts);
            need.required = scf_df_required(t, 0, r, fixed);
            return Ok(need);
        }
        if !df_k {
            return Ok(PhaseNeed::new("scf", parts));
        }
        let JkMethod::Df { aux } = self.plan.jk() else {
            unreachable!()
        };
        let auxb = self.aux_set(aux, purpose)?;
        let pairs = self.integrals.pairs(&self.basis, DerivOrder::Zero);
        let (_, occ) = self.nch_occ()?;
        let cfg = linus_jk::df::DfConfig {
            scf_occ: occ,
            ..self.df_config(self.method.omega(), 0)
        };
        let options = linus_jk::df::DfJk::scf_storage_needs(
            &self.basis,
            &pairs,
            &auxb,
            &cfg,
            self.plan.threads(),
        );
        let limit = self.plan.memory().bytes();
        let budget = limit.map(|m| (m / 4 * 3).saturating_sub(r));
        let req = |&(t, ws): &(usize, usize)| scf_df_required(t, ws, r, fixed);
        let fits = |o: &(usize, usize)| {
            limit.is_some_and(|m| budget.is_some_and(|b| o.0 <= b) && req(o) <= m)
        };
        let pick = options
            .iter()
            .find(|o| fits(o))
            .or_else(|| options.iter().min_by_key(|o| req(o)))
            .copied();
        let required = options.iter().map(req).min();
        if let Some((t, ws)) = pick {
            parts.push(Part {
                what: "DF tensors",
                bytes: t,
            });
            parts.push(Part {
                what: "DF exchange blocks",
                bytes: ws,
            });
        }
        let mut need = PhaseNeed::new("scf", parts);
        if let Some(req) = required {
            need.required = req;
        }
        Ok(need)
    }

    /// The gradient's need (SCF purpose `purpose`'s ground state): the baseline, the metric, the
    /// DF gradient's blocks (exchange on the CPU) and [`GRADIENT_REST`].
    fn gradient_need(&self, purpose: Purpose) -> Result<PhaseNeed, LinusError> {
        let (nch, occ) = self.nch_occ()?;
        let mut parts = vec![self.baseline_part()];
        parts.extend(self.metric_part(purpose)?);
        if let (JkMethod::Df { aux }, Some(xop), true) = (
            self.plan.jk(),
            self.method.exchange,
            self.placement(Family::GradientJk) != Some(Placement::Device),
        ) {
            let auxb = self.aux_set(aux, purpose)?;
            let pairs = self.integrals.pairs(&self.basis, DerivOrder::Zero);
            // the rows of the significant shell pairs (an upper bound of the DF screening's)
            let nx: usize = (0..pairs.len())
                .map(|i| pairs.pair(i))
                .filter(|p| p.nprim > 0)
                .map(|p| self.basis.shells[p.sa].nfunc() * self.basis.shells[p.sb].nfunc())
                .sum();
            let occ: &[usize] = if nch == 1 { &occ[..1] } else { &occ };
            parts.push(Part {
                what: "DF gradient blocks",
                bytes: linus_jk::df::gradient::df_gradient_workspace(
                    self.basis.nbf,
                    &auxb.shells,
                    nx,
                    occ,
                    true,
                    xop.has_erf(),
                ),
            });
        }
        parts.push(Part {
            what: "gradient matrices, XC, threads, allocator",
            bytes: GRADIENT_REST.bytes(self.basis.nbf, nch, self.plan.threads()),
        });
        Ok(PhaseNeed::new("gradient", parts))
    }

    /// A response's need: the baseline, the metric, its fixed arrays
    /// (`CalcContext::response_memory`), its shares of the limit (a Hessian's H-bar blocks with
    /// density-fitted exchange, the exchange slots, the XC kernel's accumulators) and
    /// [`RESPONSE_REST`].
    fn response_need(&self, what: ResponseUse) -> Result<PhaseNeed, LinusError> {
        let (nch, occ) = self.nch_occ()?;
        let mem = self.plan.memory();
        let mut parts = vec![self.baseline_part()];
        parts.extend(self.metric_part(Purpose::Hessian)?);
        parts.push(Part {
            what: "response arrays",
            bytes: self.response_memory(what)? as usize,
        });
        parts.push(Part {
            what: "response blocks, XC kernel, threads, allocator",
            bytes: RESPONSE_REST.bytes(self.basis.nbf, nch, self.plan.threads()),
        });
        let fixed: usize = parts.iter().map(|p| p.bytes).sum();
        let df_k = matches!(self.plan.jk(), JkMethod::Df { .. }) && self.method.exchange.is_some();
        let mut shares: Vec<(&'static str, Share)> = vec![];
        if df_k {
            if what == ResponseUse::Hessian {
                // H-bar: at most all of it, a quarter of it or 1500 MB, a quarter of the limit
                let naux = match self.plan.jk().aux() {
                    Some(a) => self.aux_set(a, Purpose::Hessian)?.nbf,
                    None => 0,
                };
                let ot: usize = occ[..nch].iter().sum();
                let all = 3 * self.basis.nbf * ot * naux * 8;
                let want = (all / 4).max(1_500_000_000).min(all);
                shares.push(("H-bar blocks", (want, 0.25)));
            }
            shares.push(("response exchange slots", (1 << 30, 0.125)));
        }
        if self.method.functional.is_some() {
            shares.push(("XC kernel accumulators", (1 << 30, 0.125)));
        }
        let required = holding(fixed, &shares.iter().map(|s| s.1).collect::<Vec<_>>());
        let at = mem.bytes().unwrap_or(required).max(required);
        parts.extend(shares.iter().map(|&(w, (cap, f))| Part {
            what: w,
            bytes: (cap as f64).min(f * at as f64) as usize,
        }));
        let name = match what {
            ResponseUse::Hessian => "hessian",
            ResponseUse::Polarizability => "polarizability",
        };
        let mut need = PhaseNeed::new(name, parts);
        need.required = required;
        Ok(need)
    }

    /// The needs of the phases of `req`, in the order they run.
    pub(crate) fn memory_needs(&self, req: Request) -> Result<Vec<PhaseNeed>, LinusError> {
        let purpose = req.purpose();
        let mut out = vec![self.scf_need(purpose)?];
        match req {
            Request::Energy => {}
            Request::Gradient => out.push(self.gradient_need(purpose)?),
            Request::Hessian { dipole_derivatives } => {
                out.push(self.response_need(ResponseUse::Hessian)?);
                if dipole_derivatives {
                    out.push(self.response_need(ResponseUse::Polarizability)?);
                }
            }
            Request::Polarizability => out.push(self.response_need(ResponseUse::Polarizability)?),
        }
        Ok(out)
    }

    /// The memory the calculation may hold holds every phase of `req` (checked before any
    /// work): `memory_gib`, else the machine's limit within 90% of the memory free now
    /// ([`linus_base::memory::MemoryLimit::run_limit`]: the checked limit only, never a budget,
    /// so no number and no storage follows it). A `host_memory` failure names the `memory_gib`
    /// that holds the calculation (the largest phase's requirement, as the plan's
    /// `memory.required_gib`) otherwise, instead of an out-of-memory kill later.
    pub(crate) fn check_memory(&self, req: Request) -> Result<(), LinusError> {
        match self.plan.memory().run_limit() {
            Some((limit, source)) => self.check_needs(req, limit, source),
            None => Ok(()),
        }
    }

    /// [`CalcContext::check_memory`] against `limit` bytes from `source`.
    fn check_needs(&self, req: Request, limit: usize, source: &str) -> Result<(), LinusError> {
        let needs = self.memory_needs(req)?;
        let Some(top) = needs.iter().max_by_key(|n| n.required()) else {
            return Ok(());
        };
        if top.required() <= limit {
            return Ok(());
        }
        // the SCF's tensors: the ways out of RI-K's 3-index tensor
        let ways = if top.phase == "scf" && top.parts.iter().any(|p| p.what == "DF tensors") {
            let compressed = if self.plan.df_storage() == linus_jk::df::DfPrecision::Auto {
                "jk.df_storage \"f48\" (compressed: relative error at most 9e-13; auto never chooses it), "
            } else {
                ""
            };
            format!(
                ". Other ways: {compressed}jk \"rij_snk\" (--jk rij-snk: RI-J with seminumerical exchange, no \
                 3-index exchange tensor)"
            )
        } else {
            String::new()
        };
        Err(LinusError::host_memory(
            format!(
                "the {} of this calculation needs {:.2} GiB ({}), more than the {:.2} GiB of the {}: memory_gib {:.1} \
                 or more holds every phase (--memory on the command line){ways}",
                top.phase,
                top.required() as f64 / GIB,
                top.describe(),
                limit as f64 / GIB,
                source,
                gib_holding(top.required()),
            ),
            MemoryNeed {
                required_gib: Some(top.required() as f64 / GIB),
                available_gib: Some(limit as f64 / GIB),
            },
        ))
    }
}

/// The smallest limit for DF tensors of `tensor` bytes with an exchange workspace of
/// `workspace` bytes beside `fixed` bytes of the SCF's other parts and `reserved` bytes reserved
/// (part of `fixed`): the tensors within 75% of the limit less the reservations, and the whole
/// within the limit.
pub(crate) fn scf_df_required(
    tensor: usize,
    workspace: usize,
    reserved: usize,
    fixed: usize,
) -> usize {
    let by_budget = ((tensor + reserved) as f64 * 4.0 / 3.0).ceil() as usize;
    by_budget.max(fixed + tensor + workspace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::report::CalcType;
    use crate::driver::{CalcOptions, Session};
    use crate::Structure;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.47\nH 0.0 -0.76 -0.47\n";

    /// The shares grow with the limit at slopes below one: the solution holds exactly.
    #[test]
    fn the_smallest_limit_holding_the_shares() {
        assert_eq!(holding(100, &[]), 100);
        // 100 + m/4 <= m: m = 400/3
        let m = holding(100, &[(1000, 0.25)]);
        assert!((m as f64 - 400.0 / 3.0).abs() <= 1.0, "{m}");
        // a capped share: 100 + 10 <= m
        assert_eq!(holding(100, &[(10, 0.5)]), 110);
    }

    /// The plan's `memory.required_gib` is what the failure below it reports (the same
    /// function), and at it the calculation runs.
    #[test]
    fn the_plan_states_what_the_failure_reports() {
        for (method, calc) in [
            ("wb97x-d3bj", CalcType::Energy),
            ("wb97x-d3bj", CalcType::Gradient),
            ("b3lyp", CalcType::Hessian),
        ] {
            let opts = |gib: f64| {
                CalcOptions::from_json(serde_json::json!({
                    "method": method, "basis": "def2-svp", "threads": 1, "memory_gib": gib
                }))
                .unwrap()
            };
            let mol = || Structure::from_xyz_str(WATER, 0, 1).unwrap();
            let plan = Session::new(mol(), opts(64.0)).unwrap().plan(calc).unwrap();
            let req = plan.memory.required_gib;
            assert!(req > 0.1 && !plan.memory.phases.is_empty(), "{req}");
            let Err(e) = Session::new(mol(), opts(req - 0.01)).unwrap().run(calc) else {
                panic!("{method} {calc:?} ran below its requirement {req}");
            };
            assert_eq!(e.code(), Some("host_memory"), "{e}");
            let need = e.memory_need().unwrap();
            assert_eq!(need.required_gib, Some(req), "{method} {calc:?}: {e}");
            Session::new(mol(), opts(req)).unwrap().run(calc).unwrap();
        }
    }

    /// Without `memory_gib` the same check runs against the memory the calculation may hold
    /// (the machine's limit within 90% of the free memory): at a small one it fails before any
    /// work with the requirement the plan states; at today's machine it passes.
    #[test]
    fn without_memory_gib_the_free_memory_is_checked() {
        let opts = CalcOptions::from_json(serde_json::json!({
            "method": "wb97x-d3bj", "basis": "def2-svp", "threads": 1
        }))
        .unwrap();
        let s = Session::new(Structure::from_xyz_str(WATER, 0, 1).unwrap(), opts).unwrap();
        let req = Request::Gradient;
        let need =
            s.cx.memory_needs(req)
                .unwrap()
                .iter()
                .map(|n| n.required())
                .max()
                .unwrap();
        let e =
            s.cx.check_needs(req, need - 1, "memory available")
                .unwrap_err();
        assert_eq!(e.code(), Some("host_memory"), "{e}");
        let plan = s.plan(CalcType::Gradient).unwrap();
        assert_eq!(
            e.memory_need().unwrap().required_gib,
            Some(plan.memory.required_gib)
        );
        assert!(s.cx.check_needs(req, need, "memory available").is_ok());
        assert!(s.cx.check_memory(req).is_ok());
    }

    /// Below what every phase needs the calculation fails before any work, naming the
    /// `memory_gib` that holds it; at that it runs: an energy and a gradient of a
    /// range-separated hybrid with density fitting, and a Hessian.
    #[test]
    fn below_the_need_the_failure_names_the_memory() {
        for (method, calc) in [
            ("wb97x-d3bj", CalcType::Energy),
            ("wb97x-d3bj", CalcType::Gradient),
            ("b3lyp", CalcType::Hessian),
        ] {
            let opts = |gib: f64| {
                CalcOptions::from_json(serde_json::json!({
                    "method": method, "basis": "def2-svp", "threads": 1, "memory_gib": gib
                }))
                .unwrap()
            };
            let mol = || Structure::from_xyz_str(WATER, 0, 1).unwrap();
            let Err(e) = Session::new(mol(), opts(0.05)).unwrap().run(calc) else {
                panic!("{method} {calc:?} ran in 0.05 GiB");
            };
            assert_eq!(e.code(), Some("host_memory"), "{e}");
            let req = e.memory_need().unwrap().required_gib.unwrap();
            assert!(req > 0.05, "{method} {calc:?}: {e}");
            Session::new(mol(), opts(req)).unwrap().run(calc).unwrap();
        }
    }
}
