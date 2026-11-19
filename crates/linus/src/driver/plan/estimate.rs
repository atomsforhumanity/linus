//! What a calculation will need from the machine, estimated before it runs: its peak host
//! memory, its device memory and the threads it uses well (the plan's `estimates`; qccompute's
//! `Plan.resources`). A scheduler packs calculations onto CPUs and GPUs by them.
//!
//! Each estimate is a model of the calculation's sizes and kind, fitted to what linus took on its
//! test molecules (water to atorvastatin, def2-SVP and def2-TZVP; HF, GGA, meta-GGA, hybrid and
//! range-separated functionals; closed and open shells; 8 threads), and carries its `band`: the
//! largest relative error over the cases it was fitted on (`cases`), rounded up to 5%, and at
//! least 25% for three cases or fewer. It is an estimate, not a bound: a molecule unlike those
//! can fall outside its band.
//!
//! The sizes, in GiB: `T` the SCF's DF tensors (naux x npair x 8 bytes, the share of the pairs
//! the Schwarz screening keeps falling with the atom count, twice for a range-separated exchange's
//! long-range tensor), `S` a matrix of the basis (nbf^2 x 8 bytes), `D` a dense 3-index DF array
//! (naux x nbf^2 x 8 bytes; the engines block it by fixed budgets, hence the caps), `G` an
//! XC grid's basis values (natoms x nbf x 8 bytes per 33 000 points).
//!
//! - Host, CPU J/K: 0.06 (the process: Python with linus loaded) + T + the calculation's working
//!   arrays (the table in [`EstimateInputs::host`]).
//! - Host, J/K on a device: 0.65 (the process with the CUDA libraries) + the arrays it keeps on
//!   the host (energies 9e-6 natoms nbf, gradients 0.1 + 330 S, Hessians 1900 S, x 1.6 for an
//!   open shell).
//! - Device: 0.54 (the CUDA context and the libraries' handles) + T + G (Hessians also
//!   2.3 + 1.5 D), within the device's limit.
//!
//! With `memory_gib` set the budgets keep the run within it, and the host estimate is at most
//! the limit.

use crate::driver::options::JkMethod;
use crate::driver::report::CalcType;
use serde::Serialize;

const GIB: f64 = 1073741824.0;

/// The threads a calculation uses well: one CCD's 8 cores. From 1 to 8 cores of one CCD the 9
/// requests measured (gradients from water to testosterone/def2-TZVP, Hessians of p-benzyne and
/// caffeine) sped up 5.2-6.5 times, small molecules as much as large ones (the step from 4 to 8
/// cores still gives each core more than half its speed), and the cores' SMT siblings added at
/// most 10% (two Hessians and testosterone ran slower). More cores, across CCDs, were not
/// measured by size; one 8-thread job per CCD gives the best throughput. The plan's
/// `estimates.threads`, and linus's own cap under a runner: a calculation given more threads and
/// no `threads` keyword uses this many (qc* decision 16; Python's `linus.Session`).
pub const THREADS_USED_WELL: usize = 8;

/// One estimate: its value, its relative error band and how it was made.
#[derive(Clone, Debug, Serialize)]
pub struct Estimate {
    /// GiB for memory, a count for threads.
    pub value: f64,
    /// The largest relative error |estimate / measured - 1| over the cases the model was
    /// checked on.
    pub band: f64,
    /// How many cases.
    pub cases: u32,
    /// How the value was made.
    pub model: &'static str,
    /// Always true: an estimate, not a measurement.
    pub estimate: bool,
}

/// The estimates of a plan.
#[derive(Clone, Debug, Serialize)]
pub struct PlannedEstimates {
    /// The process's peak resident memory during the calculation (GiB).
    pub memory_gib: Estimate,
    /// The device memory the calculation takes, with the CUDA context and the libraries'
    /// handles (GiB); absent without a device.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_memory_gib: Option<Estimate>,
    /// The threads the calculation uses well: more add less than half a thread's speed each.
    pub threads: Estimate,
}

/// The exact exchange of a method, as far as memory goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Exchange {
    /// None (semilocal functionals): J only.
    None,
    /// Global (HF, global hybrids).
    Global,
    /// Range-separated: a long-range operator beside 1/r.
    RangeSeparated,
}

/// What the estimates depend on.
pub(crate) struct EstimateInputs<'a> {
    pub calctype: CalcType,
    pub natoms: usize,
    pub nbf: usize,
    /// The DF auxiliary basis set's functions (None without density fitting).
    pub naux: Option<usize>,
    pub jk: &'a JkMethod,
    pub exchange: Exchange,
    pub open_shell: bool,
    /// `memory_gib` (bytes), when set.
    pub memory_limit: Option<usize>,
    /// The device's memory limit (bytes), with a device.
    pub device_limit: Option<usize>,
    /// Whether the SCF's J/K runs on the device.
    pub jk_on_device: bool,
}

/// An estimate's model: its value (GiB), its description and its (band, cases).
type Model = (f64, &'static str, (f64, u32));

impl EstimateInputs<'_> {
    fn s(&self) -> f64 {
        let n = self.nbf as f64;
        n * n * 8.0 / GIB
    }

    fn d(&self) -> f64 {
        self.naux.unwrap_or(0) as f64 * self.s()
    }

    /// The SCF's DF tensors (the module docs' `T`): the share of the packed pairs the
    /// screening keeps was 0.91 at 24 atoms, 0.58 at 76.
    fn t(&self) -> f64 {
        let Some(naux) = self.naux else { return 0.0 };
        let n = self.nbf as f64;
        let kept = (1.1 - 0.0065 * self.natoms as f64).clamp(0.3, 1.0);
        let lr = if self.exchange == Exchange::RangeSeparated {
            2.0
        } else {
            1.0
        };
        naux as f64 * n * (n + 1.0) / 2.0 * 8.0 * kept * lr / GIB
    }

    /// The open-shell factor of a Hessian's response arrays (both spins).
    fn open(&self) -> f64 {
        if self.open_shell {
            1.6
        } else {
            1.0
        }
    }

    /// The peak host memory with the J/K on the CPU.
    fn host_cpu(&self) -> Model {
        let (s, d) = (self.s(), self.d());
        let df = matches!(self.jk, JkMethod::Df { .. });
        let k = self.exchange != Exchange::None;
        let n3s = 3.0 * self.natoms as f64 * s;
        let energy = 0.0096 + 237.0 * s + 0.035 * d.min(1.5);
        let base = 0.06 + self.t();
        match self.calctype {
            CalcType::Energy => (
                base + energy,
                "0.06 + T + 0.0096 + 237 S + 0.035 min(D, 1.5)",
                (0.05, 6),
            ),
            CalcType::Gradient if df && k => (
                base + 0.033 + 0.79 * d.min(1.6) + 26.75 * s,
                "0.06 + T + 0.033 + 0.79 min(D, 1.6) + 27 S",
                (0.2, 13),
            ),
            CalcType::Gradient => (
                base + energy,
                "0.06 + T + 0.0096 + 237 S + 0.035 min(D, 1.5)",
                (0.1, 6),
            ),
            CalcType::Hessian if !df => (
                base + 0.83 + 53.2 * n3s * self.open(),
                "0.06 + 0.83 + 53 x 3 natoms S",
                (0.25, 3),
            ),
            CalcType::Hessian => (
                base + 0.042 + (if k { 1.46 } else { 0.73 } * d.min(0.7) + 897.0 * s) * self.open(),
                "0.06 + T + 0.042 + (1.46 min(D, 0.7) (0.73 without exact exchange) + 897 S) \
                 (x 1.6 for an open shell)",
                (0.25, 11),
            ),
        }
    }

    /// The peak host memory with the J/K on a device.
    fn host_device(&self) -> Model {
        let na = self.natoms as f64 * self.nbf as f64;
        match self.calctype {
            CalcType::Energy => (0.65 + 9e-6 * na, "0.65 + 9e-6 natoms nbf", (0.25, 3)),
            CalcType::Gradient => (0.75 + 330.0 * self.s(), "0.75 + 330 S", (0.15, 10)),
            CalcType::Hessian => (
                0.65 + 1900.0 * self.s() * self.open(),
                "0.65 + 1900 S (x 1.6 for an open shell)",
                (0.25, 3),
            ),
        }
    }

    /// The device memory, with a device: what the calculation takes with room to spare (the
    /// XC grid's cache grows into free memory, within a budget the model does not know, so
    /// the largest molecules take less: atorvastatin/def2-TZVP 27 GiB for 47 estimated).
    fn device(&self) -> Option<Model> {
        let limit = self.device_limit? as f64 / GIB;
        let g = 33e3 * self.natoms as f64 * self.nbf as f64 * 8.0 / GIB;
        let ctx = 0.54;
        let (v, model, band) = match self.calctype {
            CalcType::Energy => (ctx + self.t() + g, "0.54 + T + G", (0.75, 3)),
            CalcType::Gradient => (ctx + self.t() + g, "0.54 + T + G", (0.3, 10)),
            CalcType::Hessian => (
                ctx + self.t() + g + 2.3 + 1.5 * self.d(),
                "0.54 + T + G + 2.3 + 1.5 D",
                (0.25, 3),
            ),
        };
        Some((v.min(ctx + limit), model, band))
    }

    /// The threads used well ([`THREADS_USED_WELL`]).
    fn threads(&self) -> Model {
        (THREADS_USED_WELL as f64, "8: one CCD's cores", (0.0, 9))
    }

    /// Every estimate.
    pub(crate) fn estimates(&self) -> PlannedEstimates {
        let est = |(value, model, (band, cases)): Model| Estimate {
            value,
            band,
            cases,
            model,
            estimate: true,
        };
        let mut memory = est(if self.jk_on_device {
            self.host_device()
        } else {
            self.host_cpu()
        });
        if let Some(limit) = self.memory_limit {
            memory.value = memory.value.min(limit as f64 / GIB);
        }
        PlannedEstimates {
            memory_gib: memory,
            device_memory_gib: self.device().map(est),
            threads: est(self.threads()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::options::AuxBasis;

    fn caffeine(calctype: CalcType, jk: &JkMethod) -> EstimateInputs<'_> {
        EstimateInputs {
            calctype,
            natoms: 24,
            nbf: 246,
            naux: Some(1242),
            jk,
            exchange: Exchange::Global,
            open_shell: false,
            memory_limit: None,
            device_limit: None,
            jk_on_device: false,
        }
    }

    /// The models reproduce measured cases within their bands (caffeine B3LYP/def2-SVP on 8
    /// threads: 0.475, 0.867 and 1.544 GiB of host memory; the gradient's 2.672 GiB on an RTX
    /// PRO 6000), `memory_gib` caps the host estimate, and only a device has a device estimate.
    #[test]
    fn estimates_are_within_their_bands() {
        let df = JkMethod::Df {
            aux: AuxBasis::Auto,
        };
        for (c, measured) in [
            (CalcType::Energy, 0.475),
            (CalcType::Gradient, 0.867),
            (CalcType::Hessian, 1.544),
        ] {
            let e = caffeine(c, &df).estimates();
            let m = &e.memory_gib;
            assert!(m.estimate && m.cases > 0);
            assert!(
                (m.value / measured - 1.0).abs() <= m.band,
                "{c:?}: {}",
                m.value
            );
            assert!(e.device_memory_gib.is_none() && e.threads.value == 8.0);
        }
        let capped = EstimateInputs {
            memory_limit: Some(1 << 29),
            ..caffeine(CalcType::Hessian, &df)
        };
        assert_eq!(capped.estimates().memory_gib.value, 0.5);
        let gpu = EstimateInputs {
            device_limit: Some(8 << 30),
            jk_on_device: true,
            ..caffeine(CalcType::Gradient, &df)
        };
        let d = gpu.estimates().device_memory_gib.unwrap();
        assert!((d.value / 2.672 - 1.0).abs() <= d.band, "{}", d.value);
    }
}
