//! The device's cost models:
//! - decision D37's exchange costing on a device with little FP64 throughput: `precision.response:
//!   auto`'s choice between the stored exchange operator in FP64 and the FP32 exchange through
//!   the tensor, from the operator's plan as linus-gpu's response engine gives it
//!   (`linus_gpu::response::df::KOperatorPlan`) and the device's rates in the device policy
//!   ([`ExchangeCosting`], set up for the plan's `auto` mode by the response's recipe,
//!   `crate::backend::gpu::df::ResponseRecipe`);
//! - the DF-J/K storage ladder's rungs below the device-resident tensor with the automatic K block
//!   ([`df_shrunk_cost`], [`df_stream_cost`]): the tensor kept on the device with a smaller K block,
//!   or streamed from page-locked host memory.

use crate::driver::policy::ExchangeRates;
use crate::driver::report::ExchangeEstimate;

/// The cost model of `precision.response: "auto"`'s exchange path on devices with little FP64
/// throughput (decision D37): the stored operator in FP64 (its setup, in cuBLAS's emulation
/// under `precision.fp64` emulated, the card's auto, and its products; on the device, or
/// streamed from host memory as far as needed) against the FP32 exchange through the tensor in
/// every product. Both are far more accurate than the response needs (against the operator,
/// the FP32 exchange moves frequencies by 2e-5 to 1e-4 cm^-1 with def2-SVP, 7e-4 cm^-1 at
/// u1an39/def2-TZVP), so the cheaper runs. With an IR run's electric-field solve
/// to follow, the operator serves it too (three perturbations a product), while after the FP32
/// exchange the field solve takes its own J/K in double precision through the tensor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExchangeCosting {
    /// The device's rates (the device policy's).
    pub rates: ExchangeRates,
    /// The operator's setup in cuBLAS's FP64 emulation (else FP64).
    pub setup_emulated: bool,
    /// An IR run's electric-field solve follows the Hessian's.
    pub field: bool,
}

/// A Hessian's perturbation-products per perturbation over its response: the right-hand
/// sides' product and the CPKS iterations' (the unconverged perturbations of each), measured
/// 5.7-6.4 (la18, ts61, u1an39 with def2-SVP and def2-TZVP; B3LYP at the default tolerance;
/// decision D37, docs/internal/gpu-spike.md, section AUTOPREC).
const HESSIAN_PASSES: f64 = 6.0;

/// The products of three perturbations of an IR run's electric-field solve (10-15 iterations;
/// 12.3 at u1an39/def2-SVP).
const FIELD_PRODUCTS: f64 = 12.5;

impl ExchangeCosting {
    /// The estimated seconds of the response's exchange on the operator `plan` describes (its
    /// flops and bytes for products of `nd` perturbations, its part in host memory) and
    /// through the tensor in FP32.
    pub(crate) fn estimate(
        &self,
        plan: &linus_gpu::response::df::KOperatorPlan,
        nd: usize,
    ) -> ExchangeEstimate {
        let r = &self.rates;
        let host = plan.host_bytes as f64;
        let mut tensor = HESSIAN_PASSES * plan.tensor_flops / r.fp32_tensor;
        let setup_rate = if self.setup_emulated {
            r.setup_emulated
        } else {
            r.fp64
        };
        // a streamed operator: its host chunks copied out once during the setup (7 % of ts61's
        // setup with 22 of its 41 GiB on the host), copied back behind every product's GEMMs
        let setup = plan.setup_flops / setup_rate + host / r.pcie;
        let product = (plan.product_flops / r.fp64).max(host / r.pcie);
        let mut op = setup + HESSIAN_PASSES * product;
        if self.field && nd > 0 {
            // three perturbations a product: the operator read at the skinny GEMM's rate, the
            // tensor path's flops in FP64 (both linear in the perturbations)
            let per = 3.0 / nd as f64;
            op += FIELD_PRODUCTS
                * (plan.product_flops * per / r.fp64)
                    .max(plan.bytes as f64 / r.skinny)
                    .max(host / r.pcie);
            tensor += FIELD_PRODUCTS * plan.tensor_flops * per / r.field_tensor;
        }
        ExchangeEstimate {
            operator: op,
            tensor,
        }
    }
}

/// `auto`'s exchange path on a device with little FP64 throughput ([`ExchangeCosting`]).
#[derive(Clone, Debug)]
pub(crate) struct AutoChoice {
    /// The exchange runs in FP32 through the tensor (cheaper, or the operator does not fit or
    /// pay), else on the stored operator in FP64.
    pub single: bool,
    /// Why (`resolved.record.backend.response[].reason`).
    pub reason: String,
    /// The two estimates, when both paths were candidates.
    pub estimate: Option<ExchangeEstimate>,
}

/// Seconds for a reason string.
fn secs(x: f64) -> String {
    if x >= 10.0 {
        format!("{x:.0} s")
    } else if x >= 1.0 {
        format!("{x:.1} s")
    } else if x >= 0.1 {
        format!("{x:.2} s")
    } else {
        format!("{x:.3} s")
    }
}

/// What one more K block of a DF-J/K build costs (seconds per build): the exchange's GEMMs over
/// smaller blocks run less efficiently and launch more often. Measured (u1an39 B3LYP/def2-TZVP,
/// nbf 1034, naux 2556, the 5.49 GiB tensor on the device of an RTX PRO 6000; 16 builds per SCF):
/// K blocks of 496 (the automatic block, 6 blocks), 136 (19) and 40 (64) functions took 0.1537,
/// 0.1554 and 0.1656 s per build, 0.13 and 0.21 ms per extra block; the larger is taken.
pub(crate) const DF_K_BLOCK_S: f64 = 2.1e-4;

/// The rate at which a build streams the tensor from page-locked host memory (bytes per second,
/// the copies partly behind the exchange's GEMMs). Measured (as [`DF_K_BLOCK_S`]): streamed with
/// the automatic block, 0.2404 s per build, 0.0867 s more than on the device for 5.90e9 bytes,
/// 68 GB/s.
pub(crate) const DF_STREAM_RATE: f64 = 68e9;

/// The seconds per build a device-resident DF tensor with K blocks of `nq` auxiliary functions
/// costs beyond the automatic block `nq_auto` (`naux` auxiliary functions).
pub(crate) fn df_shrunk_cost(naux: usize, nq_auto: usize, nq: usize) -> f64 {
    let extra = naux
        .div_ceil(nq.max(1))
        .saturating_sub(naux.div_ceil(nq_auto.max(1)));
    DF_K_BLOCK_S * extra as f64
}

/// The seconds per build streaming a DF tensor of `bytes` from host memory costs beyond keeping
/// it on the device.
pub(crate) fn df_stream_cost(bytes: usize) -> f64 {
    bytes as f64 / DF_STREAM_RATE
}

/// `auto`'s path ([`ExchangeCosting`]) for the operator `plan` of `nd` perturbations
/// (`operator`: it pays and fits, on the device or streamed from host memory).
pub(crate) fn auto_choice(
    c: &ExchangeCosting,
    plan: Option<&linus_gpu::response::df::KOperatorPlan>,
    operator: bool,
    nd: Option<usize>,
) -> AutoChoice {
    let est = plan
        .filter(|_| operator)
        .map(|p| (p, c.estimate(p, nd.unwrap_or(0))));
    let kernel = "the CPKS kernel in FP32";
    let fp32 = "the FP32 exchange through the tensor";
    let (single, reason) = match est {
        Some((p, e)) if e.operator <= e.tensor => {
            let host = if p.host_bytes > 0 {
                format!(
                    ", {:.1} of its {:.1} GiB streamed from host memory",
                    p.host_bytes as f64 / 1073741824.0,
                    p.bytes as f64 / 1073741824.0
                )
            } else {
                String::new()
            };
            (
                false,
                format!(
                    "auto: the stored exchange operator in FP64{host} (estimated {} against {} \
                     for {fp32}), {kernel}",
                    secs(e.operator),
                    secs(e.tensor)
                ),
            )
        }
        Some((_, e)) => (
            true,
            format!(
                "auto: {fp32} (estimated {} against {} on the stored exchange operator in \
                 FP64), {kernel}",
                secs(e.tensor),
                secs(e.operator)
            ),
        ),
        None => (
            true,
            format!(
                "auto: {fp32}, {kernel}: the stored exchange operator{}",
                match plan.and_then(|p| p.blocked.as_deref()) {
                    Some(why) => format!(": {why}"),
                    None => " does not pay at these sizes".to_string(),
                }
            ),
        ),
    };
    AutoChoice {
        single,
        reason,
        estimate: est.map(|(_, e)| e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decision D37's cost model on the measured cases (B3LYP, the default auxiliary basis;
    /// the plans' flops and bytes as `linus_gpu`'s model gives them, docs/internal/gpu-spike.md,
    /// section AUTOPREC): on the RTX PRO 6000 a plain Hessian's exchange is cheaper in FP32
    /// through the tensor (u1an39 and ts61 with def2-SVP, ts61/def2-TZVP's streamed operator),
    /// an IR run's on the operator, which serves the field solve too; on a B200, whose FP32
    /// runs only twice its FP64, the operator always.
    #[test]
    fn exchange_costing_picks_the_cheaper_path() {
        use linus_gpu::response::df::KOperatorPlan;
        let rtx = ExchangeRates::of((12, 0), 188, 2_280_000, Some(500.0));
        for (got, want) in [
            (rtx.fp64, 1.46e12),
            (rtx.setup_emulated, 3.8e12),
            (rtx.fp32_tensor, 34.3e12),
            (rtx.field_tensor, 1.58e12),
            (rtx.skinny, 100e9),
        ] {
            assert!((got / want - 1.0).abs() < 0.01, "{got:e} vs {want:e}");
        }
        // (setup, product and tensor flops at 3N perturbations, operator GiB, of them on the host)
        let plan = |s: f64, p: f64, t: f64, gib: f64, host: f64| KOperatorPlan {
            pays: true,
            bytes: (gib * 1073741824.0) as usize,
            setup_bytes: 0,
            host_bytes: (host * 1073741824.0) as usize,
            staging_bytes: 0,
            blocked: None,
            setup_flops: s,
            product_flops: p,
            tensor_flops: t,
        };
        let u1an39 = (plan(1.260e13, 7.861e11, 3.130e13, 13.35, 0.0), 117);
        let ts61 = (plan(5.551e13, 3.800e12, 1.604e14, 41.2, 0.0), 183);
        let ts61tz = (plan(2.02e14, 1.545e13, 5.18e14, 167.4, 115.0), 183);
        let b200 = ExchangeRates::of((10, 0), 148, 1_965_000, Some(112.0));
        let cost = |rates, field, (p, nd): &(KOperatorPlan, usize)| {
            ExchangeCosting {
                rates,
                setup_emulated: true,
                field,
            }
            .estimate(p, *nd)
        };
        for (case, rates, field, operator) in [
            (&u1an39, rtx, false, false),
            (&u1an39, rtx, true, true),
            (&ts61, rtx, false, false),
            (&ts61, rtx, true, true),
            (&ts61tz, rtx, false, false),
            (&u1an39, b200, false, true),
            (&ts61, b200, false, true),
        ] {
            let e = cost(rates, field, case);
            assert_eq!(
                e.operator <= e.tensor,
                operator,
                "{e:?} (nd {}, field {field})",
                case.1
            );
        }
        // the estimates against the measured exchange seconds (operator setup and products;
        // tensor products) of u1an39/def2-SVP: 7.3 and 5.6 s
        let e = cost(rtx, false, &u1an39);
        assert!(
            (e.operator - 6.6).abs() < 0.3 && (e.tensor - 5.5).abs() < 0.3,
            "{e:?}"
        );
        let c = ExchangeCosting {
            rates: rtx,
            setup_emulated: true,
            field: false,
        };
        let a = auto_choice(&c, Some(&u1an39.0), true, Some(117));
        assert!(a.single && a.estimate.is_some(), "{a:?}");
        assert!(
            a.reason.starts_with(
                "auto: the FP32 exchange through the tensor (estimated 5.5 s against 6.6 s"
            ),
            "{}",
            a.reason
        );
        let a = auto_choice(&c, Some(&u1an39.0), false, Some(117));
        assert!(a.single && a.estimate.is_none(), "{a:?}");
    }
}
