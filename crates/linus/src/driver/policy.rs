//! The device policy: the card class and the arithmetic of every kernel family, decided once
//! per session from the `precision` settings and the device, with the reasons (decisions
//! D23, D27, D37). The datacenter (FP64-rich) and workstation (FP64-poor) paths are this
//! module's table; the engines take the [`Arith`] of their family and never read the card or the
//! settings. `precision.fp64` sets the exchange GEMMs, `precision.fp32` the families with FP32
//! paths:
//!
//! | family | `fp64` auto, FP64-rich | `fp64` auto, FP64-poor | `fp64` native | `fp64` emulated |
//! |---|---|---|---|---|
//! | SCF exchange (`scf_k`) | fp64 | ozaki 7, cublas 55, fp64 | fp64 | ozaki 7, cublas 55, fp64 |
//! | SCF fit (`scf_fit`) | fp64 | ozaki 7 | fp64 | ozaki 7 |
//! | Hessian F^x exchange (`hess_k`) | fp64 | ozaki 7, fp64 | fp64 | ozaki 7, fp64 |
//! | SCF XC potential (`vxc`) | fp64 | fp64 | fp64 | fp64 |
//!
//! | family | `fp32` off | `fp32` auto | `fp32` on |
//! |---|---|---|---|
//! | XC gradient (`grad_xc`) | fp64 | fp64 | fp32 |
//! | Hessian XC terms (`hess_xc`) | fp64 | fp64 | fp32 |
//! | VV10 SCF pair sums (`vv10_scf`) | fp64 | df64 | df64 |
//! | VV10 gradient pair sums (`vv10_grad`) | fp64 | df64 | fp32 |
//!
//! A list is a fallback chain: the first arithmetic that sets up runs, and the record says which
//! and why (the device's adapters apply it, `crate::backend::gpu::df`). VV10's pair sums follow
//! `precision.fp32`, not the card: double-float (pairs of FP32 numbers) is FP64-equal there and
//! several times faster on FP64-poor cards, and the same on every card, so results stay bitwise
//! card to card. Native K's resident dense planes (`k_dense`) and the XC products on cuBLAS's
//! grouped DGEMM (`xc_grouped`) are the FP64-rich cards'.
//!
//! A response's arithmetic (its exchange, the CPKS kernel, VV10's kernel) depends on its sizes
//! and tolerance as well, and is decided per response: [`DevicePolicy::response`] (what the
//! response J/K computes in, decisions D23 and D37) and [`DevicePolicy::vv10_cpks_fp32`] (VV10's
//! kernel, from whether the response runs single precision). The device's adapters turn these
//! decisions into their engines' settings (`crate::backend::gpu`): the exchange's fallback chains
//! and stored operator and D37's exchange costing (`df`, `costing`), and the semilocal CPKS
//! kernel's arithmetic (`xc`).
//!
//! The policy reads a device as its provider reports it ([`CardProps`]): the card's class and the
//! costing's rates are derived here.

use crate::driver::options::{Fp32Mode, Fp64Mode, ResponsePrecision};
use crate::two_electron::ResponseUse;
use linus_core::Arith;

/// The first tier of FP64 emulation: the Ozaki scheme on INT8 tensor cores with 7 slices (55
/// bits per operand, FP64 quality and smooth in the density; 6 slices leave a 6e-9 Eh bias in
/// E_K; docs/internal/gpu-spike.md, section KPS).
pub(crate) const OZAKI: Arith = Arith::Ozaki { slices: 7 };

/// The second tier: cuBLAS's fixed-mantissa emulation at 55 bits (FP64 accuracy on
/// density-fitted exchange data; docs/internal/gpu-spike.md 8.2, 9.4).
pub(crate) const CUBLAS: Arith = Arith::Fp64Emulated { bits: 55 };

/// Native FP64: nothing to set up.
pub(crate) const NATIVE: &[Arith] = &[Arith::Fp64];

/// The exchange's FP64 emulation: the Ozaki scheme on the tensor's pre-sliced planes, else
/// cuBLAS's emulation, else FP64.
pub(crate) const EMULATION: &[Arith] = &[OZAKI, CUBLAS, Arith::Fp64];

/// The Hessian F^x exchange's: the Ozaki scheme, else FP64.
pub(crate) const HESSIAN_EMULATION: &[Arith] = &[OZAKI, Arith::Fp64];

/// What the card computes fast (decision D27): `DeviceInfo::fp64_rich`'s classification, read
/// here and nowhere else.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CardClass {
    /// Native FP64 outruns its emulation (INT8 tensor cores at most about 32x FP64, or none):
    /// A100, H100/H200, V100/P100.
    Fp64Rich { int8_per_fp64: Option<f64> },
    /// FP64 emulation on INT8 tensor cores outruns native FP64: the RTX/L40/A40/T4 class, B200,
    /// B300.
    Fp64Poor { int8_per_fp64: f64 },
}

impl CardClass {
    /// The class of a device whose native FP64 outruns emulation (`fp64_rich`), with INT8
    /// tensor cores `int8_per_fp64` times its FP64 rate (None: none).
    pub(crate) fn new(fp64_rich: bool, int8_per_fp64: Option<f64>) -> CardClass {
        match int8_per_fp64 {
            Some(r) if !fp64_rich => CardClass::Fp64Poor { int8_per_fp64: r },
            r => CardClass::Fp64Rich { int8_per_fp64: r },
        }
    }

    pub(crate) fn fp64_rich(self) -> bool {
        matches!(self, CardClass::Fp64Rich { .. })
    }

    pub(crate) fn int8_per_fp64(self) -> Option<f64> {
        match self {
            CardClass::Fp64Rich { int8_per_fp64 } => int8_per_fp64,
            CardClass::Fp64Poor { int8_per_fp64 } => Some(int8_per_fp64),
        }
    }
}

/// What the policy reads of a device, as its provider reports it: the card's class
/// ([`CardProps::class`]) and the exchange costing's rates ([`CardProps::rates`]) are derived
/// from it here.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CardProps {
    /// Native FP64 outruns its emulation on the card's INT8 tensor cores.
    pub fp64_rich: bool,
    /// The INT8 tensor cores' rate in multiples of the FP64 rate (None: none).
    pub int8_per_fp64: Option<f64>,
    /// The compute capability (major, minor).
    pub cc: (i32, i32),
    /// Multiprocessors.
    pub sm_count: u32,
    /// The clock (kHz).
    pub clock_khz: u32,
}

impl CardProps {
    /// The card's class (decision D27).
    pub(crate) fn class(&self) -> CardClass {
        CardClass::new(self.fp64_rich, self.int8_per_fp64)
    }

    /// The card's rates of D37's exchange costing.
    pub(crate) fn rates(&self) -> ExchangeRates {
        ExchangeRates::of(self.cc, self.sm_count, self.clock_khz, self.int8_per_fp64)
    }
}

/// `precision.fp64` resolved for the card: what the device's FP64 families run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DevicePrecision {
    /// Native FP64.
    Double,
    /// The density-fitted exchange GEMMs (and the CPKS kernel's large ones) in FP64 emulation.
    Emulated,
}

impl DevicePrecision {
    /// The name in `resolved.record.backend.precision`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            DevicePrecision::Double => "double",
            DevicePrecision::Emulated => "emulated",
        }
    }

    /// The value of `precision.fp64` it is: `native` or `emulated`.
    pub(crate) fn setting(self) -> &'static str {
        match self {
            DevicePrecision::Double => "native",
            DevicePrecision::Emulated => "emulated",
        }
    }
}

/// The arithmetic of every kernel family of a session's device ([module docs](self)).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DevicePolicy {
    pub card: CardClass,
    /// `precision.fp64` as given.
    fp64: Fp64Mode,
    /// `precision.fp32` (VV10's arithmetic follows it, not the card).
    pub fp32: Fp32Mode,
    pub precision: DevicePrecision,
    /// `auto`'s choice for the card and why (None: an explicit setting).
    pub auto_note: Option<String>,
    /// The SCF's exchange GEMMs, a fallback chain ([`NATIVE`]: nothing to set up).
    pub scf_k: &'static [Arith],
    /// The SCF tensor's fit B = (mn|P) L^-T.
    pub scf_fit: Arith,
    /// The Hessian's F^x exchange GEMMs, a fallback chain.
    pub hess_k: &'static [Arith],
    pub vxc: Arith,
    pub grad_xc: Arith,
    pub hess_xc: Arith,
    pub vv10_scf: Arith,
    pub vv10_grad: Arith,
    /// Native K keeps its unpacked GEMM1 operand on the device between builds (FP64-rich
    /// cards, where the unpack is a large part of K: H200 23-44 %; elsewhere FP64 GEMMs dominate
    /// and the planes would take memory from the other engines for a few percent).
    pub k_dense: bool,
    /// The XC engines' FP64 per-batch products on cuBLAS's grouped DGEMM, on the FP64 tensor
    /// cores of FP64-rich cards (17.7-18.8 against vgemm's 9.4-11.0 TFLOP/s on an H200); else
    /// on the variable-size batched kernel.
    pub xc_grouped: bool,
    /// The card's rates of D37's exchange costing.
    pub rates: ExchangeRates,
}

impl DevicePolicy {
    /// The policy of the settings `fp64` and `fp32` on a card of class `card` with compute
    /// capability `cc` (the reason of `auto`) and the exchange costing's `rates`.
    pub(crate) fn new(
        fp64: Fp64Mode,
        fp32: Fp32Mode,
        card: CardClass,
        cc: &str,
        rates: ExchangeRates,
    ) -> DevicePolicy {
        let precision = match fp64 {
            Fp64Mode::Auto if !card.fp64_rich() => DevicePrecision::Emulated,
            Fp64Mode::Auto | Fp64Mode::Native => DevicePrecision::Double,
            Fp64Mode::Emulated => DevicePrecision::Emulated,
        };
        let auto_note = (fp64 == Fp64Mode::Auto).then(|| {
            let ratio = match card.int8_per_fp64() {
                Some(r) => format!("INT8 tensor cores about {r:.0}x FP64"),
                None => "no INT8 tensor cores".to_string(),
            };
            if card.fp64_rich() {
                format!(
                    "auto: native (compute capability {cc}: {ratio}; native FP64 is faster than \
                     its emulation here)"
                )
            } else {
                format!(
                    "auto: emulated (compute capability {cc}: {ratio}; FP64 emulation on INT8 \
                     tensor cores is faster than native FP64 here, at FP64 accuracy)"
                )
            }
        });
        let emulated = precision == DevicePrecision::Emulated;
        let fp32_if_on = if fp32 == Fp32Mode::On {
            Arith::Fp32
        } else {
            Arith::Fp64
        };
        let (vv10_scf, vv10_grad) = match fp32 {
            Fp32Mode::Off => (Arith::Fp64, Arith::Fp64),
            Fp32Mode::On => (Arith::DoubleFloat, Arith::Fp32),
            Fp32Mode::Auto => (Arith::DoubleFloat, Arith::DoubleFloat),
        };
        DevicePolicy {
            card,
            fp64,
            fp32,
            precision,
            auto_note,
            scf_k: if emulated { EMULATION } else { NATIVE },
            scf_fit: if emulated { OZAKI } else { Arith::Fp64 },
            hess_k: if emulated { HESSIAN_EMULATION } else { NATIVE },
            vxc: Arith::Fp64,
            grad_xc: fp32_if_on,
            hess_xc: fp32_if_on,
            vv10_scf,
            vv10_grad,
            k_dense: card.fp64_rich(),
            xc_grouped: card.fp64_rich(),
            rates,
        }
    }

    /// The device response J/K's arithmetic for `purpose` (decisions D23, D37), with
    /// the response precision `precision` (`PrecisionSettings::response_effective`), the exchange
    /// work per density n^2 o naux `work` (as the CPU's rule measures it) and `cphf_tol`.
    pub(crate) fn response(
        &self,
        purpose: ResponseUse,
        precision: ResponsePrecision,
        work: f64,
        cphf_tol: Option<f64>,
    ) -> ResponseArith {
        let (mode, why) = response_mode(purpose, precision, work, cphf_tol);
        // `precision.fp32: on` (the opt-in FP32 class): where the response runs single precision
        // (`fp32`, or auto's single-precision rule) its exchange runs on the stored operator in
        // FP32 (else the FP32 exchange through the tensor). FP64-rich devices: auto otherwise
        // stays FP64 throughout (the FP64 operator, streamed beyond the device as `fp64` does);
        // only FP64-poor cards trade the CPKS kernel and the operator's fallback down to FP32
        let fp32_operator = self.fp32 == Fp32Mode::On
            && matches!(mode, ResponseMode::AutoOperator | ResponseMode::Single);
        let (mode, why) = if fp32_operator {
            (
                ResponseMode::Single,
                "precision.fp32 on: the stored exchange operator in FP32 where it pays and fits (else the FP32 exchange through the tensor), the CPKS kernel in FP32",
            )
        } else if mode == ResponseMode::AutoOperator && self.card.fp64_rich() {
            (
                ResponseMode::Double,
                "auto: double precision (a device with full FP64 throughput)",
            )
        } else {
            (mode, why)
        };
        ResponseArith {
            mode,
            why,
            k: if mode == ResponseMode::Single {
                Arith::Fp32
            } else {
                Arith::Fp64
            },
        }
    }

    /// Every family's arithmetic with why (`resolved.record.backend.arithmetic`): (family, its
    /// arithmetic or fallback chain, reason).
    pub(crate) fn table(&self) -> Vec<(&'static str, String, String)> {
        let chain = |c: &[Arith]| {
            c.iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", else ")
        };
        let class = if self.card.fp64_rich() {
            "an FP64-rich card"
        } else {
            "an FP64-poor card"
        };
        let by = if self.fp64 == Fp64Mode::Auto {
            format!("precision.fp64 auto on {class} (D27)")
        } else {
            format!("precision.fp64 {}", self.fp64.name())
        };
        let by32 = format!("precision.fp32 {}", self.fp32.name());
        let vv10 = format!(
            "precision.fp32 {}: VV10's pair sums follow it, not the card",
            self.fp32.name()
        );
        vec![
            ("scf_k", chain(self.scf_k), by.clone()),
            ("scf_fit", self.scf_fit.to_string(), by.clone()),
            ("hess_k", chain(self.hess_k), by.clone()),
            (
                "vxc",
                self.vxc.to_string(),
                "FP64 under every setting (in FP32 the SCF needed more iterations than it saved)"
                    .into(),
            ),
            ("grad_xc", self.grad_xc.to_string(), by32.clone()),
            ("hess_xc", self.hess_xc.to_string(), by32),
            ("vv10_scf", self.vv10_scf.to_string(), vv10.clone()),
            ("vv10_grad", self.vv10_grad.to_string(), vv10),
        ]
    }

    /// VV10's CPKS products in FP32: with `precision.fp32: on`, or a single-precision response
    /// (`response_single`, as the semilocal kernel's arithmetic, `crate::backend::gpu::xc`);
    /// never with `precision.fp32: off`.
    pub(crate) fn vv10_cpks_fp32(&self, response_single: bool) -> bool {
        self.fp32 != Fp32Mode::Off && (self.fp32 == Fp32Mode::On || response_single)
    }
}

/// The arithmetic of a device response's J/K ([`DevicePolicy::response`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResponseMode {
    /// FP64 (the stored operator when it pays and fits, else the tensor), the CPKS kernel in
    /// FP64.
    Double,
    /// FP32 exchange through the tensor, the CPKS kernel in FP32.
    Single,
    /// `precision.response: auto` where the CPU would run single precision: the stored
    /// operator in FP64 or the FP32 exchange through the tensor, whichever is the cheaper on a
    /// device with little FP64 throughput (decision D37's exchange costing,
    /// `crate::backend::gpu::costing`; FP64-rich devices take [`ResponseMode::Double`] instead),
    /// the CPKS kernel in FP32.
    AutoOperator,
}

/// A device response J/K's arithmetic ([`DevicePolicy::response`]): what the plan reports of it
/// (`linus --plan`'s `response.device`); the device's adapter sets its engine up from it
/// (`crate::backend::gpu::df`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResponseArith {
    pub mode: ResponseMode,
    /// Why (`resolved.record.backend.response[].reason`; auto's costing gives its own).
    pub why: &'static str,
    /// The exchange: FP64 (in the SCF exchange's arithmetic, a fallback chain) or FP32.
    pub k: Arith,
}

/// The device response's mode for `purpose` and the settings, with the reason: single and
/// double for `fp32` and `fp64` as set; `auto` where the CPU's rule (`auto_single`: a large
/// exchange at a tolerance single precision reaches) runs single precision weighs the stored
/// operator at FP64 quality against the FP32 exchange, else double (a small exchange, or
/// `cphf_tol` below what single precision reaches); polarizabilities double.
pub(crate) fn response_mode(
    purpose: ResponseUse,
    precision: ResponsePrecision,
    work: f64,
    cphf_tol: Option<f64>,
) -> (ResponseMode, &'static str) {
    use linus_jk::df::response::auto_single;
    match (purpose, precision) {
        (ResponseUse::Polarizability, _) => (
            ResponseMode::Double,
            "polarizability: double precision",
        ),
        (ResponseUse::Hessian, ResponsePrecision::Fp32) => {
            (ResponseMode::Single, "precision.response: fp32")
        }
        (ResponseUse::Hessian, ResponsePrecision::Fp64) => {
            (ResponseMode::Double, "precision.response: fp64")
        }
        (ResponseUse::Hessian, ResponsePrecision::Auto) if auto_single(work, cphf_tol) => (
            ResponseMode::AutoOperator,
            "auto: the stored exchange operator in FP64 or the FP32 exchange through the tensor, the CPKS kernel in FP32",
        ),
        (ResponseUse::Hessian, ResponsePrecision::Auto) => (
            ResponseMode::Double,
            if cphf_tol.is_some_and(|t| t < linus_jk::df::response::SINGLE_MIN_TOL) {
                "auto: double precision (response.tol below what single precision reaches)"
            } else {
                "auto: double precision (a small exchange)"
            },
        ),
    }
}

// ------------------------------------------------------------------- D37's exchange costing

/// A device's rates for D37's exchange costing (flop/s and bytes/s,
/// `crate::backend::gpu::costing`), calibrated on the RTX PRO
/// 6000 Blackwell Max-Q (188 SMs at 2.28 GHz; the response's own products, device events,
/// docs/internal/gpu-spike.md, section AUTOPREC) and scaled to another card by its SMs times
/// clock and its class's nominal FP64 and FP32 rates per SM and clock: a fixed function of the
/// device's properties, so the choice is reproducible (no timing at run time).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExchangeRates {
    /// FP64 GEMMs: the operator's products and its FP64 setup (1.46 TFLOP/s: the mean of
    /// 1.43, 1.40 and 1.55 measured at u1an39/def2-SVP, ts61/def2-SVP and u1an39/def2-TZVP).
    pub fp64: f64,
    /// The operator's setup in cuBLAS's 55-bit FP64 emulation (3.8 TFLOP/s, 2.6x FP64: 3.3,
    /// 3.7 and 4.4; on cards whose INT8 tensor cores run about 500x their FP64, less where
    /// they run less).
    pub setup_emulated: f64,
    /// A product's exchange through the tensor in FP32 (its GEMMs, unpacking and contractions
    /// by the tensor path's flop count: 34.3 TFLOP/s; 34.4, 31.0 and 37.6).
    pub fp32_tensor: f64,
    /// A three-perturbation product through the tensor in FP64 (the field solve's own J/K:
    /// FP64 contractions, the first GEMM in the Ozaki scheme; 1.58 TFLOP/s at u1an39).
    pub field_tensor: f64,
    /// The operator read by a product of a few perturbations (cuBLAS's DGEMM at three columns
    /// runs at a fixed ~100 GB/s of the operator, far below the memory's bandwidth).
    pub skinny: f64,
    /// Host-device copies of a streamed operator's chunks (PCIe: 57 GB/s measured each way).
    pub pcie: f64,
}

impl ExchangeRates {
    /// The rates of a device with compute capability `cc`, `sms` multiprocessors at
    /// `clock_khz` and INT8 tensor cores `int8_per_fp64` times its FP64 rate.
    pub(crate) fn of(cc: (i32, i32), sms: u32, clock_khz: u32, int8_per_fp64: Option<f64>) -> Self {
        // nominal FP64 and FP32 flops per SM and clock: 2 FP64 units per SM (4 flops) on the
        // FP64-poor classes (T4, A40/A10, L40/L4, RTX and RTX PRO, B300), FP64 tensor cores on
        // B200 (cc 10.0, ~128); 64 FP32 units on Turing (cc 7.5), 128 from Ampere on (256 flops)
        let fp64_flops = if cc == (10, 0) { 128.0 } else { 4.0 };
        let fp32_flops = if cc == (7, 5) { 128.0 } else { 256.0 };
        let cycles = sms as f64 * clock_khz as f64 * 1e3;
        // the reference card: 188 SMs at 2280000 kHz, nominal FP64 1.715 TFLOP/s and FP32 109.7
        let fp64 = 0.8515 * fp64_flops * cycles;
        let size = cycles / (188.0 * 2.28e9);
        // cuBLAS's emulation gains 2.6x on the reference card (INT8 about 500x FP64), in
        // proportion to the INT8:FP64 ratio below it, and never runs slower than FP64 here
        let emu = int8_per_fp64.map_or(1.0, |r| (2.6 * (r / 500.0).min(1.0)).max(1.0));
        ExchangeRates {
            fp64,
            setup_emulated: fp64 * emu,
            fp32_tensor: 0.3126 * fp32_flops * cycles,
            field_tensor: 1.082 * fp64,
            skinny: 100e9 * size,
            pcie: 50e9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The datacenter and workstation paths: every family's arithmetic per card class and
    /// setting, as the module docs' table (decision D27; VV10 by the setting).
    #[test]
    fn the_policy_table() {
        use Arith::*;
        let rates = ExchangeRates::of((12, 0), 188, 2_280_000, Some(500.0));
        let rich = CardClass::new(true, Some(30.0));
        let poor = CardClass::new(false, Some(500.0));
        let pol = |s: (Fp64Mode, Fp32Mode), c| DevicePolicy::new(s.0, s.1, c, "x", rates);
        let auto = (Fp64Mode::Auto, Fp32Mode::Auto);
        let native = (Fp64Mode::Native, Fp32Mode::Off);
        let emulated = (Fp64Mode::Emulated, Fp32Mode::Auto);
        let mixed = (Fp64Mode::Native, Fp32Mode::On);
        type Row = (
            DevicePrecision,
            &'static [Arith],
            Arith,
            Arith,
            Arith,
            Arith,
        );
        let row = |p: &DevicePolicy| -> Row {
            (
                p.precision,
                p.scf_k,
                p.scf_fit,
                p.grad_xc,
                p.vv10_scf,
                p.vv10_grad,
            )
        };
        let fp64: Row = (
            DevicePrecision::Double,
            NATIVE,
            Fp64,
            Fp64,
            DoubleFloat,
            DoubleFloat,
        );
        let emu: Row = (
            DevicePrecision::Emulated,
            EMULATION,
            OZAKI,
            Fp64,
            DoubleFloat,
            DoubleFloat,
        );
        assert_eq!(row(&pol(auto, rich)), fp64);
        assert_eq!(row(&pol(auto, poor)), emu);
        assert_eq!(row(&pol(emulated, rich)), emu);
        assert_eq!(
            row(&pol(native, poor)),
            (DevicePrecision::Double, NATIVE, Fp64, Fp64, Fp64, Fp64)
        );
        assert_eq!(
            row(&pol(mixed, poor)),
            (
                DevicePrecision::Double,
                NATIVE,
                Fp64,
                Fp32,
                DoubleFloat,
                Fp32
            )
        );
        // fp32 on with the card's fp64: the exchange as auto, FP32 products
        assert_eq!(
            row(&pol((Fp64Mode::Auto, Fp32Mode::On), poor)),
            (
                DevicePrecision::Emulated,
                EMULATION,
                OZAKI,
                Fp32,
                DoubleFloat,
                Fp32
            )
        );
        assert!(pol(auto, rich).k_dense && !pol(auto, poor).k_dense);
        assert!(pol(native, rich).xc_grouped && !pol(native, poor).xc_grouped);
        assert_eq!(pol(emulated, poor).hess_k, HESSIAN_EMULATION);
        assert!(pol(native, poor).auto_note.is_none());
        assert!(pol(auto, poor).auto_note.unwrap().starts_with(
            "auto: emulated (compute capability x: INT8 tensor cores about 500x FP64"
        ));
        // VV10's response kernel: FP32 for single-precision responses and fp32 on, never with
        // fp32 off (the semilocal kernel's: crate::backend::gpu::xc's tests)
        let e = pol(auto, poor);
        assert!(e.vv10_cpks_fp32(true) && !e.vv10_cpks_fp32(false));
        let d = pol(native, poor);
        assert!(!d.vv10_cpks_fp32(true));
        let d = pol((Fp64Mode::Native, Fp32Mode::Auto), poor);
        assert!(d.vv10_cpks_fp32(true) && !d.vv10_cpks_fp32(false));
        assert!(pol(mixed, rich).vv10_cpks_fp32(false));
    }

    /// The device response: auto stays FP64 on FP64-rich cards and weighs the costing on the
    /// others; fp32 on moves the single-precision responses to FP32 (the engine's settings that
    /// follow: crate::backend::gpu::df's tests).
    #[test]
    fn response_arithmetic_per_card() {
        use ResponsePrecision as P;
        let rates = ExchangeRates::of((12, 0), 188, 2_280_000, Some(500.0));
        let rich = DevicePolicy::new(
            Fp64Mode::Auto,
            Fp32Mode::Auto,
            CardClass::new(true, Some(30.0)),
            "x",
            rates,
        );
        let poor = DevicePolicy::new(
            Fp64Mode::Auto,
            Fp32Mode::Auto,
            CardClass::new(false, Some(500.0)),
            "x",
            rates,
        );
        let mixed = DevicePolicy::new(
            Fp64Mode::Native,
            Fp32Mode::On,
            CardClass::new(false, Some(500.0)),
            "x",
            rates,
        );
        let h = ResponseUse::Hessian;
        let r = rich.response(h, P::Auto, 1e11, None);
        assert!(r.mode == ResponseMode::Double && r.k == Arith::Fp64);
        let r = poor.response(h, P::Auto, 1e11, None);
        assert!(r.mode == ResponseMode::AutoOperator && r.k == Arith::Fp64);
        let r = mixed.response(h, P::Auto, 1e11, None);
        assert!(r.mode == ResponseMode::Single && r.k == Arith::Fp32);
        let r = poor.response(ResponseUse::Polarizability, P::Fp32, 1e11, None);
        assert!(r.mode == ResponseMode::Double && r.k == Arith::Fp64);
    }

    /// `precision.response` on the GPU: as set; auto by the CPU's rule, preferring the stored
    /// operator where the CPU would run single precision; double for tight tolerances and
    /// polarizabilities.
    #[test]
    fn response_modes() {
        use ResponseMode::*;
        use ResponsePrecision as P;
        let (h, p) = (ResponseUse::Hessian, ResponseUse::Polarizability);
        let (large, small) = (1e11, 1e5);
        let m = |u, pr, w, t| response_mode(u, pr, w, t).0;
        assert_eq!(m(h, P::Fp32, small, None), Single);
        assert_eq!(m(h, P::Fp64, large, None), Double);
        assert_eq!(m(h, P::Auto, large, None), AutoOperator);
        assert_eq!(m(h, P::Auto, large, Some(1e-6)), AutoOperator);
        // a tolerance single precision does not reach: FP64 throughout
        assert_eq!(m(h, P::Auto, large, Some(1e-8)), Double);
        assert!(response_mode(h, P::Auto, large, Some(1e-8))
            .1
            .contains("response.tol"));
        assert_eq!(m(h, P::Auto, small, None), Double);
        for pr in [P::Auto, P::Fp32, P::Fp64] {
            assert_eq!(m(p, pr, large, None), Double);
        }
    }
}
