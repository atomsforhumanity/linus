// SPDX-License-Identifier: LGPL-3.0-or-later
//! Functional names -> dispersion parameters.
//!
//! Lookup order: the linus names of [`LINUS_FUNCTIONALS`] (the XC functionals of
//! `linus-xc`, normalized the same way: case, `-`, `_`, `/` and spaces ignored), then every
//! name the reference program accepts (s-dftd3 1.7.0 for D3(BJ), dftd4 4.3.1 for D4).
//! The tables themselves are generated from the Fortran sources (`data/params.rs`).
//!
//! ORCA's D3BJ default is two-body only (`s9 = 0`, as its manual describes); `! D3BJ ABC` adds
//! ATM.

use crate::d3::D3Params;
use crate::d4::D4Params;
use crate::data::params::{BjEntry, D3BJ_TABLE, D4_TABLE};
use crate::DispError;

/// A linus XC functional and the reference-program entries of its dispersion corrections.
#[derive(Clone, Copy, Debug)]
pub struct LinusFunctional {
    /// linus names (as `linus-xc` accepts them)
    pub names: &'static [&'static str],
    /// s-dftd3 method name of its D3(BJ) parameters (`None`: no D3(BJ) for this functional)
    pub d3bj: Option<&'static str>,
    /// dftd4 functional name of its D4 parameters (`None`: no D4 for this functional)
    pub d4: Option<&'static str>,
    /// what to know about the mapping
    pub note: &'static str,
}

/// linus functional names and their dispersion parameter sets.
pub const LINUS_FUNCTIONALS: &[LinusFunctional] = &[
    LinusFunctional { names: &["hf"], d3bj: Some("hf"), d4: Some("hf"), note: "" },
    LinusFunctional {
        names: &["b3lyp", "b3lyp/g", "b3lypg"],
        d3bj: Some("b3lypg"),
        d4: Some("b3lyp"),
        note: "VWN-RPA B3LYP (ORCA B3LYP/G): same D3(BJ) and D4 parameters as VWN5 B3LYP in s-dftd3, dftd4 and ORCA",
    },
    LinusFunctional { names: &["b3lyp5"], d3bj: Some("b3lyp"), d4: Some("b3lyp"), note: "VWN5 B3LYP (ORCA B3LYP)" },
    LinusFunctional { names: &["blyp"], d3bj: Some("blyp"), d4: Some("blyp"), note: "" },
    LinusFunctional { names: &["bp86"], d3bj: Some("bp86"), d4: Some("bp86"), note: "" },
    LinusFunctional { names: &["pbe"], d3bj: Some("pbe"), d4: Some("pbe"), note: "" },
    LinusFunctional { names: &["pbe0", "pbeh", "pbe1pbe"], d3bj: Some("pbe0"), d4: Some("pbe0"), note: "" },
    LinusFunctional { names: &["tpss"], d3bj: Some("tpss"), d4: Some("tpss"), note: "" },
    LinusFunctional { names: &["tpssh"], d3bj: Some("tpssh"), d4: Some("tpssh"), note: "" },
    LinusFunctional { names: &["scan"], d3bj: Some("scan"), d4: Some("scan"), note: "" },
    LinusFunctional { names: &["r2scan"], d3bj: Some("r2scan"), d4: Some("r2scan"), note: "" },
    LinusFunctional { names: &["cam-b3lyp"], d3bj: Some("camb3lyp"), d4: Some("cam-b3lyp"), note: "" },
    LinusFunctional {
        names: &["wb97x"],
        d3bj: None,
        d4: Some("wb97x-2008"),
        note: "Chai/Head-Gordon 2008 wB97X: no D3(BJ) parameters; D4 = dftd4 `wb97x-2008` (ORCA `wB97X D4`)",
    },
    LinusFunctional {
        names: &["wb97x-d3bj"],
        d3bj: Some("wb97x"),
        d4: None,
        note: "ORCA wB97X-D3BJ: semilocal part of wB97X-V + D3(BJ) s8 0.2641, a1 0, a2 5.4959 (s-dftd3 `wb97x`, which cites no reference)",
    },
    LinusFunctional {
        names: &["wb97x-d4"],
        d3bj: None,
        d4: Some("wb97x"),
        note: "ORCA wB97X-D4: semilocal part of wB97X-V + D4 (dftd4 `wb97x`, doi 10.1002/jcc.26411); needs the `wb97x-d3bj` XC functional",
    },
    LinusFunctional {
        names: &["wb97m-d3bj"],
        d3bj: Some("wb97m"),
        d4: None,
        note: "ORCA wB97M-D3BJ: semilocal part of wB97M-V + D3(BJ) (s-dftd3 `wb97m`, doi 10.1021/acs.jctc.8b00842)",
    },
    LinusFunctional {
        names: &["wb97m-d4"],
        d3bj: None,
        d4: Some("wb97m"),
        note: "ORCA wB97M-D4: semilocal part of wB97M-V + D4; needs the `wb97m-d3bj` XC functional",
    },
    LinusFunctional {
        names: &["b97m", "b97m-d3bj"],
        d3bj: Some("b97m"),
        d4: None,
        note: "ORCA B97M-D3BJ: semilocal part of B97M-V + D3(BJ) (s-dftd3 `b97m`, doi 10.1021/acs.jctc.8b00842)",
    },
    LinusFunctional { names: &["b97m-d4"], d3bj: None, d4: Some("b97m"), note: "ORCA B97M-D4: semilocal part of B97M-V + D4" },
    LinusFunctional {
        names: &["wb97x-v", "wb97m-v", "b97m-v"],
        d3bj: None,
        d4: None,
        note: "VV10 functionals: no D3/D4 on top (ORCA refuses D3/D4 with DFT-NL)",
    },
    LinusFunctional {
        names: &["r2scan-3c"],
        d3bj: None,
        d4: Some("r2scan-3c"),
        note: "D4 with s8 0, a1 0.42, a2 5.65, s9 2, ga 2, gc 1 (dftd4 app/driver.f90 special case; JCP 154, 064103 (2021)); gCP and mTZVPP not included",
    },
];

/// linus-style normalization: ASCII lowercase, `-`, `_`, `/` and spaces removed.
pub fn normalize(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '-' | '_' | '/' | ' '))
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn find_linus(name: &str) -> Option<&'static LinusFunctional> {
    let key = normalize(name);
    LINUS_FUNCTIONALS
        .iter()
        .find(|f| f.names.iter().any(|n| normalize(n) == key))
}

fn find_entry(table: &'static [BjEntry], name: &str) -> Option<&'static BjEntry> {
    let key = normalize(name);
    table
        .iter()
        .find(|e| e.names.iter().any(|n| normalize(n) == key))
}

/// The s-dftd3 entry of a D3(BJ) parameter set.
pub fn d3bj_entry(functional: &str) -> Result<&'static BjEntry, DispError> {
    let name = match find_linus(functional) {
        Some(f) => f.d3bj.ok_or_else(|| DispError::NoParameters {
            functional: functional.to_string(),
            model: "D3(BJ)",
            note: f.note.to_string(),
        })?,
        None => functional,
    };
    find_entry(D3BJ_TABLE, name).ok_or_else(|| DispError::NoParameters {
        functional: functional.to_string(),
        model: "D3(BJ)",
        note: String::new(),
    })
}

/// D3(BJ) parameters of a functional, two-body only (`s9 = 0`, the ORCA 6 and s-dftd3 CLI
/// default). Use [`D3Params::with_atm`] for the three-body term.
pub fn d3bj_params(functional: &str) -> Result<D3Params, DispError> {
    let e = d3bj_entry(functional)?;
    Ok(D3Params {
        s6: e.s6,
        s8: e.s8,
        a1: e.a1,
        a2: e.a2,
        s9: 0.0,
        alp: e.alp,
    })
}

/// The dftd4 entry of a D4 parameter set.
pub fn d4_entry(functional: &str) -> Result<&'static BjEntry, DispError> {
    let name = match find_linus(functional) {
        Some(f) => f.d4.ok_or_else(|| DispError::NoParameters {
            functional: functional.to_string(),
            model: "D4",
            note: f.note.to_string(),
        })?,
        None => functional,
    };
    find_entry(D4_TABLE, name).ok_or_else(|| DispError::NoParameters {
        functional: functional.to_string(),
        model: "D4",
        note: String::new(),
    })
}

/// D4 parameters of a functional as the dftd4 4.3.1 program uses them (ATM included with
/// s9 = 1 where dftd4 has `bj-eeq-atm` parameters; r2SCAN-3c: s9 = 2, ga = 2, gc = 1).
pub fn d4_params(functional: &str) -> Result<D4Params, DispError> {
    let e = d4_entry(functional)?;
    let mut p = D4Params::new(e.s6, e.s8, e.a1, e.a2);
    p.s9 = e.s9;
    p.alp = e.alp;
    if e.id == "p_r2scan_3c" {
        // dftd4 4.3.1 app/driver.f90: "special case: r2SCAN-3c (modifies s9, ga, gc)"
        p.s9 = 2.0;
        p.ga = 2.0;
        p.gc = 1.0;
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_levels() {
        let b = d3bj_params("B3LYP").unwrap();
        assert_eq!(
            (b.s6, b.s8, b.a1, b.a2, b.s9),
            (1.0, 1.9889, 0.3981, 4.4211, 0.0)
        );
        assert_eq!(d3bj_params("b3lyp5").unwrap(), b);
        assert_eq!(d3bj_params("B3LYP/G").unwrap(), b);
        let w = d3bj_params("wB97X-D3BJ").unwrap();
        assert_eq!((w.s8, w.a1, w.a2), (0.2641, 0.0, 5.4959));
        assert!(d3bj_params("wb97x").is_err());
        assert!(d3bj_params("wB97X-V").is_err());
        let r = d4_params("r2SCAN-3c").unwrap();
        assert_eq!(
            (r.s8, r.a1, r.a2, r.s9, r.ga, r.gc),
            (0.0, 0.42, 5.65, 2.0, 2.0, 1.0)
        );
        let p = d4_params("pbe0").unwrap();
        assert_eq!(
            (p.s8, p.a1, p.a2, p.s9),
            (1.20065498, 0.40085597, 5.02928789, 1.0)
        );
        // upstream names still work
        assert!(d3bj_params("dsdblyp").is_ok());
        assert!(d4_params("lc-ωpbe").is_ok());
    }
}
