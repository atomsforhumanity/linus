//! Functional names: the linus alias table and the mixture parser.
//!
//! A spec is either one name, or a sum of weighted terms:
//! `"0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP"`.
//! A name is (in this order) a linus alias (table below, case/`-`/`_`/`/`
//! insensitive), `HF`, a libxc name (`GGA_X_B88`, `xc_gga_x_b88`, any case)
//! or a libxc id (`106`). Weights go before or after the name with `*`;
//! aliases inside mixtures are scaled as a unit (`0.5*b3lyp + 0.1*hf`).
//! A `-` is a subtraction only when preceded by whitespace (or at the start),
//! so names like `wB97X-V` and `m06-2x` stay intact: write `b3lyp - 0.1*hf`.
//! PySCF's `"X,C"` comma syntax is rejected on purpose (in PySCF `PBE` before
//! the comma means *exchange only*); spell the libxc components out instead.

use crate::kernel::{Result, XcError};
use crate::libxc::libxc_id;

/// What a term refers to.
#[derive(Clone, Debug, PartialEq)]
pub enum TermKind {
    /// Full-range exact (Hartree-Fock) exchange.
    Hf,
    /// A libxc functional id.
    Libxc(i32),
}

/// One weighted term of a parsed spec.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub weight: f64,
    pub kind: TermKind,
    /// False when an alias drops the functional's VV10 (ORCA's `-D3BJ` variants).
    pub keep_vv10: bool,
}

/// Result of parsing a spec.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedSpec {
    pub terms: Vec<Term>,
    /// Dispersion correction the alias is defined with but which is *not* part
    /// of the XC kernel (e.g. `"D3BJ"` for ORCA's `wB97X-D3BJ`).
    pub dispersion: Option<&'static str>,
}

/// A named functional: libxc components with weights.
pub struct Alias {
    /// Accepted names (matched after [`normalize`], so `B3LYP/G`, `b3lypg` and
    /// `B3LYP-G` are one name).
    pub names: &'static [&'static str],
    /// `(weight, "HF" | libxc name)`.
    pub terms: &'static [(f64, &'static str)],
    /// The libxc functional carries VV10 but this alias drops it.
    pub drop_vv10: bool,
    pub dispersion: Option<&'static str>,
}

const fn a(names: &'static [&'static str], terms: &'static [(f64, &'static str)]) -> Alias {
    Alias {
        names,
        terms,
        drop_vv10: false,
        dispersion: None,
    }
}

const fn d3bj(names: &'static [&'static str], terms: &'static [(f64, &'static str)]) -> Alias {
    Alias {
        names,
        terms,
        drop_vv10: true,
        dispersion: Some("D3BJ"),
    }
}

/// linus aliases. See README.md for the comparison with ORCA/PySCF/Psi4 names.
pub static ALIASES: &[Alias] = &[
    a(&["hf"], &[(1.0, "HF")]),
    // Slater exchange only (ORCA "HFS"; PySCF "LDA"/"SLATER").
    a(&["slater", "hfs", "xalpha-2/3"], &[(1.0, "LDA_X")]),
    // LDA with VWN5 correlation: ORCA "LDA"/"LSD"/"VWN5", PySCF "SVWN".
    a(
        &["lda", "lsda", "lsd", "svwn", "svwn5", "vwn5"],
        &[(1.0, "LDA_X"), (1.0, "LDA_C_VWN")],
    ),
    // LDA with VWN's RPA fit ("functional III" of the VWN paper): Gaussian and
    // Psi4 "SVWN", ORCA "VWN3", PySCF "SLATER,VWN3". NOT libxc's LDA_C_VWN_3.
    a(
        &["svwn-rpa", "svwn3", "vwn3", "vwn-rpa"],
        &[(1.0, "LDA_X"), (1.0, "LDA_C_VWN_RPA")],
    ),
    // B88 + P86 on PZ81 local correlation (Gaussian, PySCF, Psi4).
    a(
        &["bp86", "b-p", "b88p86"],
        &[(1.0, "GGA_X_B88"), (1.0, "GGA_C_P86")],
    ),
    // B88 + P86 on VWN5 local correlation (Turbomole-style; ORCA prints
    // LDAOpt VWN-5 for BP86). Not checked against ORCA numerically.
    a(
        &["bp86-vwn5", "bp86-vwn"],
        &[(1.0, "GGA_X_B88"), (1.0, "GGA_C_P86VWN")],
    ),
    a(&["blyp"], &[(1.0, "GGA_X_B88"), (1.0, "GGA_C_LYP")]),
    a(&["pbe"], &[(1.0, "GGA_X_PBE"), (1.0, "GGA_C_PBE")]),
    a(&["pbe0", "pbeh", "pbe1pbe"], &[(1.0, "HYB_GGA_XC_PBEH")]),
    // B3LYP with VWN-RPA (libxc HYB_GGA_XC_B3LYP): Gaussian, PySCF, Psi4,
    // Q-Chem, ORCA "B3LYP/G".
    a(
        &["b3lyp", "b3lyp/g", "b3lyp-rpa"],
        &[(1.0, "HYB_GGA_XC_B3LYP")],
    ),
    // B3LYP with VWN5: ORCA "B3LYP", Turbomole "b3-lyp", Psi4 "B3LYP5".
    a(&["b3lyp5", "b3lyp-vwn5"], &[(1.0, "HYB_GGA_XC_B3LYP5")]),
    a(&["tpss"], &[(1.0, "MGGA_X_TPSS"), (1.0, "MGGA_C_TPSS")]),
    a(&["tpssh"], &[(1.0, "HYB_MGGA_XC_TPSSH")]),
    a(&["scan"], &[(1.0, "MGGA_X_SCAN"), (1.0, "MGGA_C_SCAN")]),
    a(
        &["r2scan"],
        &[(1.0, "MGGA_X_R2SCAN"), (1.0, "MGGA_C_R2SCAN")],
    ),
    a(
        &["m06-2x"],
        &[(1.0, "HYB_MGGA_X_M06_2X"), (1.0, "MGGA_C_M06_2X")],
    ),
    a(&["wb97x"], &[(1.0, "HYB_GGA_XC_WB97X")]),
    a(&["wb97x-v"], &[(1.0, "HYB_GGA_XC_WB97X_V")]),
    a(&["wb97m-v"], &[(1.0, "HYB_MGGA_XC_WB97M_V")]),
    a(&["cam-b3lyp"], &[(1.0, "HYB_GGA_XC_CAM_B3LYP")]),
    a(&["pw6b95"], &[(1.0, "HYB_MGGA_XC_PW6B95")]),
    a(&["b97m-v"], &[(1.0, "MGGA_XC_B97M_V")]),
    // Lin et al. 2013 wB97X-D3 (ORCA "wB97X-D3"; D3(0) added separately).
    Alias {
        names: &["wb97x-d3"],
        terms: &[(1.0, "HYB_GGA_XC_WB97X_D3")],
        drop_vv10: false,
        dispersion: Some("D3ZERO"),
    },
    // ORCA's -D3BJ variants of the "-V" functionals (Najibi & Goerigk 2018):
    // the -V semilocal part, VV10 removed, D3(BJ) added by the host.
    d3bj(&["wb97x-d3bj"], &[(1.0, "HYB_GGA_XC_WB97X_V")]),
    d3bj(&["wb97m-d3bj"], &[(1.0, "HYB_MGGA_XC_WB97M_V")]),
    d3bj(&["b97m-d3bj"], &[(1.0, "MGGA_XC_B97M_V")]),
];

/// Canonical form for alias lookup: lower case, `ω` -> `w`, and spaces,
/// `-`, `_`, `/`, `(`, `)` removed.
pub fn normalize(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .replace(['ω', 'Ω'], "w")
        .chars()
        .filter(|c| !c.is_whitespace() && !matches!(c, '-' | '_' | '/' | '(' | ')'))
        .collect()
}

/// Find an alias by name.
pub fn find_alias(name: &str) -> Option<&'static Alias> {
    let key = normalize(name);
    ALIASES
        .iter()
        .find(|a| a.names.iter().any(|n| normalize(n) == key))
}

/// Parse a functional spec into weighted terms.
pub fn parse(spec: &str) -> Result<ParsedSpec> {
    let err = |msg: String| XcError::Parse {
        spec: spec.to_string(),
        msg,
    };
    let s = spec.trim();
    if s.is_empty() {
        return Err(err("empty functional spec".into()));
    }
    if s.contains(',') {
        return Err(err(
            "PySCF-style 'X,C' syntax is not supported; write a sum of libxc names, e.g. 'GGA_X_PBE + GGA_C_PBE'".into(),
        ));
    }
    // A single name (handles names containing '-' such as 'wb97x-v').
    if let Some(p) = resolve_name(s, 1.0) {
        return Ok(p);
    }

    let mut out = ParsedSpec {
        terms: Vec::new(),
        dispersion: None,
    };
    for (sign, term) in split_terms(s) {
        let term = term.trim();
        if term.is_empty() {
            return Err(err("empty term".into()));
        }
        let (weight, name) = split_weight(term).map_err(err)?;
        let p = resolve_name(name, sign * weight)
            .ok_or_else(|| XcError::UnknownFunctional(name.trim().to_string()))?;
        if p.dispersion.is_some() {
            out.dispersion = p.dispersion;
        }
        out.terms.extend(p.terms);
    }
    Ok(out)
}

/// Resolve one name (alias, HF, libxc name or id) scaled by `weight`.
fn resolve_name(name: &str, weight: f64) -> Option<ParsedSpec> {
    let name = name.trim();
    if let Some(al) = find_alias(name) {
        let terms = al
            .terms
            .iter()
            .map(|&(w, n)| Term {
                weight: weight * w,
                kind: if n == "HF" {
                    TermKind::Hf
                } else {
                    TermKind::Libxc(libxc_id(n).expect(n))
                },
                keep_vv10: !al.drop_vv10,
            })
            .collect();
        return Some(ParsedSpec {
            terms,
            dispersion: al.dispersion,
        });
    }
    let kind = if name.eq_ignore_ascii_case("hf") || name.eq_ignore_ascii_case("hfx") {
        TermKind::Hf
    } else {
        TermKind::Libxc(libxc_id(name)?)
    };
    Some(ParsedSpec {
        terms: vec![Term {
            weight,
            kind,
            keep_vv10: true,
        }],
        dispersion: None,
    })
}

/// Split at '+' and at '-' preceded by whitespace (or leading), keeping signs.
fn split_terms(s: &str) -> Vec<(f64, &str)> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut sign = 1.0;
    let mut start = 0;
    for (i, &c) in bytes.iter().enumerate() {
        let is_op = c == b'+'
            && !(i > 0
                && matches!(bytes[i - 1], b'e' | b'E')
                && i >= 2
                && bytes[i - 2].is_ascii_digit())
            || c == b'-' && (i == 0 || bytes[i - 1].is_ascii_whitespace());
        if is_op {
            if !s[start..i].trim().is_empty() {
                out.push((sign, &s[start..i]));
            }
            sign = if c == b'-' { -1.0 } else { 1.0 };
            start = i + 1;
        }
    }
    out.push((sign, &s[start..]));
    out
}

/// `"0.2*HF"`, `"HF*0.2"` or `"HF"` -> (weight, name).
fn split_weight(term: &str) -> std::result::Result<(f64, &str), String> {
    let parts: Vec<&str> = term.split('*').map(str::trim).collect();
    match parts.as_slice() {
        [name] => Ok((1.0, name)),
        [l, r] => {
            if let Ok(w) = l.parse::<f64>() {
                Ok((w, r))
            } else if let Ok(w) = r.parse::<f64>() {
                Ok((w, l))
            } else {
                Err(format!("term '{term}' has no numeric weight"))
            }
        }
        _ => Err(format!("term '{term}' has more than one '*'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_resolve_to_libxc() {
        for al in ALIASES {
            for (_, n) in al.terms {
                assert!(*n == "HF" || libxc_id(n).is_some(), "{n} unknown to libxc");
            }
        }
    }

    #[test]
    fn names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for al in ALIASES {
            for n in al.names {
                assert!(seen.insert(normalize(n)), "duplicate alias {n}");
            }
        }
    }

    #[test]
    fn parse_mixture() {
        let p = parse("0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP")
            .unwrap();
        assert_eq!(p.terms.len(), 5);
        assert_eq!(p.terms[0].kind, TermKind::Hf);
        assert_eq!(p.terms[2].kind, TermKind::Libxc(106));
        assert!((p.terms[4].weight - 0.81).abs() < 1e-15);
        let p = parse("b3lyp - 0.1*hf").unwrap();
        assert_eq!(p.terms.len(), 2);
        assert_eq!(p.terms[1].weight, -0.1);
        let p = parse("0.5*wB97X-V + gga_x_pbe*0.5").unwrap();
        assert_eq!(p.terms[0].kind, TermKind::Libxc(466));
        assert_eq!(p.terms[0].weight, 0.5);
        assert_eq!(p.terms[1].kind, TermKind::Libxc(101));
        let p = parse("1e-3*LDA_X+1E+0*402").unwrap();
        assert_eq!(p.terms[0].weight, 1e-3);
        assert_eq!(p.terms[1].kind, TermKind::Libxc(402));
        assert!(parse("pbe,pbe").is_err());
        assert!(parse("nonsense").is_err());
        assert!(parse("ωB97M-V").is_ok());
    }
}
