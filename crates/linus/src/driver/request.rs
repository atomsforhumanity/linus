//! A calculation request ([`Request`]): the structure, the calctype and the settings, read
//! from one JSON object. The CLI (a request file, stdin, or flags over an XYZ file) and the
//! qcdata front end build their requests here, so the request has one grammar.
//!
//! ```json
//! {"structure": {"symbols": ["O", "H", "H"], "geometry": [[0, 0, 0.22], [0, 1.43, -0.89], [0, -1.43, -0.89]]},
//!  "calctype": "gradient", "method": "b3lyp-d3bj", "jk": "df", "threads": 8}
//! ```
//!
//! - `structure` mirrors qcdata's `Structure`: `symbols` (or `numbers`), `geometry` (n x 3,
//!   bohr), `charge` (0), `multiplicity` (1); or `xyz`, the text of an XYZ file (Å), with
//!   `charge` and `multiplicity`.
//! - `calctype`: `energy` (default), `gradient` or `hessian` (with the analysis its `freq`
//!   settings ask for).
//! - every other key is a setting ([`CalcOptions`]), `freq` and `output` included.
//!
//! [`set_path`] and [`merge`] edit a request's JSON tree the way the CLI's `--set` and
//! `--options` do: a dotted path below a group's shorthand refines the shorthand's object form
//! (`jk: "df"` then `jk.aux=...`), last wins.

use crate::driver::options::{expand_group_shorthand, CalcOptions};
use crate::driver::report::CalcType;
use crate::driver::schema::{norm_name, unknown_key};
use crate::structure::{parse_xyz, symbol_to_z, Structure};
use linus_base::error::LinusError;
use serde_json::{Map, Value};

/// A calculation: what to compute, on which structure, with which settings.
#[derive(Clone, Debug)]
pub struct Request {
    pub structure: Structure,
    /// What to compute.
    pub calctype: CalcType,
    pub settings: CalcOptions,
}

/// The keys of the request's `structure`.
const STRUCTURE_KEYS: [&str; 6] = [
    "symbols",
    "numbers",
    "geometry",
    "xyz",
    "charge",
    "multiplicity",
];

/// The structure of a request's `structure` object.
pub fn structure_from_json(v: Value) -> Result<Structure, LinusError> {
    let inp = LinusError::Input;
    let Value::Object(m) = v else {
        return Err(inp(format!(
            "structure must be an object (symbols or numbers, geometry; or xyz), not {v}"
        )));
    };
    let mut m: Map<String, Value> = m.into_iter().map(|(k, v)| (norm_name(&k), v)).collect();
    if let Some(k) = m.keys().find(|k| !STRUCTURE_KEYS.contains(&k.as_str())) {
        return Err(unknown_key("structure.", k, &STRUCTURE_KEYS));
    }
    let field = |m: &mut Map<String, Value>, k: &str| m.remove(k).filter(|v| !v.is_null());
    let charge: i32 = field(&mut m, "charge")
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| inp(format!("structure.charge: {e}")))?
        .unwrap_or(0);
    let mult: u32 = field(&mut m, "multiplicity")
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| inp(format!("structure.multiplicity: {e}")))?
        .unwrap_or(1);
    let (z, geometry) = match (
        field(&mut m, "xyz"),
        field(&mut m, "numbers"),
        field(&mut m, "symbols"),
        field(&mut m, "geometry"),
    ) {
        (Some(Value::String(xyz)), None, None, None) => parse_xyz(&xyz).map_err(inp)?,
        (None, nums, syms, Some(g)) => {
            let geometry: Vec<[f64; 3]> = serde_json::from_value(g)
                .map_err(|e| inp(format!("structure.geometry (n x 3, bohr): {e}")))?;
            let z: Vec<u32> = match (nums, syms) {
                (Some(n), None) => {
                    serde_json::from_value(n).map_err(|e| inp(format!("structure.numbers: {e}")))?
                }
                (None, Some(s)) => {
                    let s: Vec<String> = serde_json::from_value(s)
                        .map_err(|e| inp(format!("structure.symbols: {e}")))?;
                    s.iter()
                        .map(|x| {
                            symbol_to_z(x).ok_or_else(|| inp(format!("unknown element {x:?}")))
                        })
                        .collect::<Result<_, _>>()?
                }
                _ => {
                    return Err(inp(
                        "structure: give either symbols or numbers with geometry".into(),
                    ))
                }
            };
            if z.len() != geometry.len() {
                return Err(inp(format!(
                    "structure: {} atoms but {} positions in geometry",
                    z.len(),
                    geometry.len()
                )));
            }
            (z, geometry)
        }
        _ => {
            return Err(inp(
                "structure: give symbols (or numbers) and geometry (bohr), or xyz (text, Å)".into(),
            ))
        }
    };
    Structure::new(z, geometry, charge, mult).map_err(LinusError::from)
}

impl Request {
    /// A request from its JSON object (module docs).
    pub fn from_json(v: Value) -> Result<Request, LinusError> {
        let Value::Object(m) = v else {
            return Err(LinusError::Input(
                "the request must be a JSON object".into(),
            ));
        };
        let mut settings = Map::new();
        let (mut structure, mut calctype) = (None, None);
        for (k, v) in m {
            match norm_name(&k).as_str() {
                "structure" => structure = Some(v),
                "calctype" => calctype = Some(v),
                _ => {
                    settings.insert(k, v);
                }
            }
        }
        let structure = structure_from_json(structure.ok_or_else(|| {
            LinusError::Input(
                "the request needs a structure (symbols or numbers with geometry, or xyz)".into(),
            )
        })?)?;
        let kind: CalcType = match calctype {
            None | Some(Value::Null) => CalcType::Energy,
            Some(Value::String(c)) => c.parse().map_err(LinusError::Input)?,
            Some(v) => {
                return Err(LinusError::Input(format!(
                    "calctype must be a string, not {v}"
                )))
            }
        };
        Ok(Request {
            structure,
            calctype: kind,
            settings: CalcOptions::from_json(Value::Object(settings))?,
        })
    }
}

/// Sets `key` (dotted: `scf.level_shift`, `structure.charge`) in the request tree `m` to `v`,
/// making the groups on the way; a group's shorthand on the way becomes its object form
/// (`jk: "df"` then `jk.aux` is `{"kind": "df", "aux": ...}`).
pub fn set_path(m: &mut Map<String, Value>, key: &str, v: Value) -> Result<(), LinusError> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.iter().any(|p| p.is_empty()) || parts.len() > 2 {
        return Err(LinusError::Input(format!(
            "bad key {key:?}: a setting is `key` or `group.key`"
        )));
    }
    if let [group, k] = parts[..] {
        let e = m
            .entry(group.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if !e.is_object() {
            *e = Value::Object(expand_group_shorthand(group, e)?);
        }
        e.as_object_mut()
            .expect("an object")
            .insert(k.to_string(), v);
    } else {
        m.insert(key.to_string(), v);
    }
    Ok(())
}

/// Merges `over` into `base`: groups key by key (a shorthand in either refined through its
/// object form), other values replaced.
pub fn merge(base: &mut Map<String, Value>, over: Map<String, Value>) -> Result<(), LinusError> {
    for (k, v) in over {
        match (base.get_mut(&k), v) {
            (Some(b), Value::Object(o)) if !b.is_null() => {
                let mut e = expand_group_shorthand(&k, b)?;
                for (k2, v2) in o {
                    e.insert(k2, v2);
                }
                *b = Value::Object(e);
            }
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::options::{BackendKind, JkMethod};
    use serde_json::json;

    #[test]
    fn requests() {
        let r = Request::from_json(json!({
            "structure": {"symbols": ["He"], "geometry": [[0, 0, 0]], "charge": 0},
            "calctype": "gradient", "method": "pbe", "jk": "df", "grid": 2}))
        .unwrap();
        assert_eq!(r.structure.z, vec![2]);
        assert_eq!(r.calctype, CalcType::Gradient);
        assert_eq!(r.settings.method(), "pbe");
        let r =
            Request::from_json(json!({"structure": {"xyz": "1\n\nHe 0 0 0\n", "multiplicity": 1}}))
                .unwrap();
        assert_eq!(r.calctype, CalcType::Energy);
        // errors: no structure, unknown keys, mismatched sizes
        assert!(Request::from_json(json!({"method": "hf"})).is_err());
        let e =
            Request::from_json(json!({"structure": {"symbols": ["He"], "positions": [[0, 0, 0]]}}))
                .unwrap_err();
        assert!(e.to_string().contains("structure.positions"), "{e}");
        assert!(Request::from_json(
            json!({"structure": {"numbers": [2, 1], "geometry": [[0, 0, 0]]}})
        )
        .is_err());
        assert!(Request::from_json(
            json!({"structure": {"numbers": [2], "geometry": [[0, 0, 0]]}, "metod": "pbe"})
        )
        .unwrap_err()
        .to_string()
        .contains("metod"));
    }

    /// A dotted path below a shorthand refines its object form; merges likewise.
    #[test]
    fn paths_refine_shorthands() {
        let mut m = Map::new();
        set_path(&mut m, "jk", json!("df")).unwrap();
        set_path(&mut m, "jk.aux", json!("def2-universal-jkfit")).unwrap();
        set_path(&mut m, "backend", json!("auto")).unwrap();
        set_path(&mut m, "precision", json!("emulated")).unwrap();
        set_path(&mut m, "precision.fp32", json!("on")).unwrap();
        set_path(&mut m, "solvent", json!(7.4)).unwrap();
        set_path(&mut m, "solvent.model", json!("iefpcm")).unwrap();
        let o = CalcOptions::from_json(Value::Object(m.clone())).unwrap();
        assert_eq!(
            o.jk_method(),
            JkMethod::Df {
                aux: "def2-universal-jkfit".into()
            }
        );
        assert_eq!(o.backend.kind(), BackendKind::Auto);
        assert_eq!(
            (o.precision.fp64(), o.precision.fp32()),
            (
                crate::driver::options::Fp64Mode::Emulated,
                crate::driver::options::Fp32Mode::On
            )
        );
        assert_eq!(o.solvent.as_ref().unwrap().eps(), Some(7.4));
        // a value with nothing below it is an error; three levels are not a key
        set_path(&mut m, "method", json!("pbe")).unwrap();
        assert!(set_path(&mut m, "method.x", json!(1)).is_err());
        assert!(set_path(&mut m, "a.b.c", json!(1)).is_err());
        // a file's group over a flag's shorthand refines it
        let mut base = Map::from_iter([("grid".to_string(), json!("linus:2"))]);
        merge(
            &mut base,
            Map::from_iter([("grid".to_string(), json!({"nlc": "linus:1"}))]),
        )
        .unwrap();
        assert_eq!(base["grid"], json!({"xc": "linus:2", "nlc": "linus:1"}));
    }
}
