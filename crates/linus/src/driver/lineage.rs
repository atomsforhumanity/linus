//! Lineages: the chains of calculations a session carries state along (an optimization, an IRC
//! branch, an image of a path, a finite-difference star). Each lineage keeps, from its last
//! successful request, what makes its next SCF faster and its probes cheaper:
//!
//! - **The warm start.** Its last positions, occupied orbitals and converged spin densities. The
//!   next SCF of the lineage starts from the densities at the same positions, from the orbitals
//!   made orthonormal at moved atoms (Löwdin: C (CᵀSC)^-1/2; the densities if they are linearly
//!   dependent there) within [`GUESS_DISTANCE`], and from the configured guess (`scf.guess`)
//!   farther away, without history, or after a reset.
//! - **The probes.** The unset stability probes (`scf.stability_probe`,
//!   `scf.stability_closed_probe`) run at every SCF that starts fresh and not at an SCF that
//!   continues its lineage, unless the HOMO-LUMO gap fires the trigger
//!   ([`linus_scf::ProbeTrigger`]), which compares the gap with the one at the lineage's last
//!   probe.
//!
//! A lineage changes only when a whole request succeeds (`Lineage::after`): a failed SCF, or a
//! converged one whose Hessian then fails, leaves it as it was, so a session can retry a smaller
//! step or move on from its last good state. Lineages are keyed by any JSON value
//! ([`LineageKey`]); `null` is the session's default lineage.

use faer::{Mat, MatRef};
use linus_scf::{ProbeTrigger, ScfAlgorithm, ScfResult};
use serde::Serialize;
use serde_json::Value;

/// The largest atomic displacement (bohr) from a lineage's last positions at which its orbitals
/// start the next SCF; farther away the SCF starts from the configured guess. The geometry
/// optimizations of the session benchmark (bench/session) step at most 0.46 bohr.
pub const GUESS_DISTANCE: f64 = 1.0;

/// The key of a lineage: the canonical JSON text of the caller's value (object keys sorted,
/// numbers as JSON writes them), so equal values name one lineage whatever their spelling;
/// `null` is the session's default lineage.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LineageKey(String);

impl LineageKey {
    /// The key of the JSON value `v`.
    pub fn new(v: &Value) -> LineageKey {
        // serde_json's maps are sorted, so its text of a value is canonical
        LineageKey(v.to_string())
    }

    /// The key of the JSON text `s`.
    pub fn parse(s: &str) -> Result<LineageKey, linus_base::error::LinusError> {
        let v: Value = serde_json::from_str(s).map_err(|e| {
            linus_base::error::LinusError::Input(format!("a lineage is a JSON value: {e}"))
        })?;
        Ok(LineageKey::new(&v))
    }

    /// The key of a caller's canonical text, kept as it is: Python's `qcdata.lineage_key` writes
    /// it (sorted keys, compact, numbers as Python writes them), and the binding and the worker
    /// pass that text on, so a lineage is the same key in process and over a wire. It must be
    /// JSON.
    pub fn from_text(text: &str) -> Result<LineageKey, linus_base::error::LinusError> {
        serde_json::from_str::<Value>(text).map_err(|e| {
            linus_base::error::LinusError::Input(format!(
                "a lineage key is JSON text ({text:?}): {e}"
            ))
        })?;
        Ok(LineageKey(text.to_string()))
    }

    /// The key's JSON value.
    pub fn value(&self) -> Value {
        serde_json::from_str(&self.0).expect("a key is JSON text")
    }

    /// The key's canonical JSON text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for LineageKey {
    /// The session's default lineage (`null`).
    fn default() -> Self {
        LineageKey("null".into())
    }
}

impl std::fmt::Display for LineageKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a lineage keeps from its last successful request (see the module docs).
#[derive(Clone, Debug)]
pub(crate) struct Lineage {
    /// The positions of its last SCF (bohr).
    positions: Vec<[f64; 3]>,
    /// Its occupied orbitals per spin (nbf x nocc; a restricted SCF's beta ones are its alpha
    /// ones).
    c_a: Mat<f64>,
    c_b: Mat<f64>,
    restricted: bool,
    /// Its converged spin densities.
    d_a: Mat<f64>,
    d_b: Mat<f64>,
    /// The gap ([`linus_scf::gate_gap`]) at its last probe: the last SCF that started fresh, or
    /// whose gap trigger fired, or given with a continuation.
    probe_gap: Option<f64>,
}

impl Lineage {
    /// The lineage after its request succeeded with the SCF `r` at `positions`, started as
    /// `start` says: the next SCF starts from `r`, and the probes' last gap is `r`'s when its
    /// probes' gates looked at it (a fresh start, or the trigger fired), else the one it continued.
    pub fn after(start: &ScfStart, r: &ScfResult, positions: &[[f64; 3]]) -> Lineage {
        let probed = !start.continuation || r.probe_trigger.is_some_and(|t| t.fired());
        Lineage {
            positions: positions.to_vec(),
            c_a: r.c_a.subcols(0, r.nalpha).to_owned(),
            c_b: r.c_b.subcols(0, r.nbeta).to_owned(),
            restricted: r.restricted,
            d_a: r.d_a.clone(),
            d_b: r.d_b.clone(),
            probe_gap: if probed {
                linus_scf::gate_gap(&r.eps_a, &r.eps_b, r.nalpha, r.nbeta, r.restricted)
            } else {
                start.last_probe_gap
            },
        }
    }

    /// The largest atomic displacement (bohr) from the lineage's last positions to `positions`.
    pub fn distance(&self, positions: &[[f64; 3]]) -> f64 {
        self.positions
            .iter()
            .zip(positions)
            .map(|(a, b)| {
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .fold(0.0, f64::max)
    }

    /// Its occupied orbitals (alpha, beta) and whether they are restricted.
    pub fn occupied(&self) -> (MatRef<'_, f64>, MatRef<'_, f64>, bool) {
        (self.c_a.as_ref(), self.c_b.as_ref(), self.restricted)
    }

    /// Its converged spin densities.
    pub fn densities(&self) -> (Mat<f64>, Mat<f64>) {
        (self.d_a.clone(), self.d_b.clone())
    }

    /// The gap at its last probe.
    pub fn probe_gap(&self) -> Option<f64> {
        self.probe_gap
    }
}

/// Where an SCF's guess came from (`record.scf.guess.source`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GuessSource {
    /// The superposition of atomic densities (`scf.guess`).
    Sad,
    /// The core Hamiltonian (`scf.guess`).
    Core,
    /// The lineage's last orbitals or densities (a warm start).
    Lineage,
    /// Orbitals or densities the caller gave.
    Given,
}

/// Why an SCF of a lineage started from the configured guess (`record.scf.guess.reason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FreshReason {
    /// The lineage has no history: new, or reset.
    NoHistory,
    /// The positions lie farther than [`GUESS_DISTANCE`] from the lineage's last ones.
    Distance,
}

/// Where an SCF started (`record.scf.guess`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GuessRecord {
    pub source: GuessSource,
    /// The lineage of the request (its JSON value; null: the default lineage).
    pub lineage: Value,
    /// The lineage whose state started the SCF: its own, or a seed of the request (None: a
    /// fresh or given start).
    pub from_lineage: Option<Value>,
    /// The largest atomic displacement (bohr) from the lineage's last positions (None: no
    /// history).
    pub distance: Option<f64>,
    /// The orbitals were made orthonormal at these positions (false: densities used as they
    /// are, at the same positions or because the orbitals are linearly dependent here, or no
    /// orbitals).
    pub orthonormalized: bool,
    /// Why the SCF started from the configured guess.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<FreshReason>,
}

impl GuessRecord {
    /// The SCF started from its lineage's history (`record.warm_start`).
    pub fn warm_start(&self) -> bool {
        self.source == GuessSource::Lineage
    }
}

/// Where one SCF starts and the probes' rule for it, built for a request from its lineage.
#[derive(Clone, Debug)]
pub(crate) struct ScfStart {
    /// The spin densities the SCF starts from (None: the configured guess).
    pub guess: Option<(Mat<f64>, Mat<f64>)>,
    /// The SCF continues its lineage: the unset probes are off unless the gap trigger fires.
    pub continuation: bool,
    /// The gap at the lineage's last probe, which the gap trigger compares with.
    pub last_probe_gap: Option<f64>,
    pub record: GuessRecord,
}

impl ScfStart {
    /// A fresh start from the configured guess `configured` (sad or core) in `lineage`.
    pub fn fresh(
        configured: GuessSource,
        lineage: &LineageKey,
        distance: Option<f64>,
        reason: FreshReason,
    ) -> ScfStart {
        ScfStart {
            guess: None,
            continuation: false,
            last_probe_gap: None,
            record: GuessRecord {
                source: configured,
                lineage: lineage.value(),
                from_lineage: None,
                distance,
                orthonormalized: false,
                reason: Some(reason),
            },
        }
    }

    /// Hessian-vector products of the stability probe (`algo`'s, else
    /// [`linus_scf::STABILITY_PROBE_AUTO`] at a fresh start and none at a continuation).
    pub fn stability_probe(&self, algo: &ScfAlgorithm) -> usize {
        algo.stability_probe.unwrap_or(if self.continuation {
            0
        } else {
            linus_scf::STABILITY_PROBE_AUTO
        })
    }

    /// Hessian-vector products of the closed-shell probe (decision D36): `algo`'s, else
    /// [`linus_scf::CLOSED_SHELL_PROBE_AUTO`] for KS-DFT (`dft`) at a fresh start, none for
    /// Hartree-Fock and at a continuation.
    pub fn closed_shell_probe(&self, algo: &ScfAlgorithm, dft: bool) -> usize {
        algo.closed_shell_probe
            .unwrap_or(if dft && !self.continuation {
                linus_scf::CLOSED_SHELL_PROBE_AUTO
            } else {
                0
            })
    }

    /// The gap trigger of a continuation (None at a fresh start, or with both probes set in
    /// `algo`): the unset probes run when the gap fires it.
    pub fn probe_trigger(&self, algo: &ScfAlgorithm, dft: bool) -> Option<ProbeTrigger> {
        if !self.continuation {
            return None;
        }
        let t = ProbeTrigger {
            last_gap: self.last_probe_gap,
            threshold: algo.closed_shell_gap,
            stability_products: if algo.stability_probe.is_none() {
                linus_scf::STABILITY_PROBE_AUTO
            } else {
                0
            },
            closed_products: if algo.closed_shell_probe.is_none() && dft {
                linus_scf::CLOSED_SHELL_PROBE_AUTO
            } else {
                0
            },
        };
        (t.stability_products > 0 || t.closed_products > 0).then_some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Equal JSON values name one lineage whatever their spelling; null is the default.
    #[test]
    fn keys_are_canonical() {
        let a = LineageKey::parse(r#"{"b": 1, "a": [1, "x"]}"#).unwrap();
        let b = LineageKey::new(&json!({"a": [1, "x"], "b": 1}));
        assert_eq!(a, b);
        assert_eq!(a.as_str(), r#"{"a":[1,"x"],"b":1}"#);
        assert_eq!(LineageKey::new(&Value::Null), LineageKey::default());
        assert_ne!(LineageKey::new(&json!(3)), LineageKey::new(&json!("3")));
        assert_eq!(LineageKey::new(&json!("irc")).value(), json!("irc"));
        assert!(LineageKey::parse("{").is_err());
    }

    /// The probes run at a fresh start and not at a continuation, whose trigger is armed.
    #[test]
    fn probes_follow_the_start() {
        let algo = ScfAlgorithm::default();
        let fresh = ScfStart::fresh(
            GuessSource::Sad,
            &LineageKey::default(),
            None,
            FreshReason::NoHistory,
        );
        assert_eq!(
            fresh.stability_probe(&algo),
            linus_scf::STABILITY_PROBE_AUTO
        );
        assert_eq!(
            fresh.closed_shell_probe(&algo, true),
            linus_scf::CLOSED_SHELL_PROBE_AUTO
        );
        assert_eq!(fresh.closed_shell_probe(&algo, false), 0);
        assert!(fresh.probe_trigger(&algo, true).is_none());
        let cont = ScfStart {
            continuation: true,
            last_probe_gap: Some(0.2),
            ..fresh
        };
        assert_eq!(cont.stability_probe(&algo), 0);
        assert_eq!(cont.closed_shell_probe(&algo, true), 0);
        assert_eq!(cont.probe_trigger(&algo, true).unwrap().last_gap, Some(0.2));
    }
}
