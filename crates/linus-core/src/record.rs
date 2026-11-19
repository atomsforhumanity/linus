//! The run record: what a calculation did and decided, written by the code that decides, at
//! the moment it decides.
//!
//! A report is then the plan plus this record, not a reconstruction after the run. The record
//! is generic over the decision type `D`: the driver defines its decisions as one typed enum
//! (the density-fitting plan, each kernel family's placement and fallback with its reason, the
//! stability probe used, the escalation, ...), which this crate does not need to know.

use crate::timing::Timers;

/// The timings, decisions and warnings of one calculation.
#[derive(Clone, Debug)]
pub struct RunRecord<D> {
    pub timers: Timers,
    /// In the order they were made.
    pub decisions: Vec<D>,
    pub warnings: Vec<Warning>,
}

impl<D> Default for RunRecord<D> {
    fn default() -> Self {
        RunRecord {
            timers: Timers::default(),
            decisions: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

impl<D> RunRecord<D> {
    pub fn decide(&mut self, d: D) {
        self.decisions.push(d);
    }

    pub fn warn(&mut self, code: &'static str, path: &'static str, message: impl Into<String>) {
        self.warnings.push(Warning {
            code,
            message: message.into(),
            path,
        });
    }
}

/// Something the user should know about a calculation that still succeeded: a stable `code`
/// to match on (named after the condition, not the remedy), a message to read, and the `path` of
/// the setting or record it concerns (the cross-program `{code, message, path}` shape of qcdata's
/// ProgramWarning).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Warning {
    pub code: &'static str,
    pub message: String,
    pub path: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_in_order() {
        #[derive(Debug, PartialEq)]
        enum Decision {
            Aux(&'static str),
        }
        let mut r = RunRecord::default();
        r.decide(Decision::Aux("def2-universal-jkfit"));
        r.warn("backend_fallback", "backend", "device setup: out of memory");
        r.timers.add("scf", 1.0);
        assert_eq!(r.decisions, [Decision::Aux("def2-universal-jkfit")]);
        assert_eq!(r.warnings[0].code, "backend_fallback");
    }
}
