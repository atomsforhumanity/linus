//! Timings: wall-clock phase times accumulated by name ([`Timers`]), with the `resources` that
//! travel with them (sizes, counts and memory marks, not seconds), and the phase paths that name
//! them ([`Span`]).
//!
//! The report's `timings` are flat keys (`scf_jk`, `hess_xc`, `grad_2e_setup`); a [`Span`] is
//! the path behind such a key (`hess` / `xc`), so a caller can hand a sub-calculation its prefix
//! instead of renaming the sub-calculation's keys afterwards. What a calculation decided on the
//! way is not carried here: it is typed in the run record ([`crate::record`]).

use serde_json::Value;

/// A phase path; its key is the parts joined by `_` (`Span::new("hess").child("xc")` is
/// `hess_xc`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    key: String,
}

impl Span {
    /// The top-level phase `name`.
    pub fn new(name: &str) -> Self {
        Span { key: name.into() }
    }

    /// The phase `name` inside this one.
    pub fn child(&self, name: &str) -> Span {
        Span {
            key: format!("{}_{name}", self.key),
        }
    }

    /// The flat key of the report's timings.
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// Phase times (wall clock) by name, in first-use order, and the resources set along the way.
#[derive(Clone, Debug, Default)]
pub struct Timers {
    /// (name, seconds, calls)
    pub entries: Vec<(String, f64, usize)>,
    /// (name, value): storage sizes (GiB), bytes per element, block counts, peak memory (GB);
    /// the last value set wins.
    pub resources: Vec<(String, f64)>,
}

impl Timers {
    pub fn add(&mut self, name: &str, secs: f64) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.0 == name) {
            e.1 += secs;
            e.2 += 1;
        } else {
            self.entries.push((name.to_string(), secs, 1));
        }
    }
    pub fn time<T>(&mut self, name: &str, f: impl FnOnce() -> T) -> T {
        let t = std::time::Instant::now();
        let r = f();
        self.add(name, t.elapsed().as_secs_f64());
        r
    }
    /// Set a resource value (not a time).
    pub fn resource(&mut self, name: &str, value: f64) {
        match self.resources.iter_mut().find(|e| e.0 == name) {
            Some(e) => e.1 = value,
            None => self.resources.push((name.to_string(), value)),
        }
    }
    /// The seconds of `name` (None if never timed).
    pub fn get(&self, name: &str) -> Option<f64> {
        self.entries.iter().find(|e| e.0 == name).map(|e| e.1)
    }
    pub fn merge(&mut self, other: &Timers) {
        for (n, s, c) in &other.entries {
            if let Some(e) = self.entries.iter_mut().find(|e| &e.0 == n) {
                e.1 += s;
                e.2 += c;
            } else {
                self.entries.push((n.clone(), *s, *c));
            }
        }
        for (n, v) in &other.resources {
            self.resource(n, *v);
        }
    }
    /// [`Timers::merge`] with every name of `other` placed under `span` (`x` becomes
    /// `<span>_x`).
    pub fn merge_under(&mut self, span: &Span, other: &Timers) {
        for (n, s, c) in &other.entries {
            let key = span.child(n);
            match self.entries.iter_mut().find(|e| e.0 == key.key()) {
                Some(e) => {
                    e.1 += s;
                    e.2 += c;
                }
                None => self.entries.push((key.key, *s, *c)),
            }
        }
        for (n, v) in &other.resources {
            self.resource(span.child(n).key(), *v);
        }
    }
    /// Make `total` the wall time of a whole call (replacing any `total` set before).
    pub fn set_total(&mut self, secs: f64) {
        self.entries.retain(|e| e.0 != "total");
        self.entries.push(("total".into(), secs, 1));
    }
    /// The seconds per phase.
    pub fn to_json(&self) -> Value {
        Value::Object(
            self.entries
                .iter()
                .map(|(n, s, _)| (n.clone(), serde_json::json!(s)))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_name_flat_keys() {
        let hess = Span::new("hess");
        assert_eq!(hess.child("xc").child("grid").key(), "hess_xc_grid");
        let mut sub = Timers::default();
        sub.add("jk", 1.0);
        sub.add("jk", 2.0);
        sub.resource("blocks", 4.0);
        let mut t = Timers::default();
        t.add("hess_jk", 0.5);
        t.merge_under(&hess, &sub);
        assert_eq!(t.entries, [("hess_jk".to_string(), 3.5, 3)]);
        assert_eq!(t.resources, [("hess_blocks".to_string(), 4.0)]);
    }
}
