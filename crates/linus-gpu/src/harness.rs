//! Test harness: fixtures (the benchmark molecules, basis sets, grids) and comparisons of GPU
//! results with linus's CPU implementations. Molecules are linus-core's [`Structure`].

use crate::cpu::{BasisDef, BasisSet, GridSpec, MolecularGrid};
use crate::error::GpuError;
use linus_core::elements::z_to_symbol;
pub use linus_core::Structure;
use std::path::{Path, PathBuf};

/// `$LINUS_MOLECULES_DIR`, else the repository's bench/molecules.
pub fn molecules_dir() -> PathBuf {
    std::env::var_os("LINUS_MOLECULES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/molecules"))
}

/// Reads an xyz file (angstrom); the charge comes from `charge=N` or `charge N` in the
/// comment line (default 0), the multiplicity is the lowest for the electron count.
pub fn load_xyz(path: &Path) -> Result<Structure, GpuError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| GpuError::Input(format!("{}: {e}", path.display())))?;
    let comment = text.lines().nth(1).unwrap_or("");
    let words: Vec<&str> = comment
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|w| !w.is_empty())
        .collect();
    let charge = (0..words.len())
        .find_map(|i| match words[i].strip_prefix("charge=") {
            Some(v) => v.parse().ok(),
            None if words[i] == "charge" => words.get(i + 1).and_then(|v| v.parse().ok()),
            None => None,
        })
        .unwrap_or(0);
    let input = |e: String| GpuError::Input(format!("{}: {e}", path.display()));
    let (z, xyz) = linus_core::structure::parse_xyz(&text).map_err(input)?;
    let nel = z.iter().map(|&z| z as i64).sum::<i64>() - charge as i64;
    Structure::new(z, xyz, charge, 1 + nel.rem_euclid(2) as u32).map_err(|e| input(e.to_string()))
}

/// A benchmark molecule by name (`bench/molecules/<name>.xyz`).
pub fn structure(name: &str) -> Result<Structure, GpuError> {
    load_xyz(&molecules_dir().join(format!("{name}.xyz")))
}

/// The molecular formula of `mol` (Hill order: C, H, then the others alphabetically), to
/// name it in messages.
pub fn formula(mol: &Structure) -> String {
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for &z in &mol.z {
        let s = z_to_symbol(z).unwrap_or("?");
        match counts.iter_mut().find(|c| c.0 == s) {
            Some(c) => c.1 += 1,
            None => counts.push((s, 1)),
        }
    }
    let has_c = counts.iter().any(|c| c.0 == "C");
    counts.sort_by_key(|&(s, _)| match s {
        "C" if has_c => (0, s),
        "H" if has_c => (1, s),
        _ => (2, s),
    });
    counts
        .iter()
        .map(|&(s, n)| {
            if n == 1 {
                s.to_string()
            } else {
                format!("{s}{n}")
            }
        })
        .collect()
}

/// Basis set `name` (spherical) on `mol`, from linus's basis directory.
pub fn basis(mol: &Structure, name: &str) -> Result<BasisSet, GpuError> {
    let def = BasisDef::load(name, None).map_err(|e| GpuError::Contract(e.to_string()))?;
    BasisSet::new(&def, &mol.z, &mol.positions, true).map_err(|e| GpuError::Contract(e.to_string()))
}

/// The molecular grid of `spec` on `mol`.
pub fn grid(mol: &Structure, spec: &GridSpec) -> Result<MolecularGrid, GpuError> {
    crate::cpu::build_grid(&mol.z, &mol.positions, spec)
        .map_err(|e| GpuError::Contract(e.to_string()))
}

/// Element-wise comparison of a GPU result with the CPU reference.
#[derive(Clone, Debug, Default)]
pub struct ErrStats {
    pub n: usize,
    /// Largest |gpu - cpu| and its index.
    pub max_abs: f64,
    pub argmax: usize,
    /// Largest |cpu|, and max_abs relative to it.
    pub ref_max: f64,
    pub max_rel: f64,
    pub rms: f64,
    /// Entries with identical bits.
    pub identical: usize,
}

/// Compares `gpu` with `cpu` (same length).
pub fn compare(gpu: &[f64], cpu: &[f64]) -> ErrStats {
    assert_eq!(gpu.len(), cpu.len(), "compare: lengths differ");
    let mut s = ErrStats {
        n: gpu.len(),
        ..Default::default()
    };
    let mut sq = 0.0;
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        let d = (g - c).abs();
        if d > s.max_abs || d.is_nan() {
            s.max_abs = if d.is_nan() { f64::INFINITY } else { d };
            s.argmax = i;
        }
        s.ref_max = s.ref_max.max(c.abs());
        sq += d * d;
        s.identical += (g.to_bits() == c.to_bits()) as usize;
    }
    s.rms = (sq / s.n.max(1) as f64).sqrt();
    s.max_rel = if s.ref_max > 0.0 {
        s.max_abs / s.ref_max
    } else {
        s.max_abs
    };
    s
}

impl ErrStats {
    /// Ok when max_abs <= abs + rel * ref_max.
    pub fn check(&self, abs: f64, rel: f64) -> Result<(), String> {
        if self.max_abs <= abs + rel * self.ref_max {
            Ok(())
        } else {
            Err(format!("max |gpu - cpu| = {:.3e} at {} (largest |cpu| {:.3e}) exceeds {abs:.1e} + {rel:.1e} * {:.3e}", self.max_abs, self.argmax, self.ref_max, self.ref_max))
        }
    }
}

impl std::fmt::Display for ErrStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n {} max_abs {:.2e} (rel {:.2e}) rms {:.2e} identical {:.1}%",
            self.n,
            self.max_abs,
            self.max_rel,
            self.rms,
            100.0 * self.identical as f64 / self.n.max(1) as f64
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_benchmark_molecules() {
        let m = structure("u1an39").unwrap();
        assert_eq!(m.z.len(), 39);
        assert_eq!((m.charge, m.multiplicity), (-1, 1));
        let t = structure("ts113").unwrap();
        assert_eq!(t.z.len(), 113);
        assert_eq!(t.charge, 0);
        assert_eq!(t.nelectron() % 2, 0);
        let w = Structure::new(
            vec![8, 1, 1],
            vec![[0.0; 3], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
            0,
            1,
        );
        assert_eq!(formula(&w.unwrap()), "H2O");
        assert!(formula(&m).starts_with('C'));
    }
}
