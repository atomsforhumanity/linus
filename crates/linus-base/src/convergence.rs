//! The SCF's convergence thresholds ([`Convergence`], with the norm of the orbital gradient
//! they bound, [`GradNorm`]): set by the `scf` settings, applied by the SCF (linus-scf) and
//! reported by its failure ([`crate::error::ScfFailure`]).

use serde::{Deserialize, Serialize};

/// The norm of the orbital gradient that `g_tol` bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradNorm {
    /// max_pq |X^T (F_s D_s S - S D_s F_s) X|_pq over spins (D_s: spin density). In the MO basis
    /// these elements are +-F_ai, so this is max |F_ai|.
    Max,
    /// PySCF's norm_gorb: ||g||_2 with g = 2 F_vo (RHF) or g = (F^a_vo, F^b_vo) (UHF).
    Pyscf,
}

/// SCF convergence thresholds: |dE| < e_tol and orbital gradient (`g_norm`) < g_tol.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Convergence {
    pub e_tol: f64,
    pub g_tol: f64,
    pub g_norm: GradNorm,
}

impl Convergence {
    /// Named presets. ORCA's names take its TolE as e_tol and its DIIS-error tolerance TolErr
    /// as the bound on max |F_ai| (ORCA's own default check, ConvCheckMode 2, stops on the
    /// energy alone: dE < TolE and dE(1-electron) < 1e3 TolE, so these presets are stricter
    /// than ORCA's keywords of the same name). `pyscf`: PySCF's defaults (conv_tol 1e-9,
    /// conv_tol_grad sqrt(conv_tol) on its gradient norm).
    pub fn preset(name: &str) -> Option<Convergence> {
        let max = |e, g| {
            Some(Convergence {
                e_tol: e,
                g_tol: g,
                g_norm: GradNorm::Max,
            })
        };
        match name.to_ascii_lowercase().replace(['_', '-'], "").as_str() {
            "loose" => max(1e-5, 1e-4),
            "normal" => max(1e-6, 1e-5),
            "tight" => max(1e-8, 5e-7),
            "verytight" => max(1e-9, 1e-8),
            "extreme" => max(1e-12, 1e-10),
            "pyscf" => Some(Convergence {
                e_tol: 1e-9,
                g_tol: 1e-9f64.sqrt(),
                g_norm: GradNorm::Pyscf,
            }),
            _ => None,
        }
    }

    pub const PRESETS: [&'static str; 6] =
        ["loose", "normal", "tight", "verytight", "extreme", "pyscf"];
}
