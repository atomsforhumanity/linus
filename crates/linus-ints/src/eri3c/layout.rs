//! The shell pairs a density-fitting layout stores, and its sizes: one screening rule for every
//! engine that stores or plans a three-index tensor (linus-jk's `DfJk` and its gradient's
//! layouts, linus-gpu's `DfLayout`, linus's memory needs).

use crate::basis::BasisSet;
use crate::pairs::PairSet;

/// The shell pairs of `pairs` a density-fitting layout keeps, in index order: those with
/// primitive pairs whose Schwarz factor times the auxiliary basis's largest, `aux_qmax`, is at
/// least `cut` (`DfThresholds::pair`).
pub fn df_kept_pairs(pairs: &PairSet, aux_qmax: f64, cut: f64) -> Vec<usize> {
    let q = pairs.schwarz();
    (0..pairs.len())
        .filter(|&i| pairs.pair(i).nprim > 0 && q[i] * aux_qmax >= cut)
        .collect()
}

/// The sizes of a density-fitting layout over its kept shell pairs A >= B: its stored rows (one
/// per function pair (m, n), both triangles of a diagonal block) and the shell panels of the
/// device's exchange (each shell's functions against the functions of its kept partner shells,
/// both triangles: a stored row of an off-diagonal pair in the panels of both its shells).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DfRows {
    /// Stored rows.
    pub nx: usize,
    /// Panel rows: the sum over shells of their functions times their partners' functions.
    pub np: usize,
    /// Gathered-orbital columns: the sum over shells of their partners' functions.
    pub ncg: usize,
    /// The panel rows of the largest shell.
    pub max_panel_rows: usize,
    /// The stored rows of the largest kept pair (a setup batch holds at least one pair).
    pub max_pair_rows: usize,
}

impl DfRows {
    /// The sizes of the kept pairs `kept` of `pairs` over the shells of `basis`.
    pub fn new(basis: &BasisSet, pairs: &PairSet, kept: &[usize]) -> DfRows {
        DfRows::of(
            |s| basis.shells[s].nfunc(),
            kept.iter().map(|&i| (pairs.pair(i).sa, pairs.pair(i).sb)),
        )
    }

    /// The sizes of the kept shell pairs `kept` (A, B), A >= B, with `dim` functions per shell.
    pub fn of(dim: impl Fn(usize) -> usize, kept: impl Iterator<Item = (usize, usize)>) -> DfRows {
        let mut width: Vec<usize> = Vec::new();
        let (mut nx, mut max_pair_rows) = (0, 0);
        for (sa, sb) in kept {
            let (da, db) = (dim(sa), dim(sb));
            nx += da * db;
            max_pair_rows = max_pair_rows.max(da * db);
            if width.len() <= sa.max(sb) {
                width.resize(sa.max(sb) + 1, 0);
            }
            width[sa] += db;
            if sa != sb {
                width[sb] += da;
            }
        }
        let (mut np, mut ncg, mut max_panel_rows) = (0, 0, 0);
        for (s, &w) in width.iter().enumerate() {
            let rows = dim(s) * w;
            np += rows;
            ncg += w;
            max_panel_rows = max_panel_rows.max(rows);
        }
        DfRows {
            nx,
            np,
            ncg,
            max_panel_rows,
            max_pair_rows,
        }
    }
}
