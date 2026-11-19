//! Uniform cell list over atom positions for radius queries (O(1) neighbourhoods).

#[derive(Clone, Debug)]
pub(crate) struct CellList {
    origin: [f64; 3],
    h: f64,
    dims: [usize; 3],
    start: Vec<u32>,
    atoms: Vec<u32>,
    lo: [f64; 3],
    hi: [f64; 3],
}

impl CellList {
    /// Build with target cell edge `h` (bohr); the number of cells per axis is capped at 64.
    pub(crate) fn new(xyz: &[[f64; 3]], h: f64) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in xyz {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if xyz.is_empty() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f64, f64::max);
        let h = h.max(ext / 64.0).max(1e-3);
        let mut dims = [1usize; 3];
        for k in 0..3 {
            dims[k] = (((hi[k] - lo[k]) / h).floor() as usize + 1).max(1);
        }
        let ncell = dims[0] * dims[1] * dims[2];
        let mut counts = vec![0u32; ncell + 1];
        let cell_of = |p: &[f64; 3]| -> usize {
            let mut c = [0usize; 3];
            for k in 0..3 {
                c[k] = (((p[k] - lo[k]) / h).floor().max(0.0) as usize).min(dims[k] - 1);
            }
            (c[0] * dims[1] + c[1]) * dims[2] + c[2]
        };
        let cells: Vec<usize> = xyz.iter().map(cell_of).collect();
        for &c in &cells {
            counts[c + 1] += 1;
        }
        for i in 0..ncell {
            counts[i + 1] += counts[i];
        }
        let mut fill = counts.clone();
        let mut atoms = vec![0u32; xyz.len()];
        for (i, &c) in cells.iter().enumerate() {
            atoms[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        CellList {
            origin: lo,
            h,
            dims,
            start: counts,
            atoms,
            lo,
            hi,
        }
    }

    /// Calls `f(atom)` for every atom in cells overlapping the ball (p, rho). The caller must
    /// check the actual distance. Atoms are visited in no particular order.
    #[inline]
    pub(crate) fn for_each_candidate(&self, p: &[f64; 3], rho: f64, mut f: impl FnMut(usize)) {
        // whole box covered -> all atoms
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for k in 0..3 {
            if p[k] + rho < self.lo[k] || p[k] - rho > self.hi[k] {
                return;
            }
            let a = ((p[k] - rho - self.origin[k]) / self.h).floor();
            let b = ((p[k] + rho - self.origin[k]) / self.h).floor();
            lo[k] = if a < 0.0 {
                0
            } else {
                (a as usize).min(self.dims[k] - 1)
            };
            hi[k] = if b < 0.0 {
                0
            } else {
                (b as usize).min(self.dims[k] - 1)
            };
        }
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                let base = (i * self.dims[1] + j) * self.dims[2];
                let s = self.start[base + lo[2]] as usize;
                let e = self.start[base + hi[2] + 1] as usize;
                for &a in &self.atoms[s..e] {
                    f(a as usize);
                }
            }
        }
    }
}
