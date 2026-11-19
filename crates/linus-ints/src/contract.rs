//! The typed inputs of the integral contractions: what the engines contract their integrals
//! with, without spin, hybrid or solvent semantics (those belong to the callers).
//!
//! - [`DensityTerms`]: the density factor of a four-center derivative contraction,
//!   Γ_abcd = sum_t c_t L_ab R_cd (Coulomb-like) + sum_t c_t L_ac R_bd (exchange-like), as
//!   coefficient-weighted pairs of row-major nbf x nbf matrices. A restricted or unrestricted
//!   calculation, a global or range-separated hybrid, chooses its terms; the engines only
//!   sum them.
//!
//! - [`AoMatrix`], [`AoMatrixMut`]: matrices over the basis functions (nbf x nbf), the
//!   densities named operations read and the matrices they write. They are strided, so a
//!   row-major slice, a column-major one or a linear-algebra library's matrix (faer's
//!   `MatRef`, with its column stride) passes without a copy, and linus-ints depends on no
//!   linear-algebra crate.
//!
//! The charge centers of the potential engines carry their own coordinate slots
//! ([`crate::potential::ChargeSource::slot`]): which atom a surface charge moves with is the
//! caller's data, not the integrals'.

/// One term c L ⊗ R of a [`DensityTerms`] list (row-major nbf x nbf matrices).
#[derive(Clone, Copy, Debug)]
pub struct DensityTerm<'a> {
    pub coef: f64,
    pub left: &'a [f64],
    pub right: &'a [f64],
}

/// The density factor of a four-center derivative contraction (see the module docs).
#[derive(Clone, Debug, Default)]
pub struct DensityTerms<'a> {
    /// Terms c L_ab R_cd.
    pub coulomb: Vec<DensityTerm<'a>>,
    /// Terms c L_ac R_bd.
    pub exchange: Vec<DensityTerm<'a>>,
}

impl<'a> DensityTerms<'a> {
    /// A Coulomb-like term c L_ab R_cd.
    pub fn coulomb(mut self, coef: f64, left: &'a [f64], right: &'a [f64]) -> Self {
        self.coulomb.push(DensityTerm { coef, left, right });
        self
    }

    /// An exchange-like term c L_ac R_bd.
    pub fn exchange(mut self, coef: f64, left: &'a [f64], right: &'a [f64]) -> Self {
        self.exchange.push(DensityTerm { coef, left, right });
        self
    }
}

/// A read-only n x n matrix: element (i, j) at `data[i * rs + j * cs]`.
#[derive(Clone, Copy, Debug)]
pub struct AoMatrix<'a> {
    data: &'a [f64],
    n: usize,
    rs: usize,
    cs: usize,
}

impl<'a> AoMatrix<'a> {
    /// A row-major n x n matrix.
    pub fn row_major(data: &'a [f64], n: usize) -> Self {
        Self::strided(data, n, n, 1)
    }

    /// A matrix with element (i, j) at `data[i * rs + j * cs]`.
    ///
    /// # Panics
    /// When `data` is too short for an n x n matrix with these strides.
    pub fn strided(data: &'a [f64], n: usize, rs: usize, cs: usize) -> Self {
        assert!(
            n == 0 || (n - 1) * (rs + cs) < data.len(),
            "AoMatrix: {} values cannot hold {n} x {n} with strides ({rs}, {cs})",
            data.len()
        );
        AoMatrix { data, n, rs, cs }
    }

    /// The order n.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Element (i, j).
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.n && j < self.n);
        // SAFETY: i, j < n and the constructor checked (n - 1) (rs + cs) < data.len()
        unsafe { *self.data.get_unchecked(i * self.rs + j * self.cs) }
    }
}

/// A writable n x n matrix: element (i, j) at `data[i * rs + j * cs]`.
#[derive(Debug)]
pub struct AoMatrixMut<'a> {
    data: &'a mut [f64],
    n: usize,
    rs: usize,
    cs: usize,
}

impl<'a> AoMatrixMut<'a> {
    /// A row-major n x n matrix.
    pub fn row_major(data: &'a mut [f64], n: usize) -> Self {
        Self::strided(data, n, n, 1)
    }

    /// A matrix with element (i, j) at `data[i * rs + j * cs]`.
    ///
    /// # Panics
    /// When `data` is too short for an n x n matrix with these strides, or the strides let
    /// two elements share a place.
    pub fn strided(data: &'a mut [f64], n: usize, rs: usize, cs: usize) -> Self {
        assert!(
            n == 0 || (n - 1) * (rs + cs) < data.len(),
            "AoMatrixMut: {} values cannot hold {n} x {n} with strides ({rs}, {cs})",
            data.len()
        );
        assert!(
            n <= 1 || (rs >= n && cs >= 1) || (cs >= n && rs >= 1),
            "AoMatrixMut: strides ({rs}, {cs}) overlap for order {n}"
        );
        AoMatrixMut { data, n, rs, cs }
    }

    /// The order n.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Element (i, j).
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.n && j < self.n);
        // SAFETY: as in AoMatrix::get
        unsafe { *self.data.get_unchecked(i * self.rs + j * self.cs) }
    }

    /// Element (i, j), writable.
    #[inline(always)]
    pub fn at(&mut self, i: usize, j: usize) -> &mut f64 {
        debug_assert!(i < self.n && j < self.n);
        // SAFETY: as in AoMatrix::get
        unsafe { self.data.get_unchecked_mut(i * self.rs + j * self.cs) }
    }

    /// Set every element to `x`.
    pub fn fill(&mut self, x: f64) {
        for i in 0..self.n {
            for j in 0..self.n {
                *self.at(i, j) = x;
            }
        }
    }
}
