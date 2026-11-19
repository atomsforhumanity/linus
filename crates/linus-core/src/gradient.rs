//! Nuclear gradients as sums of parts.
//!
//! A gradient is one 3-vector per atom (`[f64; 3]`, Eh/bohr). The engines return
//! `Vec<[f64; 3]>`; [`Gradient`] adds the per-atom sums of their parts (`g += &part`), which the
//! engines otherwise each wrote as a helper of their own. Each component is added with one `+=`
//! in atom order, so a sum of parts has the same bits whichever helper did it.

use std::ops::{AddAssign, Deref, DerefMut, SubAssign};

/// A nuclear gradient: one 3-vector per atom.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gradient(pub Vec<[f64; 3]>);

impl Gradient {
    /// The zero gradient of `natom` atoms.
    pub fn zeros(natom: usize) -> Self {
        Gradient(vec![[0.0; 3]; natom])
    }

    /// The per-atom vectors.
    pub fn into_inner(self) -> Vec<[f64; 3]> {
        self.0
    }
}

impl From<Vec<[f64; 3]>> for Gradient {
    fn from(v: Vec<[f64; 3]>) -> Self {
        Gradient(v)
    }
}

impl From<Gradient> for Vec<[f64; 3]> {
    fn from(g: Gradient) -> Self {
        g.0
    }
}

impl Deref for Gradient {
    type Target = [[f64; 3]];
    #[inline]
    fn deref(&self) -> &[[f64; 3]] {
        &self.0
    }
}

impl DerefMut for Gradient {
    #[inline]
    fn deref_mut(&mut self) -> &mut [[f64; 3]] {
        &mut self.0
    }
}

impl AsRef<[[f64; 3]]> for Gradient {
    #[inline]
    fn as_ref(&self) -> &[[f64; 3]] {
        &self.0
    }
}

/// `g += part`, atom by atom.
impl<T: AsRef<[[f64; 3]]>> AddAssign<T> for Gradient {
    fn add_assign(&mut self, part: T) {
        let part = part.as_ref();
        for (x, y) in self.0.iter_mut().zip(part) {
            for k in 0..3 {
                x[k] += y[k];
            }
        }
    }
}

/// `g -= part`, atom by atom.
impl<T: AsRef<[[f64; 3]]>> SubAssign<T> for Gradient {
    fn sub_assign(&mut self, part: T) {
        let part = part.as_ref();
        for (x, y) in self.0.iter_mut().zip(part) {
            for k in 0..3 {
                x[k] -= y[k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_of_parts() {
        let mut g = Gradient::from(vec![[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]]);
        g += vec![[0.25, 0.0, 1.0], [1.0, 1.0, 1.0]];
        g -= &Gradient::from(vec![[1.0, 1.0, 1.0], [0.0, 0.0, 0.5]]);
        assert_eq!(g.into_inner(), vec![[0.25, 1.0, 3.0], [1.5, 1.0, -0.5]]);
        assert_eq!(Gradient::zeros(2).len(), 2);
    }
}
