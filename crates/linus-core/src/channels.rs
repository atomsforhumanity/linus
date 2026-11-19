//! Spin channels: one value per spin channel of a reference.
//!
//! A restricted (closed-shell) reference has one channel that stands for both spins, each of
//! its occupied orbitals holding two electrons; an unrestricted reference has two, alpha and
//! beta, with one electron per occupied orbital. Code written over [`Channels`] handles both
//! with one loop over [`Channels::iter`] (or a map), instead of a restricted and an
//! unrestricted copy.
//!
//! Generic code must keep each case's own arithmetic where the two differ: the total density
//! of a restricted reference is `2 D` (one channel scaled), that of an unrestricted one
//! `D_alpha + D_beta` (two channels added). [`Channels::occupation`] is the factor of the first
//! form; summing [`Channels::iter`] gives the second. Writing one as the other changes the
//! last bits of the result.

use std::ops::{Index, IndexMut};

/// One value per spin channel: `Restricted` holds one (both spins), `Unrestricted` two
/// (alpha, beta). Iteration runs over the channels in that order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channels<T> {
    /// One channel standing for both spins (occupation 2).
    Restricted(T),
    /// Alpha and beta (occupation 1 each).
    Unrestricted([T; 2]),
}

impl<T> Channels<T> {
    /// Alpha and beta channels.
    pub fn unrestricted(alpha: T, beta: T) -> Self {
        Channels::Unrestricted([alpha, beta])
    }

    /// One channel for a restricted reference, two for an unrestricted one, each from `f`
    /// (called with the channel index 0, then 1).
    pub fn from_fn(restricted: bool, mut f: impl FnMut(usize) -> T) -> Self {
        if restricted {
            Channels::Restricted(f(0))
        } else {
            let a = f(0);
            Channels::Unrestricted([a, f(1)])
        }
    }

    /// The channels of `v`: one value is a restricted set, two an unrestricted one.
    ///
    /// # Panics
    /// If `v` holds neither one nor two values.
    pub fn from_vec(v: Vec<T>) -> Self {
        let n = v.len();
        let mut it = v.into_iter();
        match (it.next(), it.next(), it.next()) {
            (Some(a), None, _) => Channels::Restricted(a),
            (Some(a), Some(b), None) => Channels::Unrestricted([a, b]),
            _ => panic!("Channels::from_vec: {n} values; a spin set has one or two channels"),
        }
    }

    pub fn is_restricted(&self) -> bool {
        matches!(self, Channels::Restricted(_))
    }

    /// The number of channels: 1 (restricted) or 2 (unrestricted).
    pub fn count(&self) -> usize {
        match self {
            Channels::Restricted(_) => 1,
            Channels::Unrestricted(_) => 2,
        }
    }

    /// Electrons per occupied orbital of a channel: 2 restricted, 1 unrestricted.
    pub fn occupation(&self) -> f64 {
        match self {
            Channels::Restricted(_) => 2.0,
            Channels::Unrestricted(_) => 1.0,
        }
    }

    /// The alpha channel (the only one of a restricted reference).
    pub fn alpha(&self) -> &T {
        &self.as_slice()[0]
    }

    /// The beta channel: the second of an unrestricted reference, the only one of a
    /// restricted reference (which stands for both spins).
    pub fn beta(&self) -> &T {
        match self {
            Channels::Restricted(x) => x,
            Channels::Unrestricted([_, b]) => b,
        }
    }

    /// The channel holding spin `s` (0 alpha, 1 beta): channel 0 for both spins of a
    /// restricted reference.
    pub fn spin(&self, s: usize) -> &T {
        assert!(s < 2, "spin index {s}: 0 (alpha) or 1 (beta)");
        if s == 0 {
            self.alpha()
        } else {
            self.beta()
        }
    }

    /// The channels as a slice of one or two values.
    pub fn as_slice(&self) -> &[T] {
        match self {
            Channels::Restricted(x) => std::slice::from_ref(x),
            Channels::Unrestricted(xs) => xs,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Channels::Restricted(x) => std::slice::from_mut(x),
            Channels::Unrestricted(xs) => xs,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// The same channels holding `f` of each value (alpha first).
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Channels<U> {
        match self {
            Channels::Restricted(x) => Channels::Restricted(f(x)),
            Channels::Unrestricted([a, b]) => {
                let a = f(a);
                Channels::Unrestricted([a, f(b)])
            }
        }
    }

    /// [`Channels::map`] with a fallible `f`: the first error stops it.
    pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<Channels<U>, E> {
        Ok(match self {
            Channels::Restricted(x) => Channels::Restricted(f(x)?),
            Channels::Unrestricted([a, b]) => {
                let a = f(a)?;
                Channels::Unrestricted([a, f(b)?])
            }
        })
    }

    /// References to the values, in the same channels.
    pub fn as_ref(&self) -> Channels<&T> {
        match self {
            Channels::Restricted(x) => Channels::Restricted(x),
            Channels::Unrestricted([a, b]) => Channels::Unrestricted([a, b]),
        }
    }

    pub fn as_mut(&mut self) -> Channels<&mut T> {
        match self {
            Channels::Restricted(x) => Channels::Restricted(x),
            Channels::Unrestricted([a, b]) => Channels::Unrestricted([a, b]),
        }
    }

    /// The values of two channel sets paired channel by channel.
    ///
    /// # Panics
    /// If one set is restricted and the other unrestricted.
    pub fn zip<U>(self, other: Channels<U>) -> Channels<(T, U)> {
        match (self, other) {
            (Channels::Restricted(x), Channels::Restricted(y)) => Channels::Restricted((x, y)),
            (Channels::Unrestricted([a, b]), Channels::Unrestricted([c, d])) => {
                Channels::Unrestricted([(a, c), (b, d)])
            }
            _ => panic!("Channels::zip: a restricted and an unrestricted set"),
        }
    }

    /// The values as a vector of one or two.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Channels::Restricted(x) => vec![x],
            Channels::Unrestricted([a, b]) => vec![a, b],
        }
    }
}

impl<T> Index<usize> for Channels<T> {
    type Output = T;
    /// Channel `i` (0 or 1; only 0 for a restricted set).
    fn index(&self, i: usize) -> &T {
        &self.as_slice()[i]
    }
}

impl<T> IndexMut<usize> for Channels<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.as_mut_slice()[i]
    }
}

impl<T> IntoIterator for Channels<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Channels<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Channels<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restricted_stands_for_both_spins() {
        let c = Channels::Restricted(7);
        assert!(c.is_restricted());
        assert_eq!((c.count(), c.occupation()), (1, 2.0));
        assert_eq!((c.alpha(), c.beta(), c.spin(1)), (&7, &7, &7));
        assert_eq!(c.iter().copied().collect::<Vec<_>>(), [7]);
        assert_eq!(c.map(|x| x * 2), Channels::Restricted(14));
        assert_eq!(Channels::from_vec(vec![7]), c);
    }

    #[test]
    fn unrestricted_iterates_alpha_then_beta() {
        let mut c = Channels::unrestricted(1, 2);
        assert_eq!((c.count(), c.occupation()), (2, 1.0));
        assert_eq!((c.alpha(), c.beta(), c[1]), (&1, &2, 2));
        for x in &mut c {
            *x += 10;
        }
        assert_eq!(c.into_vec(), [11, 12]);
        let mut order = Vec::new();
        let m = Channels::unrestricted('a', 'b').map(|x| {
            order.push(x);
            x
        });
        assert_eq!((order, m.as_slice()), (vec!['a', 'b'], &['a', 'b'][..]));
        assert_eq!(
            Channels::from_fn(false, |s| s * 3),
            Channels::unrestricted(0, 3)
        );
        assert_eq!(
            Channels::unrestricted(1, 2).zip(Channels::unrestricted('x', 'y')),
            Channels::unrestricted((1, 'x'), (2, 'y'))
        );
        let r: Result<Channels<i32>, &str> =
            Channels::unrestricted(11, 12).try_map(|x| if x > 11 { Err("big") } else { Ok(x) });
        assert_eq!(r, Err("big"));
    }

    #[test]
    #[should_panic(expected = "one or two channels")]
    fn three_values_are_no_spin_set() {
        let _ = Channels::from_vec(vec![1, 2, 3]);
    }
}
