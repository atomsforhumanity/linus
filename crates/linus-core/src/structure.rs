//! Structures: the atoms (atomic numbers and positions), charge and spin a calculation is run
//! on, as qcdata's `Structure`. A structure need not be one molecule (a complex, a cluster, a
//! reacting system). Nuclear repulsion; the XYZ reader.

use crate::elements::{symbol_to_z, z_to_symbol, MAX_Z};
use crate::error::Error;
use crate::units::BOHR_ANGSTROM;

/// Atoms (atomic numbers and positions in bohr), charge and spin. Built checked
/// ([`Structure::new`]); a calculation adds the core electrons its basis set's effective core
/// potentials replace ([`Structure::set_ecp_core_electrons`]).
#[derive(Clone, Debug)]
pub struct Structure {
    /// Atomic numbers (the API name is `numbers`, [`Structure::numbers`]).
    pub z: Vec<u32>,
    /// Cartesian positions in bohr (also [`Structure::positions`]).
    pub positions: Vec<[f64; 3]>,
    pub charge: i32,
    /// 2S + 1
    pub multiplicity: u32,
    /// Core electrons replaced by an effective core potential, per atom (empty: none). Set by
    /// the session from the basis set; the nuclear charges seen by the electrons and the
    /// other nuclei are then Z - ecp_core.
    ecp_core: Vec<u32>,
}

/// Atoms closer than this (bohr) are rejected as coincident: a duplicated XYZ line would
/// otherwise give an infinite nuclear repulsion and a singular overlap. 0.05 bohr is far below
/// any chemical contact (H2: 1.4 bohr) and still allows 0.1 Å.
pub const MIN_DISTANCE: f64 = 0.05;

/// No two atoms closer than [`MIN_DISTANCE`] (bohr).
pub fn check_distances(z: &[u32], positions: &[[f64; 3]]) -> Result<(), String> {
    for i in 0..positions.len() {
        for j in 0..i {
            let d = dist(positions[i], positions[j]);
            if d < MIN_DISTANCE {
                let sym = |k: usize| z.get(k).and_then(|&n| z_to_symbol(n)).unwrap_or("?");
                return Err(format!(
                    "atoms {} ({}) and {} ({}) are {d:.4} bohr apart, closer than {MIN_DISTANCE} bohr: \
                     check the structure for duplicated atoms",
                    j + 1,
                    sym(j),
                    i + 1,
                    sym(i)
                ));
            }
        }
    }
    Ok(())
}

/// Atomic numbers and positions (bohr) from the text of an XYZ file (Å): the atom count, a
/// comment line, then one line per atom with a symbol (or atomic number) and three
/// coordinates (further columns are ignored).
pub fn parse_xyz(text: &str) -> Result<(Vec<u32>, Vec<[f64; 3]>), String> {
    let mut lines = text.lines();
    let first = lines.next().ok_or("xyz: empty input")?;
    let n: usize = first.trim().parse().map_err(|_| {
        format!(
            "xyz: the first line must be the number of atoms, not {:?}",
            first.trim()
        )
    })?;
    lines.next();
    let mut z = Vec::with_capacity(n);
    let mut pos = Vec::with_capacity(n);
    for (i, line) in lines.take(n).enumerate() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            return Err(format!(
                "xyz: atom {} needs a symbol and three coordinates: {line:?}",
                i + 1
            ));
        }
        z.push(symbol_to_z(f[0]).ok_or_else(|| {
            format!(
                "xyz: unknown element {:?} (atom {}; linus supports H-Rn)",
                f[0],
                i + 1
            )
        })?);
        let mut r = [0.0; 3];
        for k in 0..3 {
            let x: f64 = f[k + 1]
                .parse()
                .map_err(|_| format!("xyz: bad coordinate {:?} of atom {}", f[k + 1], i + 1))?;
            if !x.is_finite() {
                return Err(format!(
                    "xyz: coordinate {:?} of atom {} is not finite",
                    f[k + 1],
                    i + 1
                ));
            }
            r[k] = x / BOHR_ANGSTROM;
        }
        pos.push(r);
    }
    if z.len() != n {
        return Err(format!("xyz: expected {n} atoms, found {}", z.len()));
    }
    if n == 0 {
        return Err("xyz: no atoms".into());
    }
    Ok((z, pos))
}

impl Structure {
    /// Atomic numbers `z`, positions in bohr, charge and multiplicity 2S + 1, checked
    /// ([`Structure::validate`]: atoms H-Rn, one finite position per atom, no two atoms closer
    /// than [`MIN_DISTANCE`], multiplicity at least 1).
    pub fn new(
        z: Vec<u32>,
        positions: Vec<[f64; 3]>,
        charge: i32,
        multiplicity: u32,
    ) -> Result<Self, Error> {
        let m = Structure {
            z,
            positions,
            charge,
            multiplicity,
            ecp_core: Vec::new(),
        };
        m.validate().map_err(Error::Input)?;
        Ok(m)
    }

    /// Read an XYZ file (Å; [`parse_xyz`]).
    pub fn from_xyz_str(text: &str, charge: i32, multiplicity: u32) -> Result<Self, Error> {
        let (z, c) = parse_xyz(text).map_err(Error::Input)?;
        Structure::new(z, c, charge, multiplicity)
    }

    /// Atomic numbers.
    #[inline]
    pub fn numbers(&self) -> &[u32] {
        &self.z
    }

    /// Positions (bohr).
    #[inline]
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }

    /// Core electrons replaced by effective core potentials, per atom (zeros without ECPs, and
    /// for a structure that no session has set up).
    pub fn ecp_core_electrons(&self) -> Vec<u32> {
        (0..self.natom())
            .map(|i| self.ecp_core.get(i).copied().unwrap_or(0))
            .collect()
    }

    /// Set the core electrons the basis set's effective core potentials replace, per atom
    /// (the session does this when it sets up the basis set).
    pub fn set_ecp_core_electrons(&mut self, core: Vec<u32>) {
        self.ecp_core = core;
    }

    /// Atoms in range (1-86), one finite position per atom, no two atoms closer than
    /// [`MIN_DISTANCE`], multiplicity at least 1.
    pub fn validate(&self) -> Result<(), String> {
        if self.z.is_empty() {
            return Err("a calculation needs at least one atom".into());
        }
        if let Some(&z) = self.z.iter().find(|&&z| !(1..=MAX_Z).contains(&z)) {
            return Err(format!(
                "atomic number {z} is not supported (1-{MAX_Z}, H-Rn)"
            ));
        }
        if self.positions.len() != self.z.len() {
            return Err(format!(
                "{} atoms but {} positions",
                self.z.len(),
                self.positions.len()
            ));
        }
        if self.positions.iter().flatten().any(|x| !x.is_finite()) {
            return Err("positions must be finite numbers".into());
        }
        check_distances(&self.z, &self.positions)?;
        if self.multiplicity < 1 {
            return Err("multiplicity must be at least 1".into());
        }
        Ok(())
    }

    #[inline]
    pub fn natom(&self) -> usize {
        self.z.len()
    }

    /// Nuclear charge of atom i (Z minus the ECP core electrons).
    #[inline]
    pub fn zeff(&self, i: usize) -> u32 {
        self.z[i] - self.ecp_core.get(i).copied().unwrap_or(0)
    }

    /// Nuclear charges of all atoms (Z minus the ECP core electrons).
    pub fn effective_z(&self) -> Vec<u32> {
        (0..self.natom()).map(|i| self.zeff(i)).collect()
    }

    /// Electrons treated explicitly (ECP cores excluded).
    pub fn nelectron(&self) -> i64 {
        (0..self.natom()).map(|i| self.zeff(i) as i64).sum::<i64>() - self.charge as i64
    }

    /// (n_alpha, n_beta)
    pub fn nalpha_nbeta(&self) -> Result<(usize, usize), String> {
        let ne = self.nelectron();
        let nu = self.multiplicity as i64 - 1;
        if ne < 0 || nu < 0 || (ne - nu) % 2 != 0 || nu > ne {
            return Err(format!(
                "{ne} electrons cannot have multiplicity {}",
                self.multiplicity
            ));
        }
        Ok((((ne + nu) / 2) as usize, ((ne - nu) / 2) as usize))
    }

    pub fn nuclear_repulsion(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..self.natom() {
            for j in 0..i {
                let d = dist(self.positions[i], self.positions[j]);
                e += (self.zeff(i) * self.zeff(j)) as f64 / d;
            }
        }
        e
    }

    /// d E_nuc / d R (natom x 3).
    pub fn nuclear_repulsion_gradient(&self) -> Vec<[f64; 3]> {
        let n = self.natom();
        let mut g = vec![[0.0; 3]; n];
        for i in 0..n {
            for j in 0..i {
                let r = [
                    self.positions[i][0] - self.positions[j][0],
                    self.positions[i][1] - self.positions[j][1],
                    self.positions[i][2] - self.positions[j][2],
                ];
                let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
                let f = -((self.zeff(i) * self.zeff(j)) as f64) / (d * d * d);
                for k in 0..3 {
                    g[i][k] += f * r[k];
                    g[j][k] -= f * r[k];
                }
            }
        }
        g
    }

    /// Nuclei as point charges (effective charges with ECPs).
    pub fn point_charges(&self) -> Vec<(f64, [f64; 3])> {
        self.positions
            .iter()
            .enumerate()
            .map(|(i, &c)| (self.zeff(i) as f64, c))
            .collect()
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_construction_and_xyz() {
        let w = Structure::from_xyz_str(
            "3\nwater\nO 0 0 0\nH 0 0.757 0.587\n1 0 -0.757 0.587\n",
            0,
            1,
        )
        .unwrap();
        assert_eq!(w.numbers(), [8, 1, 1]);
        assert_eq!(w.positions[1][1], 0.757 / BOHR_ANGSTROM);
        assert_eq!((w.nelectron(), w.nalpha_nbeta()), (10, Ok((5, 5))));
        let e = Structure::new(vec![0], vec![[0.0; 3]], 0, 1).unwrap_err();
        assert!(e.is_input());
        let mut m = Structure::new(vec![53, 1], vec![[0.0; 3], [0.0, 0.0, 3.0]], 0, 1).unwrap();
        m.set_ecp_core_electrons(vec![28, 0]);
        assert_eq!((m.zeff(0), m.ecp_core_electrons()), (25, vec![28, 0]));
        assert!(Structure::new(vec![1, 1], vec![[0.0; 3]; 2], 0, 1).is_err());
    }
}
