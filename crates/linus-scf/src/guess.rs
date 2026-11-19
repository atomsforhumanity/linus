//! Initial guesses. SAD: superposition of spin-averaged atomic Hartree-Fock densities
//! computed in the molecular basis (fractional occupation of the open shell), cached per
//! element and its basis functions and ECP (the definition itself, not the basis set's name,
//! so equally named sets from different `basis_dir`s do not share entries). Atoms with an
//! effective core potential carry only their valence electrons (Z - ncore) in the ECP's field.

use crate::{diagonalize, Diis};
use faer::{Mat, Par};
use linus_base::linalg::{
    ao_view, ao_view_mut, canonical_orthogonalizer, dot, mat_from_rowmajor, max_abs, mm,
};
use linus_ints::basis::BasisDef;
use linus_ints::{
    AoMatrixMut, BasisSet, DerivOrder, EcpPlan, JkArgs, JkOptions, JkOut, JkPlan, JkShape,
    OneEPlan, PairOptions, PairSet,
};
use std::collections::HashMap;
use std::sync::Mutex;

/// Atomic SAD densities by (the element's shells and ECP, atomic number, spherical functions).
type AtomDensities = HashMap<(String, u32, bool), Mat<f64>>;

static CACHE: Mutex<Option<AtomDensities>> = Mutex::new(None);

/// Fractional aufbau occupations (0..2 per spatial orbital) for `nel` electrons; orbitals
/// within `degen` of the highest occupied level share the open-shell electrons equally. A basis
/// with fewer orbitals than the electrons need (a short custom basis) fills them all (the SAD
/// guess is scaled to the molecule's electron count).
fn frac_occupations(eps: &[f64], nel: f64, degen: f64) -> Vec<f64> {
    let n = eps.len();
    let mut occ = vec![0.0; n];
    let nfull = (nel / 2.0).floor() as usize;
    let rem = nel - 2.0 * nfull as f64;
    let nf = if rem > 1e-12 { nfull + 1 } else { nfull };
    if nf == 0 {
        return occ;
    }
    if nf > n {
        return vec![2.0; n];
    }
    let ef = eps[nf - 1];
    let first = (0..n).find(|&i| (eps[i] - ef).abs() < degen).unwrap();
    let last = (0..n).rev().find(|&i| (eps[i] - ef).abs() < degen).unwrap();
    for o in occ.iter_mut().take(first) {
        *o = 2.0;
    }
    let shared = nel - 2.0 * first as f64;
    let m = (last - first + 1) as f64;
    for o in occ.iter_mut().take(last + 1).skip(first) {
        *o = shared / m;
    }
    occ
}

/// Spin-averaged atomic HF density (total, alpha + beta) for element `z` in basis `def`.
/// Computed on one thread (`par` is ignored): atomic matrices are small, and the result (cached
/// and shared by all sessions) must not depend on the caller's thread count.
pub fn atomic_density(def: &BasisDef, z: u32, pure: bool, _par: Par) -> Mat<f64> {
    // the element's shells and ECP (exact: Debug prints f64 round-trip), not the set's name
    let key = (
        format!("{:?} {:?}", def.elements.get(&z), def.ecp.get(&z)),
        z,
        pure,
    );
    if let Some(c) = CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return c.clone();
    }
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let d = one.install(|| atomic_scf(def, z, pure));
    CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, d.clone());
    d
}

fn atomic_scf(def: &BasisDef, z: u32, pure: bool) -> Mat<f64> {
    let par = Par::Seq;
    let basis = BasisSet::new(def, &[z], &[[0.0; 3]], pure).expect("atom basis");
    let n = basis.nbf;
    let pairs = PairSet::build(&basis, PairOptions::default());
    let one_e = OneEPlan::new(&basis, DerivOrder::Zero);
    let (mut s, mut t) = (vec![0.0; n * n], vec![0.0; n * n]);
    one_e.overlap_kinetic(
        &mut AoMatrixMut::row_major(&mut s, n),
        &mut AoMatrixMut::row_major(&mut t, n),
    );
    // ECP atoms: core electrons removed from the charge and the electron count, ECP in h
    let ncore = def.ecp.get(&z).map_or(0, |e| e.ncore);
    let zeff = (z - ncore) as f64;
    let charges = [(zeff, [0.0; 3])];
    let mut v = vec![0.0; n * n];
    one_e
        .with_charges(&pairs, &charges)
        .expect("the potential takes pairs of any order")
        .potential(&mut AoMatrixMut::row_major(&mut v, n));
    let ecp = linus_ints::one_e::ecp::EcpSet::new(def, &[z], &[[0.0; 3]]).expect("atom ECP");
    if !ecp.is_empty() {
        let mut u = vec![0.0; n * n];
        let plan = EcpPlan::new(&basis, &pairs, &ecp, DerivOrder::Zero);
        plan.matrix(&mut AoMatrixMut::row_major(&mut u, n));
        v.iter_mut().zip(&u).for_each(|(a, b)| *a += b);
    }
    let s = mat_from_rowmajor(n, n, &s);
    let h = mat_from_rowmajor(n, n, &t) + mat_from_rowmajor(n, n, &v);
    let x = canonical_orthogonalizer(s.as_ref(), 1e-8, par);
    // the four-center J/K engine of every exact-J/K SCF (linus-ints' JkPlan, as the facade's FastJk
    // runs it at this threshold), on this one-thread pool
    let thresh = 1e-12;
    let plan = JkPlan::new(
        &basis,
        &pairs,
        JkOptions {
            thresh,
            ..Default::default()
        },
    )
    .expect("orbital pairs");
    let mut ws = plan.workspace(
        JkShape {
            ndens: 1,
            j: true,
            k: true,
        },
        rayon::current_num_threads(),
    );
    let nel = zeff;
    let (eps, c) = diagonalize(h.as_ref(), x.as_ref(), par);
    let occ = frac_occupations(&eps, nel, 1e-3);
    let mut d = weighted_density(&c, &occ, par);
    let mut diis = Diis::new(8);
    let mut e_old = 0.0;
    for _ in 0..100 {
        let (mut j, mut k) = (Mat::<f64>::zeros(n, n), Mat::<f64>::zeros(n, n));
        plan.jk(
            &JkArgs {
                dens: &[ao_view(d.as_ref())],
                thresh,
            },
            &mut JkOut {
                j: &mut [ao_view_mut(j.as_mut())],
                k: &mut [ao_view_mut(k.as_mut())],
            },
            &mut ws,
        );
        // closed-shell-averaged Fock for total density D: h + J[D] - K[D]/2
        let f = h.clone() + &j - &k * 0.5;
        let e = 0.5 * dot(d.as_ref(), (h.clone() + &f).as_ref());
        let fds = mm(mm(f.as_ref(), d.as_ref(), par).as_ref(), s.as_ref(), par);
        let err = mm(
            x.transpose(),
            mm((&fds - fds.transpose()).as_ref(), x.as_ref(), par).as_ref(),
            par,
        );
        let g = max_abs(err.as_ref());
        if (e - e_old).abs() < 1e-9 && g < 1e-6 {
            break;
        }
        e_old = e;
        diis.push(vec![f], vec![err]);
        let fx = diis.extrapolate().remove(0);
        let (eps, c) = diagonalize(fx.as_ref(), x.as_ref(), par);
        let occ = frac_occupations(&eps, nel, 1e-3);
        d = weighted_density(&c, &occ, par);
    }
    d
}

fn weighted_density(c: &Mat<f64>, occ: &[f64], par: Par) -> Mat<f64> {
    let n = c.nrows();
    let k = occ.iter().filter(|&&o| o > 0.0).count();
    let cw = Mat::from_fn(n, k, |i, j| c[(i, j)] * occ[j].sqrt());
    mm(cw.as_ref(), cw.transpose(), par)
}

/// SAD guess: block-diagonal superposition of atomic densities, returned as spin densities
/// scaled to the molecule's alpha and beta electron counts.
pub fn sad_guess(
    def: &BasisDef,
    basis: &BasisSet,
    z: &[u32],
    nalpha: usize,
    nbeta: usize,
    par: Par,
) -> (Mat<f64>, Mat<f64>) {
    let n = basis.nbf;
    let mut d = Mat::<f64>::zeros(n, n);
    let mut ntot = 0.0;
    for (atom, range) in basis.atom_shells.iter().enumerate() {
        if range.is_empty() {
            continue;
        }
        let da = atomic_density(def, z[atom], basis.pure, par);
        let off = basis.shells[range.start].ao_start;
        for j in 0..da.ncols() {
            for i in 0..da.nrows() {
                d[(off + i, off + j)] = da[(i, j)];
            }
        }
        ntot += (z[atom] - def.ecp.get(&z[atom]).map_or(0, |e| e.ncore)) as f64;
    }
    let ne = (nalpha + nbeta) as f64;
    let fa = if ntot > 0.0 {
        nalpha as f64 / ntot
    } else {
        0.0
    };
    let fb = if ntot > 0.0 { nbeta as f64 / ntot } else { 0.0 };
    let _ = ne;
    (&d * fa, &d * fb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equally named basis definitions with different functions (two `basis_dir`s) get their
    /// own atomic densities; a repeated request is served from the cache unchanged.
    #[test]
    fn atomic_density_cache_is_keyed_by_definition() {
        let def = |e: &str| {
            let json = format!(
                r#"{{"name":"same","elements":{{"1":{{"electron_shells":[{{"function_type":"gto","angular_momentum":[0],"exponents":["{e}","0.5"],"coefficients":[["0.4","0.6"]]}},{{"function_type":"gto","angular_momentum":[0],"exponents":["0.15"],"coefficients":[["1.0"]]}}]}}}}}}"#
            );
            BasisDef::from_bse_json(&json).unwrap()
        };
        let (a, b) = (def("3.4"), def("9.1"));
        let da = atomic_density(&a, 1, true, Par::Seq);
        let db = atomic_density(&b, 1, true, Par::Seq);
        assert!(max_abs((&da - &db).as_ref()) > 1e-4, "{da:?} {db:?}");
        assert_eq!(
            max_abs((&atomic_density(&a, 1, true, Par::Seq) - &da).as_ref()),
            0.0
        );
    }
}
