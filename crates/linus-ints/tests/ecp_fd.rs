//! ECP integrals: closed forms of one-center integrals, derivatives against finite
//! differences, translational invariance and independence of the thread count.

use linus_ints::basis::BasisDef;
use linus_ints::one_e::ecp::EcpSet;
use linus_ints::reference::oneshot;
use linus_ints::{BasisSet, PairOptions, PairSet};

struct Sys {
    def: BasisDef,
    z: Vec<u32>,
    xyz: Vec<[f64; 3]>,
}

impl Sys {
    fn new(basis: &str, z: &[u32], xyz: &[[f64; 3]]) -> Self {
        Sys {
            def: BasisDef::load(basis, None).unwrap(),
            z: z.to_vec(),
            xyz: xyz.to_vec(),
        }
    }
    fn at(&self, xyz: &[[f64; 3]]) -> (BasisSet, PairSet, EcpSet) {
        let b = BasisSet::new(&self.def, &self.z, xyz, true).unwrap();
        let p = PairSet::build(&b, PairOptions::default());
        let e = EcpSet::new(&self.def, &self.z, xyz).unwrap();
        (b, p, e)
    }
    fn matrix(&self, xyz: &[[f64; 3]]) -> Vec<f64> {
        let (b, p, e) = self.at(xyz);
        oneshot::ecp_matrix(&b, &p, &e)
    }
}

/// A Pd complex fragment (Pd, Cl, P, H; d and f functions on Pd), bohr.
fn pd() -> Sys {
    Sys::new(
        "def2-svp",
        &[46, 17, 1, 15],
        &[
            [0.0, 0.0, 0.0],
            [4.3, 0.3, -0.2],
            [-0.4, 2.9, 0.5],
            [-0.2, -0.5, -4.3],
        ],
    )
}

/// Iodine with two hydrogens and an off-axis carbon (def2-TZVP: up to f on I).
fn iodine() -> Sys {
    Sys::new(
        "def2-tzvp",
        &[53, 1, 6],
        &[[0.1, -0.2, 0.0], [0.3, 0.2, 3.04], [-3.2, 1.1, 2.6]],
    )
}

fn random_density(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let k = n / 3 + 2;
    let c: Vec<f64> = (0..n * k).map(|_| rnd()).collect();
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = (0..k).map(|q| c[i * k + q] * c[j * k + q]).sum::<f64>() / n as f64;
        }
    }
    d
}

#[test]
fn one_center_s_matches_closed_form() {
    // <s|U|s> for s functions on the ECP atom: 4 pi sum c_i c_j sum_k d_k sqrt(pi)/(4 p^{3/2})
    // over the local and l = 0 channels (r^0 terms)
    for (basis, zz) in [("def2-tzvp", 78u32), ("def2-svp", 53), ("def2-svp", 37)] {
        let def = BasisDef::load(basis, None).unwrap();
        let b = BasisSet::new(&def, &[zz], &[[0.3, -0.1, 0.2]], true).unwrap();
        let p = PairSet::build(&b, PairOptions::default());
        let e = EcpSet::new(&def, &[zz], &[[0.3, -0.1, 0.2]]).unwrap();
        let v = oneshot::ecp_matrix(&b, &p, &e);
        let ed = &def.ecp[&zz];
        let terms: Vec<_> = ed.local.iter().chain(&ed.semilocal[0]).collect();
        let n = b.nbf;
        for sh in b.shells.iter().filter(|s| s.l == 0) {
            let mut exact = 0.0;
            for (&ai, &ci) in sh.exps.iter().zip(&sh.coefs) {
                for (&aj, &cj) in sh.exps.iter().zip(&sh.coefs) {
                    for t in &terms {
                        let q = ai + aj + t.zeta;
                        exact += 4.0
                            * std::f64::consts::PI
                            * ci
                            * cj
                            * t.coef
                            * std::f64::consts::PI.sqrt()
                            / (4.0 * q.powf(1.5));
                    }
                }
            }
            let a = sh.ao_start;
            let got = v[a * n + a];
            assert!(
                (got - exact).abs() < 2e-13 * exact.abs().max(1.0),
                "{basis} Z={zz} shell at {a}: {got} vs {exact}"
            );
        }
    }
}

#[test]
fn derivative_matrices_match_finite_differences() {
    for s in [pd(), iodine()] {
        let (b, p, e) = s.at(&s.xyz);
        let n = b.nbf;
        let n2 = n * n;
        let dv = oneshot::ecp_deriv(&b, &p, &e);
        let h = 1e-4;
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for a in 0..s.z.len() {
            for k in 0..3 {
                let mut xp = s.xyz.clone();
                let mut xm = s.xyz.clone();
                xp[a][k] += h;
                xm[a][k] -= h;
                let vp = s.matrix(&xp);
                let vm = s.matrix(&xm);
                let blk = &dv[(3 * a + k) * n2..(3 * a + k + 1) * n2];
                for i in 0..n2 {
                    let fd = (vp[i] - vm[i]) / (2.0 * h);
                    worst = worst.max((fd - blk[i]).abs());
                    scale = scale.max(fd.abs());
                }
            }
        }
        // central differences: O(h^2) truncation ~ 1e-8 relative
        assert!(
            worst < 2e-7 * scale.max(1.0),
            "worst {worst:e} (max {scale:e})"
        );
    }
}

#[test]
fn gradient_contracts_derivatives_and_is_translation_invariant() {
    for s in [pd(), iodine()] {
        let (b, p, e) = s.at(&s.xyz);
        let n = b.nbf;
        let n2 = n * n;
        let d = random_density(n, 11);
        let g = oneshot::ecp_grad(&b, &p, &e, &d);
        let dv = oneshot::ecp_deriv(&b, &p, &e);
        for a in 0..s.z.len() {
            for k in 0..3 {
                let r: f64 = dv[(3 * a + k) * n2..(3 * a + k + 1) * n2]
                    .iter()
                    .zip(&d)
                    .map(|(x, y)| x * y)
                    .sum();
                assert!(
                    (r - g[a][k]).abs() < 1e-11,
                    "atom {a} k {k}: {} vs {r}",
                    g[a][k]
                );
            }
        }
        for k in 0..3 {
            let t: f64 = g.iter().map(|x| x[k]).sum();
            assert!(t.abs() < 1e-11, "translation: {t:e}");
        }
    }
}

#[test]
fn hessian_matches_finite_differences_of_gradient() {
    for s in [pd(), iodine()] {
        let (b, p, e) = s.at(&s.xyz);
        let n = b.nbf;
        let d = random_density(n, 5);
        let hs = oneshot::ecp_hess(&b, &p, &e, &d);
        let n3 = 3 * s.z.len();
        let h = 1e-4;
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for a in 0..s.z.len() {
            for k in 0..3 {
                let mut xp = s.xyz.clone();
                let mut xm = s.xyz.clone();
                xp[a][k] += h;
                xm[a][k] -= h;
                let (bp, pp, ep) = s.at(&xp);
                let (bm, pm, em) = s.at(&xm);
                let gp = oneshot::ecp_grad(&bp, &pp, &ep, &d);
                let gm = oneshot::ecp_grad(&bm, &pm, &em, &d);
                for c in 0..s.z.len() {
                    for l in 0..3 {
                        let fd = (gp[c][l] - gm[c][l]) / (2.0 * h);
                        let an = hs[(3 * a + k) * n3 + 3 * c + l];
                        worst = worst.max((fd - an).abs());
                        scale = scale.max(fd.abs());
                    }
                }
            }
        }
        assert!(
            worst < 2e-7 * scale.max(1.0),
            "worst {worst:e} (max {scale:e})"
        );
        // symmetric, rows sum to zero (translation)
        for i in 0..n3 {
            for j in 0..n3 {
                assert!((hs[i * n3 + j] - hs[j * n3 + i]).abs() < 1e-10);
            }
            for l in 0..3 {
                let r: f64 = (0..s.z.len()).map(|c| hs[i * n3 + 3 * c + l]).sum();
                assert!(r.abs() < 1e-10, "row {i} sum {r:e}");
            }
        }
    }
}

#[test]
fn independent_of_thread_count() {
    let s = pd();
    let run = |t: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
            .install(|| {
                let (b, p, e) = s.at(&s.xyz);
                let d = random_density(b.nbf, 3);
                (
                    oneshot::ecp_matrix(&b, &p, &e),
                    oneshot::ecp_grad(&b, &p, &e, &d),
                    oneshot::ecp_hess(&b, &p, &e, &d),
                    oneshot::ecp_deriv(&b, &p, &e),
                )
            })
    };
    let r1 = run(1);
    for t in [2, 5, 8] {
        let r = run(t);
        assert!(
            r.0 == r1.0 && r.1 == r1.1 && r.2 == r1.2 && r.3 == r1.3,
            "{t} threads differ"
        );
    }
}
