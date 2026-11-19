//! The device SCF loop: its reductions against compensated host sums, Hartree-Fock energies of
//! water against PySCF (Fock terms from the dense ERI tensor through the CPU fallback), the
//! guess-density path, bitwise reproducibility run to run, the eigensolver plans, and the loop
//! on linus-gpu's own DF-J/K and XC engines (DeviceTerms). The iteration-by-iteration
//! comparison with linus's CPU run_scf is bench/gpu/dscf (it needs the linus crate).
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::GridSpec;
use linus_gpu::df::{GpuDfJk, GpuDfJkOptions};
use linus_gpu::error::GpuError;
use linus_gpu::harness;
use linus_gpu::scf::kernels::{Ops, Task};
use linus_gpu::scf::{
    run_scf, DeviceTerms, Eigensolver, FockTerms, HostFockTerms, HostState, LoopConfig,
    Observation, OnHost, ScfDecide, ScfInput, ScfResult, Step, StepPlan, Stop, Verdict,
};
use linus_gpu::solver::{EighMethod, Solver};
use linus_gpu::xc::{GpuXc, GpuXcOptions};
use linus_gpu::Gpu;
use linus_ints::{eri4c, BasisSet, PairOptions, PairSet};

/// PySCF 2.15, def2-SVP (spherical), tests/molecules/water.xyz, conv_tol 1e-12.
const E_RHF_PYSCF: f64 = -75.96096983360252;
/// The cation, UHF doublet; <S^2> 0.7562465239950211.
const E_UHF_CATION_PYSCF: f64 = -75.56216566586;
const S2_UHF_CATION_PYSCF: f64 = 0.7562465239950211;
/// The same with density fitting, auxbasis def2-universal-jkfit (113 functions).
const E_DF_RHF_PYSCF: f64 = -75.96091387001898;
const E_DF_UHF_CATION_PYSCF: f64 = -75.5621216106112;

/// The loop's controller in these tests: linus's default decisions for a closed or open shell that
/// converges without help (converged when |dE| < 1e-10 and max |F_ai| < 1e-8, not before iteration
/// 2, 3 after a guess density; DIIS from the first density of orbitals, the largest error leaving a
/// full subspace; the level shift of 0.25 Eh while max |F_ai| > 1e-2). linus's own controller is the
/// host loop's controls (crates/linus/src/scf/device.rs).
struct Plain;

impl ScfDecide for Plain {
    fn observe(&mut self, o: &Observation<'_>) -> Verdict {
        if o.iter > 1 && o.iter > o.guess_iter + 1 && o.de.abs() < 1e-10 && o.gmax < 1e-8 {
            return Verdict::Converged;
        }
        let push = o.iter > o.guess_iter && o.diis_capacity > 0;
        let n = o.diis_norms.len();
        let victim = (push && n >= o.diis_capacity)
            .then(|| (0..n).max_by(|&a, &b| o.diis_norms[a].total_cmp(&o.diis_norms[b])))
            .flatten();
        Verdict::Step(StepPlan {
            lowest: false,
            diis_push: push,
            diis_victim: victim,
            extrapolate: n + usize::from(push) > 0 && o.iter > o.guess_iter,
            level_shift: (o.gmax > 1e-2).then_some(0.25),
            hand_over: false,
        })
    }
}

/// Neumaier sum of the rounded products, as linus's `util::linalg::dot`.
fn host_dot(x: &[f64], y: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for (a, b) in x.iter().zip(y) {
        let v = a * b;
        let t = s + v;
        c += if s.abs() >= v.abs() {
            (s - t) + v
        } else {
            (v - t) + s
        };
        s = t;
    }
    s + c
}

#[test]
fn reductions_match_compensated_host_sums_and_repeat() {
    let Some(gpu) = common::gpu() else { return };
    let n = 1_000_003;
    let mut rng = Rng(17);
    let x: Vec<f64> = (0..n).map(|_| rng.wide()).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.wide() * 1e3).collect();
    let mut z = y.clone();
    z[12345] = f64::NAN;
    z[777] = f64::INFINITY;
    let (dx, dy, dz) = (
        gpu.upload(&x).unwrap(),
        gpu.upload(&y).unwrap(),
        gpu.upload(&z).unwrap(),
    );
    let mut ops = Ops::new(&gpu, 8, 4).unwrap();
    let mut out = gpu.alloc_zeros::<f64>(8).unwrap();
    let tasks = [
        Task::dot((&dx, 0), (&dy, 0), n).unwrap(),
        Task::dot((&dx, 5), (&dy, 0), n - 5).unwrap(),
        Task::max_abs((&dy, 0), n).unwrap(),
        Task::non_finite((&dz, 0), n).unwrap(),
        Task::dot((&dx, 0), (&dx, 0), 7).unwrap(),
    ];
    ops.reduce(&gpu, &tasks, &mut out).unwrap();
    let r = gpu.download(&out).unwrap();
    let want = [
        host_dot(&x, &y),
        host_dot(&x[5..], &y[..n - 5]),
        y.iter().fold(0.0f64, |m, v| m.max(v.abs())),
        2.0,
        host_dot(&x[..7], &x[..7]),
    ];
    for (i, (g, w)) in r.iter().zip(want).enumerate() {
        let tol = 4.0 * f64::EPSILON * w.abs();
        assert!((g - w).abs() <= tol, "task {i}: {g:e} vs {w:e}");
    }
    // the same bits again, and the same bits in another batch (the partition is fixed)
    ops.reduce(&gpu, &tasks, &mut out).unwrap();
    let r2 = gpu.download(&out).unwrap();
    assert!(r.iter().zip(&r2).all(|(a, b)| a.to_bits() == b.to_bits()));
    ops.reduce(&gpu, &tasks[1..2], &mut out).unwrap();
    assert_eq!(gpu.download(&out).unwrap()[0].to_bits(), r[1].to_bits());
}

/// Exact J and K from the dense ERI tensor (chemist's order, row-major n^4).
struct DenseJk {
    n: usize,
    eri: Vec<f64>,
}

impl HostFockTerms for DenseJk {
    fn coulomb_exchange(
        &mut self,
        st: &HostState<'_>,
        j: &mut [f64],
        mut k: Option<&mut [f64]>,
    ) -> Result<(), GpuError> {
        let n = self.n;
        let (n2, n3) = (n * n, n * n * n);
        for s in 0..st.nocc.len() {
            let d = st.density(s);
            for p in 0..n {
                for q in 0..n {
                    let (mut jv, mut kv) = (0.0, 0.0);
                    for r in 0..n {
                        for t in 0..n {
                            let drt = d[r + t * n];
                            jv += self.eri[p * n3 + q * n2 + r * n + t] * drt;
                            kv += self.eri[p * n3 + r * n2 + q * n + t] * drt;
                        }
                    }
                    j[s * n2 + p + q * n] = jv;
                    if let Some(k) = k.as_deref_mut() {
                        k[s * n2 + p + q * n] = kv;
                    }
                }
            }
        }
        Ok(())
    }
}

struct Water {
    mol: harness::Structure,
    basis: BasisSet,
    n: usize,
    h: Vec<f64>,
    s: Vec<f64>,
    e_nuc: f64,
    nel: usize,
    eri: Vec<f64>,
}

fn water(charge: i32) -> Water {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/molecules/water.xyz");
    let mut mol = harness::load_xyz(&path).unwrap();
    mol.charge = charge;
    let basis: BasisSet = harness::basis(&mol, "def2-svp").unwrap();
    let pairs = PairSet::build(&basis, PairOptions::default());
    let (s, t) = linus_ints::reference::oneshot::overlap_kinetic(&basis);
    let charges: Vec<(f64, [f64; 3])> = mol
        .z
        .iter()
        .zip(&mol.positions)
        .map(|(&z, &p)| (z as f64, p))
        .collect();
    let v = linus_ints::reference::oneshot::potential(&basis, &pairs, &charges);
    let mut e_nuc = 0.0;
    for a in 0..mol.z.len() {
        for b in 0..a {
            let r = (0..3)
                .map(|c| (mol.positions[a][c] - mol.positions[b][c]).powi(2))
                .sum::<f64>()
                .sqrt();
            e_nuc += (mol.z[a] * mol.z[b]) as f64 / r;
        }
    }
    Water {
        n: basis.nbf,
        h: t.iter().zip(&v).map(|(a, b)| a + b).collect(),
        s,
        e_nuc,
        nel: mol.nelectron() as usize,
        eri: eri4c::eri_full(&basis, &pairs, &linus_ints::operator::Operator::Coulomb),
        mol,
        basis,
    }
}

fn run(gpu: &Gpu, w: &Water, guess: Option<(&[f64], &[f64])>, cfg: &LoopConfig) -> ScfResult {
    let mut terms = OnHost::new(DenseJk {
        n: w.n,
        eri: w.eri.clone(),
    });
    run_with(gpu, w, guess, cfg, &mut terms, 1.0)
}

fn run_with(
    gpu: &Gpu,
    w: &Water,
    guess: Option<(&[f64], &[f64])>,
    cfg: &LoopConfig,
    terms: &mut dyn FockTerms,
    exchange_fraction: f64,
) -> ScfResult {
    let (na, nb) = (w.nel.div_ceil(2), w.nel / 2);
    let inp = ScfInput {
        nbf: w.n,
        h: &w.h,
        s: &w.s,
        e_nuc: w.e_nuc,
        nalpha: na,
        nbeta: nb,
        restricted: na == nb,
        exchange_fraction,
        guess,
    };
    run_scf(gpu, &inp, terms, cfg, &mut Plain).unwrap()
}

#[test]
fn rhf_water_matches_pyscf_and_repeats_bitwise() {
    let Some(gpu) = common::gpu() else { return };
    let w = water(0);
    let cfg = LoopConfig::default();
    let r = run(&gpu, &w, None, &cfg);
    assert!(r.converged && r.stop == Stop::Converged, "{:?}", r.stop);
    assert!(r.restricted && r.nmo == w.n && r.dropped_functions == 0);
    assert!(
        (r.energy - E_RHF_PYSCF).abs() < 1e-8,
        "{} vs {E_RHF_PYSCF}",
        r.energy
    );
    assert!(r.iterations < 25, "{} iterations", r.iterations);
    assert_eq!(r.history.last().unwrap().step, Step::Converged);
    assert!(r.history[0].level_shift && r.history.iter().any(|h| h.step == Step::Diis));
    assert_eq!(r.s2, 0.0);
    // the virial-free consistency: E = e_one + e_coulomb + e_exchange + E_nuc
    let parts = r.e_one + r.e_coulomb + r.e_exchange + r.e_xc + r.e_environment + r.e_nuc;
    assert_eq!(parts, r.energy);
    // occupied orbital energies below the virtual ones; the HOMO of water is about -0.5 Eh
    assert!(r.eps_a[4] < -0.4 && r.eps_a[5] > 0.0, "{:?}", &r.eps_a[..6]);
    // bitwise the same run to run
    let r2 = run(&gpu, &w, None, &cfg);
    assert_eq!(r.history.len(), r2.history.len());
    for (a, b) in r.history.iter().zip(&r2.history) {
        assert_eq!(
            a.energy.to_bits(),
            b.energy.to_bits(),
            "iteration {}",
            a.iter
        );
        assert_eq!(a.gmax.to_bits(), b.gmax.to_bits());
    }
    let (c1, c2) = (
        gpu.download(&r.orbitals[0]).unwrap(),
        gpu.download(&r2.orbitals[0]).unwrap(),
    );
    assert!(c1.iter().zip(&c2).all(|(p, q)| p.to_bits() == q.to_bits()));

    // from the converged density as a guess: iteration 1 is the guess's (outside DIIS), the
    // earliest convergence is iteration 3, as in linus's CPU loop
    let d = gpu.download(&r.densities).unwrap();
    let r3 = run(&gpu, &w, Some((&d, &d)), &cfg);
    assert!(r3.converged);
    assert_eq!(r3.iterations, 3);
    assert_eq!(r3.history[0].step, Step::Roothaan);
    assert!((r3.energy - r.energy).abs() < 1e-11);
}

#[test]
fn uhf_water_cation_matches_pyscf() {
    let Some(gpu) = common::gpu() else { return };
    // from the neutral molecule's density scaled to 5 and 4 electrons (as linus's SAD guess
    // scales): the core Hamiltonian's orbitals lead to the 2A1 state, 2.2 eV above 2B1
    let d0 = {
        let r0 = run(&gpu, &water(0), None, &LoopConfig::default());
        gpu.download(&r0.densities).unwrap()
    };
    let (ga, gb): (Vec<f64>, Vec<f64>) = d0.iter().map(|&x| (x, 0.8 * x)).unzip();
    let w = water(1);
    let r = run(&gpu, &w, Some((&ga, &gb)), &LoopConfig::default());
    assert!(r.converged, "{:?}", r.stop);
    assert_eq!(r.history[0].step, Step::Roothaan);
    assert!(!r.restricted && r.orbitals.len() == 2);
    assert_eq!((r.nalpha, r.nbeta), (5, 4));
    assert!(
        (r.energy - E_UHF_CATION_PYSCF).abs() < 1e-8,
        "{} vs {E_UHF_CATION_PYSCF}",
        r.energy
    );
    assert!((r.s2 - S2_UHF_CATION_PYSCF).abs() < 1e-6, "<S^2> {}", r.s2);
    // a closed-shell restricted run refuses an open shell
    let inp = ScfInput {
        nbf: w.n,
        h: &w.h,
        s: &w.s,
        e_nuc: w.e_nuc,
        nalpha: 5,
        nbeta: 4,
        restricted: true,
        exchange_fraction: 1.0,
        guess: None,
    };
    let mut terms = OnHost::new(DenseJk {
        n: w.n,
        eri: vec![],
    });
    assert!(matches!(
        run_scf(&gpu, &inp, &mut terms, &LoopConfig::default(), &mut Plain),
        Err(GpuError::Input(_))
    ));
}

#[test]
fn eigh_plans_recover_a_known_spectrum_and_repeat() {
    let Some(gpu) = common::gpu() else { return };
    let n = 97;
    // A = H diag(lambda) H with a Householder reflection H
    let mut rng = Rng(23);
    let v: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let h = |i: usize, j: usize| (if i == j { 1.0 } else { 0.0 }) - 2.0 * v[i] * v[j] / vv;
    let lambda: Vec<f64> = (0..n).map(|i| -5.0 + 0.1 * i as f64).collect();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[j * n + i] = (0..n).map(|p| h(i, p) * lambda[p] * h(p, j)).sum();
        }
    }
    let solver = Solver::new(&gpu).unwrap();
    for method in [EighMethod::Syevd, EighMethod::Xsyevd, EighMethod::Syevj] {
        let mut plan = solver.eigh_plan(&gpu, n, method, 2).unwrap();
        let mut runs = vec![];
        for slot in 0..2 {
            let mut da = gpu.upload(&a).unwrap();
            let mut dw = gpu.alloc_zeros::<f64>(n).unwrap();
            solver
                .eigh_queued(&mut plan, &mut da, &mut dw, slot)
                .unwrap();
            runs.push((gpu.download(&da).unwrap(), gpu.download(&dw).unwrap()));
        }
        assert_eq!(gpu.download(plan.info()).unwrap(), vec![0, 0], "{method:?}");
        let (x, w) = &runs[0];
        for k in 0..n {
            assert!(
                (w[k] - lambda[k]).abs() < 1e-12,
                "{method:?} eigenvalue {k}"
            );
            let r: f64 = (0..n)
                .map(|i| {
                    let ax: f64 = (0..n).map(|j| a[j * n + i] * x[k * n + j]).sum();
                    (ax - w[k] * x[k * n + i]).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            assert!(r < 1e-12, "{method:?} residual {k}: {r}");
        }
        let same = |p: &[f64], q: &[f64]| p.iter().zip(q).all(|(s, t)| s.to_bits() == t.to_bits());
        assert!(
            same(&runs[0].0, &runs[1].0) && same(&runs[0].1, &runs[1].1),
            "{method:?} repeat"
        );
    }
}

/// A factorization's numerical failure is a numerical error, which does not fall back to the CPU
/// at a setup (it was `Unsupported`, which does: a silent CPU run).
#[test]
fn factorization_failures_are_numerical_errors() {
    let Some(gpu) = common::gpu() else { return };
    let solver = Solver::new(&gpu).unwrap();
    // symmetric, eigenvalues 3 and -1: the leading minor of order 2 is not positive definite;
    // a rank-one matrix: U(2, 2) is zero
    let mut a = gpu.upload(&[1.0, 2.0, 2.0, 1.0]).unwrap();
    let e = solver.potrf_lower(&gpu, &mut a, 2).unwrap_err();
    assert!(
        matches!(
            e,
            GpuError::Numerical {
                op: "cuSOLVER potrf",
                info: 2
            }
        ),
        "{e:?}"
    );
    assert!(!e.falls_back_at_setup());
    assert_eq!(
        e.to_string(),
        "cuSOLVER potrf: the leading minor of order 2 is not positive definite"
    );
    let mut a = gpu.upload(&[1.0, 2.0, 2.0, 4.0]).unwrap();
    let mut ipiv = gpu.alloc_zeros::<i32>(2).unwrap();
    let e = solver.getrf(&gpu, &mut a, 2, &mut ipiv).unwrap_err();
    assert!(
        matches!(
            e,
            GpuError::Numerical {
                op: "cuSOLVER getrf",
                info: 2
            }
        ),
        "{e:?}"
    );
    assert!(!e.falls_back_at_setup());
}

fn df_terms(gpu: &Gpu, w: &Water, functional: Option<&str>) -> DeviceTerms {
    let aux = harness::basis(&w.mol, "def2-universal-jkfit").unwrap();
    let jk = GpuDfJk::new(gpu, &w.basis, &aux, &GpuDfJkOptions::default()).unwrap();
    let xc = functional.map(|f| {
        let grid = harness::grid(&w.mol, &GridSpec::pyscf(3)).unwrap();
        GpuXc::new(gpu, &w.basis, &grid, f, &GpuXcOptions::default()).unwrap()
    });
    DeviceTerms::new(jk, xc)
}

/// The loop on GpuDfJk: DF-HF of water and its cation against PySCF's DF energies, from the
/// core guess and from guess densities (iteration 1's K from the factored density), with
/// each eigensolver.
#[test]
fn device_terms_df_hf_matches_pyscf() {
    let Some(gpu) = common::gpu() else { return };
    let w = water(0);
    let mut terms = df_terms(&gpu, &w, None);
    let cfg = LoopConfig::default();
    let r = run_with(&gpu, &w, None, &cfg, &mut terms, 1.0);
    assert!(r.converged, "{:?}", r.stop);
    assert!(
        (r.energy - E_DF_RHF_PYSCF).abs() < 1e-8,
        "{} vs {E_DF_RHF_PYSCF}",
        r.energy
    );
    let d = gpu.download(&r.densities).unwrap();
    let r2 = run_with(&gpu, &w, Some((&d, &d)), &cfg, &mut terms, 1.0);
    assert!(
        r2.converged && r2.iterations == 3,
        "{} iterations",
        r2.iterations
    );
    assert!((r2.energy - r.energy).abs() < 1e-10);
    // the same terms from another guess: nothing of the previous run's state is reused
    let d97: Vec<f64> = d.iter().map(|x| 0.97 * x).collect();
    let r4 = run_with(&gpu, &w, Some((&d97, &d97)), &cfg, &mut terms, 1.0);
    assert!(r4.converged && (r4.energy - r.energy).abs() < 1e-10);
    assert!((r4.history[0].energy - r2.history[0].energy).abs() > 1e-3);
    for e in [Eigensolver::DivideConquer64, Eigensolver::Jacobi] {
        let c = LoopConfig {
            eigensolver: e,
            ..Default::default()
        };
        let r3 = run_with(&gpu, &w, None, &c, &mut terms, 1.0);
        assert_eq!(r3.iterations, r.iterations, "{e:?}");
        assert!((r3.energy - r.energy).abs() < 1e-10, "{e:?}");
    }

    let wc = water(1);
    let mut tc = df_terms(&gpu, &wc, None);
    let (ga, gb): (Vec<f64>, Vec<f64>) = d.iter().map(|&x| (x, 0.8 * x)).unzip();
    let rc = run_with(&gpu, &wc, Some((&ga, &gb)), &cfg, &mut tc, 1.0);
    assert!(rc.converged, "{:?}", rc.stop);
    assert!(
        (rc.energy - E_DF_UHF_CATION_PYSCF).abs() < 1e-8,
        "{} vs {E_DF_UHF_CATION_PYSCF}",
        rc.energy
    );
}

/// The loop on GpuDfJk and GpuXc (B3LYP): the guess-density path (the XC density from the
/// factored density) reaches the core-guess result, and the run repeats bitwise.
#[test]
fn device_terms_b3lyp_guess_density_and_repeat() {
    let Some(gpu) = common::gpu() else { return };
    let w = water(0);
    let mut terms = df_terms(&gpu, &w, Some("b3lyp"));
    let cfg = LoopConfig::default();
    let r = run_with(&gpu, &w, None, &cfg, &mut terms, 0.2);
    assert!(r.converged, "{:?}", r.stop);
    assert!(r.e_xc < -7.0 && r.e_xc > -8.0, "E_xc {}", r.e_xc);
    let r1 = run_with(&gpu, &w, None, &cfg, &mut terms, 0.2);
    assert!(r
        .history
        .iter()
        .zip(&r1.history)
        .all(|(p, q)| p.energy.to_bits() == q.energy.to_bits()));
    let d = gpu.download(&r.densities).unwrap();
    let r2 = run_with(&gpu, &w, Some((&d, &d)), &cfg, &mut terms, 0.2);
    assert!(
        r2.converged && r2.iterations == 3,
        "{} iterations",
        r2.iterations
    );
    assert!(
        (r2.energy - r.energy).abs() < 1e-10,
        "{} vs {}",
        r2.energy,
        r.energy
    );
}
