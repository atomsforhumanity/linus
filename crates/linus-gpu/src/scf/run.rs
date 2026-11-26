//! The loop of [`run_scf`] (see the module docs of [`super`]).

use super::diis::{DiisBook, DiisStore};
use super::kernels::{Ops, Task};
use super::{
    next_generation, Eigensolver, FockState, FockTerms, HandOver, Iteration, LoopConfig,
    LoopMemory, Observation, PhaseTimes, ScfDecide, ScfInput, ScfResult, Step, Stop, Verdict,
};
use crate::blas::{GemmDims, MatMut, MatRef, Op};
use crate::error::GpuError;
use crate::linalg::{Arith, Gemm};
use crate::solver::{EighMethod, EighPlan, Solver};
use crate::{DeviceBuf, Gpu};
use std::time::Instant;

/// Reductions per batch: 8 per iteration plus one per DIIS entry.
const BASE_TASKS: usize = 8;

/// The device's linear algebra for the loop: cuBLAS, cuSOLVER and the loop's kernels.
struct Dense<'g> {
    gpu: &'g Gpu,
    gemm: Gemm,
    solver: Solver,
    n: usize,
    nmo: usize,
}

impl Dense<'_> {
    /// C = op(A) op(B) for column-major operands given as (buffer, offset, leading dim).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        a: (&DeviceBuf<f64>, usize, usize),
        b: (&DeviceBuf<f64>, usize, usize),
        c: (&mut DeviceBuf<f64>, usize, usize),
    ) -> Result<(), GpuError> {
        self.gemm.gemm(
            Arith::Fp64,
            GemmDims::new(ta, tb, m, n, k),
            (1.0, 0.0),
            MatRef::new(a.0, a.1, a.2),
            MatRef::new(b.0, b.1, b.2),
            MatMut::new(c.0, c.1, c.2),
        )
    }

    /// The orbital gradient X^T (F D S - S D F) X of F and D at element `off` into e at
    /// `eoff` (nmo x nmo), as the CPU: FDS = (F D) S, E = FDS - FDS^T, X^T (E X).
    #[allow(clippy::too_many_arguments)]
    fn gradient(
        &self,
        ops: &Ops,
        f: &DeviceBuf<f64>,
        d: &DeviceBuf<f64>,
        off: usize,
        s: &DeviceBuf<f64>,
        x: &DeviceBuf<f64>,
        t1: &mut DeviceBuf<f64>,
        t2: &mut DeviceBuf<f64>,
        tx: &mut DeviceBuf<f64>,
        e: &mut DeviceBuf<f64>,
        eoff: usize,
    ) -> Result<(), GpuError> {
        let (n, nmo) = (self.n, self.nmo);
        self.gemm(Op::N, Op::N, n, n, n, (f, off, n), (d, off, n), (t1, 0, n))?;
        self.gemm(Op::N, Op::N, n, n, n, (t1, 0, n), (s, 0, n), (t2, 0, n))?;
        ops.antisym(self.gpu, n, t2, t1)?;
        self.gemm(Op::N, Op::N, n, nmo, n, (t1, 0, n), (x, 0, n), (tx, 0, n))?;
        self.gemm(
            Op::T,
            Op::N,
            nmo,
            nmo,
            n,
            (x, 0, n),
            (tx, 0, n),
            (e, eoff, nmo),
        )
    }

    /// Queues the eigendecomposition of X^T M X (M at element `off`): eigenvalues into `w`,
    /// eigenvectors C' into `cp` (nmo x nmo), status into slot `slot` of the plan.
    #[allow(clippy::too_many_arguments)]
    fn eigh_in_basis(
        &self,
        plan: &mut EighPlan,
        m: &DeviceBuf<f64>,
        off: usize,
        x: &DeviceBuf<f64>,
        tx: &mut DeviceBuf<f64>,
        cp: &mut DeviceBuf<f64>,
        w: &mut DeviceBuf<f64>,
        slot: usize,
    ) -> Result<(), GpuError> {
        let (n, nmo) = (self.n, self.nmo);
        self.gemm(Op::N, Op::N, n, nmo, n, (m, off, n), (x, 0, n), (tx, 0, n))?;
        self.gemm(
            Op::T,
            Op::N,
            nmo,
            nmo,
            n,
            (x, 0, n),
            (tx, 0, n),
            (cp, 0, nmo),
        )?;
        self.solver.eigh_queued(plan, cp, w, slot)
    }

    /// The first `ncol` orbitals X C'[:, :ncol] into `out` at element `ooff`.
    fn orbitals(
        &self,
        x: &DeviceBuf<f64>,
        cp: &DeviceBuf<f64>,
        ncol: usize,
        out: &mut DeviceBuf<f64>,
        ooff: usize,
    ) -> Result<(), GpuError> {
        let (n, nmo) = (self.n, self.nmo);
        if ncol == 0 {
            return Ok(());
        }
        self.gemm(
            Op::N,
            Op::N,
            n,
            ncol,
            nmo,
            (x, 0, n),
            (cp, 0, nmo),
            (out, ooff, n),
        )
    }

    /// D = C C^T of the `ncol` orbitals at element `coff` of `c`, into `d` at `doff`.
    fn density(
        &self,
        c: &DeviceBuf<f64>,
        coff: usize,
        ncol: usize,
        d: &mut DeviceBuf<f64>,
        doff: usize,
    ) -> Result<(), GpuError> {
        let n = self.n;
        if ncol == 0 {
            return self.gemm(Op::N, Op::T, n, n, 0, (c, 0, n), (c, 0, n), (d, doff, n));
        }
        self.gemm(
            Op::N,
            Op::T,
            n,
            n,
            ncol,
            (c, coff, n),
            (c, coff, n),
            (d, doff, n),
        )
    }

    /// S M S of M at element `off` into `out` at `ooff`.
    #[allow(clippy::too_many_arguments)]
    fn sms(
        &self,
        s: &DeviceBuf<f64>,
        m: &DeviceBuf<f64>,
        off: usize,
        t1: &mut DeviceBuf<f64>,
        out: &mut DeviceBuf<f64>,
        ooff: usize,
    ) -> Result<(), GpuError> {
        let n = self.n;
        self.gemm(Op::N, Op::N, n, n, n, (s, 0, n), (m, off, n), (t1, 0, n))?;
        self.gemm(Op::N, Op::N, n, n, n, (t1, 0, n), (s, 0, n), (out, ooff, n))
    }
}

/// Indices of a reduction batch's results.
#[derive(Default)]
struct Slots {
    tasks: Vec<Task>,
}

impl Slots {
    fn add(&mut self, t: Task) -> usize {
        self.tasks.push(t);
        self.tasks.len() - 1
    }
}

/// Wall seconds since `t`.
fn seconds_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

/// Device seconds of the work `f` queues on the stream, from marks around it (waits for it).
fn device_seconds(gpu: &Gpu, f: impl FnOnce() -> Result<(), GpuError>) -> Result<f64, GpuError> {
    let start = gpu.mark()?;
    f()?;
    let end = gpu.mark()?;
    start.seconds_until(&end)
}

/// The SCF of `inp` with the Fock terms `terms`, on `gpu`, decided by `ctl` (see the module docs).
/// Errors are device and input errors; numerical failures end the loop with [`Stop::NonFinite`]
/// and a NaN energy, as linus's CPU loop.
pub fn run_scf(
    gpu: &Gpu,
    inp: &ScfInput<'_>,
    terms: &mut dyn FockTerms,
    cfg: &LoopConfig,
    ctl: &mut dyn ScfDecide,
) -> Result<ScfResult, GpuError> {
    let t_setup = Instant::now();
    let n = inp.nbf;
    let n2 = n * n;
    if inp.h.len() != n2 || inp.s.len() != n2 {
        return Err(GpuError::Contract(format!(
            "SCF: h and S need {n} x {n} elements"
        )));
    }
    let (na, nb) = (inp.nalpha, inp.nbeta);
    let restricted = inp.restricted;
    if restricted && na != nb {
        return Err(GpuError::Input(
            "restricted SCF needs a closed shell (use UHF)".into(),
        ));
    }
    let nocc: Vec<usize> = if restricted { vec![na] } else { vec![na, nb] };
    let nsp = nocc.len();
    // column offsets of each channel's occupied orbitals
    let coff: Vec<usize> = (0..nsp)
        .map(|s| n * nocc[..s].iter().sum::<usize>())
        .collect();
    let ncol: usize = nocc.iter().sum();
    let a_x = inp.exchange_fraction;
    let want_k = a_x != 0.0;

    let mut ops = Ops::new(gpu, BASE_TASKS + cfg.diis_space, cfg.diis_space.max(1))?;
    let h = gpu.upload(inp.h)?;
    let s = gpu.upload(inp.s)?;

    // canonical orthogonalization: X = U s^-1/2 over the eigenvalues above lindep (ascending,
    // so the kept columns are the last nmo)
    let solver = Solver::new(gpu)?;
    let (x, nmo) = {
        let mut u = gpu.upload(inp.s)?;
        let mut w = gpu.alloc_zeros::<f64>(n.max(1))?;
        solver.syevd(gpu, &mut u, n, &mut w)?;
        let wh = gpu.download(&w)?;
        let nmo = wh[..n].iter().filter(|&&v| v > cfg.lindep).count();
        let mut x = gpu.alloc_zeros::<f64>((n * nmo).max(1))?;
        ops.orth_columns(gpu, n, nmo, n - nmo, &u, &w, &mut x)?;
        (x, nmo)
    };
    if nmo < na.max(nb) {
        return Err(GpuError::Input(format!(
            "scf.lindep {:e} drops the basis to {nmo} orbitals, fewer than the {} occupied",
            cfg.lindep,
            na.max(nb)
        )));
    }
    let dense = Dense {
        gpu,
        gemm: Gemm::new(gpu)?,
        solver,
        n,
        nmo,
    };
    let nmo2 = nmo * nmo;
    let method = match cfg.eigensolver {
        Eigensolver::DivideConquer => EighMethod::Syevd,
        Eigensolver::DivideConquer64 => EighMethod::Xsyevd,
        Eigensolver::Jacobi => EighMethod::Syevj,
    };
    let mut plan = dense.solver.eigh_plan(gpu, nmo, method, nsp)?;

    // state and scratch
    let z = |len: usize| gpu.alloc_zeros::<f64>(len.max(1));
    let mut d = z(nsp * n2)?;
    let mut d_prev = z(nsp * n2)?;
    let mut dd = z(nsp * n2)?;
    let mut cocc = z(n * ncol)?;
    let mut j = z(nsp * n2)?;
    let mut jt = z(n2)?;
    let mut k = if want_k { Some(z(nsp * n2)?) } else { None };
    let mut vxc = z(nsp * n2)?;
    let mut venv = z(n2)?;
    let mut f = z(nsp * n2)?;
    let mut fx = z(nsp * n2)?;
    let mut sds = z(nsp * n2)?;
    let mut dt = z(n2)?;
    let mut e = z(nsp * nmo2)?;
    let mut t1 = z(n2)?;
    let mut t2 = z(n2)?;
    let mut tx = z(n * nmo)?;
    // the eigenvectors of each channel's last diagonalization (the orbitals of D_n in the
    // basis X, kept for a hand-over)
    let mut cps: Vec<DeviceBuf<f64>> = (0..nsp).map(|_| z(nmo2)).collect::<Result<_, _>>()?;
    // the occupied orbitals of the lowest-energy iterate before the current one (for a
    // hand-over: the stability analysis' first trial direction, the escalation's start)
    let mut cocc_min = if cfg.keep_lowest {
        Some(z(n * ncol)?)
    } else {
        None
    };
    let mut eps: Vec<DeviceBuf<f64>> = (0..nsp).map(|_| z(nmo)).collect::<Result<_, _>>()?;
    let mut scal = z(BASE_TASKS + cfg.diis_space)?;
    let state_bytes = 8
        * (h.len()
            + s.len()
            + x.len()
            + d.len()
            + d_prev.len()
            + dd.len()
            + cocc.len()
            + j.len()
            + jt.len()
            + k.as_ref().map_or(0, |b| b.len())
            + vxc.len()
            + venv.len()
            + f.len()
            + fx.len()
            + sds.len()
            + dt.len()
            + e.len()
            + t1.len()
            + t2.len()
            + tx.len()
            + nsp * nmo2
            + cocc_min.as_ref().map_or(0, |b| b.len())
            + nsp * nmo
            + scal.len())
        + plan.bytes()
        + ops.bytes();

    // the step's checks: non-finite orbital energies or orbitals, and the eigensolver's status
    let check_step = |ops: &mut Ops,
                      eps: &[DeviceBuf<f64>],
                      cocc: &DeviceBuf<f64>,
                      plan: &EighPlan,
                      scal: &mut DeviceBuf<f64>|
     -> Result<bool, GpuError> {
        let mut tasks = Vec::with_capacity(nsp + 1);
        for w in eps {
            tasks.push(Task::non_finite((w, 0), nmo)?);
        }
        tasks.push(Task::non_finite((cocc, 0), n * ncol)?);
        ops.reduce(gpu, &tasks, scal)?;
        let bad = gpu.download(scal)?[..tasks.len()].iter().any(|&c| c != 0.0);
        let info = gpu.download(plan.info())?;
        Ok(!bad && info[..nsp].iter().all(|&i| i == 0))
    };

    // the guess: densities, or the core Hamiltonian's orbitals
    let mut have_orbitals = false;
    match inp.guess {
        Some((ga, gb)) => {
            if ga.len() != n2 || gb.len() != n2 {
                return Err(GpuError::Contract(format!(
                    "SCF: guess densities need {n} x {n} elements"
                )));
            }
            if restricted && ga.iter().zip(gb).any(|(p, q)| p.to_bits() != q.to_bits()) {
                return Err(GpuError::Contract(
                    "SCF: a closed-shell guess needs D_b = D_a".into(),
                ));
            }
            gpu.upload_into(&mut d, 0, ga)?;
            if !restricted {
                gpu.upload_into(&mut d, n2, gb)?;
            }
        }
        None => {
            // one eigendecomposition for both channels (the beta status slot and eigenvalues
            // are still zero, so the check below reads them as fine)
            dense.eigh_in_basis(&mut plan, &h, 0, &x, &mut tx, &mut cps[0], &mut eps[0], 0)?;
            if let [c0, rest @ ..] = cps.as_mut_slice() {
                for c in rest {
                    gpu.copy_range(c0, 0, c, 0, nmo2)?;
                }
            }
            for sp in 0..nsp {
                dense.orbitals(&x, &cps[sp], nocc[sp], &mut cocc, coff[sp])?;
                dense.density(&cocc, coff[sp], nocc[sp], &mut d, sp * n2)?;
            }
            have_orbitals = true;
            if !check_step(&mut ops, &eps, &cocc, &plan, &mut scal)? {
                return Err(GpuError::NotFinite(
                    "SCF: the core Hamiltonian's orbitals are not finite".into(),
                ));
            }
        }
    }
    // the iteration whose density came from a guess (not from orbitals) stays out of DIIS
    let guess_iter = if have_orbitals { 0 } else { 1 };
    let setup_seconds = seconds_since(t_setup);

    let mut book = DiisBook::new(cfg.diis_space);
    let mut store = DiisStore::default();
    let mut diis_bytes = 0usize;
    let mut history: Vec<Iteration> = Vec::new();
    let mut e_old = 0.0;
    let mut energy = 0.0;
    let mut parts = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut handover: Option<HandOver> = None;
    // with keep_occupied: the occupied orbitals of the converged state's density
    let mut occupied_kept: Option<DeviceBuf<f64>> = None;
    let mut stop = Stop::MaxIterations;
    let mut iters = 0;
    let mut has_prev = false;
    // the build is at the same state, of the densities alone ([`Verdict::Confirm`])
    let mut rebuild = false;

    for it in 1..=cfg.max_iter {
        iters = it;
        let rebuilt = std::mem::take(&mut rebuild);
        // a confirming build, and the budget's last iteration, are of the densities alone
        if rebuilt || it == cfg.max_iter {
            terms.reset_incremental();
        }
        // the densities change once per iteration; numbered across runs, so that terms
        // reused by another SCF never take its state for a cached one
        let generation = next_generation();
        let t_it = Instant::now();
        let mut times = PhaseTimes::default();
        // the Fock terms at D_n
        let st = FockState {
            nbf: n,
            restricted,
            densities: &d,
            orbitals: have_orbitals.then_some(&cocc),
            nocc: &nocc,
            generation,
        };
        let t = Instant::now();
        terms.coulomb_exchange(gpu, &st, &mut j, k.as_mut())?;
        gpu.synchronize()?;
        times.jk = seconds_since(t);
        let t = Instant::now();
        let exc = terms.exchange_correlation(gpu, &st, &mut vxc)?;
        gpu.synchronize()?;
        times.xc = seconds_since(t);
        let t = Instant::now();
        let env = terms.environment(gpu, &st, &mut venv)?;
        gpu.synchronize()?;
        times.environment = seconds_since(t);
        let incremental = terms.built_incrementally();

        // Fock matrices, energy terms, orbital gradients, and the scalars the host decides on
        let mut sl = Slots::default();
        let (mut i_ek, mut i_dd, mut i_ddmax) = (None, None, None);
        let (i_eone, i_ej, i_ess, i_emax, i_fbad);
        times.evaluate = device_seconds(gpu, || {
            if let Some(k) = k.as_mut() {
                ops.scale(gpu, nsp * n2, k, a_x)?;
            }
            // J of the total density: 2 J[D_a] for a closed shell, J[D_a] + J[D_b] otherwise
            let jb = if restricted { 0 } else { n2 };
            ops.add(gpu, n2, (&j, 0), (&j, jb), 1.0, (&mut jt, 0))?;
            ops.fock(
                gpu,
                nsp,
                n2,
                &h,
                &jt,
                k.as_ref(),
                exc.is_some().then_some(&vxc),
                matches!(env, Some((_, true))).then_some(&venv),
                &mut f,
            )?;
            let db = if restricted { 0 } else { n2 };
            ops.add(gpu, n2, (&d, 0), (&d, db), 1.0, (&mut dt, 0))?;
            for sp in 0..nsp {
                dense.gradient(
                    &ops,
                    &f,
                    &d,
                    sp * n2,
                    &s,
                    &x,
                    &mut t1,
                    &mut t2,
                    &mut tx,
                    &mut e,
                    sp * nmo2,
                )?;
            }
            if has_prev {
                ops.add(gpu, nsp * n2, (&d, 0), (&d_prev, 0), -1.0, (&mut dd, 0))?;
            }
            Ok(())
        })?;
        i_eone = sl.add(Task::dot((&dt, 0), (&h, 0), n2)?);
        i_ej = sl.add(Task::dot((&dt, 0), (&jt, 0), n2)?);
        if let Some(k) = k.as_ref() {
            i_ek = Some(sl.add(Task::dot((&d, 0), (k, 0), nsp * n2)?));
        }
        i_ess = sl.add(Task::dot((&e, 0), (&e, 0), nsp * nmo2)?);
        i_emax = sl.add(Task::max_abs((&e, 0), nsp * nmo2)?);
        i_fbad = sl.add(Task::non_finite((&f, 0), nsp * n2)?);
        if has_prev {
            i_dd = Some(sl.add(Task::dot((&dd, 0), (&dd, 0), nsp * n2)?));
            i_ddmax = Some(sl.add(Task::max_abs((&dd, 0), nsp * n2)?));
        }
        // the new error's inner products with the stored ones (the DIIS matrix's new row)
        let row0 = sl.tasks.len();
        for ent in &store.entries {
            sl.add(Task::dot((&e, 0), (&ent.e[0], 0), nsp * nmo2)?);
        }
        times.evaluate += device_seconds(gpu, || ops.reduce(gpu, &sl.tasks, &mut scal))?;
        let sv = gpu.download(&scal)?;

        let e_one = sv[i_eone];
        let e_j = 0.5 * sv[i_ej];
        let e_k = match (terms.exchange_energy(), i_ek) {
            // the builder's own energy (compensated grid-K), with the fraction a_x
            (Some(ek), _) => -a_x * if restricted { 2.0 * ek } else { ek },
            // a closed shell: D_b = D_a and K_b = K_a, as the CPU's two terms
            (None, Some(i)) if restricted => -0.5 * (sv[i] + sv[i]),
            (None, Some(i)) => -0.5 * sv[i],
            (None, None) => -0.5 * (0.0 + 0.0),
        };
        let exc_v = exc.unwrap_or(0.0);
        let e_env = env.map_or(0.0, |x| x.0);
        energy = e_one + e_j + e_k + (exc_v + e_env) + inp.e_nuc;
        parts = (e_one, e_j, e_k, exc_v, e_env);
        let ss = sv[i_ess];
        let gmax = sv[i_emax];
        let cnt = (nsp * nmo2).max(1) as f64;
        let grms = (ss / cnt).sqrt();
        let gnorm_pyscf = if restricted {
            (2.0 * ss).sqrt()
        } else {
            (0.5 * ss).sqrt()
        };
        let (ddmax, ddrms) = match (i_ddmax, i_dd) {
            (Some(m), Some(q)) => (sv[m], (sv[q] / (nsp * n2) as f64).sqrt()),
            _ => (0.0, 0.0),
        };
        let non_finite = !(energy.is_finite() && sv[i_fbad] == 0.0);
        let de = energy - e_old;
        let mut rec = Iteration {
            iter: it,
            energy: if non_finite { f64::NAN } else { energy },
            de,
            gmax,
            grms,
            gnorm_pyscf,
            ddmax,
            ddrms,
            incremental,
            step: Step::Converged,
            level_shift: false,
            diis_size: book.len(),
            times,
        };
        if non_finite {
            rec.step = Step::NonFinite;
            rec.times.wall = seconds_since(t_it);
            history.push(rec);
            stop = Stop::NonFinite;
            break;
        }
        // the controller's decision
        let norms = book.norms();
        let verdict = ctl.observe(&Observation {
            iter: it,
            guess_iter,
            energy,
            de,
            gmax,
            gnorm_pyscf,
            previous: [
                history.last().map(|h| h.energy),
                history.len().checked_sub(2).map(|i| history[i].energy),
            ],
            diis_norms: &norms,
            diis_capacity: cfg.diis_space,
            error_norm: ss,
            rebuilt,
            incremental,
        });
        let step = match verdict {
            Verdict::Step(p) => Some(p),
            Verdict::Confirm => {
                // no step: the next build is at D_n again (and |D - D_prev| then 0, as on the
                // host loop)
                rec.step = Step::Rebuild;
                rebuild = true;
                e_old = energy;
                gpu.copy(&d, &mut d_prev)?;
                has_prev = true;
                rec.times.wall = seconds_since(t_it);
                history.push(rec);
                continue;
            }
            Verdict::Converged => {
                stop = Stop::Converged;
                if cfg.keep_occupied {
                    let mut o = z(n * ncol)?;
                    gpu.copy(&cocc, &mut o)?;
                    occupied_kept = Some(o);
                }
                None
            }
            Verdict::Analyse => {
                rec.step = Step::StabilityCheck;
                stop = Stop::StabilityCheck;
                None
            }
            Verdict::TwoCycle => {
                rec.step = Step::TwoCycle;
                stop = Stop::TwoCycle;
                None
            }
            Verdict::Stop => {
                rec.step = Step::Stopped;
                stop = Stop::Requested;
                None
            }
        };
        let Some(step) = step else {
            if matches!(stop, Stop::StabilityCheck | Stop::TwoCycle) {
                // the orbitals D_n came from: X C' of each channel's last eigenvectors
                let mut step_orbitals = Vec::with_capacity(nsp);
                for cp in &cps {
                    let mut c = z(n * nmo)?;
                    dense.orbitals(&x, cp, nmo, &mut c, 0)?;
                    step_orbitals.push(c);
                }
                let mut occupied = z(n * ncol)?;
                gpu.copy(&cocc, &mut occupied)?;
                let prev_densities = if has_prev {
                    let mut p = z(nsp * n2)?;
                    gpu.copy(&d_prev, &mut p)?;
                    Some(p)
                } else {
                    None
                };
                handover = Some(HandOver {
                    occupied,
                    step_orbitals,
                    prev_densities,
                    e_prev: e_old,
                    min_orbitals: cocc_min.take(),
                    guess_iter,
                });
            }
            rec.times.wall = seconds_since(t_it);
            history.push(rec);
            break;
        };
        if step.lowest {
            if let Some(m) = cocc_min.as_mut() {
                gpu.copy(&cocc, m)?;
            }
        }
        e_old = energy;

        // DIIS: the new entry, then the extrapolated Fock matrices
        rec.times.diis = device_seconds(gpu, || {
            if step.diis_push {
                let row = &sv[row0..row0 + book.len()];
                book.push(row, ss, step.diis_victim);
                store.push(gpu, step.diis_victim, &[&f], &[&e])?;
            }
            Ok(())
        })?;
        let use_diis = step.extrapolate;
        if use_diis {
            let c = book.coefficients();
            let tm = device_seconds(gpu, || {
                let terms: Vec<(&DeviceBuf<f64>, f64)> = store
                    .entries
                    .iter()
                    .zip(&c)
                    .map(|(ent, &ci)| (&ent.f[0], ci))
                    .collect();
                ops.combine(gpu, nsp * n2, &terms, &mut fx)
            })?;
            rec.times.diis += tm;
        }
        diis_bytes = diis_bytes.max(store.bytes());
        rec.step = if use_diis { Step::Diis } else { Step::Roothaan };
        rec.diis_size = book.len();

        // level shift F + shift (S - S D S) when the controller asks, and diagonalization
        let shift_on = step.level_shift.is_some();
        let shift_val = step.level_shift.unwrap_or(0.0);
        rec.level_shift = shift_on;
        rec.times.diag = device_seconds(gpu, || {
            if shift_on {
                if !use_diis {
                    gpu.copy(&f, &mut fx)?;
                }
                for sp in 0..nsp {
                    dense.sms(&s, &d, sp * n2, &mut t1, &mut sds, sp * n2)?;
                }
                ops.level_shift(gpu, nsp, n2, &s, &sds, shift_val, &mut fx)?;
            }
            let src = if use_diis || shift_on { &fx } else { &f };
            for sp in 0..nsp {
                dense.eigh_in_basis(
                    &mut plan,
                    src,
                    sp * n2,
                    &x,
                    &mut tx,
                    &mut cps[sp],
                    &mut eps[sp],
                    sp,
                )?;
                dense.orbitals(&x, &cps[sp], nocc[sp], &mut cocc, coff[sp])?;
            }
            Ok(())
        })?;
        // the new densities (D_n is kept for |D_n+1 - D_n|)
        std::mem::swap(&mut d, &mut d_prev);
        rec.times.density = device_seconds(gpu, || {
            for sp in 0..nsp {
                dense.density(&cocc, coff[sp], nocc[sp], &mut d, sp * n2)?;
            }
            Ok(())
        })?;
        let t = Instant::now();
        let ok = check_step(&mut ops, &eps, &cocc, &plan, &mut scal)?;
        rec.times.density += seconds_since(t);
        has_prev = true;
        have_orbitals = true;
        rec.times.wall = seconds_since(t_it);
        if !ok {
            rec.step = Step::NonFinite;
            history.push(rec);
            stop = Stop::NonFinite;
            break;
        }
        history.push(rec);
        if step.hand_over {
            stop = Stop::HandOver;
            break;
        }
    }

    // after a step the controller hands over at, or at max_iter: the state after the last
    // iteration's step, for the caller to continue (linus's escalation) or to analyse (linus's
    // plateau; before the final diagonalization reuses cps)
    let at_max = stop == Stop::MaxIterations && cfg.handover_at_max_iter && iters == cfg.max_iter;
    if (stop == Stop::HandOver || at_max) && has_prev {
        let mut step_orbitals = Vec::with_capacity(nsp);
        for cp in &cps {
            let mut c = z(n * nmo)?;
            dense.orbitals(&x, cp, nmo, &mut c, 0)?;
            step_orbitals.push(c);
        }
        let mut occupied = z(n * ncol)?;
        gpu.copy(&cocc, &mut occupied)?;
        let mut p = z(nsp * n2)?;
        gpu.copy(&d_prev, &mut p)?;
        handover = Some(HandOver {
            occupied,
            step_orbitals,
            prev_densities: Some(p),
            e_prev: e_old,
            min_orbitals: cocc_min.take(),
            guess_iter,
        });
    }
    // final orbitals: the unextrapolated Fock matrices
    let mut orbitals: Vec<DeviceBuf<f64>> = Vec::new();
    let mut eps_h: Vec<Vec<f64>> = vec![vec![f64::NAN; nmo]; nsp];
    let mut final_ok = stop != Stop::NonFinite;
    if final_ok {
        for sp in 0..nsp {
            dense.eigh_in_basis(
                &mut plan,
                &f,
                sp * n2,
                &x,
                &mut tx,
                &mut cps[sp],
                &mut eps[sp],
                sp,
            )?;
            let mut c = z(n * nmo)?;
            dense.orbitals(&x, &cps[sp], nmo, &mut c, 0)?;
            orbitals.push(c);
            eps_h[sp] = gpu.download(&eps[sp])?[..nmo].to_vec();
        }
        let info = gpu.download(plan.info())?;
        final_ok = info[..nsp].iter().all(|&i| i == 0)
            && eps_h.iter().all(|v| v.iter().all(|x| x.is_finite()));
    }
    let converged = final_ok && stop == Stop::Converged;
    if !final_ok {
        energy = f64::NAN;
        for v in eps_h.iter_mut() {
            v.iter_mut().for_each(|x| *x = f64::NAN);
        }
        if stop != Stop::NonFinite {
            stop = Stop::NonFinite;
        }
    }
    let sz = 0.5 * (na as f64 - nb as f64);
    let s2 = if restricted {
        sz * (sz + 1.0)
    } else {
        dense.sms(&s, &d, 0, &mut t1, &mut t2, 0)?;
        ops.reduce(gpu, &[Task::dot((&t2, 0), (&d, n2), n2)?], &mut scal)?;
        let v = gpu.download(&scal)?[0];
        sz * (sz + 1.0) + nb as f64 - v
    };
    let (eps_a, eps_b) = if restricted {
        (eps_h[0].clone(), eps_h[0].clone())
    } else {
        (eps_h[0].clone(), eps_h[1].clone())
    };
    let orbital_bytes = 8 * orbitals.iter().map(|b| b.len()).sum::<usize>();
    Ok(ScfResult {
        energy,
        e_nuc: inp.e_nuc,
        e_one: parts.0,
        e_coulomb: parts.1,
        e_exchange: parts.2,
        e_xc: parts.3,
        e_environment: parts.4,
        converged,
        stop,
        iterations: iters,
        restricted,
        nalpha: na,
        nbeta: nb,
        nmo,
        dropped_functions: n - nmo,
        eps_a,
        eps_b,
        s2,
        history,
        orbitals,
        densities: d,
        fock: f,
        setup_seconds,
        memory: LoopMemory {
            loop_bytes: state_bytes + diis_bytes + orbital_bytes,
            diis_bytes,
            pool_peak: gpu.memory().peak,
        },
        handover,
        occupied: occupied_kept,
    })
}
