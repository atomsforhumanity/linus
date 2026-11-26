//! linus-gpu's own engines as the loop's Fock terms: [`DeviceTerms`] runs J on [`DeviceJ`]
//! (the DF engine [`GpuDfJk`], or the memory-lean [`GpuDfJDirect`] without a stored tensor),
//! K on the DF engine or on compensated grid-K ([`GridKTerm`]), XC on [`GpuXc`] and the VV10
//! term of -V functionals on [`GpuVv10`], with every matrix on the device.
//!
//! A density without orbitals (the guess) is factored on the device as linus's
//! `factor_symmetric` does: D = U w U^T, P = U_+ w_+^1/2 and N = U_- (-w_-)^1/2 over the
//! eigenvalues beyond 1e-14 of the largest, so that K\[D\] = K\[P P^T\] - K\[N N^T\] and the XC
//! density comes from P and N with occupations +1 and -1.

use super::kernels::Ops;
use super::{FockState, FockTerms};
use crate::df::{GpuDfJk, GpuJkDeviceRequest};
use crate::error::GpuError;
use crate::gridk::GridKTerm;
use crate::jdirect::GpuDfJDirect;
use crate::solver::Solver;
use crate::xc::vv10::GpuVv10;
use crate::xc::GpuXc;
use crate::{DeviceBuf, Gpu};

/// The factors of a guess density: the columns [P_a N_a (P_b N_b)], their counts and the
/// occupations (+1 for P, -1 for N), for one SCF state.
struct GuessFactors {
    generation: u64,
    c: DeviceBuf<f64>,
    /// [p_a, n_a, (p_b, n_b)]
    ncols: Vec<usize>,
    occ: DeviceBuf<f64>,
}

/// The engine of the loop's J.
pub enum DeviceJ {
    /// The DF engine: J, and K when the terms have no other exchange engine.
    Df(Box<GpuDfJk>),
    /// The memory-lean DF-J (no stored tensor): J alone.
    Direct(Box<GpuDfJDirect>),
}

impl DeviceJ {
    /// The DF engine, if J runs on it.
    pub fn df(&self) -> Option<&GpuDfJk> {
        match self {
            DeviceJ::Df(jk) => Some(jk.as_ref()),
            DeviceJ::Direct(_) => None,
        }
    }
}

/// J on a [`DeviceJ`], K on the DF engine or a [`GridKTerm`] and, for Kohn-Sham, XC on a
/// [`GpuXc`] plus, for -V functionals, VV10 on a [`GpuVv10`] (see the module docs).
pub struct DeviceTerms {
    pub j: DeviceJ,
    /// rij_cgk's exchange: K and E_K of compensated grid-K; None: the DF engine's K.
    pub gridk: Option<GridKTerm>,
    pub xc: Option<GpuXc>,
    /// VV10 of a -V functional: its energy joins E_xc, its potential both spins' V_xc.
    pub vv10: Option<GpuVv10>,
    ops: Option<Ops>,
    guess: Option<GuessFactors>,
}

impl DeviceTerms {
    pub fn new(j: DeviceJ, xc: Option<GpuXc>) -> Self {
        DeviceTerms {
            j,
            gridk: None,
            xc,
            vv10: None,
            ops: None,
            guess: None,
        }
    }

    /// With compensated grid-K's exchange `gridk` (rij_cgk's; None: the DF engine's K).
    pub fn with_gridk(mut self, gridk: Option<GridKTerm>) -> Self {
        self.gridk = gridk;
        self
    }

    /// With the VV10 term `vv10` (a -V functional's; None: none).
    pub fn with_vv10(mut self, vv10: Option<GpuVv10>) -> Self {
        self.vv10 = vv10;
        self
    }
}

/// The loop's kernels in `slot` (compiled on first use).
fn ops<'a>(slot: &'a mut Option<Ops>, gpu: &Gpu) -> Result<&'a Ops, GpuError> {
    if slot.is_none() {
        *slot = Some(Ops::new(gpu, 1, 1)?);
    }
    slot.as_ref()
        .ok_or_else(|| GpuError::Contract("the loop's kernels".into()))
}

/// The factors of the state's densities in `guess` (computed once per state), with the loop's
/// kernels in `ops_slot`.
fn factors<'a>(
    guess: &'a mut Option<GuessFactors>,
    ops_slot: &mut Option<Ops>,
    gpu: &Gpu,
    st: &FockState<'_>,
) -> Result<&'a GuessFactors, GpuError> {
    if guess
        .as_ref()
        .is_some_and(|g| g.generation != st.generation)
    {
        *guess = None;
    }
    if guess.is_none() {
        let ops = ops(ops_slot, gpu)?;
        let n = st.nbf;
        let n2 = n * n;
        let solver = Solver::new(gpu)?;
        let mut parts = Vec::new();
        for s in 0..st.nspin() {
            let mut u = gpu.alloc_zeros::<f64>(n2.max(1))?;
            gpu.copy_range(st.densities, s * n2, &mut u, 0, n2)?;
            let mut w = gpu.alloc_zeros::<f64>(n.max(1))?;
            solver.syevd(gpu, &mut u, n, &mut w)?;
            let wh = gpu.download(&w)?;
            let wmax = wh[..n].iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            let cut = 1e-14 * wmax.max(1e-300);
            // ascending: the negative eigenvalues first, the positive ones last
            let nneg = wh[..n].iter().filter(|&&x| x < -cut).count();
            let npos = wh[..n].iter().filter(|&&x| x > cut).count();
            parts.push((u, w, npos, nneg));
        }
        let ncols: Vec<usize> = parts.iter().flat_map(|p| [p.2, p.3]).collect();
        let total: usize = ncols.iter().sum();
        let mut c = gpu.alloc_zeros::<f64>((n * total).max(1))?;
        let mut occ = Vec::with_capacity(total);
        let mut col = 0;
        for (u, w, npos, nneg) in &parts {
            ops.factor_columns(gpu, n, *npos, n - npos, u, w, 1.0, (&mut c, n * col))?;
            col += npos;
            ops.factor_columns(gpu, n, *nneg, 0, u, w, -1.0, (&mut c, n * col))?;
            col += nneg;
            occ.extend(std::iter::repeat_n(1.0, *npos));
            occ.extend(std::iter::repeat_n(-1.0, *nneg));
        }
        *guess = Some(GuessFactors {
            generation: st.generation,
            c,
            ncols,
            occ: gpu.upload(&occ)?,
        });
    }
    guess
        .as_ref()
        .ok_or_else(|| GpuError::Contract("the guess density's factors".into()))
}

impl FockTerms for DeviceTerms {
    fn coulomb_exchange(
        &mut self,
        gpu: &Gpu,
        st: &FockState<'_>,
        j: &mut DeviceBuf<f64>,
        k: Option<&mut DeviceBuf<f64>>,
    ) -> Result<(), GpuError> {
        let nsp = st.nspin();
        let req = |factors, ncols| GpuJkDeviceRequest {
            dens: Some(st.densities),
            ndens: nsp,
            factors,
            ncols,
        };
        let df_k = self.gridk.is_none() && k.is_some();
        let jk = match &mut self.j {
            DeviceJ::Df(jk) if df_k => jk,
            // J alone on its engine (of the densities, no factors); K on grid-K
            j_eng => {
                match j_eng {
                    DeviceJ::Df(jk) => jk.compute_device(gpu, &req(None, &[]), Some(j), None)?,
                    DeviceJ::Direct(e) => e.coulomb(st.densities, nsp, j)?,
                }
                return match (self.gridk.as_mut(), k) {
                    (Some(g), Some(k)) => g.exchange(st.densities, nsp, k),
                    (None, Some(_)) => Err(GpuError::Contract(
                        "the memory-lean J has no exchange: K needs the DF engine or grid-K".into(),
                    )),
                    (_, None) => Ok(()),
                };
            }
        };
        match (k, st.orbitals) {
            (None, _) => jk.compute_device(gpu, &req(None, &[]), Some(j), None),
            (Some(k), Some(c)) => jk.compute_device(gpu, &req(Some(c), st.nocc), Some(j), Some(k)),
            (Some(k), None) => {
                // K[D_s] = K[P_s P_s^T] - K[N_s N_s^T]
                let n2 = st.nbf * st.nbf;
                let g = factors(&mut self.guess, &mut self.ops, gpu, st)?;
                let mut kk = gpu.alloc_zeros::<f64>(2 * nsp * n2)?;
                jk.compute_device(gpu, &req(Some(&g.c), &g.ncols), Some(j), Some(&mut kk))?;
                let ops = ops(&mut self.ops, gpu)?;
                for s in 0..nsp {
                    ops.add(
                        gpu,
                        n2,
                        (&kk, 2 * s * n2),
                        (&kk, (2 * s + 1) * n2),
                        -1.0,
                        (&mut *k, s * n2),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Compensated grid-K's E_K (its K is affine in D); None for the DF engine's K.
    fn exchange_energy(&self) -> Option<f64> {
        self.gridk.as_ref().and_then(GridKTerm::energy)
    }

    fn exchange_correlation(
        &mut self,
        gpu: &Gpu,
        st: &FockState<'_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<Option<f64>, GpuError> {
        if self.xc.is_none() {
            return Ok(None);
        }
        let n2 = st.nbf * st.nbf;
        let nsp = st.nspin();
        // with VV10, the semilocal potential into a work buffer, then V_s = V_xc,s + V_nl
        let mut vx = match self.vv10 {
            Some(_) => Some(gpu.alloc_zeros::<f64>((nsp * n2).max(1))?),
            None => None,
        };
        let mut vnl = match self.vv10 {
            Some(_) => Some(gpu.alloc_zeros::<f64>(n2.max(1))?),
            None => None,
        };
        let (exc, e_nl) = match st.orbitals {
            Some(c) => {
                let out = vx.as_mut().unwrap_or(&mut *v);
                let e = self.xc.as_mut().unwrap().compute_device(
                    gpu,
                    c,
                    st.nocc,
                    None,
                    st.restricted,
                    out,
                )?;
                let enl = match (self.vv10.as_mut(), vnl.as_mut()) {
                    (Some(nl), Some(vn)) => {
                        Some(nl.compute_device(gpu, c, st.nocc, None, st.restricted, vn)?)
                    }
                    _ => None,
                };
                (e, enl)
            }
            None => {
                let g = factors(&mut self.guess, &mut self.ops, gpu, st)?;
                // one channel per spin: [P_s N_s] with occupations +1, -1
                let per: Vec<usize> = g.ncols.chunks(2).map(|p| p[0] + p[1]).collect();
                let out = vx.as_mut().unwrap_or(&mut *v);
                let e = self.xc.as_mut().unwrap().compute_device(
                    gpu,
                    &g.c,
                    &per,
                    Some(&g.occ),
                    st.restricted,
                    out,
                )?;
                let enl = match (self.vv10.as_mut(), vnl.as_mut()) {
                    (Some(nl), Some(vn)) => {
                        Some(nl.compute_device(gpu, &g.c, &per, Some(&g.occ), st.restricted, vn)?)
                    }
                    _ => None,
                };
                (e, enl)
            }
        };
        let Some(e_nl) = e_nl else {
            return Ok(Some(exc));
        };
        let ops = ops(&mut self.ops, gpu)?;
        let (vx, vnl) = (vx.unwrap(), vnl.unwrap());
        for s in 0..nsp {
            ops.add(gpu, n2, (&vx, s * n2), (&vnl, 0), 1.0, (&mut *v, s * n2))?;
        }
        Ok(Some(exc + e_nl))
    }
}
