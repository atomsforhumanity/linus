//! Orbital gradients of rotated unrestricted determinants on the device: the expensive part of
//! linus's stability analysis (`scf::stability`), whose Hessian-vector products are finite
//! differences of the orbital gradient g_s(k) = C_s(k)_v^T F_s[D(k)] C_s(k)_o of the orbitals
//! C_s(k) = C_s exp(K_s) (kappa_s: virtual x occupied).
//!
//! The rotation is linus's `soscf::rotate` in closed form. With the occupied-occupied functions
//! of kappa^T kappa from the host ([`Rotation`]: cos, sinc and (cos - 1)/x^2 of its square root,
//! no x no each),
//!   C_o' = C_o cos + (C_v K) sinc,
//!   C_v' = C_v + ((C_v K) cosm1 - C_o sinc) K^T,
//! D_s' = C_o' C_o'^T, the Fock terms run at D' through [`FockTerms`] (the device loop's own
//! engines), F_s' = h + J - a_x K_s + V_s, and g_s = C_v'^T F_s' C_o' comes back to the host
//! (nv x no per spin, column-major: the host's flattening). The Davidson iterations and every
//! decision stay in linus, on the host.
//!
//! A closed-shell reference (one channel of orbitals, linus's closed-shell probe, decision D36)
//! is the unrestricted determinant with alpha = beta: its rotations are UHF/UKS ones (UKS-type
//! Fock builds of two spin densities), its Fock matrix one channel for both spins, and
//! [`Rotations::rebuild`] puts back the restricted state of the SCF.

use super::kernels::Ops;
use super::{next_generation, FockState, FockTerms};
use crate::blas::{GemmDims, MatMut, MatRef, Op};
use crate::error::GpuError;
use crate::linalg::{Arith, Gemm};
use crate::{DeviceBuf, Gpu};

/// One spin's rotation: kappa (nv x no) and the functions of kappa^T kappa (no x no each),
/// column-major.
pub struct Rotation<'a> {
    pub kappa: &'a [f64],
    pub cos: &'a [f64],
    pub sinc: &'a [f64],
    pub cosm1: &'a [f64],
}

/// The orbital gradients of rotations of a reference UHF/UKS determinant (see the module docs).
pub struct Rotations {
    n: usize,
    nmo: usize,
    nocc: [usize; 2],
    /// A closed-shell reference: one channel of orbitals and of the Fock matrix, alpha = beta.
    closed: bool,
    a_x: f64,
    h: DeviceBuf<f64>,
    /// The reference orbitals, alpha then beta, nbf x nmo each (occupied first).
    c: DeviceBuf<f64>,
    /// The rotated virtual orbitals, alpha then beta, nbf x nmo each (the first nv columns).
    cv: DeviceBuf<f64>,
    /// The rotated occupied orbitals side by side (the Fock terms' layout).
    cocc: DeviceBuf<f64>,
    d: DeviceBuf<f64>,
    j: DeviceBuf<f64>,
    jt: DeviceBuf<f64>,
    k: Option<DeviceBuf<f64>>,
    vxc: DeviceBuf<f64>,
    venv: DeviceBuf<f64>,
    f: DeviceBuf<f64>,
    /// nbf x nmo work (C_v K, then F C_o).
    t1: DeviceBuf<f64>,
    /// nbf x nmo work.
    t2: DeviceBuf<f64>,
    /// kappa and its functions, uploaded per product.
    kap: DeviceBuf<f64>,
    fun: DeviceBuf<f64>,
    /// The gradient, both spins (sum nv no).
    g: DeviceBuf<f64>,
    ops: Ops,
    gemm: Gemm,
}

impl Rotations {
    /// Rotations of the orbitals `orbitals` (alpha and beta, nbf x nmo each, occupied first;
    /// `nocc` occupied per spin; one channel: a closed-shell reference, alpha = beta, with equal
    /// `nocc`) with the core Hamiltonian `h` (column-major nbf x nbf) and the exchange fraction
    /// a_x of the loop ([`super::ScfInput::exchange_fraction`]).
    pub fn new(
        gpu: &Gpu,
        h: &[f64],
        orbitals: &[DeviceBuf<f64>],
        nmo: usize,
        nocc: [usize; 2],
        exchange_fraction: f64,
    ) -> Result<Rotations, GpuError> {
        let n2 = h.len();
        let n = (n2 as f64).sqrt().round() as usize;
        let closed = orbitals.len() == 1;
        if n * n != n2
            || !(orbitals.len() == 2 || (closed && nocc[0] == nocc[1]))
            || nocc.iter().any(|&no| no > nmo)
            || nmo > n
        {
            return Err(GpuError::Contract(
                "Rotations: nbf x nbf h, two channels of nbf x nmo orbitals (or one with equal \
                 nocc), nocc <= nmo"
                    .into(),
            ));
        }
        let z = |len: usize| gpu.alloc_zeros::<f64>(len.max(1));
        let mut c = z(2 * n * nmo)?;
        for s in 0..2 {
            let o = &orbitals[if closed { 0 } else { s }];
            if o.len() < n * nmo {
                return Err(GpuError::Contract(
                    "Rotations: orbitals of nbf x nmo".into(),
                ));
            }
            gpu.copy_range(o, 0, &mut c, s * n * nmo, n * nmo)?;
        }
        let mo = nocc[0].max(nocc[1]);
        let len: usize = nocc.iter().map(|&no| (nmo - no) * no).sum();
        Ok(Rotations {
            n,
            nmo,
            nocc,
            closed,
            a_x: exchange_fraction,
            h: gpu.upload(h)?,
            c,
            cv: z(2 * n * nmo)?,
            cocc: z(n * (nocc[0] + nocc[1]))?,
            d: z(2 * n2)?,
            j: z(2 * n2)?,
            jt: z(n2)?,
            k: if exchange_fraction != 0.0 {
                Some(z(2 * n2)?)
            } else {
                None
            },
            vxc: z(2 * n2)?,
            venv: z(n2)?,
            f: z(2 * n2)?,
            t1: z(n * nmo)?,
            t2: z(n * nmo)?,
            kap: z(nmo * mo)?,
            fun: z(3 * mo * mo)?,
            g: z(len)?,
            ops: Ops::new(gpu, 1, 1)?,
            gemm: Gemm::new(gpu)?,
        })
    }

    /// Reference orbital `p` of spin `spin` (nbf coefficients), on the host.
    pub fn orbital(&self, gpu: &Gpu, spin: usize, p: usize) -> Result<Vec<f64>, GpuError> {
        if spin > 1 || p >= self.nmo {
            return Err(GpuError::Contract(format!(
                "Rotations: no orbital {p} of spin {spin}"
            )));
        }
        gpu.download_range(&self.c, (spin * self.nmo + p) * self.n, self.n)
    }

    /// Length of a rotation vector: sum over spins of nv no.
    pub fn len(&self) -> usize {
        self.nocc.iter().map(|&no| (self.nmo - no) * no).sum()
    }

    /// Whether there is nothing to rotate (no occupied-virtual pairs).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device bytes held.
    pub fn bytes(&self) -> usize {
        8 * (self.h.len()
            + self.c.len()
            + self.cv.len()
            + self.cocc.len()
            + self.d.len()
            + self.j.len()
            + self.jt.len()
            + self.k.as_ref().map_or(0, |b| b.len())
            + self.vxc.len()
            + self.venv.len()
            + self.f.len()
            + self.t1.len()
            + self.t2.len()
            + self.kap.len()
            + self.fun.len()
            + self.g.len())
            + self.ops.bytes()
    }

    /// C = alpha op(A) op(B) + beta C on (buffer, offset, leading dimension) operands; nothing
    /// for an empty product (C is then scaled by beta, i.e. zeroed when beta is 0).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        g: &Gemm,
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: (&DeviceBuf<f64>, usize, usize),
        b: (&DeviceBuf<f64>, usize, usize),
        beta: f64,
        c: (&mut DeviceBuf<f64>, usize, usize),
    ) -> Result<(), GpuError> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        g.gemm(
            Arith::Fp64,
            GemmDims::new(ta, tb, m, n, k),
            (alpha, beta),
            MatRef::new(a.0, a.1, a.2),
            MatRef::new(b.0, b.1, b.2),
            MatMut::new(c.0, c.1, c.2),
        )
    }

    /// The orbital gradient C_v'^T F C_o' per spin of the rotated orbitals (`cv`, `cocc`) and
    /// the Fock matrices `f`, downloaded.
    fn gradient_of_rotated(&mut self, gpu: &Gpu, rotated: bool) -> Result<Vec<f64>, GpuError> {
        let (n, nmo) = (self.n, self.nmo);
        let n2 = n * n;
        let mut goff = 0;
        let mut coff = 0;
        for s in 0..2 {
            let no = self.nocc[s];
            let nv = nmo - no;
            // the occupied and virtual orbitals: rotated, or the reference ones
            let (o, ooff) = if rotated {
                (&self.cocc, n * coff)
            } else {
                (&self.c, s * n * nmo)
            };
            Self::gemm(
                &self.gemm,
                Op::N,
                Op::N,
                n,
                no,
                n,
                1.0,
                (&self.f, s * n2, n),
                (o, ooff, n),
                0.0,
                (&mut self.t1, 0, n),
            )?;
            let (v, voff) = if rotated {
                (&self.cv, s * n * nmo)
            } else {
                (&self.c, s * n * nmo + n * no)
            };
            Self::gemm(
                &self.gemm,
                Op::T,
                Op::N,
                nv,
                no,
                n,
                1.0,
                (v, voff, n),
                (&self.t1, 0, n),
                0.0,
                (&mut self.g, goff, nv.max(1)),
            )?;
            goff += nv * no;
            coff += no;
        }
        if goff == 0 {
            return Ok(Vec::new());
        }
        Ok(gpu.download(&self.g)?[..goff].to_vec())
    }

    /// The orbital gradient of the reference orbitals for the Fock matrices `fock` (alpha then
    /// beta, nbf x nbf each; a closed-shell reference's one channel for both): the k = 0
    /// gradient of one-sided products.
    pub fn gradient_at_reference(
        &mut self,
        gpu: &Gpu,
        fock: &DeviceBuf<f64>,
    ) -> Result<Vec<f64>, GpuError> {
        let n2 = self.n * self.n;
        if self.closed {
            gpu.copy_range(fock, 0, &mut self.f, 0, n2)?;
            gpu.copy_range(fock, 0, &mut self.f, n2, n2)?;
        } else {
            gpu.copy_range(fock, 0, &mut self.f, 0, 2 * n2)?;
        }
        self.gradient_of_rotated(gpu, false)
    }

    /// The rotation kappa (flattened as the gradients) from the reference orbitals towards the
    /// occupied orbitals `occ` (side by side) to first order, linus's `rotation_towards`:
    /// kappa_s = P_v P_o^T with P = C_s^T S O_s (`s`: the overlap, nbf x nbf).
    pub fn towards(
        &mut self,
        gpu: &Gpu,
        s: &DeviceBuf<f64>,
        occ: &DeviceBuf<f64>,
    ) -> Result<Vec<f64>, GpuError> {
        let (n, nmo) = (self.n, self.nmo);
        let mut goff = 0;
        let mut coff = 0;
        for sp in 0..2 {
            let no = self.nocc[sp];
            let nv = nmo - no;
            Self::gemm(
                &self.gemm,
                Op::N,
                Op::N,
                n,
                no,
                n,
                1.0,
                (s, 0, n),
                (occ, n * coff, n),
                0.0,
                (&mut self.t1, 0, n),
            )?;
            Self::gemm(
                &self.gemm,
                Op::T,
                Op::N,
                nmo,
                no,
                n,
                1.0,
                (&self.c, sp * n * nmo, n),
                (&self.t1, 0, n),
                0.0,
                (&mut self.t2, 0, nmo),
            )?;
            Self::gemm(
                &self.gemm,
                Op::N,
                Op::T,
                nv,
                no,
                no,
                1.0,
                (&self.t2, no, nmo),
                (&self.t2, 0, nmo),
                0.0,
                (&mut self.g, goff, nv.max(1)),
            )?;
            goff += nv * no;
            coff += no;
        }
        if goff == 0 {
            return Ok(Vec::new());
        }
        Ok(gpu.download(&self.g)?[..goff].to_vec())
    }

    /// The orbital gradient of the reference orbitals rotated by `rot` (alpha and beta): one
    /// Fock build through `terms` at the rotated density.
    pub fn gradient(
        &mut self,
        gpu: &Gpu,
        terms: &mut dyn FockTerms,
        rot: &[Rotation<'_>; 2],
    ) -> Result<Vec<f64>, GpuError> {
        let (n, nmo) = (self.n, self.nmo);
        let n2 = n * n;
        let mut coff = 0;
        for (s, r) in rot.iter().enumerate() {
            let no = self.nocc[s];
            let nv = nmo - no;
            if r.kappa.len() != nv * no
                || [r.cos, r.sinc, r.cosm1].iter().any(|m| m.len() != no * no)
            {
                return Err(GpuError::Contract(format!(
                    "Rotations: kappa of {nv} x {no} and functions of {no} x {no}"
                )));
            }
            let cbase = s * n * nmo;
            if no > 0 {
                if nv > 0 {
                    gpu.upload_into(&mut self.kap, 0, r.kappa)?;
                }
                gpu.upload_into(&mut self.fun, 0, r.cos)?;
                gpu.upload_into(&mut self.fun, no * no, r.sinc)?;
                gpu.upload_into(&mut self.fun, 2 * no * no, r.cosm1)?;
                // t2 = C_v K (nbf x no; zero without virtual orbitals)
                Self::gemm(
                    &self.gemm,
                    Op::N,
                    Op::N,
                    n,
                    no,
                    nv,
                    1.0,
                    (&self.c, cbase + n * no, n),
                    (&self.kap, 0, nv.max(1)),
                    0.0,
                    (&mut self.t2, 0, n),
                )?;
                // C_o' = C_o cos + (C_v K) sinc, into the side-by-side occupied orbitals
                Self::gemm(
                    &self.gemm,
                    Op::N,
                    Op::N,
                    n,
                    no,
                    no,
                    1.0,
                    (&self.c, cbase, n),
                    (&self.fun, 0, no),
                    0.0,
                    (&mut self.cocc, n * coff, n),
                )?;
                Self::gemm(
                    &self.gemm,
                    Op::N,
                    Op::N,
                    n,
                    no,
                    no,
                    1.0,
                    (&self.t2, 0, n),
                    (&self.fun, no * no, no),
                    1.0,
                    (&mut self.cocc, n * coff, n),
                )?;
            }
            if nv > 0 {
                // C_v' = C_v + ((C_v K) cosm1 - C_o sinc) K^T
                gpu.copy_range(&self.c, cbase + n * no, &mut self.cv, cbase, n * nv)?;
                if no > 0 {
                    Self::gemm(
                        &self.gemm,
                        Op::N,
                        Op::N,
                        n,
                        no,
                        no,
                        1.0,
                        (&self.t2, 0, n),
                        (&self.fun, 2 * no * no, no),
                        0.0,
                        (&mut self.t1, 0, n),
                    )?;
                    Self::gemm(
                        &self.gemm,
                        Op::N,
                        Op::N,
                        n,
                        no,
                        no,
                        -1.0,
                        (&self.c, cbase, n),
                        (&self.fun, no * no, no),
                        1.0,
                        (&mut self.t1, 0, n),
                    )?;
                    Self::gemm(
                        &self.gemm,
                        Op::N,
                        Op::T,
                        n,
                        nv,
                        no,
                        1.0,
                        (&self.t1, 0, n),
                        (&self.kap, 0, nv),
                        1.0,
                        (&mut self.cv, cbase, n),
                    )?;
                }
            }
            // D_s' = C_o' C_o'^T
            self.gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::N, Op::T, n, n, no),
                (1.0, 0.0),
                MatRef::new(&self.cocc, n * coff, n),
                MatRef::new(&self.cocc, n * coff, n),
                MatMut::new(&mut self.d, s * n2, n),
            )?;
            coff += no;
        }
        self.build_and_gradient(gpu, terms)
    }

    /// The Fock terms at the rotated state (`d`, `cocc`), F_s' = ((h + J) - a_x K_s) +
    /// (V_s + V_env), and the orbital gradient of the rotated orbitals.
    fn build_and_gradient(
        &mut self,
        gpu: &Gpu,
        terms: &mut dyn FockTerms,
    ) -> Result<Vec<f64>, GpuError> {
        let n = self.n;
        let n2 = n * n;
        let nocc = self.nocc;
        let st = FockState {
            nbf: n,
            restricted: false,
            densities: &self.d,
            orbitals: Some(&self.cocc),
            nocc: &nocc,
            generation: next_generation(),
        };
        terms.coulomb_exchange(gpu, &st, &mut self.j, self.k.as_mut())?;
        let exc = terms.exchange_correlation(gpu, &st, &mut self.vxc)?;
        let env = terms.environment(gpu, &st, &mut self.venv)?;
        if let Some(k) = self.k.as_mut() {
            self.ops.scale(gpu, 2 * n2, k, self.a_x)?;
        }
        self.ops
            .add(gpu, n2, (&self.j, 0), (&self.j, n2), 1.0, (&mut self.jt, 0))?;
        self.ops.fock(
            gpu,
            2,
            n2,
            &self.h,
            &self.jt,
            self.k.as_ref(),
            exc.is_some().then_some(&self.vxc),
            matches!(env, Some((_, true))).then_some(&self.venv),
            &mut self.f,
        )?;
        self.gradient_of_rotated(gpu, true)
    }

    /// Runs the Fock terms at the state (`densities`, alpha then beta; `occupied`, their
    /// occupied orbitals side by side; a closed-shell reference: the restricted state, one
    /// channel of each), discarding the matrices: puts back what the terms keep of the last
    /// state they saw (a solvent's surface charges, VV10's last energy) after rotated builds.
    pub fn rebuild(
        &mut self,
        gpu: &Gpu,
        terms: &mut dyn FockTerms,
        densities: &DeviceBuf<f64>,
        occupied: &DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        let nocc = self.nocc;
        let st = FockState {
            nbf: self.n,
            restricted: self.closed,
            densities,
            orbitals: Some(occupied),
            nocc: &nocc[..if self.closed { 1 } else { 2 }],
            generation: next_generation(),
        };
        terms.coulomb_exchange(gpu, &st, &mut self.j, self.k.as_mut())?;
        terms.exchange_correlation(gpu, &st, &mut self.vxc)?;
        terms.environment(gpu, &st, &mut self.venv)?;
        gpu.synchronize()
    }
}
