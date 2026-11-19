//! The CPU fallback of the Fock terms: [`OnHost`] downloads the densities (and occupied
//! orbitals) once per SCF state, runs host implementations ([`HostFockTerms`]) on them and
//! uploads their matrices. Same layout as on the device: column-major, channels back to back.

use super::{FockState, FockTerms};
use crate::error::GpuError;
use crate::{DeviceBuf, Gpu};
use std::time::Instant;

/// Host copies of the SCF state (see [`FockState`]).
pub struct HostState<'a> {
    pub nbf: usize,
    pub restricted: bool,
    /// Spin densities back to back (one for a closed shell).
    pub densities: &'a [f64],
    /// Occupied orbitals side by side, nbf x sum(nocc); None for a guess density.
    pub orbitals: Option<&'a [f64]>,
    pub nocc: &'a [usize],
}

impl HostState<'_> {
    /// Channel `s`'s density (nbf x nbf).
    pub fn density(&self, s: usize) -> &[f64] {
        let n2 = self.nbf * self.nbf;
        &self.densities[s * n2..(s + 1) * n2]
    }

    /// Channel `s`'s occupied orbitals (nbf x nocc\[s\]).
    pub fn occupied(&self, s: usize) -> Option<&[f64]> {
        let n = self.nbf;
        let c0: usize = self.nocc[..s].iter().sum();
        self.orbitals.map(|c| &c[n * c0..n * (c0 + self.nocc[s])])
    }
}

/// Fock terms computed on the host, the counterpart of [`FockTerms`] on host slices (outputs
/// are zeroed before each call).
pub trait HostFockTerms {
    /// J\[D_s\] per channel into `j`, and K\[D_s\] into `k` when given.
    fn coulomb_exchange(
        &mut self,
        st: &HostState<'_>,
        j: &mut [f64],
        k: Option<&mut [f64]>,
    ) -> Result<(), GpuError>;

    /// E_xc and V_xc per channel into `v`; None without an XC term.
    fn exchange_correlation(
        &mut self,
        _st: &HostState<'_>,
        _v: &mut [f64],
    ) -> Result<Option<f64>, GpuError> {
        Ok(None)
    }

    /// E_env and whether `v` (one nbf x nbf matrix) holds the term's Fock matrix; None without
    /// an environment term.
    fn environment(
        &mut self,
        _st: &HostState<'_>,
        _v: &mut [f64],
    ) -> Result<Option<(f64, bool)>, GpuError> {
        Ok(None)
    }
}

/// [`FockTerms`] from [`HostFockTerms`]: the CPU fallback (and the validation path of the loop
/// against the CPU SCF).
pub struct OnHost<T> {
    pub terms: T,
    generation: Option<u64>,
    dens: Vec<f64>,
    orbs: Option<Vec<f64>>,
    out: Vec<f64>,
    out2: Vec<f64>,
    /// Seconds spent in transfers (downloads of the state, uploads of the matrices).
    pub transfer_seconds: f64,
}

impl<T> OnHost<T> {
    pub fn new(terms: T) -> Self {
        OnHost {
            terms,
            generation: None,
            dens: Vec::new(),
            orbs: None,
            out: Vec::new(),
            out2: Vec::new(),
            transfer_seconds: 0.0,
        }
    }

    /// Downloads the state unless this generation is already on the host.
    fn fetch(&mut self, gpu: &Gpu, st: &FockState<'_>) -> Result<(), GpuError> {
        if self.generation == Some(st.generation) {
            return Ok(());
        }
        let t = Instant::now();
        let n = st.nbf;
        let nd = st.nspin() * n * n;
        let mut d = gpu.download(st.densities)?;
        d.truncate(nd);
        self.dens = d;
        self.orbs = match st.orbitals {
            Some(c) => {
                let mut v = gpu.download(c)?;
                v.truncate(n * st.nocc.iter().sum::<usize>());
                Some(v)
            }
            None => None,
        };
        self.generation = Some(st.generation);
        self.transfer_seconds += t.elapsed().as_secs_f64();
        Ok(())
    }

    fn upload(&mut self, gpu: &Gpu, dst: &mut DeviceBuf<f64>, src: &[f64]) -> Result<(), GpuError> {
        let t = Instant::now();
        gpu.upload_into(dst, 0, src)?;
        self.transfer_seconds += t.elapsed().as_secs_f64();
        Ok(())
    }
}

fn zeroed(v: &mut Vec<f64>, len: usize) {
    v.clear();
    v.resize(len, 0.0);
}

impl<T: HostFockTerms> FockTerms for OnHost<T> {
    fn coulomb_exchange(
        &mut self,
        gpu: &Gpu,
        st: &FockState<'_>,
        j: &mut DeviceBuf<f64>,
        k: Option<&mut DeviceBuf<f64>>,
    ) -> Result<(), GpuError> {
        self.fetch(gpu, st)?;
        let len = st.nspin() * st.nbf * st.nbf;
        zeroed(&mut self.out, len);
        if k.is_some() {
            zeroed(&mut self.out2, len);
        }
        let hs = HostState {
            nbf: st.nbf,
            restricted: st.restricted,
            densities: &self.dens,
            orbitals: self.orbs.as_deref(),
            nocc: st.nocc,
        };
        let kh = k.is_some().then_some(&mut self.out2[..]);
        self.terms.coulomb_exchange(&hs, &mut self.out, kh)?;
        let (jh, kh) = (
            std::mem::take(&mut self.out),
            std::mem::take(&mut self.out2),
        );
        self.upload(gpu, j, &jh)?;
        if let Some(k) = k {
            self.upload(gpu, k, &kh)?;
        }
        (self.out, self.out2) = (jh, kh);
        Ok(())
    }

    fn exchange_correlation(
        &mut self,
        gpu: &Gpu,
        st: &FockState<'_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<Option<f64>, GpuError> {
        self.fetch(gpu, st)?;
        zeroed(&mut self.out, st.nspin() * st.nbf * st.nbf);
        let hs = HostState {
            nbf: st.nbf,
            restricted: st.restricted,
            densities: &self.dens,
            orbitals: self.orbs.as_deref(),
            nocc: st.nocc,
        };
        let r = self.terms.exchange_correlation(&hs, &mut self.out)?;
        if r.is_some() {
            let vh = std::mem::take(&mut self.out);
            self.upload(gpu, v, &vh)?;
            self.out = vh;
        }
        Ok(r)
    }

    fn environment(
        &mut self,
        gpu: &Gpu,
        st: &FockState<'_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<Option<(f64, bool)>, GpuError> {
        self.fetch(gpu, st)?;
        zeroed(&mut self.out, st.nbf * st.nbf);
        let hs = HostState {
            nbf: st.nbf,
            restricted: st.restricted,
            densities: &self.dens,
            orbitals: self.orbs.as_deref(),
            nocc: st.nocc,
        };
        let r = self.terms.environment(&hs, &mut self.out)?;
        if let Some((_, true)) = r {
            let vh = std::mem::take(&mut self.out);
            self.upload(gpu, v, &vh)?;
            self.out = vh;
        }
        Ok(r)
    }
}
