//! The pointwise step of the XC engines: the functional evaluated on the host (libxc through
//! linus-xc), or, in its place, a caller's function on the device (VV10's pair sums).
//!
//! [`HostFunctional`] is the one evaluator of the XC family: the energy and potential
//! (`GpuXc`), the gradient (`crate::grad::xc`), the Hessian's XC terms (`crate::hess::xc`) and
//! the CPKS kernel (`crate::response::xc`) all evaluate through it. The grid's points go to
//! libxc in chunks of [`CHUNK`] points on the current rayon pool, and the chunks' results are
//! placed in chunk order, so the outputs do not depend on the number of threads.
//! - Order 1 ([`HostFunctional::first`]) gives the energy density and the first derivatives
//!   (zk, vrho, vsigma, vtau in libxc's layout), concatenated over the chunks;
//!   `first_device` (feature `gpu`) does the same on the device's grid arrays, with the
//!   transfers.
//! - Order 2 ([`HostFunctional::second`]) hands each chunk's libxc output, with the chunk's
//!   density gradients in the kernel weights' layout, to the caller, which forms what it keeps
//!   (the Hessian's potentials and weights, the CPKS kernel's weights).
//!
//! The pointwise path (`pointwise`, feature `gpu`) chooses between this evaluator and a
//! caller's device function ([`PointFn`]) that sees the grid's densities ([`PointInputs`]) and
//! returns the same derivatives ([`PointOutputs`]).

use crate::cpu::{Functional, XcInput, XcOutput};
use crate::error::GpuError;
use crate::xc::XcKind;
use rayon::prelude::*;

/// Points per libxc call. The chunking fixes nothing numerically (libxc evaluates each point on
/// its own), but every engine uses the same one, so a change of it is one change.
pub(crate) const CHUNK: usize = 4096;

/// The functional of an XC engine on the host: its unpolarized and polarized libxc objects and
/// the semilocal ingredients the engine evaluates it with.
pub(crate) struct HostFunctional {
    unpolarized: Functional,
    polarized: Functional,
    kind: XcKind,
    /// Where order 1 is evaluated when no device function replaces it.
    #[cfg(feature = "gpu")]
    place: FunctionalPlace,
    /// The functional compiled for the device, per spin treatment (unpolarized, polarized), on
    /// first use; Err: why libxc on the host evaluates it.
    #[cfg(feature = "gpu")]
    device: [std::sync::OnceLock<Result<crate::xc::devfunc::DevFunc, String>>; 2],
}

/// Where the XC engines evaluate the functional's first derivatives (`crate::xc::devfunc`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FunctionalPlace {
    /// libxc's code compiled for the device where linus-xc has it and it agrees with libxc at
    /// the setup's check, else libxc on the host (the default).
    #[default]
    Device,
    /// libxc on the host.
    Host,
}

/// The order-1 derivatives of every point in libxc's layout: `zk` (one per point), `vrho` (one
/// per spin), `vsigma` (one or three, GGA and meta-GGA only, else empty) and `vtau` (one per
/// spin, meta-GGA only, else empty).
pub(crate) struct FirstDerivs {
    pub(crate) zk: Vec<f64>,
    pub(crate) vrho: Vec<f64>,
    pub(crate) vsigma: Vec<f64>,
    pub(crate) vtau: Vec<f64>,
}

impl HostFunctional {
    /// The functional `name` (a linus-xc name), evaluated with the ingredients `kind` (None:
    /// the functional's own; VV10's engine on the NLC grid is a GGA of the total density
    /// whatever the functional's family).
    pub(crate) fn new(name: &str, kind: Option<XcKind>) -> Result<HostFunctional, GpuError> {
        let fe = |e: linus_xc::XcError| GpuError::Input(format!("functional {name}: {e}"));
        let unpolarized = Functional::new(name, false).map_err(fe)?;
        let polarized = Functional::new(name, true).map_err(fe)?;
        let kind = kind.unwrap_or_else(|| XcKind::of(&unpolarized));
        Ok(HostFunctional {
            unpolarized,
            polarized,
            kind,
            #[cfg(feature = "gpu")]
            place: FunctionalPlace::default(),
            #[cfg(feature = "gpu")]
            device: Default::default(),
        })
    }

    /// Where order 1 is evaluated (the default: on the device where it can be).
    #[cfg(feature = "gpu")]
    pub(crate) fn with_place(mut self, place: FunctionalPlace) -> Self {
        self.place = place;
        self
    }

    /// The plan of a device evaluation for `nsp` spin channels (linus-xc's), or why there is
    /// none.
    #[cfg(feature = "gpu")]
    pub(crate) fn device_plan(&self, nsp: usize) -> Result<crate::cpu::DevicePlan, String> {
        self.libxc(nsp).device_plan()
    }

    /// The ingredients it is evaluated with.
    pub(crate) fn kind(&self) -> XcKind {
        self.kind
    }

    /// The functional needs the density's Laplacian (no device engine covers that).
    pub(crate) fn needs_lapl(&self) -> bool {
        self.unpolarized.needs_lapl()
    }

    /// The functional's VV10 parameters, if it has a nonlocal term.
    pub(crate) fn vv10(&self) -> Option<linus_xc::Vv10> {
        self.unpolarized.vv10()
    }

    /// The libxc object for `nsp` spin channels (1: unpolarized, 2: polarized).
    fn libxc(&self, nsp: usize) -> &Functional {
        if nsp == 1 {
            &self.unpolarized
        } else {
            &self.polarized
        }
    }

    /// libxc's input of one chunk for the engine's ingredients.
    fn input<'a>(&self, rho: &'a [f64], sigma: &'a [f64], tau: &'a [f64]) -> XcInput<'a> {
        if self.kind.mgga {
            XcInput::mgga(rho, sigma, tau)
        } else if self.kind.gga {
            XcInput::gga(rho, sigma)
        } else {
            XcInput::lda(rho)
        }
    }

    /// Order 1 at `npts` points with `nsp` spin channels: `rho` (nsp per point), `sigma` (1 or
    /// 3 per point; GGA and meta-GGA) and `tau` (nsp per point; meta-GGA) in libxc's layout.
    pub(crate) fn first(
        &self,
        nsp: usize,
        npts: usize,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
    ) -> Result<FirstDerivs, GpuError> {
        let (gga, mgga) = (self.kind.gga, self.kind.mgga);
        let nsig = if !gga {
            0
        } else if nsp == 1 {
            1
        } else {
            3
        };
        let func = self.libxc(nsp);
        let parts: Vec<Result<FirstDerivs, GpuError>> = (0..npts.div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| {
                let (a, e) = (c * CHUNK, ((c + 1) * CHUNK).min(npts));
                let r = &rho[a * nsp..e * nsp];
                let sg = if gga {
                    &sigma[a * nsig..e * nsig]
                } else {
                    &[][..]
                };
                let t = if mgga {
                    &tau[a * nsp..e * nsp]
                } else {
                    &[][..]
                };
                let mut out = XcOutput::new();
                func.eval(1, &self.input(r, sg, t), &mut out)
                    .map_err(|e| GpuError::Contract(format!("libxc: {e}")))?;
                Ok(FirstDerivs {
                    zk: out.zk().to_vec(),
                    vrho: out.vrho().to_vec(),
                    vsigma: out.vsigma().to_vec(),
                    vtau: out.vtau().to_vec(),
                })
            })
            .collect();
        let mut d = FirstDerivs {
            zk: Vec::with_capacity(npts),
            vrho: Vec::with_capacity(npts * nsp),
            vsigma: Vec::with_capacity(npts * nsig),
            vtau: Vec::with_capacity(if mgga { npts * nsp } else { 0 }),
        };
        for p in parts {
            let p = p?;
            d.zk.extend_from_slice(&p.zk);
            d.vrho.extend_from_slice(&p.vrho);
            if gga {
                d.vsigma.extend_from_slice(&p.vsigma);
            }
            if mgga {
                d.vtau.extend_from_slice(&p.vtau);
            }
        }
        if d.zk.len() != npts || d.vrho.len() != npts * nsp {
            return Err(GpuError::Contract(
                "libxc returned arrays of the wrong size".into(),
            ));
        }
        Ok(d)
    }

    /// Order 2 at `npts` points with `nsp` spin channels: `rho` (nsp per point), `grad` (the
    /// density gradients, nsp x 3 per point: (s 3 + k)) and `tau` (nsp per point; meta-GGA).
    /// Per chunk of points p0..p1, libxc's output and the chunk's gradients in the kernel
    /// weights' layout (gr\[(s 3 + k) np + g\], np = p1 - p0; empty for LDA) go to `chunk`,
    /// whose results come back in chunk order.
    pub(crate) fn second<T, F>(
        &self,
        nsp: usize,
        npts: usize,
        rho: &[f64],
        grad: &[f64],
        tau: &[f64],
        chunk: F,
    ) -> Result<Vec<T>, GpuError>
    where
        T: Send,
        F: Fn(std::ops::Range<usize>, &XcOutput, &[f64]) -> T + Sync,
    {
        let (gga, mgga) = (self.kind.gga, self.kind.mgga);
        let func = self.libxc(nsp);
        (0..npts.div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| {
                let (p0, p1) = (c * CHUNK, ((c + 1) * CHUNK).min(npts));
                let np = p1 - p0;
                let r = &rho[p0 * nsp..p1 * nsp];
                // grad rho in the weights' layout, and sigma per point
                let mut gr = vec![0.0; if gga { nsp * 3 * np } else { 0 }];
                let mut sg = Vec::new();
                if gga {
                    for g in 0..np {
                        let a = &grad[(p0 + g) * nsp * 3..(p0 + g + 1) * nsp * 3];
                        for s in 0..nsp {
                            for k in 0..3 {
                                gr[(s * 3 + k) * np + g] = a[s * 3 + k];
                            }
                        }
                        let dot = |x: usize, y: usize| {
                            a[x * 3] * a[y * 3]
                                + a[x * 3 + 1] * a[y * 3 + 1]
                                + a[x * 3 + 2] * a[y * 3 + 2]
                        };
                        if nsp == 1 {
                            sg.push(dot(0, 0));
                        } else {
                            sg.extend([dot(0, 0), dot(0, 1), dot(1, 1)]);
                        }
                    }
                }
                let t = if mgga {
                    &tau[p0 * nsp..p1 * nsp]
                } else {
                    &[][..]
                };
                let mut out = XcOutput::new();
                func.eval(2, &self.input(r, &sg, t), &mut out)
                    .map_err(|e| GpuError::Contract(format!("libxc: {e}")))?;
                Ok(chunk(p0..p1, &out, &gr))
            })
            .collect()
    }
}

#[cfg(feature = "gpu")]
pub(crate) use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::{DeviceBuf, Gpu};

    /// The grid's densities as a pointwise functional of an engine's caller sees them
    /// (crate-internal: VV10 replaces the host libxc step with its pair sums). Grid arrays in
    /// libxc's layout (`crate::xc`'s module docs); `x`, `y`, `z` and `w` the points and weights.
    pub(crate) struct PointInputs<'a> {
        pub(crate) npts: usize,
        pub(crate) restricted: bool,
        pub(crate) x: &'a DeviceBuf<f64>,
        pub(crate) y: &'a DeviceBuf<f64>,
        pub(crate) z: &'a DeviceBuf<f64>,
        pub(crate) w: &'a DeviceBuf<f64>,
        pub(crate) rho: &'a DeviceBuf<f64>,
        pub(crate) grad: &'a DeviceBuf<f64>,
    }

    /// What such a functional returns at every point: zk (E = sum w rho zk; in the gradient
    /// rho zk is dE/dw, the weights' derivative factor), vrho, vsigma and vtau in libxc's
    /// layout, and optionally the explicit derivative of the energy with respect to each
    /// point's position (3 npts; the gradient adds it to the point-motion term, which goes to
    /// the point's owner).
    pub(crate) struct PointOutputs {
        pub(crate) zk: DeviceBuf<f64>,
        pub(crate) vrho: DeviceBuf<f64>,
        pub(crate) vsigma: DeviceBuf<f64>,
        pub(crate) vtau: DeviceBuf<f64>,
        pub(crate) point_force: Option<DeviceBuf<f64>>,
    }

    /// A pointwise functional of the caller in place of libxc.
    pub(crate) type PointFn<'f> =
        dyn FnMut(&Gpu, &PointInputs<'_>) -> Result<PointOutputs, GpuError> + 'f;

    /// Seconds of a pointwise step: all of it (with a device function, from events; on the
    /// host, the wall clock from the transfers' start) and of it libxc alone (wall clock, all
    /// host threads; 0 with a device function).
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct PointTimes {
        pub(crate) functional: f64,
        pub(crate) libxc: f64,
    }

    impl HostFunctional {
        /// The functional compiled for the device for `nsp` spin channels (compiled and checked
        /// on first use), or None: libxc on the host evaluates it ([`HostFunctional::device_status`]
        /// says why).
        pub(crate) fn device(&self, gpu: &Gpu, nsp: usize) -> Option<&crate::xc::devfunc::DevFunc> {
            if self.place != FunctionalPlace::Device {
                return None;
            }
            self.device[nsp - 1]
                .get_or_init(|| crate::xc::devfunc::DevFunc::new(gpu, self, nsp))
                .as_ref()
                .ok()
        }

        /// Where order 1 was evaluated for `nsp` spin channels: Some(Ok) on the device, Some(Err)
        /// on the host with the reason, None not yet (or on the host by the engine's choice).
        pub(crate) fn device_status(&self, nsp: usize) -> Option<Result<(), String>> {
            self.device[nsp - 1]
                .get()
                .map(|r| r.as_ref().map(|_| ()).map_err(|e| e.clone()))
        }

        /// Order 1 on the device's grid arrays (`rho`, `sigma`, `tau` as for
        /// [`HostFunctional::first`]): synchronizes, downloads them, evaluates, and uploads zk,
        /// vrho, vsigma and vtau (a one-element placeholder for an ingredient the functional
        /// does not have).
        pub(crate) fn first_device(
            &self,
            gpu: &Gpu,
            npts: usize,
            restricted: bool,
            rho: &DeviceBuf<f64>,
            sigma: &DeviceBuf<f64>,
            tau: &DeviceBuf<f64>,
        ) -> Result<(PointOutputs, PointTimes), GpuError> {
            let nsp = if restricted { 1 } else { 2 };
            let (gga, mgga) = (self.kind.gga, self.kind.mgga);
            gpu.synchronize()?;
            let tf = std::time::Instant::now();
            let hrho = gpu.download(rho)?;
            let hsig = if gga { gpu.download(sigma)? } else { vec![] };
            let htau = if mgga { gpu.download(tau)? } else { vec![] };
            let tl = std::time::Instant::now();
            let d = self.first(nsp, npts, &hrho, &hsig, &htau)?;
            let libxc = tl.elapsed().as_secs_f64();
            let out = PointOutputs {
                zk: gpu.upload(&d.zk)?,
                vrho: gpu.upload(&d.vrho)?,
                vsigma: gpu.upload(if gga { &d.vsigma } else { &[0.0] })?,
                vtau: gpu.upload(if mgga { &d.vtau } else { &[0.0] })?,
                point_force: None,
            };
            let functional = tf.elapsed().as_secs_f64();
            Ok((out, PointTimes { functional, libxc }))
        }

        /// The pointwise step at the grid's densities `inp` (`sigma` and `tau` the libxc
        /// inputs beside them): libxc on the host ([`HostFunctional::first_device`]), or the
        /// caller's `func` on the device (timed by events; synchronized).
        pub(crate) fn pointwise(
            &self,
            gpu: &Gpu,
            inp: &PointInputs<'_>,
            sigma: &DeviceBuf<f64>,
            tau: &DeviceBuf<f64>,
            func: Option<&mut PointFn<'_>>,
        ) -> Result<(PointOutputs, PointTimes), GpuError> {
            let Some(f) = func else {
                let nsp = if inp.restricted { 1 } else { 2 };
                if let Some(d) = self.device(gpu, nsp) {
                    let m = gpu.mark()?;
                    let [zk, vrho, vsigma, vtau] = d.eval(gpu, inp.npts, inp.rho, sigma, tau)?;
                    let functional = m.seconds_until(&gpu.mark()?)?;
                    let out = PointOutputs {
                        zk,
                        vrho,
                        vsigma,
                        vtau,
                        point_force: None,
                    };
                    let t = PointTimes {
                        functional,
                        libxc: 0.0,
                    };
                    return Ok((out, t));
                }
                return self.first_device(gpu, inp.npts, inp.restricted, inp.rho, sigma, tau);
            };
            let m = gpu.mark()?;
            let out = f(gpu, inp)?;
            let functional = m.seconds_until(&gpu.mark()?)?;
            Ok((
                out,
                PointTimes {
                    functional,
                    libxc: 0.0,
                },
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points of a smooth spin-polarized density: rho, sigma (aa, ab, bb), tau and the
    /// gradients behind sigma.
    fn points(npts: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rho = Vec::new();
        let mut grad = Vec::new();
        let mut tau = Vec::new();
        for i in 0..npts {
            let x = 0.01 + i as f64 * 1e-4;
            rho.extend([0.3 * x, 0.2 * x]);
            let ga = [0.1 * x, -0.05 * x, 0.02 * x];
            let gb = [0.07 * x, 0.01 * x, -0.03 * x];
            grad.extend(ga.iter().chain(&gb));
            tau.extend([0.5 * x, 0.4 * x]);
        }
        let sigma = grad
            .chunks(6)
            .flat_map(|g| {
                let dot = |x: usize, y: usize| {
                    g[x * 3] * g[y * 3] + g[x * 3 + 1] * g[y * 3 + 1] + g[x * 3 + 2] * g[y * 3 + 2]
                };
                [dot(0, 0), dot(0, 1), dot(1, 1)]
            })
            .collect();
        (rho, sigma, tau, grad)
    }

    /// Order 1 over several chunks is libxc on all points at once, bit for bit; order 2's
    /// chunks see libxc's output of their points and come back in order.
    #[test]
    fn chunks_are_libxc_on_all_points() {
        let npts = 2 * CHUNK + 123;
        let (rho, sigma, tau, grad) = points(npts);
        let f = HostFunctional::new("r2scan", None).unwrap();
        assert!(f.kind().mgga);
        let d = f.first(2, npts, &rho, &sigma, &tau).unwrap();
        let mut all = XcOutput::new();
        f.polarized
            .eval(1, &XcInput::mgga(&rho, &sigma, &tau), &mut all)
            .unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&d.zk), bits(all.zk()));
        assert_eq!(bits(&d.vrho), bits(all.vrho()));
        assert_eq!(bits(&d.vsigma), bits(all.vsigma()));
        assert_eq!(bits(&d.vtau), bits(all.vtau()));
        let mut all2 = XcOutput::new();
        f.polarized
            .eval(2, &XcInput::mgga(&rho, &sigma, &tau), &mut all2)
            .unwrap();
        let parts = f
            .second(2, npts, &rho, &grad, &tau, |r, out, gr| {
                assert_eq!(gr.len(), 6 * r.len());
                (r, out.v2rho2().to_vec())
            })
            .unwrap();
        let mut next = 0;
        let n2 = all2.v2rho2().len() / npts;
        for (r, v) in parts {
            assert_eq!(r.start, next);
            next = r.end;
            assert_eq!(bits(&v), bits(&all2.v2rho2()[r.start * n2..r.end * n2]));
        }
        assert_eq!(next, npts);
    }
}
