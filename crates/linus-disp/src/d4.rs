// SPDX-License-Identifier: LGPL-3.0-or-later
//! DFT-D4 (Caldeweyher et al., JCP 150, 154122 (2019)) as dftd4 4.3.1 evaluates it for
//! molecules: the D4 model, damping and ATM term of axtb-d4 (bit-identical to dftd4 4.3.1
//! at given charges) with EEQ 2019 charges from [`crate::eeq`].
//!
//! `E = E2(R, CN(R), q(R)) + E3(R, CN(R))`, the three-body term at zero charges as in
//! dftd4. Gradient: fixed-charge gradient plus `V . dq/dR` with `V = dE2/dq` (adjoint EEQ
//! solve). Hessian: fixed-charge Hessian of axtb-d4 plus
//! `P Q + Q^T P^T + Q^T K Q + sum_i V_i d2q_i/dRdR` with `Q = dq/dR`, `P = d2E/dRdq`,
//! `K = d2E/dqdq`.

use crate::eeq::EeqModel;
use crate::par;
use crate::{DispError, DispResult, Order};
use axtb_d4::{ChargeModel, D4Dispersion, D4Model, RationalDamping, RealspaceCutoff};

/// D4 parameters: rational damping, ATM scaling and the model's charge scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D4Params {
    pub s6: f64,
    pub s8: f64,
    pub a1: f64,
    pub a2: f64,
    /// ATM three-body scaling (dftd4 default 1; r2SCAN-3c 2)
    pub s9: f64,
    /// exponent of the ATM zero damping (dftd4 default 16)
    pub alp: f64,
    /// charge scaling height (dftd4 default 3; r2SCAN-3c 2)
    pub ga: f64,
    /// charge scaling steepness (dftd4 default 2; r2SCAN-3c 1)
    pub gc: f64,
    /// CN interpolation weighting factor (dftd4 default 6)
    pub wf: f64,
}

impl D4Params {
    /// Parameters with dftd4's defaults for s9 (1), alp (16), ga (3), gc (2), wf (6).
    pub fn new(s6: f64, s8: f64, a1: f64, a2: f64) -> Self {
        Self {
            s6,
            s8,
            a1,
            a2,
            s9: 1.0,
            alp: 16.0,
            ga: 3.0,
            gc: 2.0,
            wf: 6.0,
        }
    }
}

/// Real-space cutoffs (bohr), hard as in dftd4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D4Cutoffs {
    pub cn: f64,
    pub disp2: f64,
    pub disp3: f64,
}

impl D4Cutoffs {
    /// dftd4 4.3.1 defaults: CN 30, two-body 60, three-body 40 (EEQ CN: 25, fixed).
    pub const DFTD4: Self = Self {
        cn: 30.0,
        disp2: 60.0,
        disp3: 40.0,
    };
}

impl Default for D4Cutoffs {
    fn default() -> Self {
        Self::DFTD4
    }
}

/// Options of [`d4`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct D4Options {
    pub order: Order,
    pub cutoffs: D4Cutoffs,
}

/// DFT-D4 energy, gradient and Hessian of a molecule with total charge `charge` (the
/// EEQ charges depend on it).
///
/// `z`: atomic numbers (1-103); `xyz_bohr`: positions in bohr. Runs on the current rayon
/// pool; the result is bitwise identical for any number of threads.
pub fn d4(
    z: &[u32],
    xyz_bohr: &[[f64; 3]],
    charge: f64,
    params: &D4Params,
    opts: &D4Options,
) -> Result<DispResult, DispError> {
    if z.len() != xyz_bohr.len() {
        return Err(DispError::Shape(format!(
            "{} atomic numbers but {} positions",
            z.len(),
            xyz_bohr.len()
        )));
    }
    let n = z.len();
    let eeq = EeqModel::new(z)?;
    let numbers: Vec<usize> = z.iter().map(|&v| v as usize).collect();
    let model = D4Model::with_scaling(&numbers, ChargeModel::Eeq, params.ga, params.gc, params.wf)
        .map_err(|e| {
            let axtb_d4::D4Error::UnsupportedElement { atom, number } = e;
            DispError::UnsupportedElement {
                model: "D4",
                atom,
                number: number as u32,
            }
        })?;
    if n == 0 {
        let hessian = (opts.order == Order::Hessian).then(Vec::new);
        return Ok(DispResult {
            energy: 0.0,
            gradient: Vec::new(),
            hessian,
        });
    }
    let damping = RationalDamping {
        s6: params.s6,
        s8: params.s8,
        s9: params.s9,
        a1: params.a1,
        a2: params.a2,
        alp: params.alp,
    };
    let c = &opts.cutoffs;
    let cutoff = RealspaceCutoff {
        cn: c.cn,
        disp2: c.disp2,
        disp3: c.disp3,
        width2: 0.0,
        width3: 0.0,
    };
    let disp = D4Dispersion::new(model, damping, cutoff);
    let st = eeq.solve(xyz_bohr, charge)?;
    let q = &st.q;
    let cache = disp.update(xyz_bohr);

    let mut e_atom = vec![0.0; n];
    let mut vat = vec![0.0; n];
    disp.energy_and_potential(&cache, q, &mut e_atom, &mut vat);
    let order = opts.order;
    let mut gradient = Vec::new();
    if order == Order::Energy {
        disp.three_body(&cache, &mut e_atom, None, true);
    } else {
        gradient = vec![[0.0; 3]; n];
        disp.gradient(&cache, q, &mut gradient);
        disp.three_body(&cache, &mut e_atom, Some(&mut gradient), true);
        eeq.add_charge_gradient(xyz_bohr, &st, &vat, &mut gradient);
    }
    let energy: f64 = e_atom.iter().sum();

    let hessian = if order == Order::Hessian {
        let n3 = 3 * n;
        let mut h = vec![0.0; n3 * n3];
        // fixed charges (two-body at q, ATM at q = 0, CN chain rule)
        disp.hessian(&cache, q, &mut h, true);
        // charge response
        let qj = eeq.charge_jacobian(xyz_bohr, &st); // n x 3n
        let mut dv = vec![0.0; n * n3]; // dV_i/dR (n x 3n) = P^T
        disp.potential_derivative(&cache, q, &mut dv);
        let mut kq = vec![0.0; n * n];
        disp.charge_hessian(&cache, q, &mut kq);
        // u.(x_ab - A_ab y) into h; D = (u^T A_a)_j
        let d = eeq.add_charge_hessian_parts(xyz_bohr, &st, &vat, &mut h);
        // E = P - D (3n x n); H += E Q + (E Q)^T
        let mut e = par::transpose(n, n3, &dv);
        for (ev, dd) in e.iter_mut().zip(&d) {
            *ev -= dd;
        }
        let mut x = vec![0.0; n3 * n3];
        par::gemm_acc(n3, n, n3, &e, &qj, &mut x);
        par::add_sym(n3, &x, &mut h);
        // H += Q^T K Q
        let mut kqj = vec![0.0; n * n3];
        par::gemm_acc(n, n, n3, &kq, &qj, &mut kqj);
        let qt = par::transpose(n, n3, &qj);
        par::gemm_acc(n3, n, n3, &qt, &kqj, &mut h);
        Some(h)
    } else {
        None
    };
    Ok(DispResult {
        energy,
        gradient,
        hessian,
    })
}

/// EEQ charges, D4 coordination numbers and the energy split (two-body, three-body) of a
/// molecule, for checks against dftd4's `--property` output.
pub fn d4_properties(
    z: &[u32],
    xyz_bohr: &[[f64; 3]],
    charge: f64,
    params: &D4Params,
    cutoffs: &D4Cutoffs,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), DispError> {
    let n = z.len();
    let eeq = EeqModel::new(z)?;
    let numbers: Vec<usize> = z.iter().map(|&v| v as usize).collect();
    let model = D4Model::with_scaling(&numbers, ChargeModel::Eeq, params.ga, params.gc, params.wf)
        .map_err(|e| {
            let axtb_d4::D4Error::UnsupportedElement { atom, number } = e;
            DispError::UnsupportedElement {
                model: "D4",
                atom,
                number: number as u32,
            }
        })?;
    let damping = RationalDamping {
        s6: params.s6,
        s8: params.s8,
        s9: params.s9,
        a1: params.a1,
        a2: params.a2,
        alp: params.alp,
    };
    let cutoff = RealspaceCutoff {
        cn: cutoffs.cn,
        disp2: cutoffs.disp2,
        disp3: cutoffs.disp3,
        width2: 0.0,
        width3: 0.0,
    };
    let disp = D4Dispersion::new(model, damping, cutoff);
    let st = eeq.solve(xyz_bohr, charge)?;
    let cache = disp.update(xyz_bohr);
    let e2 = disp.energy_total(&cache, &st.q);
    let mut e3 = vec![0.0; n];
    disp.three_body(&cache, &mut e3, None, false);
    Ok((st.q.clone(), cache.cn.clone(), e2, e3.iter().sum()))
}
