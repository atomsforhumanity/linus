//! One `Functional` shared by many rayon workers gives bit-identical results to
//! a serial evaluation, and functionals can be created concurrently.

use linus_xc::{Functional, Out, XcInput, XcOutput};
use rayon::prelude::*;

/// Deterministic, physical-ish inputs (xorshift; no rand dependency).
fn inputs(np: usize, polarized: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut u = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let ns = if polarized { 2 } else { 1 };
    let rho: Vec<f64> = (0..np * ns).map(|_| 10f64.powf(-6.0 + 7.0 * u())).collect();
    let mut sigma = Vec::with_capacity(np * (2 * ns - 1));
    let mut tau = Vec::with_capacity(np * ns);
    for p in 0..np {
        let r = &rho[p * ns..(p + 1) * ns];
        let s2: Vec<f64> = r
            .iter()
            .map(|x| (2.0 * u() * x.powf(4.0 / 3.0)).powi(2))
            .collect();
        if polarized {
            let c = 2.0 * u() - 1.0;
            sigma.extend([s2[0], c * (s2[0] * s2[1]).sqrt(), s2[1]]);
        } else {
            sigma.push(s2[0]);
        }
        for (x, s2) in r.iter().zip(&s2) {
            tau.push(s2 / (8.0 * x) + u() * 4.6 * x.powf(5.0 / 3.0));
        }
    }
    (rho, sigma, tau)
}

#[test]
fn shared_functional_parallel_equals_serial() {
    let np = 50_000;
    for polarized in [false, true] {
        let (rho, sigma, tau) = inputs(np, polarized);
        let ns = if polarized { 2 } else { 1 };
        for name in [
            "b3lyp",
            "r2scan",
            "wb97m-v",
            "0.2*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP",
        ] {
            let f = Functional::new(name, polarized).unwrap();
            let mut serial = XcOutput::new();
            f.eval(
                2,
                &XcInput {
                    rho: &rho,
                    sigma: Some(&sigma),
                    lapl: None,
                    tau: Some(&tau),
                },
                &mut serial,
            )
            .unwrap();

            let chunk = 777; // odd size so chunk boundaries move around
            let f = &f;
            let parts: Vec<XcOutput> = (0..np.div_ceil(chunk))
                .into_par_iter()
                .map(|c| {
                    let (a, b) = (c * chunk, ((c + 1) * chunk).min(np));
                    let ns3 = 2 * ns - 1;
                    let inp = XcInput {
                        rho: &rho[a * ns..b * ns],
                        sigma: Some(&sigma[a * ns3..b * ns3]),
                        lapl: None,
                        tau: Some(&tau[a * ns..b * ns]),
                    };
                    let mut out = XcOutput::new();
                    f.eval(2, &inp, &mut out).unwrap();
                    out
                })
                .collect();
            for o in Out::ALL {
                let joined: Vec<f64> = parts
                    .iter()
                    .flat_map(|p| p.get(o).iter().copied())
                    .collect();
                let s = serial.get(o);
                assert_eq!(joined.len(), s.len(), "{name} {}", o.name());
                assert!(
                    joined
                        .iter()
                        .zip(s)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{name} polarized={polarized} {} differs between parallel and serial",
                    o.name()
                );
            }
        }
    }
}

#[test]
fn concurrent_construction() {
    // HSE/HJS variants share function-local statics in libxc's init code;
    // INIT_LOCK must keep them apart. Each must equal a serially built copy.
    let names = [
        "HYB_GGA_XC_HSE03",
        "HYB_GGA_XC_HSE06",
        "HYB_GGA_XC_HJS_PBE",
        "HYB_GGA_XC_HJS_B88",
        "b3lyp",
        "r2scan",
    ];
    let reference: Vec<_> = names
        .iter()
        .map(|n| Functional::new(n, false).unwrap().exact_exchange())
        .collect();
    let built: Vec<_> = (0..256)
        .into_par_iter()
        .map(|i| {
            (
                i % names.len(),
                Functional::new(names[i % names.len()], i % 2 == 0)
                    .unwrap()
                    .exact_exchange(),
            )
        })
        .collect();
    for (k, x) in built {
        assert_eq!(x, reference[k], "{}", names[k]);
    }
}
