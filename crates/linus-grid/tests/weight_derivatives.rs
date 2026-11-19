//! Analytic partition-weight derivatives vs central finite differences.

use linus_grid::*;

#[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
fn molecules() -> Vec<(&'static str, Vec<u32>, Vec<[f64; 3]>)> {
    vec![
        (
            "water",
            vec![8, 1, 1],
            vec![
                [0.0, 0.0, 0.2226],
                [0.0, 1.4276, -0.8904],
                [0.05, -1.4276, -0.8904],
            ],
        ),
        (
            "ch3so2cl",
            vec![16, 17, 8, 8, 6, 1, 1, 1],
            vec![
                [0.0, 0.0, 0.0],
                [3.874, 0.019, -0.038],
                [-0.813, 2.513, 0.605],
                [-0.775, -1.361, -2.211],
                [-0.983, -1.871, 2.646],
                [-3.042, -1.928, 2.721],
                [-0.208, -3.779, 2.494],
                [-0.265, -1.020, 4.384],
            ],
        ),
    ]
}

fn schemes() -> Vec<(&'static str, PartitionScheme, SizeAdjust)> {
    let bragg = RadiiTable::BraggPyscf;
    vec![
        ("becke", PartitionScheme::Becke, SizeAdjust::None),
        (
            "becke+treutler",
            PartitionScheme::Becke,
            SizeAdjust::Treutler(bragg),
        ),
        (
            "becke+becke",
            PartitionScheme::Becke,
            SizeAdjust::Becke(bragg),
        ),
        ("ssf", PartitionScheme::SSF, SizeAdjust::None),
        (
            "ssf+treutler",
            PartitionScheme::SSF,
            SizeAdjust::Treutler(bragg),
        ),
        (
            "ssf+treutler-beyond-row2",
            PartitionScheme::SSF,
            SizeAdjust::TreutlerBeyondRow2(bragg),
        ),
        (
            "becke+treutler-beyond-row2",
            PartitionScheme::Becke,
            SizeAdjust::TreutlerBeyondRow2(bragg),
        ),
        ("lko", PartitionScheme::LKO, SizeAdjust::None),
        (
            "lko+treutler",
            PartitionScheme::LKO,
            SizeAdjust::Treutler(bragg),
        ),
    ]
}

/// deterministic pseudo-random unit vectors
fn directions(n: usize) -> Vec<[f64; 3]> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let z = 2.0 * next() - 1.0;
            let ph = 2.0 * std::f64::consts::PI * next();
            let r = (1.0 - z * z).sqrt();
            [r * ph.cos(), r * ph.sin(), z]
        })
        .collect()
}

#[test]
fn weight_derivatives_match_finite_differences() {
    let h = 1e-5;
    let dirs = directions(40);
    let radii = [0.15, 0.6, 1.2, 2.0, 3.0, 4.5];
    let mut report = String::new();
    for (mname, z, xyz) in molecules() {
        for (sname, scheme, adj) in schemes() {
            let part = Partitioner::new(&z, &xyz, scheme, adj);
            let mut scr = part.scratch();
            let mut out = Vec::new();
            let natm = z.len();
            let (mut max_err, mut max_grad, mut max_tsum) = (0.0f64, 0.0f64, 0.0f64);
            let mut npts = 0;
            for a in 0..natm {
                for (k, d) in dirs.iter().enumerate() {
                    let r = radii[k % radii.len()];
                    let off = [r * d[0], r * d[1], r * d[2]];
                    let p = [xyz[a][0] + off[0], xyz[a][1] + off[1], xyz[a][2] + off[2]];
                    let w0 = part.weight_gradient(p, a, 1.0, &mut scr, &mut out);
                    let w1 = part.weight(p, a, 1.0, &mut scr);
                    assert!((w0 - w1).abs() < 1e-14, "weight mismatch {w0} {w1}");
                    let mut ana = vec![[0.0f64; 3]; natm];
                    for &(b, g) in &out {
                        ana[b] = g;
                    }
                    let mut tsum = [0.0f64; 3];
                    for g in &ana {
                        for x in 0..3 {
                            tsum[x] += g[x];
                        }
                    }
                    max_tsum = max_tsum.max(tsum.iter().fold(0.0f64, |m, v| m.max(v.abs())));
                    for b in 0..natm {
                        for x in 0..3 {
                            let mut wp = [0.0; 2];
                            for (si, sgn) in [1.0, -1.0].iter().enumerate() {
                                let mut c = xyz.clone();
                                c[b][x] += sgn * h;
                                let pp = [c[a][0] + off[0], c[a][1] + off[1], c[a][2] + off[2]];
                                let pd = Partitioner::new(&z, &c, scheme, adj);
                                let mut s2 = pd.scratch();
                                wp[si] = pd.weight(pp, a, 1.0, &mut s2);
                            }
                            let fd = (wp[0] - wp[1]) / (2.0 * h);
                            max_err = max_err.max((fd - ana[b][x]).abs());
                            max_grad = max_grad.max(ana[b][x].abs());
                        }
                    }
                    npts += 1;
                }
            }
            report += &format!(
                "{mname:9} {sname:15} points={npts:4} max|dw/dR|={max_grad:.3e} max|analytic-FD|={max_err:.2e} max|sum_D dw/dR_D|={max_tsum:.1e}\n"
            );
            assert!(max_err < 2e-7, "{mname} {sname}: FD error {max_err:e}");
            assert!(max_tsum < 1e-12);
        }
    }
    println!("{report}");
}

#[test]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn grid_contraction_matches_pointwise_sum() {
    let (z, xyz) = (
        vec![8u32, 1, 1],
        vec![
            [0.0, 0.0, 0.2226],
            [0.0, 1.4276, -0.8904],
            [0.05, -1.4276, -0.8904],
        ],
    );
    for scheme in [
        PartitionScheme::Becke,
        PartitionScheme::SSF,
        PartitionScheme::LKO,
    ] {
        let mut spec = GridSpec::pyscf(1);
        spec.partition = scheme;
        let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        let f: Vec<f64> = (0..g.len())
            .map(|i| (-(g.x[i] * g.x[i] + g.y[i] * g.y[i] + g.z[i] * g.z[i]) * 0.3).exp())
            .collect();
        let c = g.weight_gradient_contract(&f);
        let mut scr = g.partitioner().scratch();
        let mut out = Vec::new();
        let mut d = [[0.0; 3]; 3];
        for i in 0..g.len() {
            g.point_weight_gradient(i, &mut scr, &mut out);
            for &(a, v) in &out {
                for k in 0..3 {
                    d[a][k] += f[i] * v[k];
                }
            }
        }
        for a in 0..3 {
            for k in 0..3 {
                assert!((c[a][k] - d[a][k]).abs() < 1e-10 * (1.0 + d[a][k].abs()));
            }
        }
    }
}

/// The block contraction (Becke) and the point-wise one (all schemes) against the per-point
/// reference, for every partition and size adjustment; bitwise identical for 1, 3 and 8 threads.
#[test]
fn contraction_matches_reference_and_is_deterministic() {
    for (mname, z, xyz) in molecules() {
        for (sname, scheme, adj) in schemes() {
            let mut spec = GridSpec::pyscf(1);
            spec.partition = scheme;
            spec.size_adjust = adj;
            let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
            let f: Vec<f64> = (0..g.len())
                .map(|i| {
                    (-(g.x[i] * g.x[i] + 0.7 * g.y[i] * g.y[i] + 1.3 * g.z[i] * g.z[i]) * 0.2).exp()
                        * (1.0 + 0.1 * g.x[i])
                })
                .collect();
            let r = g.weight_gradient_contract_pointwise(&f);
            let run = |t: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
                    .install(|| g.weight_gradient_contract(&f))
            };
            let c = run(1);
            let scale = r.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            let err = r
                .iter()
                .flatten()
                .zip(c.iter().flatten())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                err <= 1e-13 * scale,
                "{mname} {sname}: {err:e} (scale {scale:e})"
            );
            for t in [3, 8] {
                assert_eq!(run(t), c, "{mname} {sname}: {t} threads");
            }
        }
    }
}
