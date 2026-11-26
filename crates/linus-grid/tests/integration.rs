//! Promolecular-density integration, screening exactness, batching invariants.

use linus_grid::*;

const ANG: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM_PYSCF;

#[allow(clippy::approx_constant)] // arbitrary random-angle scales, not pi
fn water_cluster(n: usize) -> (Vec<u32>, Vec<[f64; 3]>) {
    // waters on a jittered cubic lattice, 3.1 A apart, deterministic orientations
    let mut z = Vec::new();
    let mut xyz = Vec::new();
    let m = (n as f64).cbrt().ceil() as usize;
    let mut s: u64 = 12345;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut k = 0;
    'outer: for i in 0..m {
        for j in 0..m {
            for l in 0..m {
                if k == n {
                    break 'outer;
                }
                let o = [
                    (i as f64 * 3.1 + 0.3 * rnd()) * ANG,
                    (j as f64 * 3.1 + 0.3 * rnd()) * ANG,
                    (l as f64 * 3.1 + 0.3 * rnd()) * ANG,
                ];
                let th = rnd() * 6.283;
                let ph = rnd() * 3.14;
                let u = [ph.sin() * th.cos(), ph.sin() * th.sin(), ph.cos()];
                let v0 = [-u[1], u[0], 0.0];
                let nv = (v0[0] * v0[0] + v0[1] * v0[1]).sqrt().max(1e-6);
                let v = [v0[0] / nv, v0[1] / nv, 0.0];
                let (c, sn) = ((52.25f64).to_radians().cos(), (52.25f64).to_radians().sin());
                let r = 0.9572 * ANG;
                z.extend([8, 1, 1]);
                xyz.push(o);
                for sg in [1.0, -1.0] {
                    xyz.push([
                        o[0] + r * (c * u[0] + sg * sn * v[0]),
                        o[1] + r * (c * u[1] + sg * sn * v[1]),
                        o[2] + r * (c * u[2] + sg * sn * v[2]),
                    ]);
                }
                k += 1;
            }
        }
    }
    (z, xyz)
}

/// Normalised Slater-type promolecular density: (electrons, exponent) shells per element.
fn shells(z: u32) -> Vec<(f64, f64)> {
    let zf = z as f64;
    match z {
        1 => vec![(1.0, 2.0)],
        2 => vec![(2.0, 2.0 * 1.6875)],
        3..=10 => vec![(2.0, 2.0 * (zf - 0.3)), (zf - 2.0, 0.65 * zf.sqrt() + 0.2)],
        _ => vec![(2.0, 2.0 * (zf - 0.3)), (8.0, zf - 4.15), (zf - 10.0, 1.2)],
    }
}

#[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
fn promolecular(z: &[u32], xyz: &[[f64; 3]]) -> impl Fn(f64, f64, f64) -> f64 + Sync {
    let atoms: Vec<([f64; 3], Vec<(f64, f64)>)> =
        z.iter().zip(xyz).map(|(&zi, &p)| (p, shells(zi))).collect();
    move |x, y, zz| {
        let mut rho = 0.0;
        for (p, sh) in &atoms {
            let r = ((x - p[0]).powi(2) + (y - p[1]).powi(2) + (zz - p[2]).powi(2)).sqrt();
            for &(n, zeta) in sh {
                rho += n * zeta.powi(3) / (8.0 * std::f64::consts::PI) * (-zeta * r).exp();
            }
        }
        rho
    }
}

fn benzene() -> (Vec<u32>, Vec<[f64; 3]>) {
    let c = [
        [0.0, 1.396792, 0.0],
        [1.209657, 0.698396, 0.0],
        [1.209657, -0.698396, 0.0],
        [0.0, -1.396792, 0.0],
        [-1.209657, -0.698396, 0.0],
        [-1.209657, 0.698396, 0.0],
        [0.0, 2.484212, 0.0],
        [2.151390, 1.242106, 0.0],
        [2.151390, -1.242106, 0.0],
        [0.0, -2.484212, 0.0],
        [-2.151390, -1.242106, 0.0],
        [-2.151390, 1.242106, 0.0],
    ];
    (
        vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
        c.iter()
            .map(|p| [p[0] * ANG, p[1] * ANG, p[2] * ANG])
            .collect(),
    )
}

fn ch3so2cl() -> (Vec<u32>, Vec<[f64; 3]>) {
    let c = [
        [0.0, 0.0, 0.0],
        [2.05, 0.01, -0.02],
        [-0.43, 1.33, 0.32],
        [-0.41, -0.72, -1.17],
        [-0.52, -0.99, 1.40],
        [-1.61, -1.02, 1.44],
        [-0.11, -2.00, 1.32],
        [-0.14, -0.54, 2.32],
    ];
    (
        vec![16, 17, 8, 8, 6, 1, 1, 1],
        c.iter()
            .map(|p| [p[0] * ANG, p[1] * ANG, p[2] * ANG])
            .collect(),
    )
}

#[test]
fn promolecular_electron_count() {
    let mut report = String::new();
    let mut failures: Vec<String> = Vec::new();
    for (mname, (z, xyz)) in [("benzene", benzene()), ("ch3so2cl", ch3so2cl())] {
        let nel: f64 = z.iter().map(|&v| v as f64).sum();
        let rho = promolecular(&z, &xyz);
        let mut presets: Vec<(String, GridSpec, f64)> = Vec::new();
        let tol = [2e-2, 1e-3, 5e-4, 5e-4, 5e-5, 5e-6, 5e-6, 1e-6, 1e-6, 2e-7];
        for l in 0..10u8 {
            presets.push((format!("pyscf({l})"), GridSpec::pyscf(l), tol[l as usize]));
        }
        presets.push(("psi4_default".into(), GridSpec::psi4_default(), 1e-4));
        // ORCA's DefGrids: compact radial grids that lose part of the promolecular tails of S
        // and Cl
        for d in 1..=3u8 {
            presets.push((
                format!("orca_defgrid({d})"),
                GridSpec::orca_defgrid(d),
                5e-3,
            ));
        }
        for l in 1..=4u8 {
            presets.push((
                format!("linus({l})"),
                GridSpec::linus(l),
                [3e-3, 1e-3, 5e-4, 1e-4][l as usize - 1],
            ));
        }
        for (name, spec, tol) in presets {
            let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
            let n = g.integrate(&rho);
            let err = n - nel;
            report += &format!(
                "{mname:9} {name:16} points={:8} N-N_exact={err:+.3e}\n",
                g.len()
            );
            if err.abs() >= tol {
                failures.push(format!(
                    "{mname} {name}: electron count error {err:e} > {tol:e}"
                ));
            }
        }
    }
    println!("{report}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn screening_is_exact() {
    let (z0, xyz) = water_cluster(24);
    // every fourth water an H2S at the same place: S-H pairs (adjusted) and the others
    let z_s: Vec<u32> = z0
        .iter()
        .enumerate()
        .map(|(i, &zi)| if zi == 8 && (i / 3) % 4 == 0 { 16 } else { zi })
        .collect();
    let dirs = 37;
    for (scheme, adj, z) in [
        (PartitionScheme::SSF, SizeAdjust::None, &z0),
        (
            PartitionScheme::SSF,
            SizeAdjust::Treutler(RadiiTable::BraggPyscf),
            &z0,
        ),
        (
            PartitionScheme::LKO,
            SizeAdjust::Treutler(RadiiTable::BraggPyscf),
            &z0,
        ),
        (
            PartitionScheme::SSF,
            SizeAdjust::TreutlerBeyondRow2(RadiiTable::BraggPyscf),
            &z_s,
        ),
    ] {
        let z = z.clone();
        let p1 = Partitioner::new(&z, &xyz, scheme, adj);
        assert!(p1.is_screened());
        let p0 = Partitioner::new(&z, &xyz, scheme, adj).with_screening(false);
        let (mut s0, mut s1) = (p0.scratch(), p1.scratch());
        let (mut o0, mut o1) = (Vec::new(), Vec::new());
        let mut nnz = 0;
        for a in 0..z.len() {
            for k in 0..dirs {
                let r = 0.05 * (1.0 + k as f64).powf(1.6);
                let t = k as f64 * 2.399963;
                let zc = 1.0 - 2.0 * (k as f64 + 0.5) / dirs as f64;
                let s = (1.0 - zc * zc).sqrt();
                let p = [
                    xyz[a][0] + r * s * t.cos(),
                    xyz[a][1] + r * s * t.sin(),
                    xyz[a][2] + r * zc,
                ];
                let w0 = p0.weight(p, a, 1.0, &mut s0);
                let w1 = p1.weight(p, a, 1.0, &mut s1);
                // screened products run in distance order: identical up to rounding
                assert!(
                    (w0 - w1).abs() <= 1e-14 * w0.abs(),
                    "screened weight differs: {w0} vs {w1}"
                );
                if w0 != 0.0 {
                    nnz += 1;
                }
                let g0 = p0.weight_gradient(p, a, 1.0, &mut s0, &mut o0);
                let g1 = p1.weight_gradient(p, a, 1.0, &mut s1, &mut o1);
                assert!((g0 - g1).abs() <= 1e-15);
                let mut d0 = vec![[0.0; 3]; z.len()];
                let mut d1 = vec![[0.0; 3]; z.len()];
                for &(b, v) in &o0 {
                    d0[b] = v;
                }
                for &(b, v) in &o1 {
                    d1[b] = v;
                }
                for b in 0..z.len() {
                    for x in 0..3 {
                        assert!(
                            (d0[b][x] - d1[b][x]).abs() < 1e-13,
                            "screened derivative differs"
                        );
                    }
                }
            }
        }
        assert!(nnz > 100);
    }
}

#[test]
fn batches_cover_grid() {
    let (z, xyz) = water_cluster(10);
    for per_atom in [false, true] {
        let mut spec = GridSpec::linus(2);
        spec.batching = Batching {
            max_points: 128,
            per_atom,
        };
        let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        let mut next = 0;
        for b in &g.batches {
            assert_eq!(b.start, next);
            assert!(b.len() <= 128 && !b.is_empty());
            next = b.end;
            for i in b.start..b.end {
                let d = ((g.x[i] - b.center[0]).powi(2)
                    + (g.y[i] - b.center[1]).powi(2)
                    + (g.z[i] - b.center[2]).powi(2))
                .sqrt();
                assert!(d <= b.radius * (1.0 + 1e-12) + 1e-12);
            }
            if per_atom {
                assert!(g.atom[b.start..b.end].iter().all(|&a| a == g.atom[b.start]));
            }
        }
        assert_eq!(next, g.len());
        let avg = g.len() as f64 / g.batches.len() as f64;
        assert!(avg > 64.0, "average batch size {avg}");
        // the sub-batches: consecutive pieces of the batches, at most 40 points, each inside its
        // bounding sphere
        let subs = g.sub_batches(40);
        let mut next = 0;
        for b in &subs {
            assert_eq!(b.start, next);
            assert!(b.len() <= 40 && !b.is_empty());
            assert!(g
                .batches
                .iter()
                .any(|p| p.start <= b.start && b.end <= p.end));
            next = b.end;
            for i in b.start..b.end {
                let d = ((g.x[i] - b.center[0]).powi(2)
                    + (g.y[i] - b.center[1]).powi(2)
                    + (g.z[i] - b.center[2]).powi(2))
                .sqrt();
                assert!(d <= b.radius);
            }
        }
        assert_eq!(next, g.len());
    }
}

#[test]
fn standard_orientation_is_rotation_invariant() {
    let (z, xyz) = ch3so2cl();
    let mut spec = GridSpec::linus(2);
    spec.orientation = Orientation::Standard;
    let rho = promolecular(&z, &xyz);
    let n0 = MolecularGrid::build(&z, &xyz, &spec)
        .unwrap()
        .integrate(&rho);
    // rotate the molecule
    let (a, b) = (0.7f64, -1.1f64);
    let rz = [
        [a.cos(), -a.sin(), 0.0],
        [a.sin(), a.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ];
    let rx = [
        [1.0, 0.0, 0.0],
        [0.0, b.cos(), -b.sin()],
        [0.0, b.sin(), b.cos()],
    ];
    let rot = |p: [f64; 3], m: [[f64; 3]; 3]| {
        [
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
        ]
    };
    let xyz2: Vec<[f64; 3]> = xyz.iter().map(|&p| rot(rot(p, rz), rx)).collect();
    let rho2 = promolecular(&z, &xyz2);
    let n1 = MolecularGrid::build(&z, &xyz2, &spec)
        .unwrap()
        .integrate(&rho2);
    assert!((n0 - n1).abs() < 1e-10, "{n0} vs {n1}");
    // without the standard orientation the result changes
    spec.orientation = Orientation::Identity;
    let m0 = MolecularGrid::build(&z, &xyz, &spec)
        .unwrap()
        .integrate(&rho);
    let m1 = MolecularGrid::build(&z, &xyz2, &spec)
        .unwrap()
        .integrate(&rho2);
    assert!((m0 - m1).abs() > 1e-9);
}

/// linus:3's size adjustment (`TreutlerBeyondRow2`): a molecule of atoms up to neon gets exactly
/// the weights of no adjustment; with an atom beyond neon the pairs of H with it are adjusted
/// (Treutler) and the others are not.
#[test]
fn size_adjustment_beyond_row2_only() {
    let (z, xyz) = water_cluster(8);
    let t = SizeAdjust::TreutlerBeyondRow2(RadiiTable::BraggPyscf);
    let light = |adj| {
        MolecularGrid::build(
            &z,
            &xyz,
            &GridSpec {
                size_adjust: adj,
                ..GridSpec::linus(3)
            },
        )
        .unwrap()
    };
    let (a, b) = (light(t), light(SizeAdjust::None));
    assert_eq!(a.w.len(), b.w.len());
    assert!(
        a.w.iter()
            .zip(&b.w)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "CHNO weights changed"
    );
    // H2S + H2O: the S-H pairs adjusted, the O-H and S-O pairs as without adjustment
    let z2 = vec![16, 1, 1, 8, 1, 1];
    let xyz2 = vec![
        [0.0, 0.0, 0.0],
        [0.0, 1.81, 1.75],
        [0.0, -1.81, 1.75],
        [0.0, 0.0, 6.0],
        [0.0, 1.43, 7.1],
        [0.0, -1.43, 7.1],
    ];
    let g = |adj| {
        MolecularGrid::build(
            &z2,
            &xyz2,
            &GridSpec {
                size_adjust: adj,
                ..GridSpec::linus(3)
            },
        )
        .unwrap()
    };
    let (mixed, none, full) = (
        g(t),
        g(SizeAdjust::None),
        g(SizeAdjust::Treutler(RadiiTable::BraggPyscf)),
    );
    assert!(mixed.w.iter().zip(&none.w).any(|(x, y)| x != y));
    assert!(mixed.w.iter().zip(&full.w).any(|(x, y)| x != y));
    // BeCl2 (atoms beyond neon, no H): exactly no adjustment
    let (z3, xyz3) = (
        vec![4, 17, 17],
        vec![[0.0; 3], [0.0, 0.0, 3.4], [0.0, 0.0, -3.4]],
    );
    let g3 = |adj| {
        MolecularGrid::build(
            &z3,
            &xyz3,
            &GridSpec {
                size_adjust: adj,
                ..GridSpec::linus(3)
            },
        )
        .unwrap()
    };
    let (a3, b3) = (g3(t), g3(SizeAdjust::None));
    assert!(
        a3.w.iter()
            .zip(&b3.w)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "BeCl2 weights changed"
    );
}
