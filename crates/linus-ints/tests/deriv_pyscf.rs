//! Comparison of derivative integrals and gradients with PySCF (libcint) reference data from
//! tests/python/gen_deriv_fixtures.py (fixtures in tests/fixtures/).
//!
//! PySCF's "ip" integrals differentiate the electron coordinate of the marked function:
//! (∇i j|kl) = -∂(ij|kl)/∂A where A is the center of i; second derivatives carry two signs.

mod common;
use common::*;
use linus_ints::basis::{BasisDef, ShellDef};
use linus_ints::contract::DensityTerms;
use linus_ints::core::deriv::{deriv2_pair, pair_with_deriv, D2Kind};
use linus_ints::eri4c::deriv::DerivWork;
use linus_ints::potential::{matrix_block, PointCharges};
use linus_ints::reference::oneshot;
use linus_ints::reference::Dens3c;
use linus_ints::{one_e, BasisSet, DerivOrder, PairOptions, PairSet};
use serde_json::Value;
use std::collections::HashMap;

fn fixture(name: &str) -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn arr(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn coords_of(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let r = arr(r);
            [r[0], r[1], r[2]]
        })
        .collect()
}

fn zs_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn basis_from(v: &Value) -> BasisDef {
    let mut el = HashMap::new();
    for (z, shells) in v.as_object().unwrap() {
        let mut s: Vec<ShellDef> = shells
            .as_array()
            .unwrap()
            .iter()
            .map(|sh| {
                let a = sh.as_array().unwrap();
                ShellDef {
                    l: a[0].as_u64().unwrap() as usize,
                    exps: arr(&a[1]),
                    coefs: arr(&a[2]),
                }
            })
            .collect();
        s.sort_by_key(|x| x.l);
        el.insert(z.parse::<u32>().unwrap(), s);
    }
    BasisDef {
        name: "fixture".into(),
        elements: el,
        ecp_electrons: HashMap::new(),
        ecp: HashMap::new(),
    }
}

/// out[c][off[keep] + x_keep] += f * sum blk[c][x0][x1]... prod_{p != keep} vec_p[off_p + x_p]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
fn acc_proj(
    blk: &[f64],
    ncomp: usize,
    dims: &[usize],
    offs: &[usize],
    keep: usize,
    vecs: &[&[f64]],
    f: f64,
    nkeep: usize,
    out: &mut [f64],
) {
    let m = dims.len();
    let size: usize = dims.iter().product();
    let mut idx = vec![0usize; m];
    for c in 0..ncomp {
        for flat in 0..size {
            let mut r = flat;
            for p in (0..m).rev() {
                idx[p] = r % dims[p];
                r /= dims[p];
            }
            let mut w = f * blk[c * size + flat];
            if w == 0.0 {
                continue;
            }
            for p in 0..m {
                if p != keep {
                    w *= vecs[p][offs[p] + idx[p]];
                }
            }
            out[c * nkeep + offs[keep] + idx[keep]] += w;
        }
    }
}

struct Fx {
    v: Value,
    def: BasisDef,
    adef: BasisDef,
    z: Vec<u32>,
    positions: Vec<[f64; 3]>,
}

impl Fx {
    fn load() -> Fx {
        let v = fixture("pyscf_deriv_ints.json");
        let def = basis_from(&v["basis"]);
        let adef = basis_from(&v["aux"]);
        let z = zs_of(&v["z"]);
        let positions = coords_of(&v["coords"]);
        Fx {
            v,
            def,
            adef,
            z,
            positions,
        }
    }
    fn vec(&self, k: &str) -> Vec<f64> {
        arr(&self.v[k])
    }
    fn reference(&self, name: &str) -> Vec<f64> {
        arr(&self.v["ints"][name]["data"])
    }
}

/// Point charges (q, C) as a potential source (their slots do not matter to a matrix block).
fn point(charges: &[(f64, [f64; 3])]) -> PointCharges<'_> {
    PointCharges {
        charges,
        first_slot: 0,
    }
}

fn cmp(name: &str, ours: &[f64], fx: &Fx, table: &mut Vec<(String, f64)>) {
    let r = fx.reference(name);
    let e = check(&format!("vs PySCF {name}"), ours, &r, 1e-10);
    table.push((name.to_string(), e));
}

#[test]
#[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
fn pyscf_one_electron_derivatives() {
    let fx = Fx::load();
    let basis = make_basis(&fx.def, &fx.z, &fx.positions);
    let n = basis.nbf;
    assert_eq!(n as u64, fx.v["nbf"].as_u64().unwrap());
    let sh = &basis.shells;
    let v2 = fx.vec("v2");
    let ones = vec![1.0; n];
    let charges: Vec<(f64, [f64; 3])> =
        fx.z.iter()
            .zip(&fx.positions)
            .map(|(&q, &c)| (q as f64, c))
            .collect();
    let mut tab = Vec::new();
    let mut p = HashMap::<String, Vec<f64>>::new();
    let names = [
        "ipovlp", "ipkin", "ipnuc", "irp", "ipipovlp", "ipovlpip", "ipipkin", "ipkinip", "ipipnuc",
        "ipnucip",
    ];
    let ncomps = [3, 3, 3, 9, 9, 9, 9, 9, 9, 9];
    for (nm, nc) in names.iter().zip(ncomps) {
        p.insert(nm.to_string(), vec![0.0; nc * n]);
    }
    let natom = fx.z.len();
    for ia in 0..natom {
        for nm in ["iprinv", "ipiprinv", "iprinvip"] {
            p.insert(
                format!("{nm}@{ia}"),
                vec![0.0; if nm == "iprinv" { 3 } else { 9 } * n],
            );
        }
    }
    for i in 0..sh.len() {
        for j in 0..sh.len() {
            let (a, b) = (&sh[i], &sh[j]);
            let (na, nb) = (a.nfunc(), b.nfunc());
            let nab = na * nb;
            let dims = [na, nb];
            let offs = [a.ao_start, b.ao_start];
            let vecs: [&[f64]; 2] = [&ones, &v2];
            let mut add = |key: &str, blk: &[f64], nc: usize, f: f64| {
                acc_proj(
                    blk,
                    nc,
                    &dims,
                    &offs,
                    0,
                    &vecs,
                    f,
                    n,
                    p.get_mut(key).unwrap(),
                )
            };
            let (mut s1, mut t1) = (vec![0.0; 3 * nab], vec![0.0; 3 * nab]);
            one_e::deriv::overlap_kinetic_deriv1_block(a, b, &mut s1, &mut t1);
            add("ipovlp", &s1, 3, -1.0);
            add("ipkin", &t1, 3, -1.0);
            let (mut s2, mut t2) = (vec![0.0; 9 * nab], vec![0.0; 9 * nab]);
            one_e::deriv::overlap_kinetic_deriv2_block(a, b, &mut s2, &mut t2);
            add("ipipovlp", &s2, 9, 1.0);
            add("ipovlpip", &s2, 9, -1.0);
            add("ipipkin", &t2, 9, 1.0);
            add("ipkinip", &t2, 9, -1.0);
            // dipole: irp[c*3+k] = <i| r_c d/dr_k |j> = -d/dB_k <i|r_c|j>; ours [k][c]
            let (mut da, mut db) = (vec![0.0; 9 * nab], vec![0.0; 9 * nab]);
            one_e::deriv::dipole_deriv1_block(a, b, [0.0; 3], &mut da, &mut db);
            let mut irp = vec![0.0; 9 * nab];
            for k in 0..3 {
                for c in 0..3 {
                    irp[(c * 3 + k) * nab..(c * 3 + k + 1) * nab]
                        .copy_from_slice(&db[(k * 3 + c) * nab..(k * 3 + c + 1) * nab]);
                }
            }
            add("irp", &irp, 9, -1.0);
            // potential
            let (pr, pd) = pair_with_deriv(&basis, i, j, 2);
            let pot = |key: &str,
                       ch: &[(f64, [f64; 3])],
                       add: &mut dyn FnMut(&str, &[f64], usize, f64)| {
                let (mut oa, mut ob, mut oc) = (
                    vec![0.0; 3 * nab],
                    vec![0.0; 3 * nab],
                    vec![0.0; 3 * nab * ch.len()],
                );
                one_e::deriv::potential_deriv1_block(&pr, &pd, ch, &mut oa, &mut ob, &mut oc);
                let first = if key == "nuc" {
                    "ipnuc".to_string()
                } else {
                    format!("iprinv@{}", &key[2..])
                };
                add(&first, &oa, 3, -1.0);
                let mut buf = vec![0.0; 9 * nab];
                matrix_block(&deriv2_pair(&pr, &pd, D2Kind::AA), &point(ch), &mut buf);
                let k2 = if key == "nuc" {
                    "ipipnuc".to_string()
                } else {
                    format!("ipiprinv@{}", &key[2..])
                };
                add(&k2, &buf, 9, 1.0);
                matrix_block(&deriv2_pair(&pr, &pd, D2Kind::AB), &point(ch), &mut buf);
                let k3 = if key == "nuc" {
                    "ipnucip".to_string()
                } else {
                    format!("iprinvip@{}", &key[2..])
                };
                add(&k3, &buf, 9, 1.0);
            };
            pot("nuc", &charges, &mut add);
            for ia in 0..natom {
                pot(&format!("r@{ia}"), &[(-1.0, fx.positions[ia])], &mut add);
            }
        }
    }
    for (k, v) in &p {
        cmp(&format!("int1e_{}", k), v, &fx, &mut tab);
    }
    // core Hamiltonian derivative matrices (PySCF grad hcore_generator)
    let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
    let hd = oneshot::hcore_deriv(&basis, &pairs, &charges);
    for ia in 0..natom {
        let mut out = vec![0.0; 3 * n];
        for c in 0..3 {
            let m = &hd[(3 * ia + c) * n * n..(3 * ia + c + 1) * n * n];
            for i in 0..n {
                out[c * n + i] = (0..n).map(|j| m[i * n + j] * v2[j]).sum();
            }
        }
        cmp(&format!("hcore_deriv@{ia}"), &out, &fx, &mut tab);
    }
    // second-derivative contraction of the core Hamiltonian
    let dm = fx.v["ints"]["hcore_hess_dmat"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    let mut h = oneshot::overlap_kinetic_hess(&basis, None, Some(&dm));
    let hv = oneshot::nuclear_attraction_hess(&basis, &pairs, &charges, &dm);
    for (x, y) in h.iter_mut().zip(&hv) {
        *x += y;
    }
    cmp("hcore_hess_contract", &h, &fx, &mut tab);
    tab.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, e) in &tab {
        println!("SUMMARY pyscf {k:28} {e:.2e}");
    }
}

#[test]
fn pyscf_two_electron_derivatives() {
    let fx = Fx::load();
    let basis = make_basis(&fx.def, &fx.z, &fx.positions);
    let aux = make_basis(&fx.adef, &fx.z, &fx.positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    assert_eq!(naux as u64, fx.v["naux"].as_u64().unwrap());
    let sh = &basis.shells;
    let ns = sh.len();
    let (v1, v2, v3, v4, u1, u2) = (
        fx.vec("v1"),
        fx.vec("v2"),
        fx.vec("v3"),
        fx.vec("v4"),
        fx.vec("u1"),
        fx.vec("u2"),
    );
    let one = vec![1.0; n.max(naux)];
    let mut tab = Vec::new();
    let prs: Vec<Vec<_>> = (0..ns)
        .map(|i| (0..ns).map(|j| pair_with_deriv(&basis, i, j, 2)).collect())
        .collect();
    let mut w = DerivWork::new();
    // 4-center
    let mut ip1 = vec![0.0; 3 * n];
    let (mut ipip1, mut ipvip1, mut ip1ip2) =
        (vec![0.0; 9 * n], vec![0.0; 9 * n], vec![0.0; 9 * n]);
    for i in 0..ns {
        for j in 0..ns {
            for k in 0..ns {
                for l in 0..ns {
                    let (bra, bd) = &prs[i][j];
                    let (ket, kd) = &prs[k][l];
                    let blk = bra.nab() * ket.nab();
                    let dims = [bra.na, bra.nb, ket.na, ket.nb];
                    let offs = [
                        sh[i].ao_start,
                        sh[j].ao_start,
                        sh[k].ao_start,
                        sh[l].ao_start,
                    ];
                    let vecs: [&[f64]; 4] = [&one, &v2, &v3, &v4];
                    let mut d1 = vec![0.0; 12 * blk];
                    linus_ints::reference::eri_quartet_deriv1(bra, bd, ket, kd, &mut d1, &mut w);
                    acc_proj(&d1[..3 * blk], 3, &dims, &offs, 0, &vecs, -1.0, n, &mut ip1);
                    let mut d2 = vec![0.0; 90 * blk];
                    linus_ints::reference::eri_quartet_deriv2_ref(
                        bra,
                        bd,
                        ket,
                        kd,
                        &linus_ints::operator::Operator::Coulomb,
                        &mut d2,
                    );
                    acc_proj(
                        &d2[0..9 * blk],
                        9,
                        &dims,
                        &offs,
                        0,
                        &vecs,
                        1.0,
                        n,
                        &mut ipip1,
                    );
                    acc_proj(
                        &d2[9 * blk..18 * blk],
                        9,
                        &dims,
                        &offs,
                        0,
                        &vecs,
                        1.0,
                        n,
                        &mut ipvip1,
                    );
                    acc_proj(
                        &d2[18 * blk..27 * blk],
                        9,
                        &dims,
                        &offs,
                        0,
                        &vecs,
                        1.0,
                        n,
                        &mut ip1ip2,
                    );
                }
            }
        }
    }
    cmp("int2e_ip1", &ip1, &fx, &mut tab);
    cmp("int2e_ipip1", &ipip1, &fx, &mut tab);
    cmp("int2e_ipvip1", &ipvip1, &fx, &mut tab);
    cmp("int2e_ip1ip2", &ip1ip2, &fx, &mut tab);
    // 3-center: bra (P, 1), ket (i, j); PySCF order (i, j, P)
    let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
    let ash = &aux.shells;
    let (mut t_ip1, mut t_ip2) = (vec![0.0; 3 * n], vec![0.0; 3 * naux]);
    let (mut t_ipip1, mut t_ipvip1, mut t_ip1ip2, mut t_ipip2) = (
        vec![0.0; 9 * n],
        vec![0.0; 9 * n],
        vec![0.0; 9 * n],
        vec![0.0; 9 * naux],
    );
    #[allow(clippy::needless_range_loop, reason = "shells and their pairs in step")]
    for ps in 0..ash.len() {
        for i in 0..ns {
            for j in 0..ns {
                let (ket, kd) = &prs[i][j];
                let bra = ap.pair(ps);
                let blk = bra.nab() * ket.nab();
                let dims = [bra.na, 1, ket.na, ket.nb];
                let offs = [ash[ps].ao_start, 0, sh[i].ao_start, sh[j].ao_start];
                let vk_i: [&[f64]; 4] = [&u1, &one, &one, &v2];
                let vk_p: [&[f64]; 4] = [&one, &one, &v1, &v2];
                let mut d1 = vec![0.0; 12 * blk];
                linus_ints::reference::eri_quartet_deriv1(
                    bra,
                    ap.deriv(ps),
                    ket,
                    kd,
                    &mut d1,
                    &mut w,
                );
                acc_proj(
                    &d1[6 * blk..9 * blk],
                    3,
                    &dims,
                    &offs,
                    2,
                    &vk_i,
                    -1.0,
                    n,
                    &mut t_ip1,
                );
                acc_proj(
                    &d1[0..3 * blk],
                    3,
                    &dims,
                    &offs,
                    0,
                    &vk_p,
                    -1.0,
                    naux,
                    &mut t_ip2,
                );
                let mut d2 = vec![0.0; 90 * blk];
                linus_ints::reference::eri_quartet_deriv2_ref(
                    bra,
                    ap.deriv(ps),
                    ket,
                    kd,
                    &linus_ints::operator::Operator::Coulomb,
                    &mut d2,
                );
                // CC (pair 7), CD (pair 8), AC (pair 2) transposed, AA (pair 0)
                acc_proj(
                    &d2[63 * blk..72 * blk],
                    9,
                    &dims,
                    &offs,
                    2,
                    &vk_i,
                    1.0,
                    n,
                    &mut t_ipip1,
                );
                acc_proj(
                    &d2[72 * blk..81 * blk],
                    9,
                    &dims,
                    &offs,
                    2,
                    &vk_i,
                    1.0,
                    n,
                    &mut t_ipvip1,
                );
                let mut act = vec![0.0; 9 * blk];
                for k in 0..3 {
                    for l in 0..3 {
                        act[(k * 3 + l) * blk..(k * 3 + l + 1) * blk].copy_from_slice(
                            &d2[(18 + l * 3 + k) * blk..(18 + l * 3 + k + 1) * blk],
                        );
                    }
                }
                acc_proj(&act, 9, &dims, &offs, 2, &vk_i, 1.0, n, &mut t_ip1ip2);
                acc_proj(
                    &d2[0..9 * blk],
                    9,
                    &dims,
                    &offs,
                    0,
                    &vk_p,
                    1.0,
                    naux,
                    &mut t_ipip2,
                );
            }
        }
    }
    cmp("int3c2e_ip1", &t_ip1, &fx, &mut tab);
    cmp("int3c2e_ip2", &t_ip2, &fx, &mut tab);
    cmp("int3c2e_ipip1", &t_ipip1, &fx, &mut tab);
    cmp("int3c2e_ipvip1", &t_ipvip1, &fx, &mut tab);
    cmp("int3c2e_ip1ip2", &t_ip1ip2, &fx, &mut tab);
    cmp("int3c2e_ipip2", &t_ipip2, &fx, &mut tab);
    // 2-center
    let (mut c_ip1, mut c_ipip1, mut c_ip1ip2) = (
        vec![0.0; 3 * naux],
        vec![0.0; 9 * naux],
        vec![0.0; 9 * naux],
    );
    for p in 0..ash.len() {
        for q in 0..ash.len() {
            let (bra, ket) = (ap.pair(p), ap.pair(q));
            let blk = bra.na * ket.na;
            let dims = [bra.na, 1, ket.na, 1];
            let offs = [ash[p].ao_start, 0, ash[q].ao_start, 0];
            let vecs: [&[f64]; 4] = [&one, &one, &u2, &one];
            let mut d1 = vec![0.0; 12 * blk];
            linus_ints::reference::eri_quartet_deriv1(
                bra,
                ap.deriv(p),
                ket,
                ap.deriv(q),
                &mut d1,
                &mut w,
            );
            acc_proj(
                &d1[0..3 * blk],
                3,
                &dims,
                &offs,
                0,
                &vecs,
                -1.0,
                naux,
                &mut c_ip1,
            );
            let mut d2 = vec![0.0; 90 * blk];
            linus_ints::reference::eri_quartet_deriv2_ref(
                bra,
                ap.deriv(p),
                ket,
                ap.deriv(q),
                &linus_ints::operator::Operator::Coulomb,
                &mut d2,
            );
            acc_proj(
                &d2[0..9 * blk],
                9,
                &dims,
                &offs,
                0,
                &vecs,
                1.0,
                naux,
                &mut c_ipip1,
            );
            acc_proj(
                &d2[18 * blk..27 * blk],
                9,
                &dims,
                &offs,
                0,
                &vecs,
                1.0,
                naux,
                &mut c_ip1ip2,
            );
        }
    }
    cmp("int2c2e_ip1", &c_ip1, &fx, &mut tab);
    cmp("int2c2e_ipip1", &c_ipip1, &fx, &mut tab);
    cmp("int2c2e_ip1ip2", &c_ip1ip2, &fx, &mut tab);
    for (k, e) in &tab {
        println!("SUMMARY pyscf {k:28} {e:.2e}");
    }
}

// ---------------------------------------------------------------------------------------
// End-to-end RHF gradients with PySCF densities
// ---------------------------------------------------------------------------------------

fn nuc_grad(z: &[u32], c: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let n = z.len();
    let mut g = vec![[0.0; 3]; n];
    for i in 0..n {
        for j in 0..i {
            let r = [c[i][0] - c[j][0], c[i][1] - c[j][1], c[i][2] - c[j][2]];
            let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            let f = -((z[i] * z[j]) as f64) / (d * d * d);
            for k in 0..3 {
                g[i][k] += f * r[k];
                g[j][k] -= f * r[k];
            }
        }
    }
    g
}

/// Cholesky factor L (lower, row-major) of a symmetric positive definite matrix.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        let d = s.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    l
}

/// Solve (L L^T) X = B in place for B (n x m row-major).
fn chol_solve(l: &[f64], n: usize, b: &mut [f64], m: usize) {
    for c in 0..m {
        for i in 0..n {
            let mut s = b[i * m + c];
            for k in 0..i {
                s -= l[i * n + k] * b[k * m + c];
            }
            b[i * m + c] = s / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = b[i * m + c];
            for k in i + 1..n {
                s -= l[k * n + i] * b[k * m + c];
            }
            b[i * m + c] = s / l[i * n + i];
        }
    }
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let x = a[i * n + k];
            if x == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += x * b[k * n + j];
            }
        }
    }
    c
}

#[test]
fn pyscf_rhf_gradients() {
    let fx = fixture("pyscf_rhf_grad.json");
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let z = zs_of(&case["z"]);
        let positions = coords_of(&case["coords"]);
        let def = BasisDef::load(case["basis"].as_str().unwrap(), None).unwrap();
        let basis = BasisSet::new(&def, &z, &positions, true).unwrap();
        let n = basis.nbf;
        assert_eq!(n as u64, case["nbf"].as_u64().unwrap());
        let nocc = case["nocc"].as_u64().unwrap() as usize;
        let c = arr(&case["c_occ"]);
        let eps = arr(&case["e_occ"]);
        let (mut d, mut wm) = (vec![0.0; n * n], vec![0.0; n * n]);
        for i in 0..n {
            for j in 0..n {
                for k in 0..nocc {
                    d[i * n + j] += 2.0 * c[i * nocc + k] * c[j * nocc + k];
                    wm[i * n + j] -= 2.0 * eps[k] * c[i * nocc + k] * c[j * nocc + k];
                }
            }
        }
        let charges: Vec<(f64, [f64; 3])> = z
            .iter()
            .zip(&positions)
            .map(|(&q, &x)| (q as f64, x))
            .collect();
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
        let t0 = std::time::Instant::now();
        let mut g = oneshot::overlap_kinetic_grad(&basis, Some(&wm), Some(&d));
        let gv = oneshot::nuclear_attraction_grad(&basis, &pairs, &charges, &d);
        let gn = nuc_grad(&z, &positions);
        let g2 = if case["aux"].is_null() {
            let thr = std::env::var("LINUS_GRAD_THRESH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-12);
            oneshot::grad_4c(
                &basis,
                &pairs,
                &DensityTerms::default()
                    .coulomb(0.5, &d, &d)
                    .exchange(-0.25 * 1.0, &d, &d),
                thr,
                &linus_ints::operator::Operator::Coulomb,
            )
        } else {
            let adef = BasisDef::load(case["aux"].as_str().unwrap(), None).unwrap();
            let aux = BasisSet::new(&adef, &z, &positions, true).unwrap();
            let naux = aux.nbf;
            let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
            let v = linus_ints::reference::eri_2c(&aux, &ap);
            let j3 = linus_ints::reference::eri_3c(&basis, &pairs, &aux, &ap);
            let l = cholesky(&v, naux);
            let n2 = n * n;
            // c = V^-1 (P|mn) D_mn
            let mut cv: Vec<f64> = (0..naux)
                .map(|p| {
                    j3[p * n2..(p + 1) * n2]
                        .iter()
                        .zip(&d)
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect();
            chol_solve(&l, naux, &mut cv, 1);
            // B~ = V^-1 (P|mn)
            let mut bt = j3.clone();
            chol_solve(&l, naux, &mut bt, n2);
            // Γ_K = -1/2 D B~_P D ; γ = -1/2 c c^T + 1/4 tr(B~_P D B~_Q D)
            let mut gk = vec![0.0; naux * n2];
            for p in 0..naux {
                let x = matmul(&matmul(&d, &bt[p * n2..(p + 1) * n2], n), &d, n);
                for (o, y) in gk[p * n2..(p + 1) * n2].iter_mut().zip(&x) {
                    *o = -0.5 * y;
                }
            }
            let mut gam = vec![0.0; naux * naux];
            for p in 0..naux {
                for q in 0..naux {
                    let tr: f64 = gk[p * n2..(p + 1) * n2]
                        .iter()
                        .zip(&bt[q * n2..(q + 1) * n2])
                        .map(|(a, b)| a * b)
                        .sum();
                    gam[p * naux + q] = -0.5 * cv[p] * cv[q] - 0.5 * tr;
                }
            }
            let dens = Dens3c {
                full: Some(&gk),
                aux_shells: 0..aux.shells.len(),
                separable: vec![(1.0, &cv, &d)],
            };
            let g3 = linus_ints::reference::eri_grad_3c(
                &basis,
                &pairs,
                &aux,
                &ap,
                &dens,
                1e-13,
                &linus_ints::operator::Operator::Coulomb,
            );
            let g2c = linus_ints::reference::oneshot::grad_2c(
                &aux,
                &ap,
                &gam,
                1e-13,
                &linus_ints::operator::Operator::Coulomb,
            );
            g3.iter()
                .zip(&g2c)
                .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                .collect()
        };
        for i in 0..z.len() {
            for k in 0..3 {
                g[i][k] += gv[i][k] + gn[i][k] + g2[i][k];
            }
        }
        let t = t0.elapsed().as_secs_f64();
        let r = arr(&case["grad"]);
        let e = check(&format!("RHF gradient {name}"), &flat(&g), &r, 1e-9);
        println!("SUMMARY pyscf RHF gradient {name:45} max|diff| {e:.2e}  ({t:.3} s)");
    }
}
