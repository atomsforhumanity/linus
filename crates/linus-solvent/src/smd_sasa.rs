//! Exact solvent-accessible surface area (SASA) of a union of spheres, with analytic first and
//! second derivatives with respect to the sphere centres.
//!
//! The free (accessible) part of sphere k is the unit sphere minus the union of the caps cut by
//! the overlapping spheres j: cap j = {x : x.n_j >= c_j} with n_j = (r_j - r_k)/d_j and
//! c_j = (R_k^2 + d_j^2 - R_j^2) / (2 R_k d_j). Its solid angle follows from Gauss-Bonnet,
//!
//!   A = 2 pi chi + sum_j c_j L_j - sum_v angle_v,       chi = 2 N_components - N_loops,
//!
//! where L_j is the free azimuth on circle j and angle_v the angle between the two circles at
//! each boundary vertex. The number of free components is not traced: A lies in (0, 4 pi), so it
//! is fixed by reducing the rest modulo 4 pi, the device Liotard's DAREAL uses for nested
//! polygons. The boundary is found by sweeping the intersection points around each circle
//! (entry/exit of the other caps, a depth count), and boundary loops are counted by following
//! arc ends to the next arc's start.
//!
//! Derivatives use the boundary-variation identity (only the boundary moves, vertices drop out):
//!
//!   dA = sum_j [ -dn_j . W_j + dc_j L_j ],   W_j = int_free(j) x dphi = c_j L_j n_j + n_j x P_j,
//!
//! with P_j = sum over free arcs (start - end). In terms of d_j = r_j - r_k this gives
//! dA/dd_j = -(n_j x P_j)/|d_j| + c'_j L_j n_j. The Hessian differentiates that expression,
//! including the motion of the arc end points (each the intersection of two circles, whose
//! Jacobian follows from x.n_j = c_j, x.n_p = c_p, x.x = 1).
//!
//! Smoothness of the exact SASA (inherent, any algorithm): the area is continuous, but its
//! gradient jumps when two spheres start to touch (a new cap's area grows linearly with the
//! overlap), and its Hessian diverges as delta^-1/2 when two caps on a sphere start to overlap
//! (lens area ~ delta^3/2). Away from such events everything is analytic.
//!
//! Degenerate configurations (a boundary point shared by three caps) are regularized as in
//! DAREAL: the sphere's radius is increased by a relative 4e-11 and the sphere redone.
//!
//! Written for linus from the geometry above; the modulo-4 pi device and the radius
//! regularization follow D. Liotard's DAREAL (Minnesota/NWChem mnsol.F, ECL-2.0), whose SASA
//! this reproduces to ~1e-13 relative (bench/results/2026-11-07-smd-cds).

use rayon::prelude::*;

pub(crate) type V3 = [f64; 3];
pub(crate) type M3 = [[f64; 3]; 3];

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

#[inline]
pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub(crate) fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub(crate) fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub(crate) fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub(crate) fn outer(a: V3, b: V3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i] * b[j];
        }
    }
    m
}
/// skew(a) v = a x v
#[inline]
fn skew(a: V3) -> M3 {
    [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
}
#[inline]
fn matmul(a: &M3, b: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    m
}
/// vᵀ M (row vector times matrix)
#[inline]
fn vecmat(v: V3, m: &M3) -> V3 {
    [
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    ]
}
/// I - n nᵀ
#[inline]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn proj(n: V3) -> M3 {
    let mut m = outer(n, n);
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = if i == j { 1.0 } else { 0.0 } - m[i][j];
        }
    }
    m
}

/// Accessible area of one sphere and its derivatives.
#[derive(Clone, Debug, Default)]
pub struct SphereSasa {
    /// Accessible area, R^2 times the free solid angle (units of the radii squared).
    pub area: f64,
    /// Atoms whose spheres cut this one (the derivative variables).
    pub neighbors: Vec<usize>,
    /// dA/dr_j for each neighbor j (same order); dA/dr_self = -sum of these.
    pub grad: Vec<V3>,
    /// d2A/(dr_a dr_b) over the neighbors, (3m x 3m) row-major; empty unless order >= 2.
    /// Blocks involving the sphere itself follow from translation invariance.
    pub hess: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Cap {
    atom: usize,
    n: V3,
    c: f64,
    s: f64,
    d: f64,
    /// dc/dd and d2c/dd2 at fixed radii
    cp: f64,
    cpp: f64,
}

#[derive(Clone, Copy)]
struct Event {
    phi: f64,
    entry: bool,
    other: usize,
    key: (usize, usize, bool),
    p: V3,
}

#[derive(Clone, Copy)]
struct Arc {
    circle: usize,
    start_other: usize,
    end_other: usize,
    start_key: (usize, usize, bool),
    end_key: (usize, usize, bool),
    a: V3,
    b: V3,
    dphi: f64,
}

enum Outcome {
    Done(SphereSasa),
    Degenerate,
}

/// Accessible area of sphere `k` (radius `rad[k]`, centre `pos[k]`) in the union of all spheres;
/// spheres with radius <= 0 are ignored. `order` 0: area, 1: + gradient, 2: + Hessian.
/// `contact_width` > 0 smooths the onset of each cap (see [`contact_distance`]); 0 is the exact
/// SASA.
pub fn sphere_sasa(
    k: usize,
    pos: &[V3],
    rad: &[f64],
    order: usize,
    contact_width: f64,
) -> SphereSasa {
    if rad[k] <= 0.0 {
        return SphereSasa::default();
    }
    let mut rk = rad[k];
    for _ in 0..64 {
        match sphere_sasa_r(k, rk, pos, rad, order, contact_width) {
            Outcome::Done(mut s) => {
                // area and derivatives scale with the true radius squared (the perturbed radius
                // only resolves a degenerate topology)
                let f = (rad[k] / rk) * (rad[k] / rk);
                if f != 1.0 {
                    s.area *= f;
                    s.grad.iter_mut().for_each(|g| *g = scale(*g, f));
                    s.hess.iter_mut().for_each(|h| *h *= f);
                }
                return s;
            }
            Outcome::Degenerate => rk *= 1.0 + 4e-11,
        }
    }
    panic!("smd_sasa: sphere {k} stays degenerate after 64 radius perturbations");
}

fn sphere_sasa_r(
    k: usize,
    rk: f64,
    pos: &[V3],
    rad: &[f64],
    order: usize,
    contact_width: f64,
) -> Outcome {
    let mut caps: Vec<Cap> = Vec::new();
    for j in 0..pos.len() {
        let rj = rad[j];
        if j == k || rj <= 0.0 {
            continue;
        }
        let dv = sub(pos[j], pos[k]);
        let d = dot(dv, dv).sqrt();
        if d >= rk + rj {
            continue;
        }
        if d <= rj - rk {
            // sphere k inside sphere j
            return Outcome::Done(SphereSasa::default());
        }
        if d <= rk - rj {
            continue; // sphere j inside sphere k: no cap
        }
        let n = scale(dv, 1.0 / d);
        // cap of the sphere at distance de: c = (rk^2 + de^2 - rj^2) / (2 rk de) and derivatives,
        // chained through de(d) (de = d unless the contact is smoothed)
        let (de, s1, s2) = contact_distance(d, rk + rj, contact_width);
        let c = (rk * rk + de * de - rj * rj) / (2.0 * rk * de);
        let q = (rk * rk - rj * rj) / (2.0 * rk);
        let c1 = 0.5 / rk - q / (de * de);
        let c2 = 2.0 * q / (de * de * de);
        let (cp, cpp) = (c1 * s1, c2 * s1 * s1 + c1 * s2);
        caps.push(Cap {
            atom: j,
            n,
            c,
            s: (1.0 - c * c).max(0.0).sqrt(),
            d,
            cp,
            cpp,
        });
    }
    let full_area = rk * rk * FOUR_PI;
    if caps.is_empty() {
        return Outcome::Done(SphereSasa {
            area: full_area,
            ..Default::default()
        });
    }
    let m = caps.len();

    // frames for azimuths
    let frames: Vec<(V3, V3)> = caps
        .iter()
        .map(|cap| {
            let n = cap.n;
            let a = if n[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let u = sub(a, scale(n, dot(a, n)));
            let u = scale(u, 1.0 / dot(u, u).sqrt());
            (u, cross(n, u))
        })
        .collect();

    // intersection points of all circle pairs
    let mut events: Vec<Vec<Event>> = vec![Vec::new(); m];
    for a in 0..m {
        for b in a + 1..m {
            let (na, ca) = (caps[a].n, caps[a].c);
            let (nb, cb) = (caps[b].n, caps[b].c);
            let g = dot(na, nb);
            let den = 1.0 - g * g;
            if den <= 1e-28 {
                if (caps[a].c - caps[b].c).abs() < 1e-14 && g > 0.0 {
                    return Outcome::Degenerate; // coincident circles
                }
                continue;
            }
            let al = (ca - g * cb) / den;
            let be = (cb - g * ca) / den;
            let h2 = (1.0 - al * ca - be * cb) / den;
            if h2 <= 0.0 {
                continue;
            }
            let h = h2.sqrt();
            let w = cross(na, nb);
            let base = add(scale(na, al), scale(nb, be));
            for plus in [true, false] {
                let p = add(base, scale(w, if plus { h } else { -h }));
                let key = (a, b, plus);
                // on circle a (lower index) p+ leaves cap b; on circle b p+ enters cap a
                for (ci, other, entry) in [(a, b, !plus), (b, a, plus)] {
                    let (u, v) = frames[ci];
                    let phi = dot(p, v).atan2(dot(p, u)).rem_euclid(TWO_PI);
                    events[ci].push(Event {
                        phi,
                        entry,
                        other,
                        key,
                        p,
                    });
                }
            }
        }
    }

    let inside_any = |x: V3, skip: usize| -> usize {
        caps.iter()
            .enumerate()
            .filter(|(t, cap)| *t != skip && dot(x, cap.n) > cap.c)
            .count()
    };

    let mut arcs: Vec<Arc> = Vec::new();
    let mut full: Vec<usize> = Vec::new();
    for i in 0..m {
        let ev = &mut events[i];
        let (n, c, s) = (caps[i].n, caps[i].c, caps[i].s);
        let (u, v) = frames[i];
        if ev.is_empty() {
            let x = add(scale(n, c), scale(u, s));
            if inside_any(x, i) == 0 {
                full.push(i);
            }
            continue;
        }
        ev.sort_by(|x, y| x.phi.total_cmp(&y.phi).then(x.key.cmp(&y.key)));
        let ne = ev.len();
        let phi0 = ev[0].phi;
        let mut phi1 = ev[1 % ne].phi;
        if phi1 <= phi0 {
            phi1 += TWO_PI;
        }
        let mid = 0.5 * (phi0 + phi1);
        let x = add(
            scale(n, c),
            scale(add(scale(u, mid.cos()), scale(v, mid.sin())), s),
        );
        let mut depth = inside_any(x, i) as i64;
        for q in 0..ne {
            if q > 0 {
                depth += if ev[q].entry { 1 } else { -1 };
            }
            if depth < 0 {
                return Outcome::Degenerate;
            }
            if depth == 0 {
                let e0 = ev[q];
                let e1 = ev[(q + 1) % ne];
                if e0.entry || !e1.entry {
                    return Outcome::Degenerate;
                }
                let mut dphi = e1.phi - e0.phi;
                if q + 1 == ne {
                    dphi += TWO_PI;
                }
                if dphi <= 0.0 {
                    return Outcome::Degenerate;
                }
                arcs.push(Arc {
                    circle: i,
                    start_other: e0.other,
                    end_other: e1.other,
                    start_key: e0.key,
                    end_key: e1.key,
                    a: e0.p,
                    b: e1.p,
                    dphi,
                });
            }
        }
    }
    if arcs.is_empty() && full.is_empty() {
        // caps exist but no free boundary: the sphere is buried
        return Outcome::Done(SphereSasa {
            area: 0.0,
            ..Default::default()
        });
    }

    // boundary loops: each arc's end vertex starts exactly one arc
    let mut starts: Vec<((usize, usize, bool), usize)> = arcs
        .iter()
        .enumerate()
        .map(|(t, a)| (a.start_key, t))
        .collect();
    starts.sort_by_key(|x| x.0);
    for w in starts.windows(2) {
        if w[0].0 == w[1].0 {
            return Outcome::Degenerate;
        }
    }
    let mut seen = vec![false; arcs.len()];
    let mut nloops = full.len();
    for t in 0..arcs.len() {
        if seen[t] {
            continue;
        }
        nloops += 1;
        let mut q = t;
        while !seen[q] {
            seen[q] = true;
            match starts.binary_search_by(|x| x.0.cmp(&arcs[q].end_key)) {
                Ok(pos) => q = starts[pos].1,
                Err(_) => return Outcome::Degenerate,
            }
        }
        if q != t {
            return Outcome::Degenerate;
        }
    }

    // Gauss-Bonnet
    let mut len = vec![0.0; m];
    let mut pvec = vec![[0.0; 3]; m];
    for &i in &full {
        len[i] += TWO_PI;
    }
    let mut angles = 0.0;
    for arc in &arcs {
        len[arc.circle] += arc.dphi;
        pvec[arc.circle] = add(pvec[arc.circle], sub(arc.a, arc.b));
        let (ci, co) = (&caps[arc.circle], &caps[arc.end_other]);
        let cosang = (dot(ci.n, co.n) - ci.c * co.c) / (ci.s * co.s);
        angles += cosang.clamp(-1.0, 1.0).acos();
    }
    let mut sum_cl = 0.0;
    for i in 0..m {
        sum_cl += caps[i].c * len[i];
    }
    let rest = sum_cl - angles - TWO_PI * nloops as f64;
    let mut a = rest.rem_euclid(FOUR_PI);
    // resolve the 0 / 4 pi ambiguity with bounds: every cap is buried
    let max_cap = caps
        .iter()
        .map(|c| TWO_PI * (1.0 - c.c))
        .fold(0.0, f64::max);
    let sum_cap: f64 = caps.iter().map(|c| TWO_PI * (1.0 - c.c)).sum();
    let tol = 1e-9;
    if a > FOUR_PI - max_cap + tol {
        a -= FOUR_PI;
    } else if a < FOUR_PI - sum_cap - tol {
        a += FOUR_PI;
    }
    let a = a.clamp(0.0, FOUR_PI);
    let r2 = rk * rk;
    let mut out = SphereSasa {
        area: r2 * a,
        neighbors: caps.iter().map(|c| c.atom).collect(),
        ..Default::default()
    };
    if order == 0 {
        return Outcome::Done(out);
    }

    // gradient: dA/dd_i = -(n_i x P_i)/d_i + c'_i L_i n_i
    out.grad = (0..m)
        .map(|i| {
            let cap = &caps[i];
            let g = add(
                scale(cross(cap.n, pvec[i]), -1.0 / cap.d),
                scale(cap.n, cap.cp * len[i]),
            );
            scale(g, r2)
        })
        .collect();
    if order == 1 {
        return Outcome::Done(out);
    }

    // Hessian over the d_i variables
    let dim = 3 * m;
    let mut h = vec![0.0; dim * dim];
    let add_block = |h: &mut Vec<f64>, i: usize, j: usize, b: &M3, f: f64| {
        for x in 0..3 {
            for y in 0..3 {
                h[(3 * i + x) * dim + 3 * j + y] += f * b[x][y];
            }
        }
    };
    // own-geometry part of c' L n with L fixed: L (c'' n nᵀ + c' (I - n nᵀ)/d)
    let own_cl = |cap: &Cap, l: f64| -> M3 {
        let nn = outer(cap.n, cap.n);
        let p = proj(cap.n);
        let mut b = [[0.0; 3]; 3];
        for x in 0..3 {
            for y in 0..3 {
                b[x][y] = l * (cap.cpp * nn[x][y] + cap.cp * p[x][y] / cap.d);
            }
        }
        b
    };
    for &i in &full {
        let b = own_cl(&caps[i], TWO_PI);
        add_block(&mut h, i, i, &b, 1.0);
    }
    // vertex Jacobians: x on circles i and o; returns (dx/dd_i, dx/dd_o)
    let vjac = |x: V3, i: usize, o: usize| -> (M3, M3) {
        let (ci, co) = (&caps[i], &caps[o]);
        let no_x = cross(co.n, x);
        let det = dot(ci.n, no_x);
        let m1 = scale(no_x, 1.0 / det);
        let m2 = scale(cross(x, ci.n), 1.0 / det);
        let ei = sub(
            scale(ci.n, ci.cp),
            scale(sub(x, scale(ci.n, ci.c)), 1.0 / ci.d),
        );
        let eo = sub(
            scale(co.n, co.cp),
            scale(sub(x, scale(co.n, co.c)), 1.0 / co.d),
        );
        (outer(m1, ei), outer(m2, eo))
    };
    for arc in &arcs {
        let i = arc.circle;
        let cap = &caps[i];
        let (n, c, d, cp) = (cap.n, cap.c, cap.d, cap.cp);
        let (a, b) = (arc.a, arc.b);
        let (p, q) = (arc.start_other, arc.end_other);
        let (ja_i, ja_p) = vjac(a, i, p);
        let (jb_i, jb_q) = vjac(b, i, q);
        let pn = proj(n);
        // dphi derivatives: dphi = atan2(Y, X), Y = n.(a x b), X = a.b - c^2
        let axb = cross(a, b);
        let y = dot(n, axb);
        let xx = dot(a, b) - c * c;
        let r = xx * xx + y * y;
        let (wy, wx) = (xx / r, y / r);
        let bxn = cross(b, n);
        let nxa = cross(n, a);
        let ta = sub(scale(bxn, wy), scale(b, wx)); // coefficient of da
        let tb = sub(scale(nxa, wy), scale(a, wx)); // coefficient of db
        let mut dphi_i = scale(vecmat(axb, &pn), wy / d);
        dphi_i = add(dphi_i, scale(n, wx * 2.0 * c * cp));
        dphi_i = add(dphi_i, vecmat(ta, &ja_i));
        dphi_i = add(dphi_i, vecmat(tb, &jb_i));
        let dphi_p = vecmat(ta, &ja_p);
        let dphi_q = vecmat(tb, &jb_q);
        let sn = skew(n);
        // own block
        let amb = sub(a, b);
        let mut bi = matmul(&skew(amb), &pn);
        let t2 = outer(cross(n, amb), n);
        for x in 0..3 {
            for z in 0..3 {
                bi[x][z] = (bi[x][z] + t2[x][z]) / (d * d);
            }
        }
        let bcl = own_cl(cap, arc.dphi);
        let mut jdiff = ja_i;
        for x in 0..3 {
            for z in 0..3 {
                jdiff[x][z] -= jb_i[x][z];
            }
        }
        let sj = matmul(&sn, &jdiff);
        let ndp = outer(n, dphi_i);
        for x in 0..3 {
            for z in 0..3 {
                bi[x][z] += bcl[x][z] - sj[x][z] / d + cp * ndp[x][z];
            }
        }
        add_block(&mut h, i, i, &bi, 1.0);
        // start vertex: other circle p
        let sp = matmul(&sn, &ja_p);
        let ndp = outer(n, dphi_p);
        let mut bp = [[0.0; 3]; 3];
        for x in 0..3 {
            for z in 0..3 {
                bp[x][z] = -sp[x][z] / d + cp * ndp[x][z];
            }
        }
        add_block(&mut h, i, p, &bp, 1.0);
        // end vertex: other circle q
        let sq = matmul(&sn, &jb_q);
        let ndq = outer(n, dphi_q);
        let mut bq = [[0.0; 3]; 3];
        for x in 0..3 {
            for z in 0..3 {
                bq[x][z] = sq[x][z] / d + cp * ndq[x][z];
            }
        }
        add_block(&mut h, i, q, &bq, 1.0);
    }
    h.iter_mut().for_each(|v| *v *= r2);
    out.hess = h;
    Outcome::Done(out)
}

/// Accessible areas of all spheres (rayon over spheres; the result is independent of the
/// thread count because every sphere is computed sequentially).
pub fn sasa(pos: &[V3], rad: &[f64], order: usize, contact_width: f64) -> Vec<SphereSasa> {
    (0..pos.len())
        .into_par_iter()
        .map(|k| sphere_sasa(k, pos, rad, order, contact_width))
        .collect()
}

/// Effective centre distance for a cap, with its first and second derivatives with respect to the
/// true distance `d`. Exact SASA (`width` = 0): `d` itself. With `width` w > 0, a contact
/// with overlap delta = `touch` - d < w is softened: de = touch - w S(delta/w), with
/// S(s) = 6 s^3 - 8 s^4 + 3 s^5 (S(0) = S'(0) = S''(0) = 0, S(1) = 1, S'(1) = 1, S''(1) = 0), so a
/// new cap grows from zero with zero slope and curvature instead of linearly. This removes the
/// gradient jump of the exact SASA at sphere contact (C2 there) and changes nothing for
/// overlaps >= w.
pub fn contact_distance(d: f64, touch: f64, width: f64) -> (f64, f64, f64) {
    let delta = touch - d;
    if width <= 0.0 || delta >= width {
        return (d, 1.0, 0.0);
    }
    let s = delta / width;
    let (s2, s3) = (s * s, s * s * s);
    let f = width * (6.0 * s3 - 8.0 * s2 * s2 + 3.0 * s3 * s2);
    let f1 = 18.0 * s2 - 32.0 * s3 + 15.0 * s2 * s2; // dS/ddelta
    let f2 = (36.0 * s - 96.0 * s2 + 60.0 * s3) / width; // d2S/ddelta2
                                                         // de = touch - f(delta), ddelta/dd = -1
    (touch - f, f1, -f2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(pos: &[V3], rad: &[f64]) -> f64 {
        sasa(pos, rad, 0, 0.0).iter().map(|s| s.area).sum()
    }

    /// full gradient and Hessian of the total area (3N, 3N x 3N)
    fn full_derivs(pos: &[V3], rad: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nat = pos.len();
        let n3 = 3 * nat;
        let mut g = vec![0.0; n3];
        let mut h = vec![0.0; n3 * n3];
        for (k, s) in sasa(pos, rad, 2, 0.0).iter().enumerate() {
            let m = s.neighbors.len();
            let idx = |t: usize| if t == m { k } else { s.neighbors[t] };
            for t in 0..m {
                for x in 0..3 {
                    g[3 * s.neighbors[t] + x] += s.grad[t][x];
                    g[3 * k + x] -= s.grad[t][x];
                }
            }
            // d variables: d_t = r_t - r_k; chain to coordinates
            for t in 0..=m {
                for u in 0..=m {
                    for x in 0..3 {
                        for y in 0..3 {
                            let mut v = 0.0;
                            for a in 0..m {
                                for b in 0..m {
                                    let fa = if t == m {
                                        -1.0
                                    } else if a == t {
                                        1.0
                                    } else {
                                        0.0
                                    };
                                    let fb = if u == m {
                                        -1.0
                                    } else if b == u {
                                        1.0
                                    } else {
                                        0.0
                                    };
                                    if fa != 0.0 && fb != 0.0 {
                                        v += fa * fb * s.hess[(3 * a + x) * 3 * m + 3 * b + y];
                                    }
                                }
                            }
                            h[(3 * idx(t) + x) * n3 + 3 * idx(u) + y] += v;
                        }
                    }
                }
            }
        }
        (g, h)
    }

    fn cluster() -> (Vec<V3>, Vec<f64>) {
        // a non-symmetric cluster with several overlapping spheres (multiple boundary loops)
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [-0.9, 0.7, 0.3],
            [0.3, -1.0, 0.6],
            [0.2, 0.4, 1.2],
            [-0.4, -0.3, -1.1],
            [2.0, 1.0, 0.5],
        ];
        let rad = vec![1.6, 1.3, 1.5, 1.2, 1.4, 1.35, 1.7];
        (pos, rad)
    }

    #[test]
    fn two_spheres_closed_form() {
        let (r1, r2, d) = (1.5f64, 1.2f64, 2.0f64);
        let pos = vec![[0.0, 0.0, 0.0], [d, 0.0, 0.0]];
        let a = total(&pos, &[r1, r2]);
        let c1 = (r1 * r1 + d * d - r2 * r2) / (2.0 * r1 * d);
        let c2 = (r2 * r2 + d * d - r1 * r1) / (2.0 * r2 * d);
        let exact = TWO_PI * r1 * r1 * (1.0 + c1) + TWO_PI * r2 * r2 * (1.0 + c2);
        assert!((a - exact).abs() < 1e-12, "{a} {exact}");
    }

    #[test]
    fn gradient_and_hessian_match_finite_differences() {
        let (pos, rad) = cluster();
        let (g, h) = full_derivs(&pos, &rad);
        let n3 = 3 * pos.len();
        let step = 1e-5;
        let mut max_g: f64 = 0.0;
        let mut max_h: f64 = 0.0;
        for q in 0..n3 {
            let mut pp = pos.clone();
            let mut pm = pos.clone();
            pp[q / 3][q % 3] += step;
            pm[q / 3][q % 3] -= step;
            let fd = (total(&pp, &rad) - total(&pm, &rad)) / (2.0 * step);
            max_g = max_g.max((fd - g[q]).abs());
            let (gp, _) = full_derivs(&pp, &rad);
            let (gm, _) = full_derivs(&pm, &rad);
            for r in 0..n3 {
                let fdh = (gp[r] - gm[r]) / (2.0 * step);
                max_h = max_h.max((fdh - h[q * n3 + r]).abs());
            }
        }
        assert!(max_g < 1e-8, "gradient error {max_g}");
        assert!(max_h < 1e-6, "hessian error {max_h}");
        // symmetry
        let mut asym: f64 = 0.0;
        for q in 0..n3 {
            for r in 0..n3 {
                asym = asym.max((h[q * n3 + r] - h[r * n3 + q]).abs());
            }
        }
        assert!(asym < 1e-10, "hessian asymmetry {asym}");
    }

    #[test]
    fn buried_and_free_spheres() {
        // a small sphere inside a big one has no area; the big one keeps its full sphere
        let pos = vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]];
        let s = sasa(&pos, &[2.0, 1.0], 1, 0.0);
        assert!((s[0].area - FOUR_PI * 4.0).abs() < 1e-12);
        assert_eq!(s[1].area, 0.0);
        // a central sphere covered by six neighbours on the axes
        let mut pos = vec![[0.0, 0.0, 0.0]];
        for x in 0..3 {
            for sgn in [-1.0, 1.0] {
                let mut p = [0.0; 3];
                p[x] = sgn;
                pos.push(p);
            }
        }
        let rad = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let s = sasa(&pos, &rad, 1, 0.0);
        assert_eq!(s[0].area, 0.0);
    }

    #[test]
    fn ring_gives_two_components() {
        // a sphere in the centre of a planar ring of six spheres: free top and bottom caps
        let mut pos = vec![[0.0, 0.0, 0.0]];
        for t in 0..6 {
            let a = t as f64 * std::f64::consts::PI / 3.0;
            pos.push([1.5 * a.cos(), 1.5 * a.sin(), 0.0]);
        }
        let rad = vec![1.2; 7];
        let s = sasa(&pos, &rad, 0, 0.0);
        // compare with a fine Monte-Carlo-free quadrature: integrate over the sphere in (theta, phi)
        let (nt, np) = (2000, 2000);
        let mut free = 0.0;
        for it in 0..nt {
            let th = (it as f64 + 0.5) * std::f64::consts::PI / nt as f64;
            for ip in 0..np {
                let ph = (ip as f64 + 0.5) * TWO_PI / np as f64;
                let x = [
                    1.2 * th.sin() * ph.cos(),
                    1.2 * th.sin() * ph.sin(),
                    1.2 * th.cos(),
                ];
                let buried = pos[1..].iter().any(|p| {
                    let d = sub(x, *p);
                    dot(d, d) < 1.44
                });
                if !buried {
                    free += th.sin();
                }
            }
        }
        let quad = free * 1.44 * (std::f64::consts::PI / nt as f64) * (TWO_PI / np as f64);
        assert!(
            (s[0].area - quad).abs() < 2e-3 * quad,
            "{} {quad}",
            s[0].area
        );
        assert!(s[0].area > 0.0 && s[0].area < 0.5 * FOUR_PI * 1.44);
    }
}
