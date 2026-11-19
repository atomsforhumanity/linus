#![allow(dead_code)]
use linus_grid::MolecularGrid;
use serde_json::Value;
use std::collections::HashMap;

pub fn load(name: &str) -> Value {
    let p = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    serde_json::from_str(&s).unwrap()
}

pub fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

pub fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

pub fn atoms(case: &Value) -> (Vec<u32>, Vec<[f64; 3]>) {
    let z = usizes(&case["z"]).into_iter().map(|v| v as u32).collect();
    let xyz = case["xyz"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let r = f64s(r);
            [r[0], r[1], r[2]]
        })
        .collect();
    (z, xyz)
}

/// Max deviations between reference sample points and the grid (matched by coordinates).
pub struct SampleCmp {
    pub n: usize,
    pub max_dx: f64,
    pub max_dw_abs: f64,
    pub max_dw_rel: f64,
    /// max |dw| / w_atomic: deviation of the partition function value P_A/Z
    pub max_dw_part: f64,
    pub unmatched: usize,
    pub atom_mismatch: usize,
}

pub struct PointIndex {
    map: HashMap<(i64, i64, i64), Vec<usize>>,
    h: f64,
}

impl PointIndex {
    pub fn new(g: &MolecularGrid) -> Self {
        let h = 1e-6;
        let mut map: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::with_capacity(g.len());
        for i in 0..g.len() {
            let k = (
                (g.x[i] / h).floor() as i64,
                (g.y[i] / h).floor() as i64,
                (g.z[i] / h).floor() as i64,
            );
            map.entry(k).or_default().push(i);
        }
        PointIndex { map, h }
    }
    /// index of the grid point (owned by `atom`) closest to p within 1e-9
    pub fn find(&self, g: &MolecularGrid, p: [f64; 3], atom: Option<u32>) -> Option<usize> {
        let k = (
            (p[0] / self.h).floor() as i64,
            (p[1] / self.h).floor() as i64,
            (p[2] / self.h).floor() as i64,
        );
        let mut best: Option<(usize, f64)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.map.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        for &i in v {
                            if let Some(a) = atom {
                                if g.atom[i] != a {
                                    continue;
                                }
                            }
                            let d = ((g.x[i] - p[0]).powi(2)
                                + (g.y[i] - p[1]).powi(2)
                                + (g.z[i] - p[2]).powi(2))
                            .sqrt();
                            if d < 1e-9 && best.is_none_or(|b| d < b.1) {
                                best = Some((i, d));
                            }
                        }
                    }
                }
            }
        }
        best.map(|b| b.0)
    }
}

pub fn compare_sample(g: &MolecularGrid, sample: &Value, use_atom: bool) -> SampleCmp {
    let x = f64s(&sample["x"]);
    let y = f64s(&sample["y"]);
    let z = f64s(&sample["z"]);
    let w = f64s(&sample["w"]);
    let atom: Vec<usize> = sample["atom"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap().max(-1) as usize)
        .collect();
    let idx = PointIndex::new(g);
    let mut c = SampleCmp {
        n: x.len(),
        max_dx: 0.0,
        max_dw_abs: 0.0,
        max_dw_rel: 0.0,
        max_dw_part: 0.0,
        unmatched: 0,
        atom_mismatch: 0,
    };
    for k in 0..x.len() {
        let a = if use_atom { Some(atom[k] as u32) } else { None };
        match idx.find(g, [x[k], y[k], z[k]], a) {
            None => c.unmatched += 1,
            Some(i) => {
                if atom[k] as i64 >= 0 && g.atom[i] as usize != atom[k] {
                    c.atom_mismatch += 1;
                }
                let dx = (g.x[i] - x[k])
                    .abs()
                    .max((g.y[i] - y[k]).abs())
                    .max((g.z[i] - z[k]).abs());
                c.max_dx = c.max_dx.max(dx);
                let dw = (g.w[i] - w[k]).abs();
                c.max_dw_abs = c.max_dw_abs.max(dw);
                c.max_dw_part = c.max_dw_part.max(dw / g.w_atomic[i].abs().max(1e-300));
                if w[k].abs() > 1e-300 {
                    c.max_dw_rel = c.max_dw_rel.max(dw / w[k].abs());
                } else if dw > 0.0 {
                    c.max_dw_rel = c.max_dw_rel.max(dw / 1e-300);
                }
            }
        }
    }
    c
}

/// Kahan-compensated per-atom sums of w, w*x, w*y, w*z, w*|r-R_A|^2
pub fn per_atom_sums(g: &MolecularGrid) -> Vec<[f64; 6]> {
    let natm = g.atom_z.len();
    let mut s = vec![[0.0f64; 6]; natm];
    let mut c = vec![[0.0f64; 6]; natm];
    let mut n = vec![0usize; natm];
    for i in 0..g.len() {
        let a = g.atom[i] as usize;
        let r = g.atom_xyz[a];
        let d2 = (g.x[i] - r[0]).powi(2) + (g.y[i] - r[1]).powi(2) + (g.z[i] - r[2]).powi(2);
        let v = [
            g.w[i],
            g.w[i] * g.x[i],
            g.w[i] * g.y[i],
            g.w[i] * g.z[i],
            g.w[i] * d2,
            n[a] as f64,
        ];
        for k in 0..5 {
            let y = v[k] - c[a][k];
            let t = s[a][k] + y;
            c[a][k] = (t - s[a][k]) - y;
            s[a][k] = t;
        }
        n[a] += 1;
        s[a][5] = n[a] as f64;
    }
    s
}
