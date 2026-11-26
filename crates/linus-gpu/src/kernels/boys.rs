//! The Boys function table every three-center kernel interpolates from (the generated
//! integral, derivative and PCM kernels), computed exactly as linus-ints computes its own, so
//! the device's Boys values are the CPU's.

use crate::cpu::boys_series;

/// Rows of the table: F_m(T) at T_i = 0.1 i, i < 1172.
pub const BOYS_ROWS: usize = 1172;
/// Columns: F_0..F_25 and exp(-T).
pub const BOYS_COLS: usize = 27;
/// Highest total angular momentum l_a + l_b + l_k the table supports (as linus-ints).
pub const MAX_LT: usize = 18;
/// The kernels interpolate the table below this T (the nearest row and the Taylor terms above
/// it lie within [`BOYS_ROWS`]) and take the asymptotic form from here on (as linus-ints'
/// eri3c and tools/intgen/mdmath.py's `BOYS_TMAX`).
pub const BOYS_TMAX: f64 = 117.0;

/// The Boys table (row-major, BOYS_ROWS x BOYS_COLS), computed as linus-ints computes its own.
pub fn boys_table() -> Vec<f64> {
    let mut f = vec![0.0; BOYS_ROWS * BOYS_COLS];
    for i in 0..BOYS_ROWS {
        let t = i as f64 * 0.1;
        let row = &mut f[i * BOYS_COLS..(i + 1) * BOYS_COLS];
        boys_series(BOYS_COLS - 2, t, &mut row[..BOYS_COLS - 1]);
        row[BOYS_COLS - 1] = (-t).exp();
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boys_table_matches_the_cpu_kernels_table_layout() {
        let t = boys_table();
        // F_0(0) = 1, F_m(0) = 1/(2m+1), exp column
        assert!((t[0] - 1.0).abs() < 1e-15);
        assert!((t[3] - 1.0 / 7.0).abs() < 1e-15);
        assert!((t[BOYS_COLS + BOYS_COLS - 1] - (-0.1f64).exp()).abs() < 1e-16);
        assert_eq!(t.len(), BOYS_ROWS * BOYS_COLS);
    }
}
