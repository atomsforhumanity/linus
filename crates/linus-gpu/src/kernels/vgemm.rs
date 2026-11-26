//! Variable-size batched FP64 GEMMs: C_p = alpha op(A_p) op(B_p) + beta C_p for many small
//! products of different shapes in one launch (the XC engines' per-batch products), one
//! thread block per tile of one product's C, from a tile list built on the host.
//!
//! On cards with little FP64 throughput (the FP64 pipe, not memory or issue, limits a GEMM)
//! a plain shared-memory tiled kernel runs these products at the card's FP64 rate, where
//! cuBLAS's grouped GEMM reaches about 70 % of it on such shapes. Every output element is computed by one thread in a fixed
//! order over k: bitwise reproducible.
//!
//! Descriptor of product p (nine i64 at `desc[9 p]`): m, n, k, lda, ldb, ldc and the device
//! addresses of A, B and C (column-major, as cuBLAS). Tile t (three i32 at `tiles[3 t]`):
//! product, tile row, tile column.

use std::fmt::Write as _;

/// A tile configuration: bm x bn outputs per thread block, k steps of bk, tx x ty threads
/// (each computing (bm / tx) x (bn / ty) outputs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VgemmTile {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub tx: usize,
    pub ty: usize,
}

impl VgemmTile {
    /// The default: 32 x 32 tiles, 128 threads of 2 x 4 outputs (the XC products' m and n are
    /// 100-800, so small tiles waste little at the edges).
    pub const DEFAULT: VgemmTile = VgemmTile {
        bm: 32,
        bn: 32,
        bk: 16,
        tx: 16,
        ty: 8,
    };

    pub fn threads(&self) -> usize {
        self.tx * self.ty
    }
}

/// The kernel name for transpositions (ta, tb) and a tile.
pub fn kernel_name(ta: bool, tb: bool, t: &VgemmTile) -> String {
    format!(
        "vgemm_{}{}_{}x{}x{}_{}x{}",
        if ta { 't' } else { 'n' },
        if tb { 't' } else { 'n' },
        t.bm,
        t.bn,
        t.bk,
        t.tx,
        t.ty
    )
}

/// CUDA C++ of one kernel (op(A) = A^T when `ta`, op(B) = B^T when `tb`).
pub fn generate(ta: bool, tb: bool, t: &VgemmTile) -> String {
    assert!(t.bm.is_multiple_of(t.tx) && t.bn.is_multiple_of(t.ty));
    let (rm, rn) = (t.bm / t.tx, t.bn / t.ty);
    let nt = t.threads();
    let name = kernel_name(ta, tb, t);
    let mut s = String::new();
    let (bm, bn, bk, tx, ty) = (t.bm, t.bn, t.bk, t.tx, t.ty);
    // A tile element (r, kk): op(A)(r0 + r, k0 + kk)
    let a_load = if ta {
        // A stored k x m: op(A)(r, kk) = A[kk + r lda]; consecutive threads along kk
        format!(
            "const int kk = e % {bk}, r = e / {bk};
            const int gr = r0 + r, gk = k0 + kk;
            As[kk][r] = (gr < m && gk < k) ? A[gk + (long long)gr * lda] : 0.0;"
        )
    } else {
        format!(
            "const int r = e % {bm}, kk = e / {bm};
            const int gr = r0 + r, gk = k0 + kk;
            As[kk][r] = (gr < m && gk < k) ? A[gr + (long long)gk * lda] : 0.0;"
        )
    };
    let b_load = if tb {
        // B stored n x k: op(B)(kk, c) = B[c + kk ldb]; consecutive threads along c
        format!(
            "const int c = e % {bn}, kk = e / {bn};
            const int gc = c0 + c, gk = k0 + kk;
            Bs[kk][c] = (gc < n && gk < k) ? B[gc + (long long)gk * ldb] : 0.0;"
        )
    } else {
        format!(
            "const int kk = e % {bk}, c = e / {bk};
            const int gc = c0 + c, gk = k0 + kk;
            Bs[kk][c] = (gc < n && gk < k) ? B[gk + (long long)gc * ldb] : 0.0;"
        )
    };
    let _ = write!(
        s,
        r#"
// {name}: variable-size batched DGEMM (linus-gpu src/kernels/vgemm.rs, generated)
extern "C" __global__ void __launch_bounds__({nt}) {name}(const int *__restrict__ tiles,
    const long long *__restrict__ desc, double alpha, double beta)
{{
    __shared__ double As[{bk}][{bm} + 1];
    __shared__ double Bs[{bk}][{bn} + 1];
    const int t = threadIdx.x;
    const int p = tiles[3 * blockIdx.x];
    const int r0 = tiles[3 * blockIdx.x + 1] * {bm}, c0 = tiles[3 * blockIdx.x + 2] * {bn};
    const long long *d = desc + 9LL * p;
    const int m = (int)d[0], n = (int)d[1], k = (int)d[2];
    const long long lda = d[3], ldb = d[4], ldc = d[5];
    const double *__restrict__ A = (const double *)d[6];
    const double *__restrict__ B = (const double *)d[7];
    double *C = (double *)d[8];
    const int ix = t % {tx}, iy = t / {tx};
    double acc[{rm}][{rn}];
    for (int i = 0; i < {rm}; ++i)
        for (int j = 0; j < {rn}; ++j) acc[i][j] = 0.0;
    for (int k0 = 0; k0 < k; k0 += {bk}) {{
        for (int e = t; e < {bm} * {bk}; e += {nt}) {{
            {a_load}
        }}
        for (int e = t; e < {bk} * {bn}; e += {nt}) {{
            {b_load}
        }}
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < {bk}; ++kk) {{
            double a[{rm}], b[{rn}];
#pragma unroll
            for (int i = 0; i < {rm}; ++i) a[i] = As[kk][ix + {tx} * i];
#pragma unroll
            for (int j = 0; j < {rn}; ++j) b[j] = Bs[kk][iy + {ty} * j];
#pragma unroll
            for (int i = 0; i < {rm}; ++i)
#pragma unroll
                for (int j = 0; j < {rn}; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }}
        __syncthreads();
    }}
    for (int j = 0; j < {rn}; ++j) {{
        const int c = c0 + iy + {ty} * j;
        if (c >= n) continue;
        for (int i = 0; i < {rm}; ++i) {{
            const int r = r0 + ix + {tx} * i;
            if (r >= m) continue;
            double *o = C + r + (long long)c * ldc;
            *o = beta == 0.0 ? alpha * acc[i][j] : alpha * acc[i][j] + beta * *o;
        }}
    }}
}}
"#
    );
    s
}

/// The module with the four transposition variants of a tile.
pub fn module_source(t: &VgemmTile) -> String {
    let mut s = String::new();
    for ta in [false, true] {
        for tb in [false, true] {
            s.push_str(&generate(ta, tb, t));
        }
    }
    s
}

/// One product of a `VgemmPlan` (feature `gpu`): shape, leading dimensions and operand
/// addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VgemmProduct {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: u64,
    pub lda: usize,
    pub b: u64,
    pub ldb: usize,
    pub c: u64,
    pub ldc: usize,
}

/// Host halves of a plan: the descriptors and the tile list.
pub fn plan_tables(products: &[VgemmProduct], t: &VgemmTile) -> (Vec<i64>, Vec<i32>) {
    let mut desc = Vec::with_capacity(9 * products.len());
    let mut tiles = Vec::new();
    for (p, x) in products.iter().enumerate() {
        desc.extend([
            x.m as i64,
            x.n as i64,
            x.k as i64,
            x.lda as i64,
            x.ldb as i64,
            x.ldc as i64,
            x.a as i64,
            x.b as i64,
            x.c as i64,
        ]);
        if x.m == 0 || x.n == 0 {
            continue;
        }
        for tn in 0..x.n.div_ceil(t.bn) {
            for tm in 0..x.m.div_ceil(t.bm) {
                tiles.extend([p as i32, tm as i32, tn as i32]);
            }
        }
    }
    (desc, tiles)
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::error::GpuError;
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};

    /// The four kernels of a tile configuration.
    pub struct Vgemm {
        tile: VgemmTile,
        k: [Kernel; 4],
    }

    /// Products with their operand addresses fixed, ready to launch (the addresses must stay
    /// valid for the plan's life).
    pub struct VgemmPlan {
        pub nproducts: usize,
        pub ntiles: usize,
        pub flops: f64,
        desc: DeviceBuf<i64>,
        tiles: DeviceBuf<i32>,
    }

    impl Vgemm {
        pub fn new(gpu: &Gpu, tile: VgemmTile) -> Result<Vgemm, GpuError> {
            let src = module_source(&tile);
            let k = |ta, tb| gpu.kernel("vgemm", &src, &kernel_name(ta, tb, &tile));
            Ok(Vgemm {
                tile,
                k: [
                    k(false, false)?,
                    k(false, true)?,
                    k(true, false)?,
                    k(true, true)?,
                ],
            })
        }

        pub fn tile(&self) -> VgemmTile {
            self.tile
        }

        /// A plan for `products`.
        ///
        /// # Safety
        /// When the plan is launched, every address must point to a live device matrix of its
        /// product's shape (op(A) m x k, op(B) k x n, C m x n with their leading dimensions);
        /// C matrices must not overlap each other or any operand.
        pub unsafe fn plan(
            &self,
            gpu: &Gpu,
            products: &[VgemmProduct],
        ) -> Result<VgemmPlan, GpuError> {
            let (desc, tiles) = plan_tables(products, &self.tile);
            let ntiles = tiles.len() / 3;
            if ntiles > u32::MAX as usize / 2 || products.len() > i32::MAX as usize {
                return Err(GpuError::Contract("vgemm: too many tiles".into()));
            }
            let flops = products
                .iter()
                .map(|p| 2.0 * (p.m * p.n * p.k) as f64)
                .sum();
            Ok(VgemmPlan {
                nproducts: products.len(),
                ntiles,
                flops,
                desc: gpu.upload(if desc.is_empty() { &[0] } else { &desc })?,
                tiles: gpu.upload(if tiles.is_empty() { &[0] } else { &tiles })?,
            })
        }

        /// Queues every product of `plan`: C = alpha op(A) op(B) + beta C.
        ///
        /// # Safety
        /// [`Vgemm::plan`]'s contract.
        pub unsafe fn launch(
            &self,
            gpu: &Gpu,
            ta: bool,
            tb: bool,
            alpha: f64,
            beta: f64,
            plan: &VgemmPlan,
        ) -> Result<(), GpuError> {
            if plan.ntiles == 0 {
                return Ok(());
            }
            let k = &self.k[2 * usize::from(ta) + usize::from(tb)];
            let cfg = Launch {
                grid: (plan.ntiles as u32, 1, 1),
                block: (self.tile.threads() as u32, 1, 1),
                shared_bytes: 0,
            };
            // SAFETY: the kernel's signature; the plan's addresses are the caller's contract.
            unsafe {
                gpu.launch(
                    k,
                    cfg,
                    &[
                        Arg::buf(&plan.tiles),
                        Arg::buf(&plan.desc),
                        Arg::F64(alpha),
                        Arg::F64(beta),
                    ],
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_every_product() {
        let t = VgemmTile::DEFAULT;
        let p = |m, n| VgemmProduct {
            m,
            n,
            k: 3,
            a: 0,
            lda: m,
            b: 0,
            ldb: 3,
            c: 0,
            ldc: m,
        };
        let (desc, tiles) = plan_tables(&[p(33, 64), p(0, 5), p(1, 1)], &t);
        assert_eq!(desc.len(), 27);
        // 2 x 2 tiles, none, 1
        assert_eq!(tiles.len() / 3, 5);
        assert!(generate(true, false, &t).contains("vgemm_tn_32x32x16_16x8"));
    }
}
