//! Response (CPHF/CPKS) products on the device: the projected J/K of first-order densities
//! ([`df`], `GpuDfResponse`) and the XC kernel contraction ([`xc`], `GpuXcKernel`), feature
//! `gpu`. They are the device counterparts of linus's `ResponseJk` builders (the CPU's
//! `DfResponseJk`) and of `XcIntegrator::kernel_contract_occ`, with plain host matrices at the
//! boundary; linus wraps them behind its own traits.
//!
//! Every CPHF/CPKS density of a Hessian or a polarizability has the form
//! dD_{s,d} = A_{s,d} C_s^T + C_s A_{s,d}^T for spin channel s (occupied orbitals C_s, n x o_s,
//! fixed at setup) and perturbation d (factor A_{s,d}, n x o_s), and the response equations
//! only need the induced Fock matrices projected on the occupied orbitals,
//!
//!   G_{s,d} = (c_j J\[sum_s' dD_{s',d}\] - c_k K\[dD_{s,d}\] + dV_xc,s\[dD_{.,d}\]) C_s   (n x o_s).
//!
//! One channel is a closed shell (dD the alpha density; c_j = 2, and the XC kernel of the total
//! first-order density 2 dD), two are the spins of an open shell (c_j = 1).
//!
//! **J/K** (`df`), with the fitted three-center tensor B_Q = L^-1 (Q|mn) of a `GpuDfJk`:
//! - Y_{s,Q} = B_Q C_s (n x o_s) is formed once at setup and stored, Q-major: the matrix
//!   Yp_s ((n o_s) x naux, column Q holds Y_{s,Q} column-major).
//! - J: gamma_{Q,d} = 2 sum_s <Y_{s,Q}, A_{s,d}>, so gamma = 2 sum_s Yp_s^T Acat_s with the
//!   factors side by side as Acat_s ((n o_s) x nd), and J\[sum_s dD\] C_s = Yp_s gamma: two
//!   GEMMs over the stored Yp, no pass over the tensor.
//! - K: K\[dD\] C = sum_Q [Z_Q X_Q + Y_Q (A^T Y_Q)] with Z_Q = B_Q A and X_Q = C^T B_Q C. Per
//!   block of auxiliary functions the tensor is unpacked to dense planes (`GpuDfJk`'s kernel)
//!   and Z = B^T [A_1 .. A_nd] is one large GEMM for all perturbations of a group (the
//!   n^2 o naux nd flops of the response exchange). With the block's Y in the same (i, q)
//!   order, the stacked X_Q (Xs = Y^T C) and A^T Y_Q (= Z_d^T C for all d at once) make the
//!   two contractions strided-batched GEMMs over the perturbations.
//! - Range-separated hybrids (the exchange operator w = a/r + b erf(omega r)/r, coefficients
//!   included): Psi4's wK scheme, as the CPU's response and SCF builders, with a second tensor
//!   W_Q = L^-1 (Q|w|mn) on the same rows (Coulomb metric):
//!   K_w\[dD\] C = 1/2 sum_Q [Z_Q Xw_Q + Yw_Q (A^T Y_Q) + Zw_Q X_Q + Y_Q (A^T Yw_Q)],
//!   twice the exchange work of a global hybrid; J is unchanged.
//! - Precision: FP64; optionally cuBLAS's fixed-mantissa FP64 emulation for the exchange
//!   GEMMs (`KMath::Emulated`). The setup (Y) and J are always native.
//! - **The stored exchange operator** (`KOperator`). K\[dD\] C is linear in A,
//!   so per channel it is one symmetric (n o) x (n o) matrix acting on vec(A) (row (m, j),
//!   column (l, i), index m + n j):
//!   Op\[(m, j), (l, i)\] = sum_Q [B_Q\[m, l\] X_Q\[i, j\] + Y_Q\[m, i\] Y_Q\[l, j\]]
//!   (wK: 1/2 sum_Q [B_Q Xw_Q + W_Q X_Q]\[m, l\]\[i, j\] plus 1/2 sum_Q [Yw_Q\[m, i\]
//!   Y_Q\[l, j\] + Y_Q\[m, i\] Yw_Q\[l, j\]]). Built once, it turns a product into one GEMM
//!   pair per panel: 2 (n o)^2 nd flops instead of 2 n^2 o naux nd (1 + 3 o / n) through the
//!   tensor, 35-75x faster at def2-SVP sizes, and a range-separated hybrid costs what a global
//!   one does.
//!   - Storage: the lower block triangle in panels of whole orbitals (panel (j0, bj): rows of
//!     j0 <= j < j0 + bj, columns of i < j0 + bj), (n o)^2 / 2 doubles (1 + bj / o).
//!   - Setup, per panel: the Y part as one GEMM, Y\[rows of the panel's j\] Y^T, whose n x n
//!     blocks come out transposed (block (j, i) holds Y_Q\[m, j\] Y_Q\[l, i\]); the pair part
//!     B^T X over the tensor's stored rows (nx x i per orbital j, one batched GEMM with X_Q
//!     as an o^2 x naux matrix); `resp_op_fill` transposes every block in place and adds the
//!     pair part through the map of function pairs to stored rows. About (n o)^2 naux
//!     + nx naux o^2 flops (twice with wK), half of one product through the tensor.
//!   - Products: per panel out\[rows\] += -c_k P A\[columns\] and, below the diagonal block,
//!     out\[columns\] += -c_k P^T A\[rows\], in a fixed order, in native FP64. The setup's
//!     GEMMs follow the exchange arithmetic (cuBLAS's emulation with `RespMath::Emulated`,
//!     2.5x faster than FP64 there).
//!   - Policy (`KOperator::Auto`, the default): built at the first product whose setup and
//!     one product on it cost fewer flops than two products through the tensor (a Hessian's
//!     3N perturbations; not a polarizability's three), when it and its setup's temporaries
//!     fit the device beside a reserve for the other engines (the CPKS kernel's workspace);
//!     else the exchange stays on the tensor, with the reason in `resolved`. With the FP32
//!     exchange (`Single`) only as the FP32 operator (`k_operator_fp32`).
//!   - Element: the operator in FP64, or in FP32 (half the memory and SGEMM products), one
//!     implementation over the element type: the FP32 build runs the same steps as SGEMMs on
//!     FP32 copies over blocks of 256 auxiliary functions, each block's sum added in FP64, and
//!     rounds each chunk to FP32; its products add each chunk's FP32 product to the FP64
//!     result.
//!
//! **XC kernel** (`xc`), on the blocks and collocation of a `GpuXc`: the functional's second
//! derivatives depend only on the ground state, so they are evaluated once at setup (libxc on
//! the host through linus-xc, the densities from the device) and kept on the device as the
//! weighted effective second derivatives wF (nu x nu per point, the CPU's
//! `weighted_derivs`). Per call and block, with F = [phi; grad phi] (collocation, (nc np) x nf)
//! and Phi_s = F C_s, for all perturbations of a group:
//! - LDA and meta-GGA: P_s = F A_s (one GEMM); one thread per (point, perturbation) forms u1
//!   (rho1, grad rho1, tau1 of each spin) from P and Phi, t = wF u1, and overwrites P with
//!   Z = [t_rho Phi + t_k d_k Phi; t_k Phi + 1/2 t_tau d_k Phi] (k = x, y, z); then
//!   dV C = F^T Z (one GEMM).
//! - GGA, the CPU's form with half those multiply-adds: P_s = phi A_s for the values only and
//!   Q_d = Phi A_d^T (np x nf per perturbation, a batched GEMM), so that
//!   grad rho1 = 2 sum_i P_i grad Phi_i + 2 sum_m grad phi_m Q_m; the pointwise kernel
//!   overwrites P with Z = t_rho Phi + t_k d_k Phi and Q with G = t_k d_k phi, and
//!   dV C = phi^T Z + G_d^T Phi (a GEMM and a batched GEMM).
//! - dV C of the block is added to its rows of the result.
//!
//! The large GEMMs can run in cuBLAS's fixed-point FP64 emulation (`emulated_bits`).
//!
//! Determinism: every kernel has one writer per output, the blocks are added in a fixed order
//! on one stream, and cuBLAS is deterministic for a fixed configuration, so the products are
//! bitwise reproducible run to run on a device.

pub mod df;
pub mod krylov;
pub mod xc;

/// Perturbations per group and auxiliary functions per exchange block for a workspace of
/// `workspace` bytes: the unpacked planes (nq n^2, twice with an operator tensor), Z
/// (nq n W, W = group o), the stacked products (W nq o) and the block's Y and X
/// (n o nq, o^2 nq; twice with an operator tensor). The group is as large as `nq_min`
/// functions per block allow; nq then fills the workspace, at most `nq_max` and `naux`.
/// Returns (perturbations per group, nq).
#[allow(clippy::too_many_arguments)]
pub fn exchange_plan(
    n: usize,
    o: usize,
    nd: usize,
    naux: usize,
    rs: bool,
    workspace: usize,
    nq_min: usize,
    nq_max: usize,
) -> (usize, usize) {
    let r = if rs { 2 } else { 1 };
    let fixed = |w: usize| r * n * n + n * w + w * o + 2 * r * n * o + 2 * r * o * o;
    let words = workspace / 8;
    let nd = nd.max(1);
    let o1 = o.max(1);
    // the largest group whose per-function bytes allow nq_min functions
    let per_min = words / nq_min.max(1);
    let base = r * n * n + 2 * r * n * o + 2 * r * o * o;
    let group = if per_min > base {
        ((per_min - base) / (o1 * (n + o1))).clamp(1, nd)
    } else {
        1
    };
    let nq = (words / fixed(group * o).max(1)).clamp(1, nq_max.max(1));
    (group, nq.min(naux.max(1)))
}

/// Perturbations per group of the XC kernel for a workspace of `workspace` bytes, with `per`
/// words of block buffers per perturbation (all channels).
pub fn kernel_group(per: usize, nd: usize, workspace: usize) -> usize {
    (workspace / (8 * per.max(1))).clamp(1, nd.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_fits_the_workspace() {
        // u1an39/def2-SVP-like sizes, 117 perturbations
        let (n, o, nd, naux) = (430, 120, 117, 1500);
        for rs in [false, true] {
            for ws in [1usize << 30, 4 << 30, 16 << 30] {
                let (g, nq) = exchange_plan(n, o, nd, naux, rs, ws, 8, 512);
                assert!((1..=nd).contains(&g) && (1..=512).contains(&nq));
                let r = if rs { 2 } else { 1 };
                let w = g * o;
                let bytes = 8 * nq * (r * n * n + n * w + w * o + 2 * r * n * o + 2 * r * o * o);
                assert!(bytes <= ws, "{bytes} > {ws}");
            }
        }
        // a large workspace takes every perturbation in one group
        assert_eq!(exchange_plan(n, o, nd, naux, false, 64 << 30, 8, 512).0, nd);
        let per = (4 * 2048 + 300) * 120;
        assert_eq!(kernel_group(per, 117, 64 << 30), 117);
        assert_eq!(kernel_group(per, 117, 1), 1);
    }
}
