// linus-gpu portability header, pasted in front of every kernel source by the host.
//
// Kernels are written in the subset of C++ that CUDA (NVRTC/nvcc) and HIP (hiprtc/hipcc) both
// compile: C-style code, no inline PTX, no vendor intrinsics outside this header. The few
// names that differ are wrapped here. The warp (wavefront) size is never written as a number
// in a kernel: the host defines LINUS_WARP_SIZE from the device (32 on NVIDIA, 64 on AMD CDNA,
// 32 on AMD RDNA in wave32 mode) before this header.

#ifndef LINUS_PORTABLE_H
#define LINUS_PORTABLE_H

#ifndef LINUS_WARP_SIZE
#error "LINUS_WARP_SIZE must be defined by the host before portable.h"
#endif

#if defined(__HIPCC__) || defined(__HIPCC_RTC__)
#define LINUS_HIP 1
// HIP shuffles take the width; all lanes of the wavefront take part
#define linus_shfl_down(v, d) __shfl_down((v), (d), LINUS_WARP_SIZE)
#define linus_shfl_xor(v, m) __shfl_xor((v), (m), LINUS_WARP_SIZE)
#define linus_shfl(v, src) __shfl((v), (src), LINUS_WARP_SIZE)
// a 64-bit mask on either vendor (bit l: lane l)
#define linus_ballot(p) ((unsigned long long)__ballot(p))
// a wavefront runs in lockstep
#define linus_syncwarp() ((void)0)
#else
#define LINUS_CUDA 1
#define linus_shfl_down(v, d) __shfl_down_sync(0xffffffffu, (v), (d), LINUS_WARP_SIZE)
#define linus_shfl_xor(v, m) __shfl_xor_sync(0xffffffffu, (v), (m), LINUS_WARP_SIZE)
#define linus_shfl(v, src) __shfl_sync(0xffffffffu, (v), (src), LINUS_WARP_SIZE)
#define linus_ballot(p) ((unsigned long long)__ballot_sync(0xffffffffu, (p)))
#define linus_syncwarp() __syncwarp()
#endif

#endif
