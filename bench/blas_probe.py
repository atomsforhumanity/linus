"""Which BLAS/LAPACK a process uses, its version and the CPU kernel it picked; DGEMM speed.

Imported by the benchmark job scripts (bench/jobs/*) to record the BLAS libraries loaded in
the program's own process (`loaded_blas()`), and run from the command line to validate a
library or an override (OPENBLAS_CORETYPE, LD_PRELOAD):

  python bench/blas_probe.py loaded                       # BLAS libraries in this process
  python bench/blas_probe.py dgemm LIB [THREADS] [N]      # GF/s of cblas_dgemm (or dgemm_) in LIB

OpenBLAS is asked for openblas_get_config() (version, build options and, for DYNAMIC_ARCH
builds, the kernel set it picked), openblas_get_corename() and its thread count; MKL for its
version string, max threads and the code path it dispatches to (mkl_cbwr_get_auto_branch).
"""
import ctypes, json, os, re, sys, time

BLAS_RE = re.compile(r"(openblas|mkl_rt|mkl_core|libblas|libcblas|liblapack|flexiblas|blis)[^/]*\.so", re.I)
# mkl_cbwr_get_auto_branch() return values (mkl_cbwr.h)
MKL_BRANCH = {1: "BRANCH_OFF", 2: "AUTO", 3: "COMPATIBLE", 4: "SSE2", 6: "SSSE3", 7: "SSE4_1", 8: "SSE4_2", 9: "AVX",
              10: "AVX2", 11: "AVX512_MIC", 12: "AVX512", 13: "AVX512_MIC_E1", 14: "AVX512_E1"}


def _sym(lib, name, restype=None, argtypes=None):
    try:
        f = getattr(lib, name)
    except AttributeError:
        return None
    f.restype = restype
    if argtypes is not None:
        f.argtypes = argtypes
    return f


def _load(path, mode=ctypes.DEFAULT_MODE):
    """Load a library; wheel-bundled ones (PySCF's OpenBLAS) need their sibling libgfortran first."""
    try:
        return ctypes.CDLL(path, mode=mode)
    except OSError:
        import glob
        d = os.path.dirname(os.path.realpath(path))
        deps = sorted(glob.glob(os.path.join(d, "libquadmath*.so*"))) + sorted(glob.glob(os.path.join(d, "libgfortran*.so*")))
        for dep in deps:
            try:
                ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        return ctypes.CDLL(path, mode=mode)


def describe(path):
    """Version, kernel and threads of the BLAS library at `path` (loaded or loadable)."""
    info = {"path": os.path.realpath(path), "file": os.path.basename(os.path.realpath(path))}
    try:
        lib = _load(path)
    except OSError as e:
        info["error"] = str(e)
        return info
    cfg = _sym(lib, "openblas_get_config", ctypes.c_char_p)
    if cfg:
        info["kind"] = "openblas"
        info["config"] = cfg().decode()
        m = re.search(r"OpenBLAS\s+(\d+\.\d+\.\d+)", info["config"])
        info["version"] = m.group(1) if m else None
        core = _sym(lib, "openblas_get_corename", ctypes.c_char_p)
        info["core"] = core().decode() if core else None
        nt = _sym(lib, "openblas_get_num_threads", ctypes.c_int)
        info["threads"] = nt() if nt else None
        par = _sym(lib, "openblas_get_parallel", ctypes.c_int)
        info["parallel"] = {0: "sequential", 1: "pthreads", 2: "openmp"}.get(par(), par()) if par else None
        if info["version"] is None:  # 0.3.3 prints no version in its config string
            m = re.search(r"r0[-.].*?(\d+\.\d+\.\d+)\.so", info["file"]) or re.search(r"(\d+\.\d+\.\d+)\.so", info["file"])
            info["version"] = m.group(1) if m else None
        return info
    ver = _sym(lib, "MKL_Get_Version_String", None, [ctypes.c_char_p, ctypes.c_int])
    if ver:
        info["kind"] = "mkl"
        buf = ctypes.create_string_buffer(512)
        ver(buf, 512)
        info["config"] = buf.value.decode().strip()
        m = re.search(r"Version (\d{4}\.\d+(?:\.\d+)?)", info["config"])
        info["version"] = m.group(1) if m else None
        br = _sym(lib, "mkl_cbwr_get_auto_branch", ctypes.c_int)
        info["core"] = MKL_BRANCH.get(br(), br()) if br else None
        mt = _sym(lib, "MKL_Get_Max_Threads", ctypes.c_int)
        info["threads"] = mt() if mt else None
        return info
    info["kind"] = "other"
    return info


def loaded_blas():
    """BLAS/LAPACK libraries mapped into this process, described."""
    paths = []
    try:
        for line in open("/proc/self/maps"):
            p = line.split()[-1] if len(line.split()) >= 6 else ""
            if p.startswith("/") and BLAS_RE.search(os.path.basename(p)) and p not in paths:
                paths.append(p)
    except OSError:
        return []
    seen, out = set(), []
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(describe(p))
    return out


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p), ("dli_sname", ctypes.c_char_p),
                ("dli_saddr", ctypes.c_void_p)]


def symbol_binding(lib_path, sym="dgemm_"):
    """Which library a reference to `sym` from the (loaded) library `lib_path` binds to.

    The dynamic linker resolves a symbol in the global scope first (the executable,
    LD_PRELOAD libraries, RTLD_GLOBAL objects) and only then in the library's own dependency
    tree, so: the global provider if there is one, else the local one."""
    libc = ctypes.CDLL(None)
    libc.dlsym.restype = ctypes.c_void_p
    libc.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libc.dlopen.restype = ctypes.c_void_p
    libc.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]

    def where(addr):
        if not addr:
            return None
        info = _DlInfo()
        return os.path.realpath(info.dli_fname.decode()) if libc.dladdr(addr, ctypes.byref(info)) and info.dli_fname else None
    glob_ = where(libc.dlsym(None, sym.encode()))  # RTLD_DEFAULT
    h = libc.dlopen(lib_path.encode(), 4 | 1)  # RTLD_NOLOAD | RTLD_LAZY: only if already loaded
    local = where(libc.dlsym(h, sym.encode())) if h else None
    return dict(library=os.path.basename(lib_path), symbol=sym, global_provider=glob_, local_provider=local,
                binds_to=glob_ or local)


def blas_env():
    """Environment variables that steer BLAS/OpenMP behaviour."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE", "MKL_ENABLE_INSTRUCTIONS",
            "MKL_DEBUG_CPU_TYPE", "LD_PRELOAD", "RAYON_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY")
    return {k: os.environ[k] for k in keys if k in os.environ}


def dgemm_gflops(path, threads=1, n=2000, reps=None):
    """GF/s of an n x n x n DGEMM through `path` (cblas_dgemm, column-major), checked against numpy.
    BENCH_LATE_CORETYPE=X sets OPENBLAS_CORETYPE=X after NumPy's OpenBLAS is loaded, so only `path` sees it."""
    import numpy as np
    if os.environ.get("BENCH_LATE_CORETYPE"):
        os.environ["OPENBLAS_CORETYPE"] = os.environ["BENCH_LATE_CORETYPE"]
    lib = _load(path, mode=ctypes.RTLD_GLOBAL)
    d = describe(path)
    if d.get("kind") == "mkl":
        _sym(lib, "MKL_Set_Num_Threads", None, [ctypes.c_int])(threads)
    else:
        f = _sym(lib, "openblas_set_num_threads", None, [ctypes.c_int])
        if f:
            f(threads)
    I, D, P = ctypes.c_int, ctypes.c_double, ctypes.c_void_p
    g = _sym(lib, "cblas_dgemm", None, [I, I, I, I, I, I, D, P, I, P, I, D, P, I])
    rng = np.random.default_rng(1)
    a = np.asfortranarray(rng.random((n, n)))
    b = np.asfortranarray(rng.random((n, n)))
    c = np.zeros((n, n), order="F")
    reps = reps or max(3, int(4e10 * threads / (2 * n ** 3)))
    best = 1e30
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            g(102, 111, 111, n, n, n, 1.0, a.ctypes.data, n, b.ctypes.data, n, 0.0, c.ctypes.data, n)
        best = min(best, (time.perf_counter() - t) / reps)
    err = float(np.abs(c - a @ b).max())
    return dict(describe(path), gflops=2 * n ** 3 / best / 1e9, threads_requested=threads, n=n, max_abs_err=err)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dgemm":
        print(json.dumps(dgemm_gflops(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1,
                                      int(sys.argv[4]) if len(sys.argv) > 4 else 2000)))
    else:
        print(json.dumps({"loaded": loaded_blas(), "env": blas_env()}, indent=1))
