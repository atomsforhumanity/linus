//! Run-time compilation of CUDA C++ with NVRTC, cached in memory and on disk.
//!
//! Kernels are compiled for the architecture of the open device: a CUBIN for `sm_XY` when the
//! loaded NVRTC knows it, otherwise PTX for the newest `compute_XY` it knows, which the driver
//! JIT-compiles forward. The cache key is the SHA-256 of the source, the options, the target
//! and the NVRTC version, so a changed kernel, option or toolkit never reuses a stale binary.
//! Sources are self-contained (headers are pasted in by the caller), so the hash covers
//! everything the compiler sees.
//!
//! Disk cache: the directory the caller opens the device with (linus's `backend.kernel_cache`;
//! None: memory only), by default [`default_cache_dir`]: `$LINUS_GPU_CACHE` if set (`off`
//! disables it), else `$XDG_CACHE_HOME/linus/gpu-kernels`, else `~/.cache/linus/gpu-kernels`
//! (the default of linus's `backend.kernel_cache: auto`); one file per kernel,
//! `<target>/<hash>.cubin` or `.ptx`, written to a temporary name and renamed, so concurrent
//! processes never see a partial file. A file that fails to load is recompiled.
//!
//! Source dump (a debugging aid, e.g. for a manifest of the kernel sources a run compiled):
//! with `$LINUS_GPU_KERNEL_DUMP` set to a directory, the first request of each module in a
//! process also writes `<hash>.cu` (the assembled source the compiler sees) and `<hash>.json`
//! (module name, options, target, NVRTC version, the source's SHA-256) there, `<hash>` being
//! the cache key. Read once when the cache is created; without it nothing is written.

use crate::error::GpuError;
use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::{result as nv, sys as nvsys, Ptx};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Options every kernel is compiled with: C++17, FMA contraction on (FP64 results are
/// correctly rounded per operation either way; FMA only removes roundings), line info for
/// Nsight Compute source views.
pub const BASE_OPTIONS: &[&str] = &["--std=c++17", "--fmad=true", "-lineinfo"];

/// What the cache compiles to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// `sm_XY` binary.
    Cubin(u32),
    /// `sm_XYa` binary: the architecture-specific features of `sm_XY` (e.g. sm_90a's register
    /// hand-over, `setmaxnreg`), for that architecture alone.
    CubinArchSpecific(u32),
    /// `compute_XY` PTX for the driver to JIT.
    Ptx(u32),
}

impl Target {
    pub fn arch_option(&self) -> String {
        format!("--gpu-architecture={}", self.tag())
    }

    fn tag(&self) -> String {
        match self {
            Target::Cubin(a) => format!("sm_{a}"),
            Target::CubinArchSpecific(a) => format!("sm_{a}a"),
            Target::Ptx(a) => format!("compute_{a}"),
        }
    }

    fn is_cubin(&self) -> bool {
        !matches!(self, Target::Ptx(_))
    }
}

/// Counters of the cache (for tests and reports).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheStats {
    pub memory_hits: usize,
    pub disk_hits: usize,
    pub compiled: usize,
    pub compile_seconds: f64,
}

/// Compiled modules of one device.
pub struct KernelCache {
    ctx: Arc<CudaContext>,
    target: Target,
    nvrtc_version: (i32, i32),
    dir: Option<PathBuf>,
    /// `$LINUS_GPU_KERNEL_DUMP` (module docs).
    dump: Option<PathBuf>,
    modules: Mutex<HashMap<[u8; 32], Arc<CudaModule>>>,
    stats: Mutex<CacheStats>,
}

/// NVRTC's version.
pub fn nvrtc_version() -> Result<(i32, i32), GpuError> {
    let (mut a, mut b) = (0, 0);
    // SAFETY: out pointers.
    unsafe { nvsys::nvrtcVersion(&mut a, &mut b) }
        .result()
        .map_err(super::nvrtc_error("nvrtcVersion"))?;
    Ok((a, b))
}

/// Architectures (XY of sm_XY) the loaded NVRTC can compile for.
pub fn supported_archs() -> Result<Vec<u32>, GpuError> {
    let mut n = 0;
    // SAFETY: out pointers sized from the first call.
    unsafe { nvsys::nvrtcGetNumSupportedArchs(&mut n) }
        .result()
        .map_err(super::nvrtc_error("nvrtcGetNumSupportedArchs"))?;
    let mut v = vec![0i32; n.max(0) as usize];
    unsafe { nvsys::nvrtcGetSupportedArchs(v.as_mut_ptr()) }
        .result()
        .map_err(super::nvrtc_error("nvrtcGetSupportedArchs"))?;
    Ok(v.into_iter().map(|a| a as u32).collect())
}

/// The default disk-cache directory (see the module docs); None when disabled or no home.
pub fn default_cache_dir() -> Option<PathBuf> {
    match std::env::var_os("LINUS_GPU_CACHE") {
        Some(v) if v == "off" || v.is_empty() => return None,
        Some(v) => return Some(PathBuf::from(v)),
        None => {}
    }
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME").filter(|x| !x.is_empty()) {
        return Some(PathBuf::from(x).join("linus/gpu-kernels"));
    }
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(|h| PathBuf::from(h).join(".cache/linus/gpu-kernels"))
}

fn create_private_dir(dir: &Path) -> Result<(), GpuError> {
    let err = |e: std::io::Error| GpuError::Cache {
        path: dir.to_path_buf(),
        msg: e.to_string(),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .map_err(err)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir).map_err(err)
    }
}

impl KernelCache {
    /// A cache for a device of compute capability `cc`, with the disk cache in `dir`.
    pub(crate) fn new(
        ctx: Arc<CudaContext>,
        cc: (i32, i32),
        dir: Option<PathBuf>,
    ) -> Result<Self, GpuError> {
        let nvrtc_version = nvrtc_version()?;
        let dev = (cc.0 * 10 + cc.1) as u32;
        let archs = supported_archs()?;
        let target = if archs.contains(&dev) {
            Target::Cubin(dev)
        } else {
            let best = archs
                .iter()
                .copied()
                .filter(|&a| a <= dev)
                .max()
                .ok_or_else(|| {
                    GpuError::NotCovered(format!(
                        "NVRTC {}.{} cannot compile for sm_{dev} or older",
                        nvrtc_version.0, nvrtc_version.1
                    ))
                })?;
            Target::Ptx(best)
        };
        let dump = std::env::var_os("LINUS_GPU_KERNEL_DUMP")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        Ok(KernelCache {
            ctx,
            target,
            nvrtc_version,
            dir,
            dump,
            modules: Mutex::new(HashMap::new()),
            stats: Mutex::new(CacheStats::default()),
        })
    }

    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn nvrtc_version(&self) -> (i32, i32) {
        self.nvrtc_version
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Drops the in-memory modules (the next request goes to disk), for tests.
    pub fn clear_memory(&self) {
        self.modules.lock().unwrap().clear();
    }

    /// The device's architecture-specific target (`sm_XYa`): only where the cache compiles
    /// binaries for the device's own architecture (not PTX for an older one).
    pub fn arch_specific(&self) -> Result<Target, GpuError> {
        match self.target {
            Target::Cubin(a) => Ok(Target::CubinArchSpecific(a)),
            ref t => Err(GpuError::NotCovered(format!(
                "an architecture-specific kernel needs NVRTC to compile for the device's architecture \
                 (it compiles {})",
                t.tag()
            ))),
        }
    }

    fn options(&self, extra: &[&str], target: &Target) -> Vec<String> {
        let mut o: Vec<String> = BASE_OPTIONS.iter().map(|s| s.to_string()).collect();
        o.push(target.arch_option());
        o.extend(extra.iter().map(|s| s.to_string()));
        o
    }

    fn key(&self, src: &str, opts: &[String], target: &Target) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"linus-gpu kernel cache v1\n");
        h.update(
            format!(
                "nvrtc {}.{}\n{}\n",
                self.nvrtc_version.0,
                self.nvrtc_version.1,
                target.tag()
            )
            .as_bytes(),
        );
        for o in opts {
            h.update(o.as_bytes());
            h.update(b"\n");
        }
        h.update(b"\0");
        h.update(src.as_bytes());
        h.finalize().into()
    }

    fn path(&self, key: &[u8; 32], target: &Target) -> Option<PathBuf> {
        let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
        let ext = if target.is_cubin() { "cubin" } else { "ptx" };
        self.dir
            .as_ref()
            .map(|d| d.join(target.tag()).join(format!("{hex}.{ext}")))
    }

    fn load(&self, image: Vec<u8>, target: &Target) -> Result<Arc<CudaModule>, GpuError> {
        let ptx = if target.is_cubin() {
            Ptx::from_binary(image)
        } else {
            Ptx::from_src(String::from_utf8_lossy(&image).into_owned())
        };
        Ok(self.ctx.load_module(ptx)?)
    }

    /// The module compiled from `src` with the base options plus `extra`.
    pub fn module(
        &self,
        name: &str,
        src: &str,
        extra: &[&str],
    ) -> Result<Arc<CudaModule>, GpuError> {
        self.module_for(name, src, extra, &self.target.clone())
    }

    /// [`KernelCache::module`] for `target` (the device's own, or its architecture-specific
    /// one, [`KernelCache::arch_specific`]).
    pub fn module_for(
        &self,
        name: &str,
        src: &str,
        extra: &[&str],
        target: &Target,
    ) -> Result<Arc<CudaModule>, GpuError> {
        let opts = self.options(extra, target);
        let key = self.key(src, &opts, target);
        if let Some(m) = self.modules.lock().unwrap().get(&key) {
            self.stats.lock().unwrap().memory_hits += 1;
            return Ok(m.clone());
        }
        if let Some(d) = &self.dump {
            self.dump_source(d, &key, name, src, &opts, target);
        }
        let path = self.path(&key, target);
        if let Some(p) = &path {
            if let Ok(image) = std::fs::read(p) {
                if let Ok(m) = self.load(image, target) {
                    self.stats.lock().unwrap().disk_hits += 1;
                    self.modules.lock().unwrap().insert(key, m.clone());
                    return Ok(m);
                }
            }
        }
        let t0 = Instant::now();
        let (image, _log) = compile(name, src, &opts, target)?;
        {
            let mut s = self.stats.lock().unwrap();
            s.compiled += 1;
            s.compile_seconds += t0.elapsed().as_secs_f64();
        }
        if let Some(p) = &path {
            // a cache that cannot be written only costs a recompilation next time
            let _ = write_atomic(p, &image);
        }
        let m = self.load(image, target)?;
        self.modules.lock().unwrap().insert(key, m.clone());
        Ok(m)
    }

    /// `$LINUS_GPU_KERNEL_DUMP`: the module's source and its description in `dir` (module
    /// docs). Best effort: a dump that cannot be written changes nothing else.
    fn dump_source(
        &self,
        dir: &Path,
        key: &[u8; 32],
        name: &str,
        src: &str,
        opts: &[String],
        target: &Target,
    ) {
        let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
        let quote = |s: &str| -> String {
            let mut q = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => q.push_str("\\\""),
                    '\\' => q.push_str("\\\\"),
                    c if (c as u32) < 0x20 => q.push_str(&format!("\\u{:04x}", c as u32)),
                    c => q.push(c),
                }
            }
            q.push('"');
            q
        };
        let options: Vec<String> = opts.iter().map(|o| quote(o)).collect();
        let meta = format!(
            "{{\"name\": {}, \"options\": [{}], \"target\": {}, \"nvrtc\": \"{}.{}\", \"source_sha256\": \"{}\"}}\n",
            quote(name),
            options.join(", "),
            quote(&target.tag()),
            self.nvrtc_version.0,
            self.nvrtc_version.1,
            hex(&Sha256::digest(src.as_bytes()))
        );
        let key = hex(key);
        let _ = write_atomic(&dir.join(format!("{key}.cu")), src.as_bytes());
        let _ = write_atomic(&dir.join(format!("{key}.json")), meta.as_bytes());
    }

    /// Compiles `src` without caching and returns NVRTC's log (with `extra` options such as
    /// `--ptxas-options=-v`, which reports registers, spills and shared memory per kernel).
    pub fn compile_log(&self, name: &str, src: &str, extra: &[&str]) -> Result<String, GpuError> {
        Ok(compile(name, src, &self.options(extra, &self.target), &self.target)?.1)
    }
}

/// A temporary name beside `path` that no other writer uses: the process id alone repeats
/// across containers that share a cache volume (each has its own pid namespace), so the name
/// also carries 64 random bits (std's per-thread random hash keys, seeded by the OS, mixed with
/// the clock and a per-process counter).
fn temp_name(path: &Path) -> PathBuf {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    path.with_file_name(format!(
        ".{}.{}.{:016x}.tmp",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        h.finish()
    ))
}

/// Writes `data` to a unique temporary file beside `path` and renames it over `path` (atomic
/// on one file system): concurrent writers, also in other containers on a shared cache
/// volume, never see a partial file, and the last rename wins with a complete one.
fn write_atomic(path: &Path, data: &[u8]) -> Result<(), GpuError> {
    let dir = path.parent().expect("cache file has a parent");
    create_private_dir(dir)?;
    let tmp = temp_name(path);
    let err = |e: std::io::Error| GpuError::Cache {
        path: path.to_path_buf(),
        msg: e.to_string(),
    };
    let r = std::fs::write(&tmp, data).and_then(|()| std::fs::rename(&tmp, path));
    if r.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    r.map_err(err)
}

fn log_of(prog: nvsys::nvrtcProgram) -> String {
    // SAFETY: prog is a live program.
    match unsafe { nv::get_program_log(prog) } {
        Ok(v) => {
            let bytes: Vec<u8> = v
                .into_iter()
                .map(|c| c as u8)
                .take_while(|&c| c != 0)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        }
        Err(e) => format!("(no log: {e:?})"),
    }
}

/// Compiles the full source `src` ([`super::device::kernel_source`]) for `target` with the
/// cache's options ([`BASE_OPTIONS`] and the architecture), without a context and without the
/// cache: the image and the compiler log. For the compile test of every module (no device).
pub fn compile_uncached(
    name: &str,
    src: &str,
    target: &Target,
) -> Result<(Vec<u8>, String), GpuError> {
    let mut opts: Vec<String> = BASE_OPTIONS.iter().map(|s| s.to_string()).collect();
    opts.push(target.arch_option());
    compile(name, src, &opts, target)
}

/// Compiles `src`; returns the CUBIN or PTX image and the compiler log.
fn compile(
    name: &str,
    src: &str,
    opts: &[String],
    target: &Target,
) -> Result<(Vec<u8>, String), GpuError> {
    let csrc = CString::new(src)
        .map_err(|_| GpuError::Contract(format!("kernel source {name} contains a NUL byte")))?;
    let cname = CString::new(format!("{name}.cu")).unwrap();
    let prog = nv::create_program(&csrc, Some(&cname))
        .map_err(super::nvrtc_error("nvrtcCreateProgram"))?;
    // SAFETY: prog is live until destroy_program below.
    let res = unsafe { nv::compile_program(prog, opts) };
    let log = log_of(prog);
    let out = match res {
        Err(_) => Err(GpuError::Compile {
            name: name.to_string(),
            log: log.clone(),
        }),
        Ok(()) => match target {
            Target::Cubin(_) | Target::CubinArchSpecific(_) => {
                let mut n = 0usize;
                // SAFETY: sized buffer for the CUBIN of a compiled program.
                unsafe { nvsys::nvrtcGetCUBINSize(prog, &mut n) }
                    .result()
                    .map_err(super::nvrtc_error("nvrtcGetCUBINSize"))?;
                let mut v = vec![0u8; n];
                unsafe { nvsys::nvrtcGetCUBIN(prog, v.as_mut_ptr() as *mut _) }
                    .result()
                    .map_err(super::nvrtc_error("nvrtcGetCUBIN"))?;
                Ok(v)
            }
            Target::Ptx(_) => {
                // SAFETY: compiled program.
                let v = unsafe { nv::get_ptx(prog) }.map_err(super::nvrtc_error("nvrtcGetPTX"))?;
                Ok(v.into_iter().map(|c| c as u8).collect())
            }
        },
    };
    // SAFETY: created above, destroyed once.
    let _ = unsafe { nv::destroy_program(prog) };
    out.map(|img| (img, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_names_are_unique_and_beside_the_file() {
        let p = Path::new("/cache/sm_120/abc.cubin");
        let names: std::collections::HashSet<PathBuf> = (0..1000).map(|_| temp_name(p)).collect();
        assert_eq!(names.len(), 1000);
        for t in &names {
            assert_eq!(t.parent(), p.parent());
            let f = t.file_name().unwrap().to_string_lossy();
            assert!(f.starts_with(".abc.cubin.") && f.ends_with(".tmp"), "{f}");
        }
    }

    #[test]
    fn concurrent_atomic_writes_leave_one_complete_file() {
        let dir = std::env::temp_dir().join(format!("linus-gpu-nvrtc-test-{}", std::process::id()));
        let path = dir.join("k.cubin");
        let contents: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; 1 << 16]).collect();
        std::thread::scope(|s| {
            for c in &contents {
                let path = &path;
                s.spawn(move || {
                    for _ in 0..20 {
                        write_atomic(path, c).unwrap();
                    }
                });
            }
        });
        let got = std::fs::read(&path).unwrap();
        assert!(contents.contains(&got));
        let left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left.len(), 1, "temporary files left: {left:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
