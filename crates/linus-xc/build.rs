//! Build script for `linus-xc`: provides a static libxc 7.0.0 and a tiny C shim
//! used by the tests to check our hand-written `#[repr(C)]` layouts.
//!
//! Where libxc comes from, in order:
//! 1. `LIBXC_DIR=<prefix>`: a prebuilt install prefix (`<prefix>/include/xc.h`,
//!    `<prefix>/lib{,64}/libxc.a` or `libxc.so`). Must be libxc 7.0.0 exactly.
//! 2. A build cache keyed by (tarball sha256, cmake options, C flags, target, C compiler and
//!    the target macros it predefines, which tell apart compilers that default to another ISA):
//!    `$LINUS_XC_CACHE_DIR`, else `$XDG_CACHE_HOME/linus-xc`, else
//!    `$HOME/.cache/linus-xc`. Shared by all profiles, `cargo clean`, worktrees and
//!    rust-analyzer, so libxc is compiled once per machine/compiler.
//!    `LINUS_XC_CACHE_DIR=off` builds inside `OUT_DIR` instead (redone after
//!    `cargo clean` or for another profile, but never by incremental builds).
//!
//! The tarball is taken from `LIBXC_TARBALL=<file>` if set (offline builds),
//! else from the cache, else downloaded with `curl`; its sha256 is always checked.
//!
//! With the feature `device-sources` (linus-gpu's `gpu` feature turns it on), the same tarball
//! also gives the functional files a GPU compiles at run time (`device_sources`), even with
//! `LIBXC_DIR`; a build that cannot get it fails, unless `LINUS_XC_DEVICE_SOURCES=off` (the
//! GPU engines then evaluate every functional with libxc on the host).

use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const LIBXC_VERSION: &str = "7.0.0";
const LIBXC_URL: &str = "https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz";
/// sha256 of the GitLab archive above (also recorded by Spack for libxc 7.0.0).
const LIBXC_SHA256: &str = "8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77";
/// Bump when the recipe below changes, to invalidate existing caches.
const RECIPE_REVISION: u32 = 2;

/// C flags for libxc on top of CMake's Release flags (-O3 -DNDEBUG). No floating-point
/// contraction: a compiler whose default ISA has FMA would otherwise fuse a*b+c and change
/// libxc's last bits by the build host. Ubuntu 26.04's amd64v3 package variant (GitHub's
/// ubuntu-26.04 runner) ships a gcc that defaults to -march=x86-64-v3, and clang contracts by
/// default on arm64. With it, libxc computes the same bits as an x86-64 baseline build
/// (PySCF's libxc, the manylinux wheels) on any of them; baseline builds are unchanged.
const LIBXC_CFLAGS: &[&str] = &["-ffp-contract=off"];

/// CMake options for libxc. kxc/lxc off (Hessians only need fxc), no Fortran,
/// no tests, no `-march=native` (results and binaries portable across x86-64).
const CMAKE_DEFINES: &[(&str, &str)] = &[
    ("BUILD_SHARED_LIBS", "OFF"),
    ("BUILD_FPIC", "ON"),
    ("BUILD_TESTING", "OFF"),
    ("ENABLE_FORTRAN", "OFF"),
    ("ENABLE_PYTHON", "OFF"),
    ("ENABLE_CUDA", "OFF"),
    ("ENABLE_XHOST", "OFF"),
    ("DISABLE_VXC", "OFF"),
    ("DISABLE_FXC", "OFF"),
    ("DISABLE_KXC", "ON"),
    ("DISABLE_LXC", "ON"),
    ("DISABLE_FHC", "OFF"),
    ("CMAKE_INSTALL_LIBDIR", "lib"),
    ("CMAKE_INSTALL_INCLUDEDIR", "include"),
    // libxc 7.0.0 declares cmake_minimum_required(VERSION 3.1); CMake >= 4 refuses
    // that unless a policy floor is given.
    ("CMAKE_POLICY_VERSION_MINIMUM", "3.5"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/layout_check.c");
    println!("cargo:rerun-if-changed=csrc/glibc_cbrt.c");
    for v in [
        "LIBXC_DIR",
        "LIBXC_TARBALL",
        "LINUS_XC_CACHE_DIR",
        "LINUS_XC_ALLOW_VERSION_MISMATCH",
        "LINUS_XC_GLIBC_CBRT",
        "LINUS_XC_DEVICE_SOURCES",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    let prefix = match env::var_os("LIBXC_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => cached_build(),
    };
    if env::var_os("CARGO_FEATURE_DEVICE_SOURCES").is_some() {
        device_sources();
    }

    let include = prefix.join("include");
    check_version(&include);
    let libdir = ["lib", "lib64"]
        .iter()
        .map(|d| prefix.join(d))
        .find(|d| d.join("libxc.a").exists() || d.join("libxc.so").exists())
        .unwrap_or_else(|| {
            panic!(
                "no libxc.a or libxc.so under {}/lib{{,64}}",
                prefix.display()
            )
        });

    println!("cargo:rustc-link-search=native={}", libdir.display());
    if libdir.join("libxc.a").exists() {
        println!("cargo:rustc-link-lib=static=xc");
    } else {
        println!(
            "cargo:warning=linus-xc: no libxc.a in {}, linking libxc.so dynamically",
            libdir.display()
        );
        println!("cargo:rustc-link-lib=dylib=xc");
    }
    println!("cargo:rustc-link-lib=m");
    // Exported to dependents as DEP_XC_ROOT / DEP_XC_INCLUDE.
    println!("cargo:root={}", prefix.display());
    println!("cargo:include={}", include.display());
    println!("cargo:rustc-env=LINUS_XC_LIBXC_PREFIX={}", prefix.display());

    // Bind libxc's cbrt to glibc's instead of compiler-builtins' weak cbrt
    // (see csrc/glibc_cbrt.c). Opt out with LINUS_XC_GLIBC_CBRT=0.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let want_glibc_cbrt = env::var("LINUS_XC_GLIBC_CBRT").map_or(true, |v| v != "0");
    println!("cargo::rustc-check-cfg=cfg(linus_xc_glibc_cbrt)");
    if want_glibc_cbrt
        && target_os == "linux"
        && target_env == "gnu"
        && (arch == "x86_64" || arch == "aarch64")
    {
        cc::Build::new()
            .file("csrc/glibc_cbrt.c")
            .warnings(false)
            .compile("linus_xc_glibc_cbrt");
        println!("cargo:rustc-cfg=linus_xc_glibc_cbrt");
    }

    // Layout shim: sizeof/offsetof of the libxc structs we mirror in Rust,
    // compiled against the very headers we link against.
    cc::Build::new()
        .file("csrc/layout_check.c")
        .include(&include)
        .warnings(true)
        .compile("linus_xc_layout");
}

/// The directory of the libxc build cache and the tarball: [`cache_root`], else (the cache off,
/// or no home) `OUT_DIR/libxc-cache`; created.
fn build_root() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let root = cache_root().unwrap_or_else(|| out_dir.join("libxc-cache"));
    fs::create_dir_all(&root)
        .unwrap_or_else(|e| panic!("cannot create cache dir {}: {e}", root.display()));
    root
}

/// Return the install prefix of a cached libxc build, building it if needed.
fn cached_build() -> PathBuf {
    let cache_root = build_root();

    let key = cache_key();
    let prefix = cache_root.join(format!("libxc-{LIBXC_VERSION}-{key}"));
    let stamp = prefix.join(".linus-xc-complete");
    if stamp.exists() {
        return prefix;
    }

    // Serialize concurrent builds (two cargo invocations, rust-analyzer, ...).
    let lock_path = cache_root.join(format!("libxc-{LIBXC_VERSION}-{key}.lock"));
    let lock = fs::File::create(&lock_path).expect("cannot create lock file");
    lock.lock().expect("cannot lock libxc cache");
    if stamp.exists() {
        return prefix;
    }

    let tarball = tarball(&cache_root).unwrap_or_else(|e| panic!("{e}"));
    let src_root = cache_root.join(format!("libxc-{LIBXC_VERSION}-{key}.src"));
    let _ = fs::remove_dir_all(&src_root);
    let _ = fs::remove_dir_all(&prefix);
    fs::create_dir_all(&src_root).unwrap();
    // The test suite is 44 MB of reference data we never build.
    run(Command::new("tar")
        .arg("-xzf")
        .arg(&tarball)
        .arg("-C")
        .arg(&src_root)
        .arg(format!("--exclude=libxc-{LIBXC_VERSION}/testsuite")));
    let src = src_root.join(format!("libxc-{LIBXC_VERSION}"));
    assert!(
        src.join("CMakeLists.txt").exists(),
        "unexpected tarball layout"
    );

    let t0 = std::time::Instant::now();
    let mut cfg = cmake::Config::new(&src);
    cfg.out_dir(&prefix)
        .profile("Release")
        .no_default_flags(true);
    for flag in libxc_cflags() {
        cfg.cflag(flag);
    }
    for (k, v) in CMAKE_DEFINES {
        cfg.define(k, v);
    }
    cfg.build();
    println!(
        "cargo:warning=linus-xc: built libxc {LIBXC_VERSION} in {:.0} s into {}",
        t0.elapsed().as_secs_f64(),
        prefix.display()
    );

    // Keep only the install tree (lib/, include/) plus the CMake cache for provenance.
    let share = prefix.join("share").join("linus-xc");
    fs::create_dir_all(&share).unwrap();
    let _ = fs::copy(
        prefix.join("build").join("CMakeCache.txt"),
        share.join("CMakeCache.txt"),
    );
    let _ = fs::remove_dir_all(prefix.join("build"));
    let _ = fs::remove_dir_all(&src_root);
    fs::write(
        &stamp,
        format!("libxc {LIBXC_VERSION} sha256 {LIBXC_SHA256}\n"),
    )
    .unwrap();
    prefix
}

fn cache_root() -> Option<PathBuf> {
    if let Some(v) = env::var_os("LINUS_XC_CACHE_DIR") {
        let s = v.to_string_lossy().to_ascii_lowercase();
        if s == "off" || s == "0" || s == "none" || s.is_empty() {
            return None;
        }
        return Some(PathBuf::from(v));
    }
    let base = env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let dir = base.join("linus-xc");
    fs::create_dir_all(&dir).ok().map(|_| dir)
}

/// [`LIBXC_CFLAGS`] for a GCC-style compiler driver (none for MSVC).
fn libxc_cflags() -> &'static [&'static str] {
    if cc::Build::new().get_compiler().is_like_msvc() {
        &[]
    } else {
        LIBXC_CFLAGS
    }
}

/// Hash of everything that changes the produced library.
fn cache_key() -> String {
    let target = env::var("TARGET").unwrap();
    let compiler = cc::Build::new().get_compiler();
    // The C flags CMake gets: ours, plus what cmake-rs adds from the environment (CFLAGS,
    // CFLAGS_<target>, ...) through a cc::Build configured as its own (no default flags; the
    // optimization level is CMake's), so the key does not depend on the cargo profile.
    let cmake_c = cc::Build::new()
        .opt_level(0)
        .debug(false)
        .warnings(false)
        .no_default_flags(true)
        .get_compiler();
    let mut cflags: Vec<String> = libxc_cflags().iter().map(|f| f.to_string()).collect();
    cflags.extend(
        cmake_c
            .args()
            .iter()
            .map(|a| a.to_string_lossy().into_owned()),
    );
    // Its predefined macros at those flags (__FMA__, __AVX2__, ... for a compiler whose default
    // ISA is x86-64-v3): `cc --version` is the same for Ubuntu's amd64 and amd64v3 gcc.
    let macros = if compiler.is_like_msvc() {
        Vec::new()
    } else {
        Command::new(compiler.path())
            .args(&cflags)
            .args(["-dM", "-E", "-x", "c", "-"])
            .stdin(std::process::Stdio::null())
            .output()
            .map(|o| o.stdout)
            .unwrap_or_default()
    };
    let cc_version = Command::new(compiler.path())
        .arg("--version")
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_default();
    let mut h = Sha256::new();
    h.update(LIBXC_SHA256);
    h.update(RECIPE_REVISION.to_le_bytes());
    for (k, v) in CMAKE_DEFINES {
        h.update(format!("{k}={v};"));
    }
    h.update(&target);
    h.update(compiler.path().to_string_lossy().as_bytes());
    h.update(&cc_version);
    for f in &cflags {
        h.update(format!("{f};"));
    }
    h.update(&macros);
    let hex = format!("{:x}", h.finalize());
    hex[..12].to_string()
}

/// Path to a verified libxc tarball: `LIBXC_TARBALL`, else the copy in `cache_root`, else
/// downloaded into it; or why there is none.
fn tarball(cache_root: &Path) -> Result<PathBuf, String> {
    if let Some(p) = env::var_os("LIBXC_TARBALL") {
        let p = PathBuf::from(p);
        verify_sha256(&p).map_err(|e| format!("LIBXC_TARBALL {}: {e}", p.display()))?;
        return Ok(p);
    }
    let dest = cache_root.join(format!("libxc-{LIBXC_VERSION}.tar.gz"));
    if dest.exists() && verify_sha256(&dest).is_ok() {
        return Ok(dest);
    }
    let part = cache_root.join(format!("libxc-{LIBXC_VERSION}.tar.gz.part"));
    let status = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&part)
        .arg(LIBXC_URL)
        .status();
    match status {
        Ok(s) if s.success() => {}
        other => {
            return Err(format!(
                "downloading {LIBXC_URL} failed ({other:?}). Set LIBXC_TARBALL to a local copy \
                 (sha256 {LIBXC_SHA256}) or LIBXC_DIR to a prebuilt libxc {LIBXC_VERSION}."
            ))
        }
    }
    verify_sha256(&part).map_err(|e| format!("downloaded libxc tarball: {e}"))?;
    fs::rename(&part, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    Ok(dest)
}

fn verify_sha256(path: &Path) -> Result<(), String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let got = format!("{:x}", h.finalize());
    if got == LIBXC_SHA256 {
        Ok(())
    } else {
        Err(format!(
            "sha256 mismatch: got {got}, expected {LIBXC_SHA256}"
        ))
    }
}

/// Refuse anything but libxc 7.0.0 (cuEST pins the same version).
fn check_version(include: &Path) {
    let hdr = include.join("xc_version.h");
    let text =
        fs::read_to_string(&hdr).unwrap_or_else(|e| panic!("cannot read {}: {e}", hdr.display()));
    let found = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("#define XC_VERSION "))
        .map(|v| v.trim().trim_matches('"').to_string())
        .unwrap_or_default();
    if found != LIBXC_VERSION {
        let msg = format!(
            "libxc at {} is version '{found}', linus-xc pins {LIBXC_VERSION}",
            include.display()
        );
        if env::var_os("LINUS_XC_ALLOW_VERSION_MISMATCH").is_some() {
            println!("cargo:warning={msg} (allowed by LINUS_XC_ALLOW_VERSION_MISMATCH)");
        } else {
            panic!("{msg}; set LINUS_XC_ALLOW_VERSION_MISMATCH=1 to override");
        }
    }
}

fn run(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("cannot run {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed: {status}");
}

/// The libxc source files of the functionals a device can evaluate (feature `device-sources`):
/// the main file of each (its functional numbers and parameter structs) and the Maple-generated
/// code it includes. The device engines compile them at run time (NVRTC) for the leaf functionals
/// of a functional's libxc tree; anything else is evaluated by libxc on the host.
const DEVICE_FILES: &[&str] = &[
    "lda_x",
    "lda_c_vwn",
    "lda_c_vwn_rpa",
    "lda_c_pw",
    "lda_c_pz",
    "gga_x_b88",
    "gga_c_lyp",
    "gga_x_pbe",
    "gga_c_pbe",
    "gga_c_p86",
    "gga_x_pw91",
    "gga_c_pw91",
    "gga_xc_b97",
    "hyb_gga_xc_wb97",
    "mgga_x_r2scan",
    "mgga_c_r2scan",
    "mgga_x_scan",
    "mgga_c_scan",
    "mgga_x_tpss",
    "mgga_c_tpss",
    "hyb_mgga_xc_wb97mv",
    "mgga_xc_b97mv",
    "mgga_x_m06l",
    "mgga_c_m06l",
];

/// Writes `$OUT_DIR/device_sources.rs` (included by `src/device.rs`): per file of
/// [`DEVICE_FILES`] its functional numbers, parameter structs, the struct its Maple code reads and
/// that struct's size in doubles, and the Maple code up to second derivatives (the third and
/// fourth are dropped: no engine uses them). Without a libxc source tarball the table is empty and
/// says why (the device engines then evaluate every functional on the host).
fn device_sources() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dir = out_dir.join("device-src");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let gen = out_dir.join("device_sources.rs");
    let entries = match tarball(&build_root()).and_then(|t| device_extract(&t, &dir)) {
        Ok(e) => e,
        Err(why) => {
            // a GPU build without them would quietly evaluate every functional on the host
            if env::var("LINUS_XC_DEVICE_SOURCES").as_deref() != Ok("off") {
                panic!(
                    "linus-xc: the device sources need the libxc {LIBXC_VERSION} tarball ({why}); \
                     set LIBXC_TARBALL to a copy (sha256 {LIBXC_SHA256}), or \
                     LINUS_XC_DEVICE_SOURCES=off to build without them (the GPU engines then \
                     evaluate functionals on the host)"
                );
            }
            println!("cargo:warning=linus-xc: no device sources ({why})");
            fs::write(
                &gen,
                format!(
                    "pub(crate) const UNAVAILABLE: Option<&str> = Some({why:?});\n\
                     pub(crate) static SOURCES: &[DeviceSource] = &[];\n"
                ),
            )
            .unwrap();
            return;
        }
    };
    let mut s = String::from(
        "pub(crate) const UNAVAILABLE: Option<&str> = None;\n\
         pub(crate) static SOURCES: &[DeviceSource] = &[\n",
    );
    for e in &entries {
        s += &format!(
            "    DeviceSource {{ file: {:?}, family: {:?}, numbers: &{:?}, params_type: {:?}, \
             params_doubles: {:?}, typedefs: include_str!(concat!(env!(\"OUT_DIR\"), \
             \"/device-src/{}.types.c\")), maple: include_str!(concat!(env!(\"OUT_DIR\"), \
             \"/device-src/{}.maple.c\")) }},\n",
            e.file, e.family, e.numbers, e.params_type, e.params_doubles, e.file, e.file
        );
    }
    s += "];\n";
    fs::write(&gen, s).unwrap();
}

/// One file of [`DEVICE_FILES`] as `device_sources.rs` describes it.
struct DeviceEntry {
    file: String,
    family: String,
    numbers: Vec<i32>,
    params_type: Option<String>,
    params_doubles: Option<usize>,
}

/// Extracts the main and Maple files of [`DEVICE_FILES`] from `tarball` into `dir` (the Maple
/// code cut to orders 0-2 without its order macros; the main file's parameter structs) and
/// describes each.
fn device_extract(tarball: &Path, dir: &Path) -> Result<Vec<DeviceEntry>, String> {
    let tmp = dir.join("tar");
    fs::create_dir_all(&tmp).unwrap();
    let base = format!("libxc-{LIBXC_VERSION}/src");
    let untar = |members: &[String]| -> Result<(), String> {
        let st = Command::new("tar")
            .arg("-xzf")
            .arg(tarball)
            .arg("-C")
            .arg(&tmp)
            .args(members)
            .status()
            .map_err(|e| format!("tar: {e}"))?;
        if st.success() {
            Ok(())
        } else {
            Err(format!("tar failed on {}: {st}", tarball.display()))
        }
    };
    untar(
        &DEVICE_FILES
            .iter()
            .map(|f| format!("{base}/{f}.c"))
            .collect::<Vec<_>>(),
    )?;
    let mut mains = Vec::new();
    for f in DEVICE_FILES {
        let text = fs::read_to_string(tmp.join(&base).join(format!("{f}.c")))
            .map_err(|e| format!("{f}.c: {e}"))?;
        let incs: Vec<String> = text
            .lines()
            .filter_map(|l| l.trim().strip_prefix("#include \"maple2c/"))
            .map(|r| r.trim_end_matches('"').to_string())
            .collect();
        if incs.len() != 1 {
            return Err(format!("{f}.c includes {} Maple files", incs.len()));
        }
        mains.push((f.to_string(), text, incs[0].clone()));
    }
    untar(
        &mains
            .iter()
            .map(|(_, _, m)| format!("{base}/maple2c/{m}"))
            .collect::<Vec<_>>(),
    )?;
    let mut out = Vec::new();
    for (f, text, maple_rel) in mains {
        let maple = fs::read_to_string(tmp.join(&base).join("maple2c").join(&maple_rel))
            .map_err(|e| format!("{maple_rel}: {e}"))?;
        let family = maple_rel.split('_').next().unwrap_or("").to_string();
        let numbers: Vec<i32> = text
            .lines()
            .filter_map(|l| {
                let mut w = l.split_whitespace();
                if w.next() != Some("#define") || !w.next()?.starts_with("XC_") {
                    return None;
                }
                w.next()?.parse().ok()
            })
            .collect();
        let typedefs = struct_typedefs(&text);
        let params_type = maple.lines().find_map(|l| {
            let r = l.trim().strip_prefix("params = (")?;
            Some(
                r.split_whitespace()
                    .next()?
                    .trim_end_matches('*')
                    .to_string(),
            )
        });
        let params_doubles = params_type
            .as_ref()
            .and_then(|t| typedefs.iter().find(|(n, _)| n == t))
            .and_then(|(_, body)| count_doubles(body));
        fs::write(
            dir.join(format!("{f}.types.c")),
            typedefs
                .iter()
                .map(|(_, b)| b.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        fs::write(dir.join(format!("{f}.maple.c")), cut_maple(&maple)).unwrap();
        out.push(DeviceEntry {
            file: f,
            family,
            numbers,
            params_type,
            params_doubles,
        });
    }
    let _ = fs::remove_dir_all(&tmp);
    Ok(out)
}

/// The `typedef struct { ... } name;` blocks of a libxc source file, (name, text) each.
fn struct_typedefs(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("typedef struct") {
        let tail = &rest[i..];
        let Some(open) = tail.find('{') else { break };
        let Some(close) = tail[open..].find('}') else {
            break;
        };
        let after = &tail[open + close + 1..];
        let Some(semi) = after.find(';') else { break };
        let name = after[..semi].trim().to_string();
        let end = open + close + 1 + semi + 1;
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            out.push((name, tail[..end].to_string()));
        }
        rest = &tail[end..];
    }
    out
}

/// The doubles of a parameter struct whose members are all `double` scalars or arrays of a
/// literal size (None for anything else: no device evaluation then).
fn count_doubles(block: &str) -> Option<usize> {
    let body = &block[block.find('{')? + 1..block.rfind('}')?];
    // without comments
    let mut clean = String::new();
    let mut s = body;
    loop {
        match (s.find("/*"), s.find("//")) {
            (Some(a), b) if b.is_none_or(|b| a < b) => {
                clean += &s[..a];
                s = &s[a + 2..];
                s = &s[s.find("*/")? + 2..];
            }
            (_, Some(b)) => {
                clean += &s[..b];
                s = &s[b..];
                s = s.find('\n').map_or("", |n| &s[n..]);
            }
            _ => {
                clean += s;
                break;
            }
        }
    }
    let mut n = 0;
    for decl in clean.split(';').map(str::trim).filter(|d| !d.is_empty()) {
        let names = decl.strip_prefix("double")?;
        for v in names.split(',') {
            let v = v.trim();
            n += match v.find('[') {
                Some(i) => v[i + 1..v.find(']')?].trim().parse::<usize>().ok()?,
                None if !v.is_empty() && v.chars().all(|c| c.is_alphanumeric() || c == '_') => 1,
                None => return None,
            };
        }
    }
    Some(n)
}

/// Maple-generated code without its third and fourth derivatives and its order macros.
fn cut_maple(text: &str) -> String {
    let mut out = String::new();
    let (mut depth, mut skip_at) = (0usize, 0usize);
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("#if") {
            depth += 1;
            if skip_at == 0
                && (t == "#ifndef XC_DONT_COMPILE_KXC" || t == "#ifndef XC_DONT_COMPILE_LXC")
            {
                skip_at = depth;
                continue;
            }
        } else if t.starts_with("#endif") {
            let closing = depth;
            depth = depth.saturating_sub(1);
            if skip_at != 0 && closing == skip_at {
                skip_at = 0;
                continue;
            }
        }
        if skip_at != 0
            || t.starts_with("#define maple2c_order")
            || t.starts_with("#define MAPLE2C_FLAGS")
        {
            continue;
        }
        out += line;
        out.push('\n');
    }
    out
}
