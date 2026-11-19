//! `linus` command-line driver: `linus mol.xyz [options]` runs a calculation on an XYZ file (Å)
//! and prints a summary, or with `--json` the result as JSON; `linus request.json [options]`
//! runs a JSON request (`linus --json < request.json` reads it on stdin). `linus --help` lists
//! the flags, `linus --list-settings` every setting; docs/cli.md has the request and the
//! result. The request, the settings and their grammar are the library's
//! (`linus::Request`, `linus::CalcOptions`), the result its report (`linus::driver::report`).
//!
//! Each flag is one key of the request (`--conv` is `scf.convergence`, `--charge`
//! `structure.charge`); `--set KEY=VALUE` sets any key. A request file or `--options` file is
//! the base; flags, `--options` and `--set` apply in command-line order, the last one wins.
//!
//! Exit codes: 0 success, 1 a failed calculation (an SCF that does not converge, not enough
//! memory, a numerical failure, something not implemented, an internal error), 2 bad input
//! (command line, request, settings, basis set names). With `--json`, failures print
//! `{"schema_version": 1, "success": false, "error": ..., "error_kind": ..., "scf": ...}` on
//! stdout as well; `error_kind` is the library's [`LinusError::kind`] (`input`, `basis`,
//! `convergence`, `unsupported`, `resource`, `backend`, `numerical`; never `stopped`, which only
//! an in-process caller's stop request gives) or `internal` for a panic
//! (a bug), and `error_code` the library's [`LinusError::code`] where the error has one
//! (`unknown_key`, `no_effect`, `not_applicable`, `host_memory`, `device_memory`).

use linus::driver::available_functionals;
use linus::driver::molden::MOLDEN_FILE;
use linus::driver::report::{FrequencyOut, Report, SCHEMA_VERSION};
use linus::driver::request::{merge, set_path};
use linus::driver::schema;
use linus::linus_ints::basis::{available_basis_sets, search_path};
use linus::{CalcType, LinusError, Session};
use serde_json::{json, Map, Value};
use std::io::Read;

const USAGE: &str = "usage: linus mol.xyz [options]          (run `linus --help` for the options)
       linus request.json [options]     (a JSON request; `linus --json < request.json` reads stdin)
       linus schema settings|record [--calctype C]";

const HELP: &str = "linus: Hartree-Fock and Kohn-Sham DFT energies, gradients and Hessians with frequencies.

usage: linus mol.xyz [options]          XYZ file in Å; prints a summary (--json: the result as JSON)
       linus request.json [options]     a JSON request (docs/cli.md); also on stdin: linus --json < request.json
       linus schema settings|record [--calctype C]   the JSON Schema of the settings or the run record

Each flag sets one key of the request (in brackets); --set KEY=VALUE sets any. A request file or
--options FILE is the base; flags, --options and --set then apply in order, the last one wins.

Calculation (calctype; default energy):
  --grad, --hess              gradient, or Hessian with the harmonic analysis and thermochemistry
  --ir                        [freq.ir] IR intensities and the polarizability (implies --hess)
  --thermo M                  [freq.thermo] rrho (default), qrrho, qrrho-hg, quasiharmonic or off
  -T, --temperature K         [freq.temperature] (298.15; a list with --set freq.temperature=[250,300])
  --pressure PA               [freq.pressure] (101325)
  --concentration MOL/L       [freq.concentration] standard state by concentration instead
  --freq-scale X              [freq.scale] frequency scale factor (1)

Structure:
  --charge Q                  [structure.charge] total charge (0)
  --mult M                    [structure.multiplicity] spin multiplicity 2S+1 (1)

Model:
  --method NAME               [method] hf (default) or a functional: pbe, b3lyp, r2scan, wb97x-d3bj, wb97m-v,
                              libxc names, mixtures \"0.5*b3lyp + 0.5*hf\"; a dispersion suffix such as
                              b3lyp-d3bj sets the dispersion (--list-functionals)
  --basis NAME                [basis] def2-svp (default), def2-tzvp, cc-pvdz, ... (--list-basis)
  --dispersion D              [dispersion] auto, d3bj, d3bj-atm, d4 or none
  --reference R               [reference] auto, restricted (rhf, rks) or unrestricted (uhf, uks)
  --cartesian                 [cartesian] Cartesian d and higher functions
  --basis-dir DIR             [basis_dir] a directory of BSE JSON basis set files, searched first (repeatable)
  --solvent NAME|EPS          [solvent] implicit solvent by name (--list-solvents) or permittivity (C-PCM)
  --solvent-model M           [solvent.model] cpcm, cosmo, iefpcm, ssvpe, smd
  --eps X                     [solvent.eps] the permittivity (overrides the named solvent's)

Approximations:
  --jk K                      [jk] df (default), exact, incore or rij_snk (synonyms ri, rijcosx, ...)
  --grid G                    [grid.xc] auto, N or pyscf:N (0-9), linus:N (1-4), orca:N (1-3), psi4
  --precision P               [precision] the GPU's arithmetic: auto, fp64 (FP64 throughout), emulated (FP64
                              emulation of the exchange GEMMs) or mixed (opt-in FP32 products; docs/gpu.md)

SCF:
  --conv PRESET               [scf.convergence] loose, normal, tight, verytight, extreme, pyscf
  --e-tol X, --g-tol X        [scf.e_tol, scf.g_tol] energy and orbital-gradient tolerances
  --max-iter N                [scf.max_iter] (288)
  --guess G                   [scf.guess] sad (default) or core
  --stability S               [scf.stability] auto (default), off or always

Resources and output:
  -n, --threads N             [threads] (1)
  --memory GIB                [memory_gib] memory the calculation may use (default: from the machine)
  --backend B                 [backend] cpu (default), gpu or auto (docs/gpu.md)
  --orbitals                  [output.orbitals] MO coefficients in the JSON (--set output.hamiltonian=true:
                              the Fock matrices with the overlap; output.overlap, output.core_hamiltonian)
  --molden FILE               [output.molden] write the orbitals as a Molden file (the basis set's spherical
                              or Cartesian functions; --set output.molden=cartesian for Cartesian ones)
  -v, --verbose               [output.verbose] SCF progress on stderr
  --json                      print the result as JSON
  --plan                      print what the calculation will do, as JSON, without running it

Any key:
  --set KEY=VALUE             a key by its JSON name, dotted in a group, VALUE as JSON or a string:
                              --set scf.soscf=true --set jk.df_storage=f48 --set structure.charge=1
  --options FILE              keys from a JSON object (settings or a whole request)

Information:
  --list-settings [GROUP] [--all]   every setting with its type, default and meaning (--all: also the
                              development ones); the groups: top, jk, grid, precision, solvent, scf,
                              response, freq, backend, output
  --list-basis, --list-functionals, --list-solvents, --version, -h/--help

Exit codes: 0 success, 1 failed calculation, 2 bad input.";

/// A calculation: the request and what to print.
#[derive(Debug)]
struct Job {
    req: linus::Request,
    /// Print the plan instead of running (`--plan`).
    plan: bool,
    /// Write the Molden file of the orbitals here (`--molden FILE`).
    molden: Option<String>,
}

/// An error before or during the calculation, with its exit code.
struct Failure {
    code: i32,
    /// `error_kind` of the failure JSON: a [`LinusError::kind`], or `internal` for a panic.
    kind: &'static str,
    /// `error_code` of the failure JSON ([`LinusError::code`]), when the error has one.
    error_code: Option<&'static str>,
    /// `error_details` of the failure JSON (`linus::driver::report::error_details`): a memory
    /// failure's `required_gib` and `available_gib`.
    details: Option<Box<Value>>,
    msg: String,
    /// The calculation's failure report (`--json`).
    json: Option<Value>,
}

impl From<String> for Failure {
    fn from(msg: String) -> Self {
        Failure {
            code: 2,
            kind: "input",
            error_code: None,
            details: None,
            msg,
            json: None,
        }
    }
}

impl From<LinusError> for Failure {
    fn from(e: LinusError) -> Self {
        Failure {
            code: if e.is_input() { 2 } else { 1 },
            kind: e.kind(),
            error_code: e.code(),
            details: linus::driver::report::error_details(&e).map(Box::new),
            msg: e.to_string(),
            json: None,
        }
    }
}

/// A flag's or `--set`'s value: JSON if it parses, else the string.
fn value_of(s: &str) -> Value {
    serde_json::from_str(s.trim()).unwrap_or_else(|_| Value::String(s.trim().to_string()))
}

/// The key each valued flag sets.
fn flag_key(flag: &str) -> Option<&'static str> {
    Some(match flag {
        "--charge" => "structure.charge",
        "--mult" => "structure.multiplicity",
        "--method" => "method",
        "--basis" => "basis",
        "--dispersion" => "dispersion",
        "--reference" => "reference",
        "--solvent" => "solvent",
        "--solvent-model" => "solvent.model",
        "--eps" => "solvent.eps",
        "--jk" => "jk",
        "--grid" => "grid.xc",
        "--precision" => "precision",
        "--conv" => "scf.convergence",
        "--e-tol" => "scf.e_tol",
        "--g-tol" => "scf.g_tol",
        "--max-iter" => "scf.max_iter",
        "--guess" => "scf.guess",
        "--stability" => "scf.stability",
        "--thermo" => "freq.thermo",
        "--temperature" | "-T" => "freq.temperature",
        "--pressure" => "freq.pressure",
        "--concentration" => "freq.concentration",
        "--freq-scale" => "freq.scale",
        "--threads" | "-n" => "threads",
        "--memory" => "memory_gib",
        "--backend" => "backend",
        _ => return None,
    })
}

/// The key each switch sets to true.
fn switch_key(flag: &str) -> Option<&'static str> {
    Some(match flag {
        "--cartesian" => "cartesian",
        "--orbitals" => "output.orbitals",
        "--verbose" | "-v" => "output.verbose",
        "--ir" => "freq.ir",
        _ => return None,
    })
}

/// The flags, for did-you-mean on unknown ones.
const FLAGS: &[&str] = &[
    "--grad",
    "--hess",
    "--ir",
    "--thermo",
    "--temperature",
    "--pressure",
    "--concentration",
    "--freq-scale",
    "--charge",
    "--mult",
    "--method",
    "--basis",
    "--dispersion",
    "--reference",
    "--cartesian",
    "--basis-dir",
    "--solvent",
    "--solvent-model",
    "--eps",
    "--jk",
    "--grid",
    "--precision",
    "--conv",
    "--e-tol",
    "--g-tol",
    "--max-iter",
    "--guess",
    "--stability",
    "--threads",
    "--memory",
    "--backend",
    "--orbitals",
    "--molden",
    "--verbose",
    "--json",
    "--plan",
    "--set",
    "--options",
    "--list-settings",
    "--list-basis",
    "--list-functionals",
    "--list-solvents",
    "--version",
    "--help",
];

enum Parsed {
    Run(Box<Job>, bool),
    Info(String),
}

/// `linus schema settings|record [--calctype C]`.
fn schema_command(args: &[String]) -> Result<Parsed, Failure> {
    let mut what = None;
    let mut calctype = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--calctype" => {
                calctype = Some(
                    it.next()
                        .ok_or_else(|| "--calctype needs a value".to_string())?
                        .to_ascii_lowercase(),
                )
            }
            w if what.is_none() => what = Some(w.to_string()),
            other => return Err(format!("schema: unknown argument {other:?}\n{USAGE}").into()),
        }
    }
    let calctype = match calctype.as_deref() {
        None => None,
        Some(c @ ("energy" | "gradient" | "hessian")) => Some(c),
        Some(c) => {
            return Err(
                format!("schema: unknown calctype {c:?}; use energy, gradient or hessian").into(),
            )
        }
    };
    let v = match what.as_deref() {
        Some("settings") => schema::json_schema(calctype),
        Some("record") => schema::record_schema(),
        _ => return Err(format!("schema: settings or record\n{USAGE}").into()),
    };
    Ok(Parsed::Info(
        serde_json::to_string_pretty(&v).expect("a schema serializes"),
    ))
}

/// Command line -> request (or information to print).
fn parse_args(args: &[String]) -> Result<Parsed, Failure> {
    if args.first().map(String::as_str) == Some("schema") {
        return schema_command(&args[1..]);
    }
    let mut file: Option<String> = None;
    let (mut json_out, mut plan) = (false, false);
    // the request's edits in command-line order: (key, value) or an --options file's object
    enum Edit {
        Set(String, Value),
        Merge(String, Map<String, Value>),
    }
    let mut edits: Vec<Edit> = vec![];
    let mut calctype: Option<CalcType> = None;
    let mut basis_dirs: Vec<String> = vec![];
    let mut molden: Option<String> = None;
    let (mut list_basis, mut list_settings, mut all) = (false, None::<Option<String>>, false);
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{a} needs a value"))
        };
        if let Some(key) = flag_key(a) {
            let v = val()?;
            edits.push(Edit::Set(key.into(), value_of(&v)));
            continue;
        }
        if let Some(key) = switch_key(a) {
            edits.push(Edit::Set(key.into(), json!(true)));
            continue;
        }
        match a.as_str() {
            "--json" => json_out = true,
            "--plan" => plan = true,
            "--basis-dir" => basis_dirs.push(val()?),
            "--molden" => molden = Some(val()?),
            "--set" => {
                let kv = val()?;
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("--set {kv:?}: use KEY=VALUE"))?;
                edits.push(Edit::Set(k.trim().to_string(), value_of(v)));
            }
            "--options" => {
                let f = val()?;
                let text = std::fs::read_to_string(&f)
                    .map_err(|e| format!("--options: cannot read {f}: {e}"))?;
                match serde_json::from_str(&text)
                    .map_err(|e| format!("--options {f}: bad JSON: {e}"))?
                {
                    Value::Object(o) => edits.push(Edit::Merge(f, o)),
                    _ => return Err(format!("--options {f}: the file must hold a JSON object").into()),
                }
            }
            "--grad" => calctype = Some(CalcType::Gradient),
            "--hess" => calctype = Some(CalcType::Hessian),
            "--list-settings" => {
                let group = it.next_if(|g| !g.starts_with('-')).cloned();
                list_settings = Some(group);
            }
            "--all" => all = true,
            "--list-basis" => list_basis = true,
            "--list-functionals" => {
                return Ok(Parsed::Info(format!(
                    "{}\n(also libxc names such as gga_x_pbe + gga_c_pbe, and mixtures such as \"0.5*b3lyp + 0.5*hf\")",
                    available_functionals().join("\n")
                )))
            }
            "--list-solvents" => {
                let names: Vec<String> = linus::solvent::db::solvents().iter().map(|x| format!("{}\t{}{}", x.name, x.eps, if x.smd.is_some() { "\tsmd" } else { "" })).collect();
                let full = match linus::solvent::db::crc_table() {
                    Some(p) => format!("\n(and the other liquids of the CRC table {})", p.display()),
                    None => "\n(the other CRC liquids resolve where $LINUS_SOLVENT_TABLE names qcref's data/solvents/crc.csv)".into(),
                };
                return Ok(Parsed::Info(format!("name\teps\t(smd: SMD descriptors)\n{}{full}", names.join("\n"))));
            }
            "--version" | "-V" => {
                return Ok(Parsed::Info(format!(
                    "linus {} (numerics epoch {})",
                    linus::version(),
                    linus::NUMERICS_EPOCH
                )))
            }
            "-h" | "--help" => return Ok(Parsed::Info(HELP.into())),
            f if !f.starts_with('-') && file.is_none() => file = Some(f.to_string()),
            other => {
                let names: Vec<String> = FLAGS.iter().map(|s| s.to_string()).collect();
                let hint = linus::linus_ints::basis::did_you_mean(other, &names);
                return Err(format!("unknown argument {other:?}.{hint}\n{USAGE}").into());
            }
        }
    }
    if let Some(group) = list_settings {
        return schema::list_settings(group.as_deref(), all)
            .map(Parsed::Info)
            .map_err(Failure::from);
    }
    if !basis_dirs.is_empty() {
        edits.push(Edit::Set("basis_dir".into(), json!(basis_dirs)));
    }
    if list_basis {
        let dirs: Vec<std::path::PathBuf> = basis_dirs.iter().map(Into::into).collect();
        return Ok(Parsed::Info(
            available_basis_sets(&search_path(&dirs)).join("\n"),
        ));
    }
    // the base: a request file, a request on stdin, or an XYZ file's structure
    let mut m: Map<String, Value> = match &file {
        Some(path) if path.ends_with(".json") => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            match serde_json::from_str(&text)
                .map_err(|e| format!("{path}: bad JSON request: {e}"))?
            {
                Value::Object(o) => o,
                _ => return Err(format!("{path}: the request must be a JSON object").into()),
            }
        }
        Some(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            Map::from_iter([("structure".to_string(), json!({"xyz": text}))])
        }
        None if json_out => {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            match serde_json::from_str(&text).map_err(|e| format!("bad JSON request: {e}"))? {
                Value::Object(o) => o,
                _ => return Err("the request must be a JSON object".to_string().into()),
            }
        }
        None => return Err(USAGE.to_string().into()),
    };
    for e in edits {
        match e {
            Edit::Set(k, v) => set_path(&mut m, &k, v).map_err(|e| format!("--set {k}: {e}"))?,
            Edit::Merge(f, o) => merge(&mut m, o).map_err(|e| format!("--options {f}: {e}"))?,
        }
    }
    if let Some(c) = calctype {
        m.insert("calctype".into(), json!(c.name()));
    }
    // --molden FILE: output.molden as given, else the basis set's own functions
    if let Some(f) = &molden {
        let given = m.get("output").and_then(|o| o.get("molden")).cloned();
        match given {
            None => {
                let cartesian = m.get("cartesian").and_then(Value::as_bool) == Some(true);
                let form = if cartesian { "cartesian" } else { "spherical" };
                set_path(&mut m, "output.molden", json!(form))
                    .map_err(|e| format!("--molden {f}: {e}"))?;
            }
            Some(v) if v == json!("off") => {
                return Err(format!("--molden {f}: output.molden is \"off\"").into())
            }
            Some(_) => {}
        }
    }
    // --ir is a Hessian's
    let ir = m
        .get("freq")
        .and_then(|f| f.get("ir"))
        .and_then(Value::as_bool)
        == Some(true);
    if ir && calctype.is_none() && m.get("calctype").is_none() {
        m.insert("calctype".into(), json!("hessian"));
    }
    let req = linus::Request::from_json(Value::Object(m))?;
    Ok(Parsed::Run(Box::new(Job { req, plan, molden }), json_out))
}

/// Run a request: the report, or the failure (with the failure JSON).
fn run(req: linus::Request) -> Result<Report, Failure> {
    let ir = req.settings.freq.ir();
    let mut calc = Session::new(req.structure, req.settings)?;
    // a setting this one calculation does not read is an input error (decision D41)
    calc.declare(linus::CalcTypes::of(req.calctype, ir))?;
    let report = calc.run(req.calctype).map_err(|e| {
        let json = calc.failure(&e, req.calctype).to_json();
        Failure {
            code: if e.is_input() { 2 } else { 1 },
            kind: e.kind(),
            error_code: e.code(),
            details: linus::driver::report::error_details(&e).map(Box::new),
            msg: e.to_string(),
            json: Some(json),
        }
    })?;
    // what the user should know about a calculation that succeeded (also in the JSON)
    for w in &report.warnings {
        eprintln!("linus: warning: {}", w.message);
    }
    Ok(report)
}

/// A name as serde writes it (an enum's string form).
fn json_name<T: serde::Serialize>(x: &T) -> String {
    serde_json::to_value(x)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "?".into())
}

/// A threshold for people (`1e-8`).
fn sci(v: &Value) -> String {
    v.as_f64()
        .map_or_else(|| "?".into(), |x| format!("{x:.0e}"))
}

/// The result for people (the JSON output has everything).
fn summary(r: &Report) -> String {
    let res = &r.resolved;
    let kind = match (r.restricted, r.method.eq_ignore_ascii_case("hf")) {
        (true, true) => "RHF",
        (true, false) => "RKS",
        (false, true) => "UHF",
        (false, false) => "UKS",
    };
    let mut s = format!(
        "linus {}  {}/{}  {} atoms, {} basis functions, {kind}, {} alpha / {} beta electrons\n",
        r.version, r.method, r.basis, r.natoms, r.nbf, r.nalpha, r.nbeta
    );
    // one settings line: what the calculation used (resolved.settings)
    let st = &res.settings;
    let text = |v: &Value| match v {
        Value::String(x) => x.clone(),
        v => v.to_string(),
    };
    let mut set = vec![];
    let jk = &st["jk"];
    set.push(match jk.get("aux") {
        Some(a) => format!("jk {} ({})", text(&jk["kind"]), text(a)),
        None => format!("jk {}", text(&jk["kind"])),
    });
    if let Some(g) = jk.get("k_grid") {
        set.push(format!("k-grid {}", text(g)));
    }
    let grid = st.get("grid");
    if let Some(g) = grid.and_then(|g| g.get("xc")) {
        set.push(format!("grid {}", text(g)));
    }
    if let Some(g) = grid.and_then(|g| g.get("kernel")) {
        if Some(g) != grid.and_then(|g| g.get("xc")) {
            set.push(format!("kernel grid {}", text(g)));
        }
    }
    let dispersion = st.get("dispersion").map(text).filter(|d| d != "none");
    if let Some(d) = &dispersion {
        set.push(format!("dispersion {d}"));
    }
    if let Some(sv) = &res.record.solvent {
        let so = &st["solvent"];
        let name = so.get("name").map(text).unwrap_or_else(|| "-".into());
        let eps = sv.eps.map_or_else(|| "-".into(), |e| e.to_string());
        let model = so.get("model").map(text).unwrap_or_default();
        set.push(format!("solvent {name} ({model}, eps {eps})"));
    }
    set.push(format!(
        "SCF e_tol {} g_tol {}",
        sci(&st["scf"]["e_tol"]),
        sci(&st["scf"]["g_tol"])
    ));
    let threads = r.options.threads();
    set.push(format!(
        "{threads} thread{}",
        if threads == 1 { "" } else { "s" }
    ));
    s += &format!("settings: {}\n", set.join(", "));
    s += &format!(
        "SCF converged in {} iterations ({} Fock builds)\n",
        r.iterations, r.scf.fock_builds
    );
    let c = &r.components;
    let rows = [
        ("nuclear repulsion", Some(c.nuclear_repulsion), true),
        ("one-electron", Some(c.one_electron), true),
        ("Coulomb", Some(c.coulomb), true),
        // components that do not apply are left out (no exact exchange in pure DFT, ...)
        ("exact exchange", Some(c.exchange), c.exchange != 0.0),
        ("exchange-correlation", Some(c.xc), res.record.dft),
        ("dispersion", Some(c.dispersion), dispersion.is_some()),
        ("solvation (electr.)", c.solvation_electrostatic, true),
        ("solvation (CDS)", c.solvation_cds, true),
    ];
    for (label, v, applies) in rows {
        if let (Some(v), true) = (v, applies) {
            s += &format!("  {label:<22}{v:>22.12}\n");
        }
    }
    s += &format!("  {:<22}{:>22.12} Eh\n", "total energy", r.energy);
    if !r.restricted {
        s += &format!("<S^2> {:.6}\n", r.s2);
    }
    if let Some(g) = r.homo_lumo_gap {
        s += &format!(
            "HOMO-LUMO gap {g:.6} Eh ({:.4} eV)\n",
            g * 27.211_386_245_981
        );
    }
    let d = r.dipole;
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    // 1 e bohr = 2.541746473 D (CODATA 2022)
    s += &format!(
        "dipole (e bohr) {:.6} {:.6} {:.6}  |mu| {n:.6} e bohr = {:.4} D\n",
        d[0],
        d[1],
        d[2],
        n * 2.541_746_473
    );
    if let Some(g) = &r.gradient {
        s += "gradient (Eh/bohr):\n";
        let (mut gmax, mut ss) = (0.0f64, 0.0);
        for (i, v) in g.iter().enumerate() {
            let sym = r.structure.symbols.get(i).copied().unwrap_or("?");
            s += &format!(
                "  {:4} {sym:<2} {:16.10} {:16.10} {:16.10}\n",
                i + 1,
                v[0],
                v[1],
                v[2]
            );
            for x in v {
                gmax = gmax.max(x.abs());
                ss += x * x;
            }
        }
        s += &format!(
            "  max |g| {gmax:.3e}  rms {:.3e}\n",
            (ss / (3 * g.len()).max(1) as f64).sqrt()
        );
    }
    if let Some(f) = &r.frequencies {
        s += &freq_summary(f);
    }
    let parts: Vec<String> = r
        .timings
        .iter()
        .filter(|(k, _)| *k != "total")
        .map(|(k, v)| format!("{k} {v:.3}"))
        .collect();
    s += &format!(
        "time {:.3} s ({})\n",
        r.timings.get("total").copied().unwrap_or(0.0),
        parts.join(", ")
    );
    s
}

/// Frequencies and thermochemistry, for people (the JSON output has everything).
fn freq_summary(f: &FrequencyOut) -> String {
    let Some(v) = &f.vibrational_analysis else {
        return String::new();
    };
    let ir = v.ir_intensities.as_deref().unwrap_or(&[]);
    let mut s = format!(
        "point group {} (symmetry number {}, {}), {} rotor, {} imaginary\nfrequencies (cm^-1){}:\n",
        v.symmetry.point_group,
        v.symmetry.symmetry_number,
        v.symmetry.source,
        json_name(&v.rotor_type),
        v.n_imaginary,
        if ir.is_empty() { "" } else { " [IR km/mol]" }
    );
    for (i, freq) in v.frequencies.iter().enumerate() {
        let tag = if v.external.get(i) == Some(&true) {
            " (external)"
        } else {
            ""
        };
        let irs = ir.get(i).map(|x| format!(" [{x:.2}]")).unwrap_or_default();
        s += &format!("  {i:4} {freq:12.2}{irs}{tag}\n");
    }
    for t in &f.thermochemistry {
        let o = &t.options;
        let state = match o.concentration {
            Some(c) => format!("{c} mol/L"),
            None => format!("{} Pa", o.pressure),
        };
        let model = match (o.qrrho_entropy, o.qrrho_enthalpy, o.raise_low_frequencies) {
            (true, true, _) => ", quasi-RRHO (Grimme S, Head-Gordon H)",
            (true, false, _) => ", quasi-RRHO (Grimme S)",
            (_, _, Some(_)) => ", quasiharmonic",
            _ => "",
        };
        s += &format!(
            "thermochemistry at {} K, {state}{model}:\n  ZPE {:.8} Eh   E(0 K) {:.8} Eh\n  E {:.8}  H {:.8}  G {:.8} Eh   S {:.6e} Eh/K\n  corrections to E_el: E {:.8}  H {:.8}  G {:.8} Eh\n",
            o.temperature,
            t.zpe,
            t.energy_0k,
            t.total.energy,
            t.total.enthalpy,
            t.total.gibbs,
            t.total.entropy,
            t.thermal_energy_correction,
            t.enthalpy_correction,
            t.gibbs_correction
        );
    }
    let thermo_warnings = f.thermochemistry.iter().flat_map(|t| &t.warnings);
    for w in v.warnings.iter().chain(thermo_warnings) {
        s += &format!("  note: {w}\n");
    }
    s
}

/// Parse the arguments and run: what to print on stdout.
fn execute(args: &[String]) -> Result<String, Failure> {
    match parse_args(args)? {
        Parsed::Info(text) => Ok(format!("{text}\n")),
        Parsed::Run(job, _) if job.plan => {
            let req = job.req;
            let ir = req.settings.freq.ir();
            let mut calc = Session::new(req.structure, req.settings)?;
            calc.declare(linus::CalcTypes::of(req.calctype, ir))?;
            let p = calc.plan(req.calctype)?;
            Ok(format!(
                "{}\n",
                serde_json::to_string_pretty(&p).expect("JSON")
            ))
        }
        Parsed::Run(job, json_out) => {
            let r = run(job.req)?;
            if let (Some(path), Some(text)) = (&job.molden, r.files.get(MOLDEN_FILE)) {
                std::fs::write(path, text).map_err(|e| Failure {
                    code: 1,
                    kind: "resource",
                    error_code: None,
                    details: None,
                    msg: format!("--molden: cannot write {path}: {e}"),
                    json: None,
                })?;
            }
            Ok(if json_out {
                format!("{}\n", serde_json::to_string(&r.to_json()).expect("JSON"))
            } else {
                summary(&r)
            })
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json_flag = args.iter().any(|a| a == "--json");
    // A panic is a bug in linus: it fails like any other error (exit code 1, error_kind
    // "internal", the failure JSON with --json) instead of exiting with code 101 and no JSON.
    // The panic hook still prints its location on stderr.
    let out = std::panic::catch_unwind(|| execute(&args)).unwrap_or_else(|p| {
        let msg = p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic".into());
        Err(Failure {
            code: 1,
            kind: "internal",
            error_code: None,
            details: None,
            msg: format!("internal error in linus: {msg}"),
            json: None,
        })
    });
    match out {
        Ok(text) => print!("{text}"),
        Err(f) => {
            if json_flag {
                let v = f.json.unwrap_or_else(|| {
                    let mut v = json!({"schema_version": SCHEMA_VERSION, "program": "linus", "version": linus::version(), "success": false, "error": f.msg, "error_kind": f.kind});
                    if let Some(c) = f.error_code {
                        v["error_code"] = json!(c);
                    }
                    if let Some(d) = f.details {
                        v["error_details"] = *d;
                    }
                    v
                });
                println!("{v}");
            }
            eprintln!("linus: {}", f.msg);
            std::process::exit(f.code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus::driver::options::{
        Auto, BackendKind, ConvergencePreset, Dispersion, DispersionSetting, ExactStorage,
        Fp32Mode, Fp64Mode, MoldenForm, Reference,
    };
    use linus::CalcOptions;
    use linus::{GridChoice, JkMethod};

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    /// One file per test thread: a test reading it while another rewrites it read it empty.
    fn tmp(name: &str, text: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("linus-cli-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!("{:?}-{name}", std::thread::current().id()));
        std::fs::write(&p, text).unwrap();
        p
    }

    fn job_of(cmd: &str) -> Result<Job, String> {
        let xyz = tmp("he.xyz", "1\n\nHe 0 0 0\n");
        match parse_args(&args(&format!("{} {cmd}", xyz.display()))) {
            Ok(Parsed::Run(j, _)) => Ok(*j),
            Ok(Parsed::Info(_)) => Err("info".into()),
            Err(f) => Err(f.msg),
        }
    }

    fn opts_of(cmd: &str) -> Result<CalcOptions, String> {
        job_of(cmd).map(|j| j.req.settings)
    }

    #[test]
    fn flags_and_set() {
        let o = opts_of("--method b3lyp-d3bj --jk rijcosx --grid linus:2 --memory 4 --set scf.soscf=true --set jk.aux=def2-universal-jfit --set response.tol=1e-9").unwrap();
        assert_eq!(
            (o.method().as_str(), o.dispersion()),
            ("b3lyp", DispersionSetting::Model(Dispersion::D3bj))
        );
        assert!(matches!(o.jk_method(), JkMethod::RijSnk { .. }));
        assert_eq!(o.grid.xc(), GridChoice::Linus { level: 2 });
        assert_eq!(o.memory_gib(), Auto::Value(4.0));
        assert!(o.scf.soscf());
        assert_eq!(o.response.tol(), Auto::Value(1e-9));
        let o = opts_of("--conv verytight --e-tol 1e-9 --max-iter 50 --guess core --jk incore --reference uhf --cartesian").unwrap();
        assert_eq!(
            o.scf.convergence(),
            Auto::Value(ConvergencePreset::VeryTight)
        );
        assert_eq!((o.scf.e_tol(), o.scf.max_iter()), (Auto::Value(1e-9), 50));
        assert_eq!(o.scf.guess(), linus::driver::options::Guess::Core);
        assert_eq!(
            (o.jk_method(), o.jk.exact_storage()),
            (JkMethod::Exact, ExactStorage::Incore)
        );
        assert_eq!(o.reference(), Reference::Unrestricted);
        assert!(o.cartesian());
        // --backend sets the kind; --precision the arithmetic, refined by --set
        let o = opts_of("--backend auto --precision emulated --set precision.fp32=on").unwrap();
        assert_eq!(o.backend.kind(), BackendKind::Auto);
        assert_eq!(
            (o.precision.fp64(), o.precision.fp32()),
            (Fp64Mode::Emulated, Fp32Mode::On)
        );
        let o = opts_of("--precision fp64 --set device_memory_gib=20").unwrap();
        assert_eq!(
            (o.precision.fp64(), o.precision.fp32()),
            (Fp64Mode::Native, Fp32Mode::Off)
        );
        assert_eq!(o.device_memory_gib(), Auto::Value(20.0));
        assert!(opts_of("--grid linus:9").unwrap_err().contains("1-4"));
        assert!(opts_of("--set scf.sosfc=true")
            .unwrap_err()
            .contains("soscf"));
        assert!(opts_of("--dispersion d3")
            .unwrap_err()
            .contains("ambiguous"));
        // the structure flags and their keys
        let j = job_of("--charge 1 --mult 2").unwrap();
        assert_eq!(
            (j.req.structure.charge, j.req.structure.multiplicity),
            (1, 2)
        );
        let j = job_of("--set structure.multiplicity=3").unwrap();
        assert_eq!(j.req.structure.multiplicity, 3);
        // flags that are not keys any more are unknown
        for f in [
            "--aux x",
            "--pcm smd",
            "--int-thresh 1e-10",
            "--uhf",
            "--e-conv 1e-8",
        ] {
            assert!(opts_of(f).unwrap_err().contains("unknown argument"), "{f}");
        }
    }

    /// Flags, --options and --set apply in command-line order; the last one wins.
    #[test]
    fn last_wins_and_requests() {
        let o = opts_of("--threads 2 --set threads=3").unwrap();
        assert_eq!(o.threads(), 3);
        let o = opts_of("--set threads=3 --threads 2").unwrap();
        assert_eq!(o.threads(), 2);
        let f = tmp(
            "o.json",
            r#"{"backend": "auto", "jk": {"aux": "def2-universal-jkfit"}}"#,
        );
        let o = opts_of(&format!(
            "--jk df --options {} --set precision.fp64=emulated",
            f.display()
        ))
        .unwrap();
        assert_eq!(
            o.jk_method(),
            JkMethod::Df {
                aux: "def2-universal-jkfit".into()
            }
        );
        assert_eq!(
            (o.backend.kind(), o.precision.fp64()),
            (BackendKind::Auto, Fp64Mode::Emulated)
        );
        // a request file with flags over it
        let r = tmp(
            "r.json",
            r#"{"structure": {"symbols": ["He"], "geometry": [[0, 0, 0]]}, "method": "pbe", "jk": "df", "grid": 2, "solvent": "water"}"#,
        );
        let j = match parse_args(&args(&format!(
            "{} --solvent-model iefpcm --grad",
            r.display()
        ))) {
            Ok(Parsed::Run(j, _)) => j,
            _ => panic!("a request"),
        };
        assert_eq!(j.req.settings.grid.xc(), GridChoice::Pyscf { level: 2 });
        assert_eq!(
            j.req.settings.solvent.as_ref().unwrap().model(),
            linus::solvent::PcmModel::Iefpcm
        );
        assert_eq!(j.req.calctype, CalcType::Gradient);
    }

    #[test]
    fn frequency_options() {
        let j = job_of("--ir --temperature 310 --thermo qrrho-hg").unwrap();
        assert_eq!(j.req.calctype, CalcType::Hessian);
        let f = j.req.settings.freq_options();
        assert!(f.ir && f.thermo[0].qrrho_entropy && f.thermo[0].qrrho_enthalpy);
        assert_eq!(f.thermo[0].temperature, 310.0);
        let j = job_of("--hess --set freq.temperature=[250,300]").unwrap();
        assert_eq!(j.req.settings.freq_options().thermo.len(), 2);
        assert!(job_of("--freq").unwrap_err().contains("unknown argument"));
    }

    /// `--molden FILE` writes the orbitals' Molden file (the basis set's own functions unless
    /// output.molden says otherwise); the JSON carries it in `files`.
    #[test]
    fn molden_file() {
        let o = opts_of("--molden x.molden").unwrap();
        assert_eq!(o.output.molden(), MoldenForm::Spherical);
        let o = opts_of("--molden x.molden --cartesian").unwrap();
        assert_eq!(o.output.molden(), MoldenForm::Cartesian);
        let o = opts_of("--set output.molden=cartesian --molden x.molden").unwrap();
        assert_eq!(o.output.molden(), MoldenForm::Cartesian);
        assert!(opts_of("--set output.molden=off --molden x.molden")
            .unwrap_err()
            .contains("off"));
        let xyz = tmp(
            "w.xyz",
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        );
        let path = tmp("w.molden", "");
        let out = execute(&args(&format!(
            "{} --basis cc-pvdz --molden {} --json",
            xyz.display(),
            path.display()
        )))
        .unwrap_or_else(|f| panic!("{}", f.msg));
        let v: Value = serde_json::from_str(&out).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(v["files"][MOLDEN_FILE], text.as_str());
        assert!(text.starts_with("[Molden Format]") && text.contains("[5D7F]"));
        // 24 functions of 24 orbitals, one spin set (RHF)
        assert_eq!(text.matches(" Sym= A").count(), 24);
        assert_eq!(text.matches(" Spin= Alpha").count(), 24);
        assert_eq!(v["resolved"]["settings"]["output"]["molden"], "spherical");
        // a spherical file of Cartesian functions is an input error before the calculation
        let f = execute(&args(&format!(
            "{} --basis cc-pvdz --cartesian --set output.molden=spherical",
            xyz.display()
        )))
        .expect_err("an error");
        assert_eq!(f.code, 2);
        assert!(f.msg.contains("output.molden"), "{}", f.msg);
    }

    /// `--plan` prints the plan as JSON and runs nothing; `--list-settings` and `schema` print
    /// the table.
    #[test]
    fn plan_and_information() {
        let xyz = tmp(
            "w.xyz",
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        );
        let out = execute(&args(&format!(
            "{} --method pbe --basis sto-3g --grad --plan",
            xyz.display()
        )))
        .unwrap_or_else(|f| panic!("{}", f.msg));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["calctype"], "gradient");
        assert_eq!(v["settings"]["grid"]["xc"], "pyscf:3");
        assert_eq!(v["placement"]["xc"]["on"], "cpu");
        assert!(v.get("energy").is_none());
        let out = execute(&args("--list-settings scf")).unwrap_or_else(|f| panic!("{}", f.msg));
        assert!(out.contains("scf.max_iter") && !out.contains("jk.kind"));
        assert!(execute(&args("--list-settings sfc")).is_err());
        let out = execute(&args("schema settings --calctype energy"))
            .unwrap_or_else(|f| panic!("{}", f.msg));
        let s: Value = serde_json::from_str(&out).unwrap();
        assert!(s["properties"]["response"].is_null());
        assert!(s["properties"]["scf"]["properties"]["max_iter"].is_object());
        let out = execute(&args("schema record")).unwrap_or_else(|f| panic!("{}", f.msg));
        assert!(serde_json::from_str::<Value>(&out).is_ok());
    }

    /// Errors raised while setting up the session keep the library's kind and exit code.
    #[test]
    fn setup_error_kinds() {
        let req = |s: &str| linus::Request::from_json(serde_json::from_str(s).unwrap()).unwrap();
        let he = r#""structure": {"symbols": ["He"], "geometry": [[0, 0, 0]]}"#;
        let f = run(req(&format!(r#"{{{he}, "method": "wb97x-d3"}}"#))).expect_err("an error");
        assert_eq!((f.code, f.kind), (1, "unsupported"), "{}", f.msg);
        let f = run(req(&format!(r#"{{{he}, "basis": "def2svp"}}"#))).expect_err("an error");
        assert_eq!((f.code, f.kind), (2, "input"), "{}", f.msg);
        assert_eq!(Failure::from("bad flag".to_string()).kind, "input");
    }
}
