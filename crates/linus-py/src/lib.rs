//! Python bindings: `linus._linus`, wrapped by the pure-Python package `linus`
//! (`python/linus`). Settings go in as JSON and are read by the library's grammar
//! (`linus::CalcOptions::from_json`: the same names, aliases and checks as the CLI);
//! results come back as the library's report (`linus::driver::report`) converted to a dict, with numpy
//! arrays for the orbitals, densities, Fock matrices and the Hessian.
//!
//! Every computation runs with the GIL released, on the session's own thread pool.
//!
//! Errors are the classes of Python's `linus.errors` (`qcdata.ProgramError` subclasses):
//! `LinusError` (base), `InputError` (also a `ValueError`), `UnsupportedError` (also a
//! `NotImplementedError`), `ResourceError` (also a `MemoryError`), `BackendError` (the GPU backend
//! failed), `ConvergenceError`, whose `result` is the report of the unconverged SCF (its `scf`
//! history included), and `NumericalError` (also an `ArithmeticError`). Every exception has
//! `kind`, the library's error kind (`linus::LinusError::kind`; `internal` for a panic, a bug in
//! linus), and `code`.

use faer::Mat;
use linus::driver::LineageKey;
use linus::{
    CalcOptions, CalcType, Failure, LinusError as DriverError, Report, ScfResult,
    Session as Driver, Structure,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBool, PyDict, PyList, PyString, PyType};
use serde_json::{Map, Value};
use std::panic::AssertUnwindSafe;

/// Calculation types the driver can do (the Python package and the qccodec codec read this).
const CAPABILITIES: &[&str] = &["energy", "gradient", "hessian"];

/// The exception classes (Python's `linus.errors`, `qcdata.ProgramError` subclasses):
/// LinusError, InputError, UnsupportedError, ResourceError, ConvergenceError, NumericalError,
/// BackendError.
static EXCEPTIONS: GILOnceCell<[Py<PyType>; 7]> = GILOnceCell::new();

fn exceptions(py: Python<'_>) -> PyResult<&[Py<PyType>; 7]> {
    EXCEPTIONS.get_or_try_init(py, || {
        let m = py.import("linus.errors")?;
        let get = |name: &str| -> PyResult<Py<PyType>> {
            Ok(m.getattr(name)?.downcast_into::<PyType>()?.unbind())
        };
        Ok([
            get("LinusError")?,
            get("InputError")?,
            get("UnsupportedError")?,
            get("ResourceError")?,
            get("ConvergenceError")?,
            get("NumericalError")?,
            get("BackendError")?,
        ])
    })
}

/// Errors of one call into the driver.
enum Error {
    /// A driver error, with the calculation's failure value when a calculation failed.
    Driver(DriverError, Option<Box<Failure>>),
    Panic(String),
}

impl From<DriverError> for Error {
    fn from(e: DriverError) -> Self {
        Error::Driver(e, None)
    }
}

fn to_py_err(py: Python<'_>, e: Error) -> PyErr {
    // `kind` and `code` always (`code`: the error's own, `LinusError::code`: a settings error's,
    // `host_memory`, `device_memory`; `device_lost` for a device fault that poisoned the CUDA
    // context; else None), `details` where the error has them (a memory failure's
    // `required_gib` and `available_gib`: qcdata's `ErrorInfo.details`); for a failed
    // calculation `failure` (the CLI's failure JSON as a dict) and, when it reached one, `result`
    // (the report of the unconverged SCF)
    let raise = |idx: usize,
                 kind: &str,
                 code: Option<&str>,
                 details: Option<Value>,
                 msg: String,
                 failure: Option<Box<Failure>>|
     -> PyResult<PyErr> {
        let ty = exceptions(py)?[idx].bind(py);
        let kw = PyDict::new(py);
        kw.set_item("kind", kind)?;
        kw.set_item("code", code)?;
        if let Some(d) = &details {
            kw.set_item("details", to_py(py, d)?)?;
        }
        let inst = ty.call((msg,), Some(&kw))?;
        if let Some(f) = failure {
            inst.setattr("failure", to_py(py, &f.to_json())?)?;
            if let Some(r) = &f.result {
                inst.setattr("result", report_dict(py, r, false)?)?;
            }
        }
        Ok(PyErr::from_value(inst))
    };
    let r = match e {
        Error::Driver(e, result) => {
            let idx = match &e {
                DriverError::Input(_) | DriverError::Setting { .. } | DriverError::Basis(_) => 1,
                DriverError::Unsupported(_) => 2,
                DriverError::Resource { .. } => 3,
                DriverError::NotConverged(_) | DriverError::ResponseNotConverged(_) => 4,
                DriverError::Numerical(_) => 5,
                DriverError::Backend { .. } => 6,
                DriverError::Internal(_) | DriverError::Stopped(_) => 0,
            };
            let code = match &e {
                DriverError::Backend { .. } if linus::device_lost() => Some("device_lost"),
                e => e.code(),
            };
            let details = linus::driver::report::error_details(&e);
            raise(idx, e.kind(), code, details, e.to_string(), result)
        }
        Error::Panic(msg) => raise(
            0,
            "internal",
            None,
            None,
            format!("internal error in linus: {msg}"),
            None,
        ),
    };
    r.unwrap_or_else(|e| e)
}

/// An input error from the binding itself (shapes, values).
fn input_err(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    to_py_err(py, Error::Driver(DriverError::Input(msg.into()), None))
}

/// Run `f` without the GIL; a panic becomes an error instead of unwinding into Python.
fn run<T: Send>(py: Python<'_>, f: impl FnOnce() -> Result<T, Error> + Send) -> PyResult<T> {
    py.allow_threads(|| match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => Err(Error::Panic(panic_text(p.as_ref()))),
    })
    .map_err(|e| to_py_err(py, e))
}

/// The message of a caught panic.
fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown panic".into())
}

/// A JSON value as a Python object (dicts, lists, numbers, strings, None).
fn to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.into_pyobject(py)?.into_any(),
            (_, Some(u)) => u.into_pyobject(py)?.into_any(),
            _ => n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any(),
        },
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(a) => {
            let l = PyList::empty(py);
            for x in a {
                l.append(to_py(py, x)?)?;
            }
            l.into_any()
        }
        Value::Object(m) => map_to_dict(py, m)?.into_any(),
    })
}

fn map_to_dict<'py>(py: Python<'py>, m: &Map<String, Value>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, x) in m {
        d.set_item(k, to_py(py, x)?)?;
    }
    Ok(d)
}

/// A Python object (anything `json.dumps` takes) as JSON, through the json module.
fn from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let s: String = py
        .import("json")?
        .call_method1("dumps", (obj,))?
        .extract()?;
    serde_json::from_str(&s).map_err(|e| input_err(py, e.to_string()))
}

fn positions_from(
    py: Python<'_>,
    natoms: usize,
    positions: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<[f64; 3]>> {
    let shape = positions.shape();
    if shape.len() != 2 || shape[1] != 3 || shape[0] != natoms {
        return Err(input_err(
            py,
            format!("positions must have shape ({natoms}, 3) (bohr), not {shape:?}"),
        ));
    }
    let a = positions.as_array();
    let pos: Vec<[f64; 3]> = (0..natoms)
        .map(|i| [a[[i, 0]], a[[i, 1]], a[[i, 2]]])
        .collect();
    if pos.iter().flatten().any(|x| !x.is_finite()) {
        return Err(input_err(py, "positions must be finite numbers"));
    }
    Ok(pos)
}

fn mat_to_py<'py>(py: Python<'py>, m: &Mat<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut v = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            v.push(m[(i, j)]);
        }
    }
    PyArray1::from_vec(py, v).reshape([r, c])
}

fn py_to_mat(
    py: Python<'_>,
    a: &PyReadonlyArray2<'_, f64>,
    rows: usize,
    cols: Option<usize>,
    name: &str,
) -> PyResult<Mat<f64>> {
    let shape = a.shape();
    if shape[0] != rows || cols.is_some_and(|c| shape[1] != c) {
        return Err(input_err(
            py,
            format!(
                "{name} must have {rows} rows{}, not shape {shape:?}",
                cols.map(|c| format!(" and {c} columns"))
                    .unwrap_or_default()
            ),
        ));
    }
    let v = a.as_array();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(input_err(py, format!("{name} must be finite numbers")));
    }
    Ok(Mat::from_fn(shape[0], shape[1], |i, j| v[[i, j]]))
}

/// The dict of a request's report (the Python package turns it into a `qcdata.Frame`): the
/// serialized report, with the MO coefficients (and on request the Fock matrices of the SCF's
/// last build, `hamiltonian_*`, and the report's AO matrices, `overlap` and `core_hamiltonian`)
/// and the Hessian as numpy arrays.
fn report_dict<'py>(
    py: Python<'py>,
    r: &Report,
    hamiltonian: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let Value::Object(mut m) = serde_json::to_value(r).map_err(|e| input_err(py, e.to_string()))?
    else {
        return Err(input_err(py, "a report serializes to a JSON object"));
    };
    // the Hessian goes as an array, not a list
    m.remove("hessian");
    let d = map_to_dict(py, &m)?;
    let s: &ScfResult = &r.wavefunction;
    d.set_item("mo_coefficients_alpha", mat_to_py(py, &s.c_a)?)?;
    d.set_item("mo_coefficients_beta", mat_to_py(py, &s.c_b)?)?;
    if hamiltonian {
        d.set_item("hamiltonian_alpha", mat_to_py(py, &s.f_a)?)?;
        d.set_item("hamiltonian_beta", mat_to_py(py, &s.f_b)?)?;
    }
    if let Some(m) = &r.ao.overlap {
        d.set_item("overlap", mat_to_py(py, m)?)?;
    }
    if let Some(m) = &r.ao.core_hamiltonian {
        d.set_item("core_hamiltonian", mat_to_py(py, m)?)?;
    }
    if let Some(h) = &r.hessian {
        let n = (h.len() as f64).sqrt() as usize;
        d.set_item(
            "hessian",
            PyArray1::from_vec(py, h.clone()).reshape([n, n])?,
        )?;
    }
    Ok(d)
}

/// One molecule with fixed atoms, basis and settings: a session's engine and its lineages.
///
/// `options` is a JSON object of settings (the library's `CalcOptions` fields and the request
/// conveniences: `jk`/`grid` strings, top-level `aux`, `k_grid`, `pcm`, `solvent` as a name or
/// a dielectric constant, dispersion suffixes on `method`); unknown names are an error.
/// `calctypes` (energy, gradient, hessian; None: all) are the calctypes the session may run: a
/// setting that applies to none of them is an error here (decision D41). Positions are in bohr.
///
/// `compute` runs a request in a lineage, committed when it succeeds; a lineage is keyed by its
/// canonical JSON text (`qcdata.lineage_key`; `null` is the default lineage).
///
/// A session runs one call at a time: a call from a second thread while one runs raises
/// `InputError` (use one session per thread). A panic (a bug in linus) closes the session: the
/// call raises an internal error and later calls an `InputError`. `close()` frees the engine
/// (its threads, memory and device) at once.
#[pyclass(module = "linus._linus", name = "Session")]
struct PySession {
    /// The engine and its lineages; None once closed.
    inner: std::sync::Mutex<Option<Driver>>,
    /// The calctypes it may run: the declared ones with their lower derivatives, or what the
    /// method supports.
    calctypes: linus::CalcTypes,
}

/// The open engine of a session, locked ([`PySession::lock`]).
struct Engine<'a>(std::sync::MutexGuard<'a, Option<Driver>>);

impl std::ops::Deref for Engine<'_> {
    type Target = Driver;
    fn deref(&self) -> &Driver {
        self.0.as_ref().expect("an Engine holds an open session")
    }
}

impl std::ops::DerefMut for Engine<'_> {
    fn deref_mut(&mut self) -> &mut Driver {
        self.0.as_mut().expect("an Engine holds an open session")
    }
}

/// The calctypes a session declares from their names (None: every one; a Hessian declares the
/// IR intensities a request may add too).
fn declared_calctypes(
    py: Python<'_>,
    names: Option<Vec<String>>,
) -> PyResult<Option<linus::CalcTypes>> {
    let Some(names) = names else {
        return Ok(None);
    };
    let mut c = linus::CalcTypes::of(CalcType::Energy, false);
    for n in names {
        match n.parse::<CalcType>().map_err(|e| input_err(py, e))? {
            CalcType::Energy => {}
            CalcType::Gradient => c.gradient = true,
            CalcType::Hessian => c = linus::CalcTypes::ALL,
        }
    }
    Ok(Some(c))
}

/// The calctypes the method and settings of `d` can run, a session's default: every one, less
/// the Hessian (and then the gradient) when the method has no such derivatives. (A response's
/// arrays beyond `memory_gib` are not a missing capability: such a request fails when it runs.)
fn supported_calctypes(d: &Driver) -> linus::CalcTypes {
    use linus::derivatives::capabilities::Request;
    let can = |c| {
        !matches!(
            d.check_request(Request::of(c, false)),
            Err(DriverError::Unsupported(_))
        )
    };
    if can(CalcType::Hessian) {
        linus::CalcTypes::ALL
    } else if can(CalcType::Gradient) {
        linus::CalcTypes::of(CalcType::Gradient, false)
    } else {
        linus::CalcTypes::of(CalcType::Energy, false)
    }
}

/// A session request's fields beside its positions and calctype ([`PySession::compute`]).
struct RequestJson {
    lineage: String,
    seed: Vec<String>,
    commit: bool,
    freq: Option<Map<String, Value>>,
    accuracy: Option<Map<String, Value>>,
    hamiltonian: bool,
    ao: linus::AoOutputs,
}

impl RequestJson {
    /// The request of the JSON object `s`: `lineage` and `seed` keys as JSON texts, `commit`
    /// (default true), `freq` (an object), `accuracy` (an object of targets: `energy`,
    /// `gradient`, `hessian`), `hamiltonian`, `overlap` and `core_hamiltonian` (default false;
    /// the Hamiltonian comes with the overlap).
    fn parse(py: Python<'_>, s: &str) -> PyResult<RequestJson> {
        let bad = |m: String| input_err(py, format!("invalid session request: {m}"));
        let Value::Object(m) = serde_json::from_str(s).map_err(|e| bad(e.to_string()))? else {
            return Err(bad("not a JSON object".into()));
        };
        let mut r = RequestJson {
            lineage: "null".into(),
            seed: Vec::new(),
            commit: true,
            freq: None,
            accuracy: None,
            hamiltonian: false,
            ao: linus::AoOutputs::default(),
        };
        for (k, v) in m {
            match (k.as_str(), v) {
                ("lineage", Value::String(t)) => r.lineage = t,
                ("seed", Value::Array(a)) => {
                    r.seed = a
                        .into_iter()
                        .map(|x| match x {
                            Value::String(t) => Ok(t),
                            x => Err(bad(format!("seed keys are JSON texts, not {x}"))),
                        })
                        .collect::<PyResult<_>>()?
                }
                ("commit", Value::Bool(b)) => r.commit = b,
                ("freq", Value::Object(f)) => r.freq = Some(f),
                ("freq", Value::Null) => {}
                ("accuracy", Value::Object(a)) => r.accuracy = Some(a),
                ("accuracy", Value::Null) => {}
                ("hamiltonian", Value::Bool(b)) => r.hamiltonian = b,
                ("overlap", Value::Bool(b)) => r.ao.overlap = b,
                ("core_hamiltonian", Value::Bool(b)) => r.ao.core_hamiltonian = b,
                (k, v) => return Err(bad(format!("{k}: {v}"))),
            }
        }
        r.ao.overlap |= r.hamiltonian;
        Ok(r)
    }
}

/// The listener of a session's progress events: each event as a dict (`{"event": name, ...}`,
/// a non-finite number as None) to `callable`, with the GIL taken for the call. An exception it
/// raises is dropped: a display's bug must not fail the calculation.
fn listener(callable: PyObject) -> linus::progress::Listener {
    use linus::progress::Value as V;
    std::sync::Arc::new(move |e| {
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            let fill = || -> PyResult<()> {
                d.set_item("event", e.name())?;
                for (k, v) in e.fields() {
                    match v {
                        V::Str(s) => d.set_item(k, s)?,
                        V::Float(x) if x.is_finite() => d.set_item(k, x)?,
                        V::Float(_) => d.set_item(k, py.None())?,
                        V::Int(n) => d.set_item(k, n)?,
                        V::Bool(b) => d.set_item(k, b)?,
                    }
                }
                callable.call1(py, (d.clone(),))?;
                Ok(())
            };
            let _ = fill();
        })
    })
}

/// The stop check of a session's calculations: `callable()` (qccompute's
/// `ctx.stop_requested`), called with the GIL taken; an exception it raises means no stop.
fn stopper(callable: PyObject) -> linus::progress::StopCheck {
    std::sync::Arc::new(move || {
        Python::with_gil(|py| {
            callable
                .call0(py)
                .and_then(|r| r.is_truthy(py))
                .unwrap_or(false)
        })
    })
}

/// A lineage's key from its canonical JSON text.
fn lineage_key(py: Python<'_>, text: &str) -> PyResult<LineageKey> {
    LineageKey::from_text(text).map_err(|e| to_py_err(py, e.into()))
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (numbers, positions, *, charge=0, multiplicity=1, options="{}", calctypes=None, progress=None, stop=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python constructor's keyword arguments"
    )]
    fn new(
        py: Python<'_>,
        numbers: Vec<u32>,
        positions: PyReadonlyArray2<'_, f64>,
        charge: i32,
        multiplicity: u32,
        options: &str,
        calctypes: Option<Vec<String>>,
        progress: Option<PyObject>,
        stop: Option<PyObject>,
    ) -> PyResult<Self> {
        let pos = positions_from(py, numbers.len(), &positions)?;
        let v: Value = serde_json::from_str(options)
            .map_err(|e| input_err(py, format!("invalid linus options: {e}")))?;
        let declared = declared_calctypes(py, calctypes)?;
        let (inner, calctypes) = run(py, move || {
            let mol =
                Structure::new(numbers, pos, charge, multiplicity).map_err(DriverError::from)?;
            let mut d = Driver::new(mol, CalcOptions::from_json(v)?)?;
            // in process, `output.verbose`'s lines go to the output's logs, not stderr
            d.collect_progress();
            // memory_gib limits the whole Python process: its baseline in every phase's need
            d.set_process(linus::memory::Process::Python);
            if let Some(callable) = progress {
                d.listen(listener(callable));
            }
            if let Some(callable) = stop {
                d.stop_when(stopper(callable));
            }
            let calctypes = declared.unwrap_or_else(|| supported_calctypes(&d));
            if calctypes != linus::CalcTypes::ALL {
                d.declare(calctypes)?;
            }
            Ok((d, calctypes))
        })?;
        Ok(PySession {
            inner: std::sync::Mutex::new(Some(inner)),
            calctypes,
        })
    }

    /// The calctypes the session may run (`energy`, `gradient`, `hessian`): the declared ones
    /// with the lower derivatives they imply, by default every one the method supports.
    #[getter]
    fn calctypes(&self) -> Vec<&'static str> {
        let mut v = self.calctypes.names();
        v.retain(|c| *c != "ir");
        v
    }

    /// Run a request: `calctype` at `positions` (natoms x 3, bohr; None: the current ones).
    /// `request` (JSON): `lineage` (its key's JSON text; default `"null"`, the default
    /// lineage), `seed` (keys: the lineages to start from, in order, when its own has no
    /// state), `commit` (default true: a success becomes the lineage's state), `freq` (this
    /// request's analysis settings over the session's), `accuracy` (this request's targets
    /// instead of the settings' `accuracy`), `hamiltonian`, `overlap` and
    /// `core_hamiltonian` (add those matrices; computed after the calculation, they change no
    /// other number). The result dict.
    #[pyo3(signature = (positions, calctype, request="{}"))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        positions: Option<PyReadonlyArray2<'_, f64>>,
        calctype: &str,
        request: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kind: CalcType = calctype.parse().map_err(|e: String| input_err(py, e))?;
        let r = RequestJson::parse(py, request)?;
        let lineage = lineage_key(py, &r.lineage)?;
        let seed = r
            .seed
            .iter()
            .map(|s| lineage_key(py, s))
            .collect::<PyResult<Vec<_>>>()?;
        let natoms = self.lock(py)?.mol().natom();
        let positions = positions
            .map(|p| positions_from(py, natoms, &p))
            .transpose()?;
        let freq = match r.freq {
            None => None,
            Some(call) => Some(self.with(py, move |d| Ok(d.freq_options_with(call)?))?),
        };
        let accuracy = match r.accuracy {
            None => None,
            Some(a) => Some(
                linus::driver::options::AccuracySettings::from_map(a)
                    .map_err(|e| to_py_err(py, e.into()))?
                    .targets(),
            ),
        };
        let req = linus::driver::session::Request {
            positions,
            calctype: kind,
            lineage,
            seed,
            commit: r.commit,
            freq,
            accuracy,
        };
        let (want, hamiltonian) = (r.ao, r.hamiltonian);
        let (rep, logs) = self.with(py, move |d| {
            // a failure's progress lines go with it (`Failure::logs`)
            let mut rep = d.compute(&req).map_err(|e| {
                let f = d.failure(&e, kind);
                Error::Driver(e, Some(Box::new(f)))
            })?;
            let logs = d.take_logs();
            // the request's AO matrices beyond the settings' (after the calculation)
            let more = d.ao_matrices(linus::AoOutputs {
                overlap: want.overlap && rep.ao.overlap.is_none(),
                core_hamiltonian: want.core_hamiltonian && rep.ao.core_hamiltonian.is_none(),
            });
            rep.ao.overlap = rep.ao.overlap.take().or(more.overlap);
            rep.ao.core_hamiltonian = rep.ao.core_hamiltonian.take().or(more.core_hamiltonian);
            Ok((rep, logs))
        })?;
        let dict = report_dict(py, &rep, hamiltonian)?;
        if let Some(logs) = logs {
            dict.set_item("logs", logs)?;
        }
        Ok(dict)
    }

    /// Forget the lineage `lineage` (its key's JSON text; None: every lineage).
    #[pyo3(signature = (lineage=None))]
    fn reset(&self, py: Python<'_>, lineage: Option<&str>) -> PyResult<()> {
        let key = lineage.map(|s| lineage_key(py, s)).transpose()?;
        let mut d = self.lock(py)?;
        match &key {
            Some(k) => d.reset(k),
            None => d.reset_all(),
        }
        Ok(())
    }

    /// The keys (canonical JSON texts) of the lineages with a state, in key order.
    fn lineages(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(self
            .lock(py)?
            .lineages()
            .iter()
            .map(|k| k.as_str().to_string())
            .collect())
    }

    /// Free the engine: its threads, memory and device. Later calls raise `InputError`.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let d = self.guard(py)?.take();
        // the device's memory pool and handles go here, without the GIL
        py.allow_threads(move || drop(d));
        Ok(())
    }

    /// Whether the session is closed (by `close()`, or after an internal error).
    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.guard(py)?.is_none())
    }

    /// What a calculation of `calctype` (energy, gradient or hessian) will do, decided before it
    /// runs: the plan as a dict (`linus --plan`, the driver's `PlanReport`).
    fn plan<'py>(&self, py: Python<'py>, calctype: &str) -> PyResult<Bound<'py, PyAny>> {
        let kind: CalcType = calctype.parse().map_err(|e: String| input_err(py, e))?;
        let p = self.with(py, move |d| d.plan(kind).map_err(Error::from))?;
        to_py(py, &serde_json::to_value(&p).expect("the plan serializes"))
    }

    /// Start the default lineage's next SCF from occupied orbitals (nbf x nocc per spin) of an
    /// earlier SCF of this molecule, made orthonormal at the current positions (Löwdin). Used
    /// once; its record says `scf.guess.source` `given`.
    #[pyo3(signature = (c_alpha, c_beta, restricted=false))]
    fn set_guess_orbitals(
        &self,
        py: Python<'_>,
        c_alpha: PyReadonlyArray2<'_, f64>,
        c_beta: PyReadonlyArray2<'_, f64>,
        restricted: bool,
    ) -> PyResult<()> {
        let n = self.lock(py)?.basis().nbf;
        let (ca, cb) = (
            py_to_mat(py, &c_alpha, n, None, "c_alpha")?,
            py_to_mat(py, &c_beta, n, None, "c_beta")?,
        );
        self.with(py, move |d| {
            Ok(d.set_guess_orbitals(ca.as_ref(), cb.as_ref(), restricted)?)
        })
    }

    #[getter]
    fn numbers(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        Ok(self.lock(py)?.mol().numbers().to_vec())
    }

    /// Core electrons replaced by ECPs, per atom.
    #[getter]
    fn ecp_core_electrons(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        Ok(self.lock(py)?.mol().ecp_core_electrons())
    }

    /// The settings in effect (normalized `CalcOptions`), as a dict.
    #[getter]
    fn options<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = serde_json::to_value(self.lock(py)?.opts())
            .map_err(|e| input_err(py, e.to_string()))?;
        to_py(py, &v)
    }

    /// The basis set as placed on the atoms: one dict per shell in AO order (`atom`, `l`,
    /// `pure`, `exponents`, `coefficients`: contraction coefficients of the unnormalized
    /// primitives r^l exp(-a r^2) that give the x^l component unit norm).
    fn basis_shells<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.lock(py)?
            .basis()
            .shells
            .iter()
            .map(|s| {
                let d = PyDict::new(py);
                d.set_item("atom", s.atom)?;
                d.set_item("l", s.l)?;
                d.set_item("pure", s.pure)?;
                d.set_item("exponents", s.exps.clone())?;
                d.set_item("coefficients", s.coefs.clone())?;
                Ok(d)
            })
            .collect()
    }

    /// The effective core potentials as placed on the atoms: one dict per atom that has one
    /// (`atom`, `core_electrons`, `local`: the terms of U_L, `semilocal`: per l < L the terms of
    /// U_l - U_L), each term a (power, exponent, coefficient) triple of coefficient r^power
    /// exp(-exponent r^2): Basis Set Exchange's channels, with the power of r (BSE's n - 2).
    fn ecp_centers<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let terms = |ts: &[linus::linus_ints::one_e::ecp::EcpTerm]| -> Vec<(i32, f64, f64)> {
            ts.iter().map(|t| (t.rpow, t.zeta, t.coef)).collect()
        };
        self.lock(py)?
            .ecp()
            .centers
            .iter()
            .map(|c| {
                let d = PyDict::new(py);
                d.set_item("atom", c.atom)?;
                d.set_item("core_electrons", c.def.ncore)?;
                d.set_item("local", terms(&c.def.local))?;
                d.set_item(
                    "semilocal",
                    c.def.semilocal.iter().map(|s| terms(s)).collect::<Vec<_>>(),
                )?;
                Ok(d)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        match self.inner.try_lock() {
            Ok(g) => match g.as_ref() {
                Some(c) => {
                    let o = c.opts();
                    format!(
                        "Session(method={:?}, basis={:?}, natoms={}, nbf={})",
                        o.method(),
                        o.basis(),
                        c.mol().natom(),
                        c.basis().nbf
                    )
                }
                None => "Session(<closed>)".into(),
            },
            Err(_) => "Session(<running>)".into(),
        }
    }
}

impl PySession {
    /// The session's lock (open or closed), or `InputError` while another thread is running a
    /// call on it.
    fn guard(&self, py: Python<'_>) -> PyResult<std::sync::MutexGuard<'_, Option<Driver>>> {
        match self.inner.try_lock() {
            Ok(g) => Ok(g),
            // a panic is caught inside the call, which closes the session
            Err(std::sync::TryLockError::Poisoned(p)) => Ok(p.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => Err(input_err(
                py,
                "this Session is running a call in another thread; use one Session per thread",
            )),
        }
    }

    /// The open engine, or `InputError` when the session is closed or busy.
    fn lock(&self, py: Python<'_>) -> PyResult<Engine<'_>> {
        let g = self.guard(py)?;
        if g.is_none() {
            return Err(input_err(
                py,
                "this Session is closed (by close(), or after an internal error); open a new one",
            ));
        }
        Ok(Engine(g))
    }

    /// `f` on the engine without the GIL. A panic in it (a bug in linus) closes the session and
    /// raises an internal error.
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut Driver) -> Result<T, Error> + Send,
    ) -> PyResult<T> {
        let mut engine = self.lock(py)?;
        let d: &mut Driver = &mut engine;
        let out = py.allow_threads(|| std::panic::catch_unwind(AssertUnwindSafe(|| f(d))));
        match out {
            Ok(r) => r.map_err(|e| to_py_err(py, e)),
            Err(p) => {
                let d = engine.0.take();
                py.allow_threads(move || drop(d));
                Err(to_py_err(
                    py,
                    Error::Panic(format!(
                        "{}; the session is closed (open a new one)",
                        panic_text(p.as_ref())
                    )),
                ))
            }
        }
    }
}

/// Whether a GPU fault poisoned this process's CUDA context: no device work can succeed in
/// this process any more (a worker answers its request and exits).
#[pyfunction]
fn device_lost() -> bool {
    linus::device_lost()
}

/// Calculation types this build can do.
#[pyfunction]
fn capabilities() -> Vec<&'static str> {
    CAPABILITIES.to_vec()
}

/// The `freq` settings of a JSON object (keys as `freq.*`), resolved.
fn freq_options(py: Python<'_>, options: &str) -> PyResult<linus::vib::FrequencyOptions> {
    let m = match serde_json::from_str(options) {
        Ok(Value::Object(m)) => m,
        _ => {
            return Err(input_err(
                py,
                "options must be a JSON object of freq settings",
            ))
        }
    };
    let freq =
        linus::driver::options::FreqSettings::from_map(m).map_err(|e| to_py_err(py, e.into()))?;
    Ok(CalcOptions {
        freq,
        ..Default::default()
    }
    .freq_options())
}

/// Harmonic analysis of any Cartesian Hessian (3N x 3N, Eh/bohr^2) of atoms `numbers` at
/// `positions` (bohr): the driver's `VibrationalAnalysis` as a dict. `options`: the analysis's
/// `freq` settings as JSON (`mass_table`, `masses`, `isotopes`, `projection`, `symmetry_number`,
/// `symmetry_tol`, `codata`); `dipole_derivatives` (3N x 3, dmu/dX) add IR intensities.
#[pyfunction]
#[pyo3(signature = (numbers, positions, hessian, options="{}", dipole_derivatives=None))]
fn harmonic_analysis<'py>(
    py: Python<'py>,
    numbers: Vec<u32>,
    positions: PyReadonlyArray2<'_, f64>,
    hessian: PyReadonlyArray2<'_, f64>,
    options: &str,
    dipole_derivatives: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let pos = positions_from(py, numbers.len(), &positions)?;
    let n3 = 3 * numbers.len();
    if hessian.shape() != [n3, n3] {
        return Err(input_err(
            py,
            format!(
                "hessian must have shape ({n3}, {n3}), not {:?}",
                hessian.shape()
            ),
        ));
    }
    let h: Vec<f64> = hessian.as_array().iter().copied().collect();
    let apt: Option<Vec<[f64; 3]>> = match &dipole_derivatives {
        None => None,
        Some(d) if d.shape() != [n3, 3] => {
            return Err(input_err(
                py,
                format!(
                    "dipole_derivatives must have shape ({n3}, 3), not {:?}",
                    d.shape()
                ),
            ))
        }
        Some(d) if d.as_array().iter().any(|x| !x.is_finite()) => {
            return Err(input_err(py, "dipole_derivatives must be finite numbers"))
        }
        Some(d) => {
            let a = d.as_array();
            Some((0..n3).map(|i| [a[[i, 0]], a[[i, 1]], a[[i, 2]]]).collect())
        }
    };
    let opts = freq_options(py, options)?
        .analysis
        .ok_or_else(|| input_err(py, "freq.analysis is off"))?;
    // without the GIL, panics contained, on a pool of its own (one thread: no global pool)
    let out = run(py, || {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .map_err(|e| DriverError::resource(format!("cannot start a thread: {e}")))?;
        Ok(pool
            .install(|| linus::vib::harmonic_analysis(&numbers, &pos, &h, &opts, apt.as_deref()))?)
    })?;
    to_py(
        py,
        &serde_json::to_value(&out).map_err(|e| input_err(py, e.to_string()))?,
    )
}

/// Thermochemistry from a `VibrationalAnalysis` (the dict `harmonic_analysis` returns), the
/// electronic energy (Eh) and the spin multiplicity: a list of the driver's `Thermochemistry`
/// dicts, one per temperature. `options`: the thermochemistry's `freq` settings as JSON
/// (`thermo`, `temperature`, `pressure`, `concentration`, `scale`, `rotor_cutoff`, ...).
#[pyfunction]
#[pyo3(signature = (analysis, energy, multiplicity=1, options="{}"))]
fn thermochemistry<'py>(
    py: Python<'py>,
    analysis: &Bound<'py, PyAny>,
    energy: f64,
    multiplicity: u32,
    options: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let vib: linus::vib::VibrationalAnalysis = serde_json::from_value(from_py(py, analysis)?)
        .map_err(|e| input_err(py, format!("invalid analysis: {e}")))?;
    let opts = freq_options(py, options)?;
    let out = opts
        .thermo
        .iter()
        .map(|t| linus::vib::thermochemistry(&vib, energy, multiplicity, t))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| to_py_err(py, e.into()))?;
    to_py(
        py,
        &serde_json::to_value(&out).map_err(|e| input_err(py, e.to_string()))?,
    )
}

/// Default mass (u) of element `z` from `table` ("isotope" or "average").
#[pyfunction]
#[pyo3(signature = (z, table="isotope"))]
fn atomic_mass(py: Python<'_>, z: u32, table: &str) -> PyResult<f64> {
    let t: linus::vib::MassTable = serde_json::from_value(Value::String(table.into()))
        .map_err(|e| input_err(py, e.to_string()))?;
    linus::vib::masses::mass(z, t)
        .ok_or_else(|| input_err(py, format!("no mass for atomic number {z}")))
}

/// Every setting at its default (`"auto"` where the calculation decides), as a dict.
#[pyfunction]
fn default_options(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    to_py(py, &CalcOptions::default().resolved_tree())
}

/// Settings (any object `json.dumps` takes) read, normalized and checked by the library's
/// grammar: the settings given, in their canonical form, as a dict.
#[pyfunction]
fn normalize_options<'py>(
    py: Python<'py>,
    options: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let v = from_py(py, options)?;
    let o = CalcOptions::from_json(v).map_err(|e| to_py_err(py, e.into()))?;
    to_py(
        py,
        &serde_json::to_value(o).map_err(|e| input_err(py, e.to_string()))?,
    )
}

/// The JSON Schema (draft 2020-12) of the settings, of those that can apply to `calctype`
/// (energy, gradient, hessian) when given, as a JSON string.
#[pyfunction]
#[pyo3(signature = (calctype=None))]
fn settings_schema(py: Python<'_>, calctype: Option<&str>) -> PyResult<String> {
    if let Some(c) = calctype {
        if !matches!(c, "energy" | "gradient" | "hessian") {
            return Err(input_err(
                py,
                format!("unknown calctype {c:?}; use energy, gradient or hessian"),
            ));
        }
    }
    Ok(linus::driver::schema::json_schema(calctype).to_string())
}

/// The JSON Schema of a result's run record (`resolved.record`; diagnostic), as a JSON string.
#[pyfunction]
fn record_schema() -> String {
    linus::driver::schema::record_schema().to_string()
}

/// Functional names linus defines (libxc names and mixtures work too).
#[pyfunction]
fn available_functionals() -> Vec<&'static str> {
    linus::driver::available_functionals()
}

/// Solvent names of the solvent database, with their dielectric constants and whether they have
/// SMD descriptors: a list of (name, eps, smd).
#[pyfunction]
fn available_solvents() -> Vec<(String, f64, bool)> {
    linus::solvent::db::solvents()
        .iter()
        .map(|s| (s.name.clone(), s.eps, s.smd.is_some()))
        .collect()
}

/// Register the full CRC permittivity table (qcref's `data/solvents/crc.csv`) that solvent names
/// outside the shipped table resolve through (once, at import, where qcref is installed;
/// `$LINUS_SOLVENT_TABLE` overrides it).
#[pyfunction]
fn set_solvent_table(path: std::path::PathBuf) -> bool {
    linus::solvent::db::set_crc_table(path)
}

/// The full CRC permittivity table in effect (`$LINUS_SOLVENT_TABLE`, else the registered one).
#[pyfunction]
fn solvent_table() -> Option<String> {
    linus::solvent::db::crc_table().map(|p| p.display().to_string())
}

/// The basis set search path: the directories `basis_dir`, then `$LINUS_BASIS_DIR`, the
/// source tree of development builds and the package's sets (`linus_ints::basis::search_path`).
#[pyfunction]
#[pyo3(signature = (basis_dir=None))]
fn basis_search_path(basis_dir: Option<Vec<std::path::PathBuf>>) -> Vec<String> {
    linus::linus_ints::basis::search_path(&basis_dir.unwrap_or_default())
        .iter()
        .map(|p| p.display().to_string())
        .collect()
}

/// Names of the basis sets in the directories of `path` (sorted, each once).
#[pyfunction]
fn available_basis_sets(path: Vec<std::path::PathBuf>) -> Vec<String> {
    linus::linus_ints::basis::available_basis_sets(&path)
}

/// The file of basis set `name` in the first directory of `path` that has it (None: none).
#[pyfunction]
fn find_basis_file(name: &str, path: Vec<std::path::PathBuf>) -> Option<String> {
    linus::linus_ints::basis::find_basis_file(name, &path)
        .ok()
        .map(|p| p.display().to_string())
}

/// Register the basis sets the installed package ships (once, at import; the last directory
/// of the search path).
#[pyfunction]
fn set_package_basis_dir(dir: std::path::PathBuf) -> bool {
    linus::linus_ints::basis::set_package_dir(dir)
}

/// Element symbols by atomic number (index 0: "X"; linus supports H-Rn).
#[pyfunction]
fn element_symbols() -> Vec<&'static str> {
    linus::structure::SYMBOLS.to_vec()
}

#[pymodule]
fn _linus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // the distribution's version (PEP 440): what qccompute records as the program's version
    m.add("__version__", linus::version())?;
    m.add("NUMERICS_EPOCH", linus::NUMERICS_EPOCH)?;
    // the GPU build (distribution afh-linus-cuda13) and the CPU build (afh-linus) share a version
    m.add("GPU_BUILD", linus::GPU_BUILD)?;
    m.add("THREADS_USED_WELL", linus::THREADS_USED_WELL)?;
    m.add_class::<PySession>()?;
    m.add_function(wrap_pyfunction!(capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(device_lost, m)?)?;
    m.add_function(wrap_pyfunction!(default_options, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_options, m)?)?;
    m.add_function(wrap_pyfunction!(settings_schema, m)?)?;
    m.add_function(wrap_pyfunction!(record_schema, m)?)?;
    m.add_function(wrap_pyfunction!(available_functionals, m)?)?;
    m.add_function(wrap_pyfunction!(available_solvents, m)?)?;
    m.add_function(wrap_pyfunction!(basis_search_path, m)?)?;
    m.add_function(wrap_pyfunction!(available_basis_sets, m)?)?;
    m.add_function(wrap_pyfunction!(find_basis_file, m)?)?;
    m.add_function(wrap_pyfunction!(set_package_basis_dir, m)?)?;
    m.add_function(wrap_pyfunction!(set_solvent_table, m)?)?;
    m.add_function(wrap_pyfunction!(solvent_table, m)?)?;
    m.add_function(wrap_pyfunction!(element_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(harmonic_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(thermochemistry, m)?)?;
    m.add_function(wrap_pyfunction!(atomic_mass, m)?)?;
    Ok(())
}
