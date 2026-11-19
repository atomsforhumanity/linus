//! Solvent database: the single source of solvent parameters in linus.
//!
//! - **Dielectric constant** (default): the CRC Handbook of Chemistry and Physics, 106th ed.
//!   (C. Wohlfarth, "Permittivity (Dielectric Constant) of Liquids"), as packaged by AFH's qcref
//!   (v0.5.1, commit f2b9e44, MIT). linus ships the rows of the SMD solvents, the other solvents
//!   it names and about 30 common liquids (`data/crc.csv`, 207 of qcref's 1303); the other CRC
//!   liquids resolve through qcref's full table where it is installed ([`crc_table`]), with the
//!   same rules. The value is, in order of preference: (1) the tabulated value when the CRC measured it at exactly 298.15 K
//!   (25 °C; where it and the fit differ, the measurement wins, as in the Minnesota database);
//!   (2) epsilon at 298.15 K from the CRC fit a + bT + cT^2 + dT^3 when 298.15 K lies within the
//!   fit's range widened by [`FIT_EXTRAPOLATION_K`] (the Minnesota database extrapolated those
//!   fits the same way, up to 25 K) and the result is >= 1; (3) the tabulated value at its own
//!   temperature. [`EpsSource`] records which. Five SMD solvents that the CRC table lacks keep
//!   the Minnesota value. (Policy adopted 2026-11-07.)
//! - **SMD descriptors** (n, alpha, beta, gamma, phi, psi): the Minnesota Solvent Descriptor
//!   Database (Winget, Dolney, Giesen, Cramer, Truhlar; version of 2026-01-06, which includes the
//!   2021 DMSO refractive index correction and water's descriptors), `data/smd_minnesota.csv`,
//!   179 solvents incl. water. Its epsilon is kept as `eps_minnesota`.
//! - **Program epsilons** for reproducing other programs exactly (explicit override, never the
//!   default): PySCF 2.15 (`data/program_eps.csv`); ORCA 6.0.0's SMD and CPCM-only tables and
//!   CPCM refractive indices (`data/orca/solvent_eps.csv` of the source tree: values ORCA
//!   prints, included with permission of FAccTs GmbH).
//! - **Names**: qcref's UPPER_SNAKE keys, CRC names, Minnesota names, PySCF's names and aliases,
//!   and common abbreviations (`data/aliases.csv`), and the names ORCA 6.0.0 accepts
//!   (`data/orca/solvent_aliases.csv` of the source tree). [`lookup`] ignores case and every
//!   non-alphanumeric character ("THF", "tetrahydrofuran", "Tetra-hydrofuran").
//!
//! `data/README.md` records sources, versions and licenses; `tools/sync_qcref_solvents.py`
//! refreshes the CRC copy from qcref, `tools/gen_solvent_db.py` regenerates the SMD tables and
//! `tools/gen_solvent_db_orca.py` ORCA's.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const CRC_CSV: &str = include_str!("data/crc.csv");
const SMD_CSV: &str = include_str!("data/smd_minnesota.csv");
const PROGRAM_CSV: &str = include_str!("data/program_eps.csv");
const ALIAS_CSV: &str = include_str!("data/aliases.csv");
const ORCA_EPS_CSV: &str = include_str!("../../../data/orca/solvent_eps.csv");
const ORCA_ALIAS_CSV: &str = include_str!("../../../data/orca/solvent_aliases.csv");

/// Reference temperature of the default dielectric constants (K).
pub const T_REF_K: f64 = 298.15;
/// How far outside its stated range (K) a CRC fit may be evaluated at [`T_REF_K`].
pub const FIT_EXTRAPOLATION_K: f64 = 30.0;

/// SMD solvent descriptors (Minnesota Solvent Descriptor Database).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmdDescriptors {
    /// refractive index at 293 K (n_D^20)
    pub n: f64,
    /// Abraham hydrogen-bond acidity (sum alpha_2^H)
    pub alpha: f64,
    /// Abraham hydrogen-bond basicity (sum beta_2^H)
    pub beta: f64,
    /// macroscopic surface tension at 298 K in cal mol^-1 Å^-2
    pub gamma: f64,
    /// aromaticity: fraction of non-hydrogen atoms that are aromatic carbons
    pub phi: f64,
    /// electronegative halogenicity: fraction of non-hydrogen atoms that are F, Cl or Br
    pub psi: f64,
    /// use SMD's aqueous parameter set (water); then only `water` matters for G_CDS
    pub water: bool,
}

/// CRC polynomial fit epsilon(T) = a + bT + cT^2 + dT^3 (T in K).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DielectricFit {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub t_min_k: Option<f64>,
    pub t_max_k: Option<f64>,
}

impl DielectricFit {
    /// The fit at `t_k` (no range check).
    pub fn eval(&self, t_k: f64) -> f64 {
        self.a + self.b * t_k + self.c * t_k * t_k + self.d * t_k * t_k * t_k
    }
    /// Kelvin outside the stated range (negative below t_min, positive above t_max, 0 inside).
    pub fn outside(&self, t_k: f64) -> f64 {
        match (self.t_min_k, self.t_max_k) {
            (Some(lo), _) if t_k < lo => t_k - lo,
            (_, Some(hi)) if t_k > hi => t_k - hi,
            _ => 0.0,
        }
    }
}

/// One row of the CRC permittivity table (qcref).
#[derive(Clone, Debug, PartialEq)]
pub struct CrcRecord {
    pub name: String,
    pub formula: Option<String>,
    /// tabulated epsilon at `dielectric_temperature_k`
    pub dielectric: f64,
    pub dielectric_temperature_k: Option<f64>,
    pub fit: Option<DielectricFit>,
}

/// Where a solvent's default epsilon comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EpsSource {
    /// CRC fit at 298.15 K, `extrapolated_k` outside its range (0 if inside)
    CrcFit { extrapolated_k: f64 },
    /// CRC tabulated value at its temperature
    CrcTabulated { temperature_k: Option<f64> },
    /// Minnesota database (solvent not in the CRC table)
    Minnesota,
}

/// Which epsilon to use: the database default or another program's own value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EpsConvention {
    /// qcref/CRC at 298.15 K (see the module docs)
    #[default]
    Default,
    /// the Minnesota Solvent Descriptor Database (SMD's parametrization)
    Minnesota,
    /// PySCF 2.15 (`pyscf.solvent._solvent_data`): Minnesota, water 78.3553
    Pyscf,
    /// ORCA 6.0.0 with SMD (`data/orca/solvent_eps.csv`; None for solvents ORCA lacks)
    OrcaSmd,
    /// ORCA 6.0.0 CPCM without SMD: ORCA's own table (`data/orca/solvent_eps.csv`; None for
    /// solvents ORCA lacks)
    OrcaCpcm,
}

#[derive(Clone, Debug)]
pub struct Solvent {
    /// qcref key (UPPER_SNAKE of the CRC name, leading locant moved to the end), or a linus key
    /// in the same style for the few SMD solvents the CRC table lacks
    pub key: String,
    pub name: String,
    pub formula: Option<String>,
    /// default static dielectric constant (see [`EpsSource`])
    pub eps: f64,
    pub eps_source: EpsSource,
    pub crc: Option<CrcRecord>,
    /// SMD descriptors, if the solvent is in the Minnesota database
    pub smd: Option<SmdDescriptors>,
    /// refractive index at 298 K (n_D^25) from the Minnesota database, where it has one
    pub n25: Option<f64>,
    pub minnesota_name: Option<String>,
    pub eps_minnesota: Option<f64>,
    pub eps_pyscf: Option<f64>,
    pub eps_orca_smd: Option<f64>,
    pub eps_orca_cpcm: Option<f64>,
    /// refractive index ORCA 6.0.0's CPCM uses (its optical dielectric is n^2)
    pub refrac_orca_cpcm: Option<f64>,
}

impl Solvent {
    /// Epsilon under a convention; None if that program has no value for this solvent.
    pub fn eps_for(&self, c: EpsConvention) -> Option<f64> {
        match c {
            EpsConvention::Default => Some(self.eps),
            EpsConvention::Minnesota => self.eps_minnesota,
            EpsConvention::Pyscf => self.eps_pyscf,
            EpsConvention::OrcaSmd => self.eps_orca_smd,
            EpsConvention::OrcaCpcm => self.eps_orca_cpcm,
        }
    }

    /// CRC epsilon at temperature `t_k` from the fit (None without a fit; no range check).
    pub fn eps_at(&self, t_k: f64) -> Option<f64> {
        self.crc.as_ref()?.fit.map(|f| f.eval(t_k))
    }
}

struct Db {
    solvents: Vec<Solvent>,
    index: HashMap<String, Option<usize>>,
    aliases: Vec<(String, usize, String)>,
}

/// Case- and punctuation-insensitive form of a solvent name: lower-case ASCII alphanumerics.
pub fn normalize(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Minimal CSV reader (RFC 4180 quoting, no embedded newlines).
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let mut fields = Vec::new();
        let mut cur = String::new();
        let mut quoted = false;
        let mut chars = line.chars().peekable();
        while let Some(ch) = chars.next() {
            match (ch, quoted) {
                ('"', true) if chars.peek() == Some(&'"') => {
                    cur.push('"');
                    chars.next();
                }
                ('"', _) => quoted = !quoted,
                (',', false) => fields.push(std::mem::take(&mut cur)),
                _ => cur.push(ch),
            }
        }
        fields.push(cur);
        rows.push(fields);
    }
    rows
}

fn table(text: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut rows = parse_csv(text);
    let header = rows.remove(0);
    (header, rows)
}

fn col(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("solvent db: no column {name}"))
}

fn parse_num(s: &str) -> Result<Option<f64>, String> {
    let s = s.trim();
    if s.is_empty() {
        Ok(None)
    } else {
        s.parse().map(Some).map_err(|_| format!("bad number {s:?}"))
    }
}

fn num(s: &str) -> Option<f64> {
    parse_num(s).unwrap_or_else(|e| panic!("solvent db: {e}"))
}

fn default_eps(crc: &CrcRecord) -> (f64, EpsSource) {
    // a value measured at 25 °C wins over the fit evaluated there
    if crc
        .dielectric_temperature_k
        .is_some_and(|t| (t - T_REF_K).abs() < 1e-9)
    {
        return (
            crc.dielectric,
            EpsSource::CrcTabulated {
                temperature_k: crc.dielectric_temperature_k,
            },
        );
    }
    if let Some(fit) = crc.fit {
        let out = fit.outside(T_REF_K);
        let v = fit.eval(T_REF_K);
        if out.abs() <= FIT_EXTRAPOLATION_K && v >= 1.0 {
            return (
                v,
                EpsSource::CrcFit {
                    extrapolated_k: out,
                },
            );
        }
    }
    (
        crc.dielectric,
        EpsSource::CrcTabulated {
            temperature_k: crc.dielectric_temperature_k,
        },
    )
}

/// The solvents of a CRC table (qcref's `crc.csv` columns), in its order.
fn crc_solvents(text: &str) -> Result<Vec<Solvent>, String> {
    let (h, rows) = table(text);
    let col = |name: &str| {
        h.iter()
            .position(|x| x == name)
            .ok_or_else(|| format!("no column {name}"))
    };
    let (ck, cn, cf, cd, ct) = (
        col("key")?,
        col("name")?,
        col("formula")?,
        col("dielectric")?,
        col("dielectric_temperature_k")?,
    );
    let (fa, fb, fc, fd, tmin, tmax) = (
        col("fit_a")?,
        col("fit_b")?,
        col("fit_c")?,
        col("fit_d")?,
        col("t_min_k")?,
        col("t_max_k")?,
    );
    let mut solvents = Vec::with_capacity(rows.len());
    for r in &rows {
        if r.len() != h.len() {
            return Err(format!(
                "row {:?} has {} fields, the header {}",
                r[0],
                r.len(),
                h.len()
            ));
        }
        let at = |i: usize| parse_num(&r[i]).map_err(|e| format!("{}: {e}", r[ck]));
        let fit = match at(fa)? {
            Some(a) => Some(DielectricFit {
                a,
                b: at(fb)?.unwrap_or(0.0),
                c: at(fc)?.unwrap_or(0.0),
                d: at(fd)?.unwrap_or(0.0),
                t_min_k: at(tmin)?,
                t_max_k: at(tmax)?,
            }),
            None => None,
        };
        let crc = CrcRecord {
            name: r[cn].clone(),
            formula: (!r[cf].is_empty()).then(|| r[cf].clone()),
            dielectric: at(cd)?.ok_or_else(|| format!("{}: no dielectric", r[ck]))?,
            dielectric_temperature_k: at(ct)?,
            fit,
        };
        let (eps, eps_source) = default_eps(&crc);
        solvents.push(Solvent {
            key: r[ck].clone(),
            name: crc.name.clone(),
            formula: crc.formula.clone(),
            eps,
            eps_source,
            crc: Some(crc),
            smd: None,
            n25: None,
            minnesota_name: None,
            eps_minnesota: None,
            eps_pyscf: None,
            eps_orca_smd: None,
            eps_orca_cpcm: None,
            refrac_orca_cpcm: None,
        });
    }
    Ok(solvents)
}

/// Add a name form to the index; a form claimed by two solvents is ambiguous (None).
fn claim(index: &mut HashMap<String, Option<usize>>, name: &str, i: usize) {
    let n = normalize(name);
    if n.is_empty() {
        return;
    }
    match index.get(&n) {
        None => {
            index.insert(n, Some(i));
        }
        Some(Some(j)) if *j != i => {
            index.insert(n, None);
        }
        _ => {}
    }
}

fn build() -> Db {
    let mut solvents: Vec<Solvent> = Vec::new();
    let mut by_key: HashMap<String, usize> = HashMap::new();

    let crc = crc_solvents(CRC_CSV).unwrap_or_else(|e| panic!("solvent db: data/crc.csv: {e}"));
    for s in crc {
        by_key.insert(s.key.clone(), solvents.len());
        solvents.push(s);
    }

    let (h, rows) = table(SMD_CSV);
    let c = |n: &str| col(&h, n);
    let (k, mname, ncrc, n, n25, al, be, ga, ep, ph, ps) = (
        c("key"),
        c("minnesota_name"),
        c("name_if_not_crc"),
        c("n"),
        c("n25"),
        c("alpha"),
        c("beta"),
        c("gamma"),
        c("eps"),
        c("phi"),
        c("psi"),
    );
    for r in &rows {
        let idx = match by_key.get(&r[k]) {
            Some(&i) => i,
            None => {
                let eps = num(&r[ep]).expect("Minnesota eps");
                by_key.insert(r[k].clone(), solvents.len());
                solvents.push(Solvent {
                    key: r[k].clone(),
                    name: r[ncrc].clone(),
                    formula: None,
                    eps,
                    eps_source: EpsSource::Minnesota,
                    crc: None,
                    smd: None,
                    n25: None,
                    minnesota_name: None,
                    eps_minnesota: None,
                    eps_pyscf: None,
                    eps_orca_smd: None,
                    eps_orca_cpcm: None,
                    refrac_orca_cpcm: None,
                });
                solvents.len() - 1
            }
        };
        let s = &mut solvents[idx];
        s.smd = Some(SmdDescriptors {
            n: num(&r[n]).unwrap(),
            alpha: num(&r[al]).unwrap(),
            beta: num(&r[be]).unwrap(),
            gamma: num(&r[ga]).unwrap(),
            phi: num(&r[ph]).unwrap(),
            psi: num(&r[ps]).unwrap(),
            water: r[k] == "WATER",
        });
        s.n25 = num(&r[n25]);
        s.minnesota_name = Some(r[mname].clone());
        s.eps_minnesota = num(&r[ep]);
    }

    let (h, rows) = table(PROGRAM_CSV);
    let (k, py) = (col(&h, "key"), col(&h, "pyscf_2_15"));
    for r in &rows {
        let i = *by_key
            .get(&r[k])
            .unwrap_or_else(|| panic!("program_eps: unknown key {}", r[k]));
        solvents[i].eps_pyscf = num(&r[py]);
    }
    // ORCA 6.0.0's values
    let (h, rows) = table(ORCA_EPS_CSV);
    let (k, os, oc, ocn) = (
        col(&h, "key"),
        col(&h, "orca_6_0_0_smd"),
        col(&h, "orca_6_0_0_cpcm"),
        col(&h, "orca_6_0_0_cpcm_refrac"),
    );
    for r in &rows {
        let i = *by_key
            .get(&r[k])
            .unwrap_or_else(|| panic!("orca/solvent_eps: unknown key {}", r[k]));
        let s = &mut solvents[i];
        s.eps_orca_smd = num(&r[os]);
        s.eps_orca_cpcm = num(&r[oc]);
        s.refrac_orca_cpcm = num(&r[ocn]);
    }

    // name index: keys, names, Minnesota names, then aliases; a form claimed by two solvents is
    // ambiguous and resolves to nothing
    let mut index: HashMap<String, Option<usize>> = HashMap::new();
    for (i, s) in solvents.iter().enumerate() {
        claim(&mut index, &s.key, i);
        claim(&mut index, &s.name, i);
        if let Some(m) = &s.minnesota_name {
            claim(&mut index, m, i);
        }
    }
    let mut alias_rows = Vec::new();
    for text in [ALIAS_CSV, ORCA_ALIAS_CSV] {
        let (h, rows) = table(text);
        let (ca, ck, cs) = (col(&h, "alias"), col(&h, "key"), col(&h, "source"));
        for r in rows {
            alias_rows.push((r[ca].clone(), r[ck].clone(), r[cs].clone()));
        }
    }
    // the generators' order over both files (key, normalized alias)
    alias_rows.sort_by(|a, b| (&a.1, normalize(&a.0)).cmp(&(&b.1, normalize(&b.0))));
    let mut aliases = Vec::new();
    for (alias, key, source) in alias_rows {
        let i = *by_key
            .get(&key)
            .unwrap_or_else(|| panic!("aliases: unknown key {key}"));
        claim(&mut index, &alias, i);
        aliases.push((alias, i, source));
    }
    // a key always wins over any other name that normalizes to it
    for (i, s) in solvents.iter().enumerate() {
        index.insert(normalize(&s.key), Some(i));
    }
    Db {
        solvents,
        index,
        aliases,
    }
}

fn db() -> &'static Db {
    static DB: OnceLock<Db> = OnceLock::new();
    DB.get_or_init(build)
}

/// The full CRC table that names outside the shipped one resolve through: its solvents (CRC
/// rows only) and their keys and names.
struct FullTable {
    path: PathBuf,
    solvents: Vec<Solvent>,
    index: HashMap<String, Option<usize>>,
}

static REGISTERED_TABLE: OnceLock<PathBuf> = OnceLock::new();

/// Register a full CRC table (qcref's `data/solvents/crc.csv`): the Python package registers
/// qcref's at import where qcref is installed. The first registration counts; false if one was
/// there already.
pub fn set_crc_table(path: PathBuf) -> bool {
    REGISTERED_TABLE.set(path).is_ok()
}

/// The full CRC table that solvent names outside the shipped table resolve through:
/// `$LINUS_SOLVENT_TABLE` if set, else the registered one ([`set_crc_table`]). It is read once,
/// at the first name the shipped table lacks.
pub fn crc_table() -> Option<PathBuf> {
    std::env::var_os("LINUS_SOLVENT_TABLE")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| REGISTERED_TABLE.get().cloned())
}

fn read_full_table(path: &Path) -> Result<FullTable, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let solvents = crc_solvents(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut index = HashMap::new();
    for (i, s) in solvents.iter().enumerate() {
        claim(&mut index, &s.key, i);
        claim(&mut index, &s.name, i);
    }
    // a key always wins, as in the shipped table
    for (i, s) in solvents.iter().enumerate() {
        index.insert(normalize(&s.key), Some(i));
    }
    Ok(FullTable {
        path: path.to_path_buf(),
        solvents,
        index,
    })
}

/// The full CRC table, read on first use: None without one, Err if it could not be read.
fn full_table() -> Option<&'static Result<FullTable, String>> {
    static FULL: OnceLock<Option<Result<FullTable, String>>> = OnceLock::new();
    FULL.get_or_init(|| crc_table().map(|p| read_full_table(&p)))
        .as_ref()
}

/// Look a solvent up by key, name or alias (case- and punctuation-insensitive): in the shipped
/// table, and a name it does not know in the full CRC table ([`crc_table`]). A name that is
/// ambiguous in the shipped table resolves to nothing.
pub fn lookup(name: &str) -> Option<&'static Solvent> {
    let d = db();
    let n = normalize(name);
    match d.index.get(&n) {
        Some(found) => found.map(|i| &d.solvents[i]),
        None => match full_table()? {
            Ok(t) => t.index.get(&n).copied().flatten().map(|i| &t.solvents[i]),
            Err(_) => None,
        },
    }
}

/// The full CRC table a solvent came from; None for the shipped table's.
pub fn table_of(s: &Solvent) -> Option<&'static Path> {
    match full_table()? {
        Ok(t) if t.solvents.as_ptr_range().contains(&(s as *const Solvent)) => {
            Some(t.path.as_path())
        }
        _ => None,
    }
}

/// The solvents of the full CRC table ([`crc_table`]), empty without one.
pub fn full_table_solvents() -> &'static [Solvent] {
    match full_table() {
        Some(Ok(t)) => &t.solvents,
        _ => &[],
    }
}

/// Why a name outside the shipped table did not resolve: the state of the full CRC table.
pub fn full_table_note() -> String {
    match full_table() {
        None => "linus ships the permittivities of the SMD solvents and about 30 other common \
                 liquids; the rest of the CRC table resolves where qcref is installed (Python) or \
                 $LINUS_SOLVENT_TABLE names qcref's data/solvents/crc.csv."
            .into(),
        Some(Ok(t)) => format!("It is not in the CRC table {} either.", t.path.display()),
        Some(Err(e)) => format!("The CRC table could not be read ({e})."),
    }
}

/// SMD descriptors of a solvent by name.
pub fn smd_descriptors(name: &str) -> Option<&'static SmdDescriptors> {
    lookup(name)?.smd.as_ref()
}

/// The shipped solvents (CRC table order, then the SMD solvents the CRC table lacks).
pub fn solvents() -> &'static [Solvent] {
    &db().solvents
}

/// The solvents with SMD descriptors.
pub fn smd_solvents() -> impl Iterator<Item = &'static Solvent> {
    solvents().iter().filter(|s| s.smd.is_some())
}

/// Aliases recorded for a solvent key, with their source ("minnesota", "pyscf", "orca", "common").
pub fn aliases(key: &str) -> Vec<(&'static str, &'static str)> {
    let d = db();
    d.aliases
        .iter()
        .filter(|(_, i, _)| d.solvents[*i].key == key)
        .map(|(a, _, s)| (a.as_str(), s.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_common_names() {
        assert_eq!(solvents().iter().filter(|s| s.crc.is_some()).count(), 207);
        assert_eq!(smd_solvents().count(), 179);
        for (name, key) in [
            ("thf", "TETRAHYDROFURAN"),
            ("Tetrahydrofuran", "TETRAHYDROFURAN"),
            ("TETRAHYDROFURAN", "TETRAHYDROFURAN"),
            ("water", "WATER"),
            ("H2O", "WATER"),
            ("DMSO", "DIMETHYL_SULFOXIDE"),
            ("dimethylsulfoxide", "DIMETHYL_SULFOXIDE"),
            ("CH2Cl2", "DICHLOROMETHANE"),
            ("dcm", "DICHLOROMETHANE"),
            ("chloroform", "TRICHLOROMETHANE"),
            ("CHCl3", "TRICHLOROMETHANE"),
            ("1,4-dioxane", "DIOXANE_1_4"),
            ("DIOXANE_1_4", "DIOXANE_1_4"),
            ("n-hexane", "HEXANE"),
            ("hexane", "HEXANE"),
            ("ETHYL ACETATE", "ETHYL_ACETATE"),
            ("ethyl ethanoate", "ETHYL_ACETATE"),
            ("toluene", "TOLUENE"),
            ("benzene", "BENZENE"),
            ("MeCN", "ACETONITRILE"),
            ("isopropanol", "PROPANOL_2"),
            ("sulfolane", "SULFOLANE"),
            ("tetrahydrothiophene-S,S-dioxide", "SULFOLANE"),
            ("propylene carbonate", "PROPYLENE_CARBONATE"),
        ] {
            assert_eq!(lookup(name).map(|s| s.key.as_str()), Some(key), "{name}");
        }
        assert!(lookup("no such solvent").is_none());
    }

    #[test]
    fn default_eps_matches_smd_values() {
        // CRC fits at 298.15 K reproduce the Minnesota epsilons
        for (name, eps) in [
            ("water", 78.3553),
            ("thf", 7.4257),
            ("methanol", 32.613),
            ("dmso", 46.826),
            ("toluene", 2.3741),
        ] {
            let s = lookup(name).unwrap();
            assert!(
                (s.eps - eps).abs() < 2e-4 * eps,
                "{name}: {} vs {eps}",
                s.eps
            );
            assert!(matches!(s.eps_source, EpsSource::CrcFit { .. }));
        }
        // THF's fit ends at 295 K: 3.15 K extrapolation
        match lookup("thf").unwrap().eps_source {
            EpsSource::CrcFit { extrapolated_k } => assert!((extrapolated_k - 3.15).abs() < 1e-9),
            e => panic!("{e:?}"),
        }
        // measured at 298.15 K: the tabulated value wins over the fit there (fit in brackets);
        // this reproduces the Minnesota values
        for (name, eps, fit) in [
            ("1-pentanol", 15.13, 14.6928),
            ("dichloromethane", 8.93, 8.8192),
            ("m-cresol", 12.44, 12.6961),
            ("formic acid", 51.1, 51.5913),
            ("acetophenone", 17.44, 17.4258),
        ] {
            let s = lookup(name).unwrap();
            assert_eq!(s.eps, eps, "{name}");
            assert_eq!(
                s.eps_source,
                EpsSource::CrcTabulated {
                    temperature_k: Some(298.15)
                }
            );
            assert!((s.eps_at(T_REF_K).unwrap() - fit).abs() < 1e-4, "{name}");
            assert_eq!(Some(s.eps), s.eps_minnesota);
        }
        // no fit: tabulated value with its temperature
        let s = lookup("pentanal").unwrap();
        assert_eq!(s.eps, 10.0);
        assert_eq!(
            s.eps_source,
            EpsSource::CrcTabulated {
                temperature_k: Some(293.15)
            }
        );
        // not in CRC: Minnesota
        let s = lookup("sulfolane").unwrap();
        assert_eq!((s.eps, s.eps_source), (43.962, EpsSource::Minnesota));
        // every default epsilon is physical
        for s in solvents() {
            assert!(s.eps >= 1.0, "{} {}", s.key, s.eps);
        }
    }

    #[test]
    fn program_conventions() {
        let w = lookup("water").unwrap();
        assert_eq!(w.eps_for(EpsConvention::Pyscf), Some(78.3553));
        assert_eq!(w.eps_for(EpsConvention::Minnesota), Some(78.355));
        let t = lookup("thf").unwrap();
        assert!(w.smd.unwrap().water && !t.smd.unwrap().water);
        // ORCA's values: crates/linus/tests/orca_presets.rs
    }

    #[test]
    fn smd_descriptor_values() {
        let d = smd_descriptors("chloroform").unwrap();
        assert_eq!(
            (d.n, d.alpha, d.beta, d.gamma, d.phi, d.psi),
            (1.4459, 0.15, 0.02, 38.39, 0.0, 0.75)
        );
        let d = smd_descriptors("benzene").unwrap();
        assert_eq!(
            (d.n, d.alpha, d.beta, d.gamma, d.phi, d.psi),
            (1.5011, 0.0, 0.14, 40.62, 1.0, 0.0)
        );
        // 2021 correction of DMSO's refractive index
        assert_eq!(smd_descriptors("dmso").unwrap().n, 1.4783);
    }

    #[test]
    fn aliases_are_unambiguous() {
        let d = db();
        for (alias, i, _) in &d.aliases {
            assert_eq!(
                lookup(alias).map(|s| s.key.as_str()),
                Some(d.solvents[*i].key.as_str()),
                "{alias}"
            );
        }
        for s in solvents() {
            assert_eq!(lookup(&s.key).map(|x| x.key.as_str()), Some(s.key.as_str()));
        }
    }

    /// A name the shipped table lacks resolves through a full CRC table with the same rules;
    /// the shipped table's names, ambiguous ones included, never reach it.
    #[test]
    fn full_table_fallback() {
        let header = CRC_CSV.lines().next().unwrap();
        let water = CRC_CSV.lines().find(|l| l.starts_with("WATER,")).unwrap();
        let fields = header.split(',').count();
        let row = |key: &str, name: &str, eps: &str| {
            let mut f = vec![String::new(); fields];
            let c = |n: &str| header.split(',').position(|h| h == n).unwrap();
            f[c("key")] = key.into();
            f[c("name")] = name.into();
            f[c("dielectric")] = eps.into();
            f[c("dielectric_temperature_k")] = "298.15".into();
            f.join(",")
        };
        let text = [
            header.to_string(),
            water.to_string(),
            row("SOLVENT_X", "Solvent X", "12.5"),
            row("SOLVENT_Y", "Liquid Q", "3.25"),
            row("SOLVENT_Z", "Liquid Q", "4.0"),
        ]
        .join("\n");
        let dir = std::env::temp_dir().join(format!("linus-crc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("crc.csv");
        std::fs::write(&path, &text).unwrap();
        let t = read_full_table(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let find = |n: &str| t.index.get(&normalize(n)).copied().flatten();
        let x = &t.solvents[find("solvent-x").unwrap()];
        assert_eq!((x.key.as_str(), x.eps), ("SOLVENT_X", 12.5));
        assert_eq!(
            x.eps_source,
            EpsSource::CrcTabulated {
                temperature_k: Some(298.15)
            }
        );
        // two solvents with one name: only their keys resolve
        assert!(find("liquid-q").is_none());
        assert_eq!(t.solvents[find("SOLVENT_Z").unwrap()].eps, 4.0);
        // a shipped row read from the full table is the shipped solvent, bit for bit
        let w = &t.solvents[find("WATER").unwrap()];
        let shipped = lookup("WATER").unwrap();
        assert_eq!(w.eps.to_bits(), shipped.eps.to_bits());
        assert_eq!(w.crc, shipped.crc);
        assert!(table_of(shipped).is_none());
        // a malformed table is an error, not a panic
        assert!(crc_solvents("key,name\nA,a\n").is_err());
        let bad = format!("{header}\n{}", row("BAD", "bad", "x"));
        assert!(crc_solvents(&bad).unwrap_err().contains("bad number"));
    }

    /// The shipped CRC rows are qcref's, verbatim and in its order (run with
    /// QCREF_PYTHON=<python with qcref>; skipped when no such Python is found).
    #[test]
    fn vendored_crc_matches_qcref() {
        let py = std::env::var("QCREF_PYTHON").unwrap_or_else(|_| "python3".into());
        let out = std::process::Command::new(&py)
            .args(["-c", "import qcref, os; print(os.path.join(os.path.dirname(qcref.__file__), 'data', 'solvents'))"])
            .output();
        let dir = match out {
            Ok(o) if o.status.success() => String::from_utf8(o.stdout).unwrap().trim().to_string(),
            _ => {
                eprintln!("qcref not importable with {py}; skipping");
                return;
            }
        };
        let theirs = std::fs::read_to_string(format!("{dir}/crc.csv")).unwrap();
        let shipped: std::collections::HashSet<&str> = CRC_CSV.lines().collect();
        let subset: Vec<&str> = theirs.lines().filter(|l| shipped.contains(l)).collect();
        assert_eq!(
            subset,
            CRC_CSV.lines().collect::<Vec<_>>(),
            "data/crc.csv is not a subset of qcref's rows; run tools/sync_qcref_solvents.py"
        );
        // and every shipped solvent resolves to the same values through qcref's full table
        let full = read_full_table(Path::new(&format!("{dir}/crc.csv"))).unwrap();
        for s in solvents().iter().filter(|s| s.crc.is_some()) {
            let i = full.index[&normalize(&s.key)].unwrap();
            assert_eq!(full.solvents[i].crc, s.crc, "{}", s.key);
            assert_eq!(full.solvents[i].eps.to_bits(), s.eps.to_bits(), "{}", s.key);
        }
        let meta = std::fs::read_to_string(format!("{dir}/crc.meta.json")).unwrap();
        assert_eq!(meta, include_str!("data/crc.meta.json"));
    }
}
