//! The settings schema: every setting declared once, with its metadata.
//!
//! A settings group is declared with `settings_group!`: each setting's name, value type,
//! default, tier (common, expert, dev), concern (model, approximation, convergence, algorithm,
//! resource, output), unit, the lowest calctype it applies to, and its doc comment. The macro
//! generates the group's struct, whose fields are `Option`s (explicit presence: `Some` is a
//! setting the request gave, `None` unset; the D41 checks read it, so settings built in Rust are
//! checked as strictly as JSON), an accessor per setting that applies the default, the reader of
//! a JSON object (keys normalized, unknown keys an `unknown_key` error with a did-you-mean
//! suggestion, values parsed by their type) and the group's entries of the metadata table
//! ([`SettingMeta`]).
//!
//! Everything the front ends show about the settings is generated from that table:
//! - the JSON Schema of the settings (draft 2020-12; [`json_schema`]: `linus schema settings`,
//!   Python's `linus.settings_schema()`), with `x-tier`, `x-concern`, `x-unit` and
//!   `x-applies-to` on every key, as qcdata's conventions ask of every AFH program;
//! - `linus --list-settings [GROUP] [--all]` ([`list_settings`]);
//! - the settings tables of docs/cli.md ([`docs_tables`]) and the Python `TypedDict` stubs of
//!   linus-py ([`python_stubs`]), both checked against the repository by a test;
//! - the did-you-mean suggestions of unknown keys;
//! - the calctype half of decision D41's applicability (`crate::driver::settings::applies`).
//!
//! Design note: a declarative macro rather than a schema-deriving crate. Nearly every
//! non-trivial setting has a string grammar of its own (grids, J/K methods, auxiliary bases,
//! dispersion, the group shorthands), which a derive cannot see, and the metadata is needed at
//! run time; the table is plain data, and the core crate takes no procedural macro.

use linus_base::error::{did_you_mean, LinusError};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// Who a setting is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// What most calculations set.
    Common,
    /// Tuning and comparisons with other programs.
    Expert,
    /// Development and studies: hidden from `--list-settings` unless `--all`.
    Dev,
}

/// What a setting is about (qcdata's conventions; resource and output settings do not change
/// what is computed and are left out of a result's identity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Concern {
    Model,
    Approximation,
    Convergence,
    Algorithm,
    Resource,
    Output,
}

/// The lowest calctype a setting can take effect in (`x-applies-to`): the higher calctypes
/// compute the lower ones on the way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Needs {
    Energy,
    Gradient,
    Hessian,
    /// Hessians with IR intensities (`freq.ir`).
    Ir,
}

macro_rules! names {
    ($t:ty { $($v:ident => $s:literal),* $(,)? }) => {
        impl $t {
            /// The name in the schema.
            pub fn name(self) -> &'static str {
                match self { $(Self::$v => $s),* }
            }
        }
    };
}
names!(Tier { Common => "common", Expert => "expert", Dev => "dev" });
names!(Concern {
    Model => "model", Approximation => "approximation", Convergence => "convergence",
    Algorithm => "algorithm", Resource => "resource", Output => "output",
});

/// The lower-case spellings `settings_group!` declarations use (`[common, convergence]`).
#[allow(non_upper_case_globals)]
impl Tier {
    pub const common: Tier = Tier::Common;
    pub const expert: Tier = Tier::Expert;
    pub const dev: Tier = Tier::Dev;
}

#[allow(non_upper_case_globals)]
impl Concern {
    pub const model: Concern = Concern::Model;
    pub const approximation: Concern = Concern::Approximation;
    pub const convergence: Concern = Concern::Convergence;
    pub const algorithm: Concern = Concern::Algorithm;
    pub const resource: Concern = Concern::Resource;
    pub const output: Concern = Concern::Output;
}

impl Needs {
    /// The calctypes a setting with these needs applies to (`x-applies-to`).
    pub fn calctypes(self) -> &'static [&'static str] {
        match self {
            Needs::Energy => &["energy", "gradient", "hessian"],
            Needs::Gradient => &["gradient", "hessian"],
            Needs::Hessian | Needs::Ir => &["hessian"],
        }
    }
}

/// One setting of the table.
#[derive(Clone, Copy)]
pub struct SettingMeta {
    /// The dotted path (`scf.max_iter`; top-level settings have no dot).
    pub path: &'static str,
    /// The doc comment of its declaration.
    pub doc: &'static str,
    pub tier: Tier,
    pub concern: Concern,
    /// The unit, where the value has one (`Eh`, `GiB`, `K`).
    pub unit: Option<&'static str>,
    pub needs: Needs,
    /// The JSON Schema of its value.
    pub schema: fn() -> Value,
    /// Its default, as JSON (`"auto"`: decided by the calculation).
    pub default: fn() -> Value,
}

impl SettingMeta {
    /// The doc comment as one paragraph.
    pub fn text(&self) -> String {
        paragraph(self.doc)
    }

    /// The setting's key in its group (`max_iter`).
    pub fn key(&self) -> &'static str {
        self.path.rsplit('.').next().unwrap_or(self.path)
    }

    /// The group it belongs to (`""` for top-level settings).
    pub fn group(&self) -> &'static str {
        self.path.rsplit_once('.').map_or("", |(g, _)| g)
    }
}

/// One group of settings (`scf`, `solvent`): its doc comment and value shorthand.
#[derive(Clone, Copy)]
pub struct GroupMeta {
    pub path: &'static str,
    pub doc: &'static str,
    /// Absent means off (the solvent: gas phase), not "every setting at its default".
    pub optional: bool,
    /// The schema of the value that stands for the whole group (`jk: "df"`), if it has one.
    pub shorthand: Option<fn() -> Value>,
}

/// The doc comment lines as one paragraph.
fn paragraph(doc: &str) -> String {
    doc.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A key in its canonical form: lower case, `-` as `_` (`Max-Iter` is `max_iter`).
pub fn norm_name(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace('-', "_")
}

// ---------------------------------------------------------------------------------- values

/// The type of a setting's value: its JSON Schema and how it is read from JSON.
pub trait SettingType: Sized + Serialize {
    /// The JSON Schema of the value.
    fn schema() -> Value;

    /// The value from JSON. The default reads it with serde, an enum's string first normalized
    /// (lower case, `-` as `_`) and replaced by the name its schema lists that normalizes the
    /// same (so `Very-Tight` reads as `verytight` and serde's aliases, written normalized, apply).
    fn parse(v: Value) -> Result<Self, LinusError>
    where
        Self: DeserializeOwned,
    {
        let v = match (&v, Self::schema().get("enum")) {
            (Value::String(s), Some(Value::Array(names))) => {
                let k = norm_name(s);
                let squash = |x: &str| x.replace('_', "");
                names
                    .iter()
                    .find(|n| n.as_str().is_some_and(|n| norm_name(n) == k))
                    .or_else(|| {
                        names.iter().find(|n| {
                            n.as_str()
                                .is_some_and(|n| squash(&norm_name(n)) == squash(&k))
                        })
                    })
                    .cloned()
                    .unwrap_or(Value::String(k))
            }
            _ => v,
        };
        serde_json::from_value(v).map_err(|e| LinusError::Input(e.to_string()))
    }
}

/// Reads the setting at `path` from `v` (`T::parse`), naming the path in the error.
pub fn parse_setting<T: SettingType + DeserializeOwned>(
    path: &str,
    v: Value,
) -> Result<T, LinusError> {
    T::parse(v).map_err(|e| match e {
        LinusError::Input(m) if !m.starts_with(path) => LinusError::Input(format!("{path}: {m}")),
        e => e,
    })
}

impl SettingType for bool {
    fn schema() -> Value {
        json!({"type": "boolean"})
    }
}

impl SettingType for f64 {
    fn schema() -> Value {
        json!({"type": "number"})
    }
}

macro_rules! unsigned {
    ($($t:ty),*) => {$(
        impl SettingType for $t {
            fn schema() -> Value {
                json!({"type": "integer", "minimum": 0})
            }
        }
    )*};
}
unsigned!(usize, u32);

impl SettingType for String {
    fn schema() -> Value {
        json!({"type": "string"})
    }
}

/// A setting without a default (`solvent.name`): its value type's schema; the accessor gives
/// None while it is unset.
impl<T: SettingType + DeserializeOwned> SettingType for Option<T> {
    fn schema() -> Value {
        T::schema()
    }

    fn parse(v: Value) -> Result<Self, LinusError> {
        T::parse(v).map(Some)
    }
}

impl SettingType for BTreeMap<String, f64> {
    fn schema() -> Value {
        json!({"type": "object", "additionalProperties": {"type": "number"}})
    }
}

/// The schema and string forms of an enum setting: the names its schema lists (case and `-`/`_`
/// are normalized to them before serde reads the value, so serde's aliases still apply).
macro_rules! setting_enum {
    ($t:ty: $($name:literal),+ $(,)?) => {
        impl $crate::driver::schema::SettingType for $t {
            fn schema() -> serde_json::Value {
                serde_json::json!({"type": "string", "enum": [$($name),+]})
            }
        }
    };
}
pub(crate) use setting_enum;

/// A value linus may decide: `"auto"` or the value (a setting whose default is decided by the
/// calculation and that also takes a plain value: `scf.e_tol`, `memory_gib`). Serialized as
/// `"auto"` or the value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Auto<T> {
    #[default]
    Auto,
    Value(T),
}

impl<T: Copy> Auto<T> {
    /// The value, None for auto.
    pub fn value(self) -> Option<T> {
        match self {
            Auto::Auto => None,
            Auto::Value(v) => Some(v),
        }
    }
}

impl<T> Auto<T> {
    pub fn is_auto(&self) -> bool {
        matches!(self, Auto::Auto)
    }

    /// The value, or `f()` for auto.
    pub fn or_else(self, f: impl FnOnce() -> T) -> T {
        match self {
            Auto::Auto => f(),
            Auto::Value(v) => v,
        }
    }
}

impl<T> From<T> for Auto<T> {
    fn from(v: T) -> Self {
        Auto::Value(v)
    }
}

impl<T: Serialize> Serialize for Auto<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Auto::Auto => s.serialize_str("auto"),
            Auto::Value(v) => v.serialize(s),
        }
    }
}

impl<'de, T: SettingType + DeserializeOwned> serde::Deserialize<'de> for Auto<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        <Auto<T> as SettingType>::parse(Value::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

impl<T: SettingType + DeserializeOwned> SettingType for Auto<T> {
    fn schema() -> Value {
        json!({"anyOf": [{"const": "auto"}, T::schema()]})
    }

    fn parse(v: Value) -> Result<Self, LinusError> {
        match &v {
            Value::String(s) if norm_name(s) == "auto" => Ok(Auto::Auto),
            _ => T::parse(v).map(Auto::Value),
        }
    }
}

// ---------------------------------------------------------------------------------- groups

/// A group of settings, as `settings_group!` declares it (or an optional group, absent
/// meaning off).
pub trait Group: Sized + Default {
    /// Absent means off rather than "every setting at its default" (an `Option` of a group).
    const OPTIONAL: bool = false;

    /// Appends the group's settings and subgroups to the tables.
    fn meta(settings: &mut Vec<SettingMeta>, groups: &mut Vec<GroupMeta>);

    /// The group from JSON at `path`: an object of its settings, a shorthand, or null (unset).
    fn from_value(path: &str, v: Value) -> Result<Self, LinusError>;

    /// Appends the dotted paths of the settings given (`Some`).
    fn given(&self, out: &mut Vec<String>);

    /// No setting given.
    fn is_unset(&self) -> bool;

    /// Every setting, the unset ones at their defaults (`"auto"` where the calculation
    /// decides), as JSON; null for an optional group that is off.
    fn resolved(&self) -> Value;

    /// The schema of the group's shorthand value, if it has one.
    fn shorthand_schema() -> Option<Value>;
}

/// What a value that is not an object stands for, for a group with a shorthand (`jk: "df"`
/// is `{"kind": "df"}`). Groups without one take the default: an error.
pub trait Shorthand {
    /// The object the value `v` at `path` stands for.
    fn expand(path: &str, v: Value) -> Result<Map<String, Value>, String> {
        Err(format!("{path} must be an object of its settings, not {v}"))
    }

    /// The schema of the shorthand values (None: no shorthand).
    fn schema() -> Option<Value> {
        None
    }
}

/// `skip_serializing_if` of a group field: nothing given.
pub fn is_unset<G: Group>(g: &G) -> bool {
    g.is_unset()
}

impl<T: Group> Group for Option<T> {
    const OPTIONAL: bool = true;

    fn meta(settings: &mut Vec<SettingMeta>, groups: &mut Vec<GroupMeta>) {
        T::meta(settings, groups)
    }

    fn from_value(path: &str, v: Value) -> Result<Self, LinusError> {
        if v.is_null() {
            Ok(None)
        } else {
            T::from_value(path, v).map(Some)
        }
    }

    fn given(&self, out: &mut Vec<String>) {
        if let Some(g) = self {
            g.given(out)
        }
    }

    fn is_unset(&self) -> bool {
        self.is_none()
    }

    fn resolved(&self) -> Value {
        self.as_ref().map_or(Value::Null, T::resolved)
    }

    fn shorthand_schema() -> Option<Value> {
        T::shorthand_schema()
    }
}

/// The error of an unknown key `key` in the group at `prefix` (`"scf."`, `""`), with the
/// closest key of the group, or else of any group, as a suggestion.
pub fn unknown_key(prefix: &str, key: &str, keys: &[&str]) -> LinusError {
    let own: Vec<String> = keys.iter().map(|k| format!("{prefix}{k}")).collect();
    let mut hint = did_you_mean(&format!("{prefix}{key}"), &own);
    if hint.is_empty() {
        // a key of another group (`e_tol` at the top level: scf.e_tol), by its last part
        let all = table();
        let leaf: Vec<String> = all.settings.iter().map(|m| m.key().to_string()).collect();
        let near = did_you_mean(key, &leaf);
        if !near.is_empty() {
            let paths: Vec<String> = all
                .settings
                .iter()
                .filter(|m| near.contains(&format!("'{}'", m.key())))
                .map(|m| m.path.to_string())
                .collect();
            hint = did_you_mean(key, &paths);
            if hint.is_empty() && !paths.is_empty() {
                hint = format!(" Did you mean '{}'?", paths[0]);
            }
        }
    }
    LinusError::Setting {
        code: "unknown_key",
        path: format!("{prefix}{key}"),
        message: format!("not a setting.{hint} (`linus --list-settings` lists them)"),
    }
}

/// Declares a group of settings (module docs): the struct, the accessors and the [`Group`]
/// implementation. Each setting is
/// `/// doc` `name: Type = default, [tier, concern(, unit = "U")?(, needs = Calctype)?];`
/// and the optional `groups { /// doc name: GroupType, ... }` lists the subgroups.
macro_rules! settings_group {
    (
        $(#[doc = $gdoc:literal])*
        $vis:vis struct $name:ident at $prefix:literal {
            $(
                $(#[doc = $doc:literal])*
                $field:ident: $ty:ty = $default:expr,
                    [$tier:ident, $concern:ident $(, unit = $unit:literal)? $(, needs = $needs:ident)?];
            )*
        }
        $(groups {
            $(
                $(#[doc = $gfdoc:literal])*
                $gfield:ident: $gty:ty,
            )*
        })?
    ) => {
        $(#[doc = $gdoc])*
        #[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
        $vis struct $name {
            $(
                $(#[doc = $doc])*
                #[serde(skip_serializing_if = "Option::is_none")]
                pub $field: Option<$ty>,
            )*
            $($(
                $(#[doc = $gfdoc])*
                #[serde(skip_serializing_if = "crate::driver::schema::is_unset")]
                pub $gfield: $gty,
            )*)?
        }

        #[allow(dead_code)]
        impl $name {
            /// The keys of the group.
            pub const KEYS: &'static [&'static str] =
                &[$(stringify!($field),)* $($(stringify!($gfield),)*)?];

            $(
                $(#[doc = $doc])*
                ///
                /// (The value given, else the default.)
                pub fn $field(&self) -> $ty {
                    match &self.$field {
                        Some(v) => v.clone(),
                        None => $default,
                    }
                }
            )*

            /// The group from a JSON object (keys normalized; unknown keys an error).
            pub fn from_map(m: serde_json::Map<String, serde_json::Value>) -> Result<Self, linus_base::error::LinusError> {
                let mut s = Self::default();
                for (k, v) in m {
                    let key = crate::driver::schema::norm_name(&k);
                    match key.as_str() {
                        $(
                            stringify!($field) => {
                                if !v.is_null() {
                                    s.$field = Some(crate::driver::schema::parse_setting::<$ty>(
                                        concat!($prefix, stringify!($field)), v)?);
                                }
                            }
                        )*
                        $($(
                            stringify!($gfield) => {
                                s.$gfield = <$gty as crate::driver::schema::Group>::from_value(
                                    concat!($prefix, stringify!($gfield)), v)?;
                            }
                        )*)?
                        _ => return Err(crate::driver::schema::unknown_key($prefix, &key, Self::KEYS)),
                    }
                }
                Ok(s)
            }
        }

        impl crate::driver::schema::Group for $name {
            fn meta(
                settings: &mut Vec<crate::driver::schema::SettingMeta>,
                groups: &mut Vec<crate::driver::schema::GroupMeta>,
            ) {
                let _ = &groups;
                #[allow(unused_imports)]
                use crate::driver::schema::{Concern, Needs, SettingMeta, SettingType, Tier};
                $(
                    settings.push(SettingMeta {
                        path: concat!($prefix, stringify!($field)),
                        doc: concat!($($doc, "\n",)*),
                        tier: Tier::$tier,
                        concern: Concern::$concern,
                        unit: None::<&'static str>$(.or(Some($unit)))?,
                        needs: Needs::Energy$(.max(Needs::$needs))?,
                        schema: <$ty as SettingType>::schema,
                        default: || serde_json::to_value::<$ty>($default).expect("a default serializes"),
                    });
                )*
                $($(
                    groups.push(crate::driver::schema::GroupMeta {
                        path: concat!($prefix, stringify!($gfield)),
                        doc: concat!($($gfdoc, "\n",)*),
                        optional: <$gty as crate::driver::schema::Group>::OPTIONAL,
                        shorthand: <$gty as crate::driver::schema::Group>::shorthand_schema()
                            .map(|_| (|| <$gty as crate::driver::schema::Group>::shorthand_schema()
                                .expect("a shorthand")) as fn() -> serde_json::Value),
                    });
                    <$gty as crate::driver::schema::Group>::meta(settings, groups);
                )*)?
            }

            fn from_value(path: &str, v: serde_json::Value) -> Result<Self, linus_base::error::LinusError> {
                match v {
                    serde_json::Value::Null => Ok(Self::default()),
                    serde_json::Value::Object(m) => Self::from_map(m),
                    other => Self::from_map(
                        <Self as crate::driver::schema::Shorthand>::expand(path, other)
                            .map_err(linus_base::error::LinusError::Input)?,
                    ),
                }
            }

            fn given(&self, out: &mut Vec<String>) {
                $(
                    if self.$field.is_some() {
                        out.push(concat!($prefix, stringify!($field)).to_string());
                    }
                )*
                $($(
                    if <$gty as crate::driver::schema::Group>::OPTIONAL
                        && !crate::driver::schema::Group::is_unset(&self.$gfield)
                    {
                        out.push(concat!($prefix, stringify!($gfield)).to_string());
                    }
                    crate::driver::schema::Group::given(&self.$gfield, out);
                )*)?
            }

            fn is_unset(&self) -> bool {
                true $(&& self.$field.is_none())*
                    $($(&& crate::driver::schema::Group::is_unset(&self.$gfield))*)?
            }

            fn resolved(&self) -> serde_json::Value {
                let mut m = serde_json::Map::new();
                $(
                    m.insert(
                        stringify!($field).to_string(),
                        serde_json::to_value(self.$field()).expect("a setting serializes"),
                    );
                )*
                $($(
                    let g = crate::driver::schema::Group::resolved(&self.$gfield);
                    if !g.is_null() {
                        m.insert(stringify!($gfield).to_string(), g);
                    }
                )*)?
                serde_json::Value::Object(m)
            }

            fn shorthand_schema() -> Option<serde_json::Value> {
                <Self as crate::driver::schema::Shorthand>::schema()
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let v = serde_json::Value::deserialize(d)?;
                <Self as crate::driver::schema::Group>::from_value($prefix.trim_end_matches('.'), v)
                    .map_err(serde::de::Error::custom)
            }
        }
    };
}
pub(crate) use settings_group;

// ---------------------------------------------------------------------------------- the table

/// The metadata of every setting and group, in declaration order.
pub struct Table {
    pub settings: Vec<SettingMeta>,
    pub groups: Vec<GroupMeta>,
}

/// The table of the settings ([`crate::driver::CalcOptions`]), built once.
pub fn table() -> &'static Table {
    static T: std::sync::OnceLock<Table> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let (mut settings, mut groups) = (Vec::new(), Vec::new());
        <crate::driver::CalcOptions as Group>::meta(&mut settings, &mut groups);
        Table { settings, groups }
    })
}

/// The metadata of the setting at `path`.
pub fn meta(path: &str) -> Option<&'static SettingMeta> {
    table().settings.iter().find(|m| m.path == path)
}

/// One setting's property in the JSON Schema.
fn property(m: &SettingMeta) -> Value {
    let mut p = match (m.schema)() {
        Value::Object(o) => o,
        other => Map::from_iter([("anyOf".to_string(), other)]),
    };
    p.insert("description".into(), Value::String(m.text()));
    p.insert("default".into(), (m.default)());
    p.insert("x-tier".into(), json!(m.tier.name()));
    p.insert("x-concern".into(), json!(m.concern.name()));
    if let Some(u) = m.unit {
        p.insert("x-unit".into(), json!(u));
    }
    p.insert("x-applies-to".into(), json!(m.needs.calctypes()));
    if m.needs == Needs::Ir {
        p.insert("x-applies-when".into(), json!("freq.ir"));
    }
    Value::Object(p)
}

/// The JSON Schema (draft 2020-12) of the settings (the `keywords` of a qcdata input, the
/// settings of a JSON request, `resolved.settings`): for `calctype` (None: any), the settings
/// that can apply to it.
pub fn json_schema(calctype: Option<&str>) -> Value {
    let t = table();
    let applies = |m: &SettingMeta| calctype.is_none_or(|c| m.needs.calctypes().contains(&c));
    fn object(doc: Option<&str>) -> Map<String, Value> {
        let mut o = Map::new();
        o.insert("type".into(), json!("object"));
        if let Some(d) = doc {
            o.insert("description".into(), json!(paragraph(d)));
        }
        o.insert("additionalProperties".into(), json!(false));
        o.insert("properties".into(), json!({}));
        o
    }
    let mut groups: BTreeMap<&str, Map<String, Value>> = BTreeMap::new();
    let mut root = object(None);
    for m in t.settings.iter().filter(|m| applies(m)) {
        let g = m.group();
        let obj = if g.is_empty() {
            &mut root
        } else {
            groups.entry(g).or_insert_with(|| {
                let doc = t.groups.iter().find(|x| x.path == g).map(|x| x.doc);
                object(doc)
            })
        };
        obj["properties"]
            .as_object_mut()
            .expect("properties")
            .insert(m.key().to_string(), property(m));
    }
    for (path, obj) in groups {
        let gm = t.groups.iter().find(|x| x.path == path);
        let v = match gm.and_then(|g| g.shorthand) {
            Some(sh) => {
                let mut s = sh();
                if let Value::Object(o) = &mut s {
                    o.insert("description".into(), json!("shorthand"));
                }
                json!({"anyOf": [Value::Object(obj), s]})
            }
            None => Value::Object(obj),
        };
        root["properties"]
            .as_object_mut()
            .expect("properties")
            .insert(path.to_string(), v);
    }
    root.insert(
        "$schema".into(),
        json!("https://json-schema.org/draft/2020-12/schema"),
    );
    root.insert(
        "title".into(),
        json!(match calctype {
            Some(c) => format!("linus settings ({c})"),
            None => "linus settings".into(),
        }),
    );
    root.insert(
        "description".into(),
        json!("The settings of a linus calculation (docs/cli.md). Keys and enum values ignore case and `-` equals `_`; null means unset."),
    );
    Value::Object(root)
}

/// The JSON Schema of a result's run record (`resolved.record`): diagnostic, its keys may
/// change without a schema version change, so only its top-level parts are typed.
pub fn record_schema() -> Value {
    let obj = |d: &str| json!({"type": ["object", "null"], "description": d});
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "linus run record",
        "description": "What a linus calculation did and decided (`resolved.record`; docs/cli.md). Diagnostic: its keys may change between versions.",
        "type": "object",
        "properties": {
            "warm_start": {"type": "boolean", "description": "the SCF started from its lineage's history (a session's earlier orbitals), not from the input alone (scf.guess has the detail)"},
            "purpose": {"enum": ["energy", "gradient", "hessian"], "description": "what the SCF was converged for"},
            "dft": {"type": "boolean", "description": "Kohn-Sham DFT (false: Hartree-Fock)"},
            "scf": obj("the SCF as it ran: where it started (guess), dropped functions, two-cycle, plateau, escalation, probes"),
            "df": obj("the density-fitting storage of the SCF"),
            "memory": obj("the memory limit and the budgets derived from it (GiB)"),
            "response": obj("the response equations of a Hessian"),
            "solvent": obj("the implicit solvent as it ran"),
            "backend": obj("what ran each family of kernels, the device and its arithmetic"),
            "debug_env": {"type": "object", "additionalProperties": {"type": "string"}, "description": "the LINUS_* environment variables that were set"},
            "accuracy": {
                "type": "object",
                "description": "with accuracy targets (qcdata's shared record key): the targets, the thresholds they set, the estimated errors and whether every estimate is within its target",
                "properties": {
                    "target": {"type": "object", "additionalProperties": {"type": "number"}},
                    "thresholds": {"type": "object", "additionalProperties": {"type": "number"}},
                    "estimate": {"type": "object", "additionalProperties": {"type": "number"}},
                    "met": {"type": "boolean"}
                },
                "required": ["target", "thresholds", "estimate", "met"]
            }
        },
        "additionalProperties": true
    })
}

/// A value's type for people (`number`, `auto | number`, `loose | normal | ...`).
fn type_text(s: &Value) -> String {
    if let Some(e) = s.get("enum").and_then(Value::as_array) {
        return e
            .iter()
            .map(|x| x.as_str().map_or_else(|| x.to_string(), String::from))
            .collect::<Vec<_>>()
            .join(" | ");
    }
    if let Some(c) = s.get("const") {
        return c.as_str().map_or_else(|| c.to_string(), String::from);
    }
    if let Some(a) = s.get("anyOf").and_then(Value::as_array) {
        return a.iter().map(type_text).collect::<Vec<_>>().join(" | ");
    }
    if let Some(d) = s.get("x-type").and_then(Value::as_str) {
        return d.to_string();
    }
    match s.get("type") {
        Some(Value::String(t)) if t == "array" => match s.get("items") {
            Some(i) => format!("[{}]", type_text(i)),
            None => "list".into(),
        },
        Some(Value::String(t)) if t == "object" => "object".into(),
        Some(Value::String(t)) => t.clone(),
        Some(Value::Array(ts)) => ts
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" | "),
        _ => "any".into(),
    }
}

/// A default for people.
fn default_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "none".into(),
        Value::Array(a) if a.is_empty() => "[]".into(),
        Value::Object(o) if o.is_empty() => "{}".into(),
        v => v.to_string(),
    }
}

/// `linus --list-settings [GROUP] [--all]`: every setting of `group` (None: all; `top`: the
/// top-level ones), with its type, default and meaning; dev-tier settings only with `all`.
pub fn list_settings(group: Option<&str>, all: bool) -> Result<String, String> {
    let t = table();
    let want = group.map(norm_name);
    if let Some(g) = &want {
        if g != "top" && !t.groups.iter().any(|x| x.path == g) {
            let names: Vec<String> = t.groups.iter().map(|x| x.path.to_string()).collect();
            return Err(format!(
                "no settings group {g:?}.{} The groups: top, {}",
                did_you_mean(g, &names),
                names.join(", ")
            ));
        }
    }
    let mut out = String::new();
    let mut last = None;
    for m in &t.settings {
        let g = m.group();
        let show = match want.as_deref() {
            None => true,
            Some("top") => g.is_empty(),
            Some(w) => g == w,
        };
        if !show || (m.tier == Tier::Dev && !all) {
            continue;
        }
        if last != Some(g) {
            let head = if g.is_empty() { "top level" } else { g };
            let doc = t
                .groups
                .iter()
                .find(|x| x.path == g)
                .map(|x| format!(": {}", paragraph(x.doc)))
                .unwrap_or_default();
            out += &format!("{}{head}{doc}\n", if last.is_some() { "\n" } else { "" });
            last = Some(g);
        }
        let unit = m.unit.map(|u| format!(" [{u}]")).unwrap_or_default();
        let tier = match m.tier {
            Tier::Common => "",
            Tier::Expert => " (expert)",
            Tier::Dev => " (dev)",
        };
        out += &format!(
            "  {} = {}{unit}{tier}\n      {} Default: {}.\n",
            m.path,
            type_text(&(m.schema)()),
            m.text(),
            default_text(&(m.default)())
        );
    }
    if !all && t.settings.iter().any(|m| m.tier == Tier::Dev) {
        out += "\n(--all also lists the development settings)\n";
    }
    Ok(out)
}

/// The settings tables of docs/cli.md (one per group, dev-tier settings left out), as the text
/// between its `<!-- settings:begin -->` and `<!-- settings:end -->` markers.
pub fn docs_tables() -> String {
    let t = table();
    let mut out = String::new();
    let mut groups: Vec<&str> = vec![""];
    groups.extend(t.groups.iter().map(|g| g.path));
    for g in groups {
        let rows: Vec<&SettingMeta> = t
            .settings
            .iter()
            .filter(|m| m.group() == g && m.tier != Tier::Dev)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let gm = t.groups.iter().find(|x| x.path == g);
        out += &match gm {
            Some(gm) => format!("\n**`{g}`**: {}\n\n", paragraph(gm.doc)),
            None => "\n**Top level**\n\n".into(),
        };
        out += "| key | values | default | meaning |\n|---|---|---|---|\n";
        for m in rows {
            let unit = m.unit.map(|u| format!(" ({u})")).unwrap_or_default();
            let tier = if m.tier == Tier::Expert {
                " *(expert)*"
            } else {
                ""
            };
            out += &format!(
                "| `{}` | {}{unit} | `{}` | {}{tier} |\n",
                m.path,
                type_text(&(m.schema)()).replace('|', "\\|"),
                default_text(&(m.default)()).replace('|', "\\|"),
                m.text().replace('|', "\\|")
            );
        }
    }
    out
}

/// A value's type as a Python annotation.
fn py_type(s: &Value) -> String {
    if let Some(e) = s.get("enum").and_then(Value::as_array) {
        let names: Vec<String> = e.iter().map(|x| format!("{x}")).collect();
        return format!("Literal[{}]", names.join(", "));
    }
    if let Some(c) = s.get("const") {
        return format!("Literal[{c}]");
    }
    if let Some(a) = s.get("anyOf").and_then(Value::as_array) {
        return a.iter().map(py_type).collect::<Vec<_>>().join(" | ");
    }
    if let Some(d) = s.get("x-python").and_then(Value::as_str) {
        return d.to_string();
    }
    match s.get("type").and_then(Value::as_str) {
        Some("boolean") => "bool".into(),
        Some("integer") => "int".into(),
        Some("number") => "float".into(),
        Some("string") => "str".into(),
        Some("array") => match s.get("items") {
            Some(i) => format!("list[{}]", py_type(i)),
            None => "list[Any]".into(),
        },
        Some("object") => match s.get("additionalProperties") {
            Some(Value::Object(_)) => format!("dict[str, {}]", py_type(&s["additionalProperties"])),
            _ => "dict[str, Any]".into(),
        },
        _ => "Any".into(),
    }
}

/// The Python `TypedDict` stubs of the settings (linus-py's `linus/settings.py`): one class per
/// group and `Settings` for the whole tree, for editors and type checkers.
pub fn python_stubs() -> String {
    let t = table();
    let class = |g: &str| -> String {
        let base: String = g
            .split('_')
            .map(|w| {
                let mut c = w.chars();
                c.next()
                    .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                    .unwrap_or_default()
            })
            .collect();
        format!("{base}Settings")
    };
    let mut out = String::from(
        "\"\"\"The settings of a linus calculation as typed dictionaries (generated from the\n\
         library's settings table by `cargo test -p linus generated_files` with\n\
         LINUS_UPDATE_GENERATED=1; do not edit). Keys and values as in docs/cli.md;\n\
         `linus.settings_schema()` returns the JSON Schema.\"\"\"\n\n\
         from __future__ import annotations\n\n\
         from typing import Any, Literal, TypedDict\n\n\
         __all__ = [",
    );
    let mut names: Vec<String> = t.groups.iter().map(|g| class(g.path)).collect();
    names.push("Settings".into());
    out += &names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ");
    out += "]\n";
    let field = |m: &SettingMeta| -> String {
        format!(
            "    {}: {}\n    \"\"\"{}\"\"\"\n",
            m.key(),
            py_type(&(m.schema)()),
            m.text().replace('\\', "\\\\").replace("\"\"\"", "'''")
        )
    };
    for g in &t.groups {
        out += &format!(
            "\n\nclass {}(TypedDict, total=False):\n    \"\"\"{}\"\"\"\n\n",
            class(g.path),
            paragraph(g.doc).replace('\\', "\\\\")
        );
        for m in t.settings.iter().filter(|m| m.group() == g.path) {
            out += &field(m);
        }
    }
    out += "\n\nclass Settings(TypedDict, total=False):\n    \"\"\"Every setting of a calculation.\"\"\"\n\n";
    for m in t.settings.iter().filter(|m| m.group().is_empty()) {
        out += &field(m);
    }
    for g in t.groups.iter().filter(|g| !g.path.contains('.')) {
        let sh = g
            .shorthand
            .map(|s| format!(" | {}", py_type(&s())))
            .unwrap_or_default();
        out += &format!(
            "    {}: {}{sh}\n    \"\"\"{}\"\"\"\n",
            g.path,
            class(g.path),
            paragraph(g.doc).replace('\\', "\\\\")
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every setting has a doc comment, a value schema and a default that its own type reads
    /// back; paths are unique.
    #[test]
    fn the_table_is_complete() {
        let t = table();
        let mut seen = std::collections::BTreeSet::new();
        for m in &t.settings {
            assert!(seen.insert(m.path), "{} declared twice", m.path);
            assert!(!m.text().is_empty(), "{} has no doc comment", m.path);
            assert!((m.schema)().is_object(), "{}", m.path);
            // the default reads back (as a request would give it)
            let v = (m.default)();
            let o = crate::driver::CalcOptions::from_json(path_value(m.path, v.clone()));
            assert!(o.is_ok(), "{} = {v}: {:?}", m.path, o.err());
        }
        for g in &t.groups {
            assert!(
                !paragraph(g.doc).is_empty(),
                "group {} has no doc comment",
                g.path
            );
        }
    }

    /// `{"a": {"b": v}}` for `a.b`.
    fn path_value(path: &str, v: Value) -> Value {
        path.rsplit('.').fold(v, |acc, k| json!({ k: acc }))
    }

    /// The JSON Schema is a schema object with every setting, and the extensions qcdata asks for.
    #[test]
    fn the_json_schema() {
        let s = json_schema(None);
        assert_eq!(s["type"], "object");
        assert_eq!(s["$schema"], "https://json-schema.org/draft/2020-12/schema");
        for m in &table().settings {
            let p = match m.path.split_once('.') {
                Some((g, k)) => {
                    let g = &s["properties"][g];
                    let o = g.get("anyOf").map_or(g, |a| &a[0]);
                    o["properties"][k].clone()
                }
                None => s["properties"][m.path].clone(),
            };
            for x in ["description", "x-tier", "x-concern", "x-applies-to"] {
                assert!(p.get(x).is_some(), "{}: {x}", m.path);
            }
        }
        // a calctype's schema leaves out what cannot apply to it
        let e = json_schema(Some("energy"));
        for m in table().settings.iter().filter(|m| m.needs > Needs::Energy) {
            let (g, k) = m.path.split_once('.').unwrap_or(("", m.path));
            let gs = if g.is_empty() {
                &e
            } else {
                &e["properties"][g]
            };
            let o = gs.get("anyOf").map_or(gs, |a| &a[0]);
            assert!(
                o["properties"].get(k).is_none(),
                "{} in the energy schema",
                m.path
            );
        }
    }

    /// Unknown keys are `unknown_key` errors with a suggestion, at any level.
    #[test]
    fn unknown_keys_suggest() {
        let e =
            crate::driver::CalcOptions::from_json(json!({"scf": {"max_iters": 3}})).unwrap_err();
        let LinusError::Setting {
            code,
            path,
            message,
        } = &e
        else {
            panic!("{e}")
        };
        assert_eq!((*code, path.as_str()), ("unknown_key", "scf.max_iters"));
        assert!(message.contains("'scf.max_iter'"), "{message}");
        let e = crate::driver::CalcOptions::from_json(json!({"e_tol": 1e-9})).unwrap_err();
        assert!(e.to_string().contains("scf.e_tol"), "{e}");
        // keys ignore case, and - equals _
        let o = crate::driver::CalcOptions::from_json(json!({"SCF": {"Max-Iter": 7}})).unwrap();
        assert_eq!(o.scf.max_iter(), 7);
    }

    #[test]
    fn listing() {
        let all = list_settings(None, true).unwrap();
        let common = list_settings(None, false).unwrap();
        for m in &table().settings {
            assert!(all.contains(m.path), "{}", m.path);
            assert_eq!(
                common.contains(&format!("  {} =", m.path)),
                m.tier != Tier::Dev
            );
        }
        assert!(list_settings(Some("scf"), false)
            .unwrap()
            .contains("scf.max_iter"));
        assert!(list_settings(Some("sfc"), false)
            .unwrap_err()
            .contains("'scf'"));
    }

    /// The generated files in the repository are current: docs/cli.md's settings tables and
    /// linus-py's settings stubs (`LINUS_UPDATE_GENERATED=1` rewrites them).
    #[test]
    fn generated_files() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let update = std::env::var_os("LINUS_UPDATE_GENERATED").is_some();
        // docs/cli.md between its markers
        let cli = root.join("docs/cli.md");
        let text = std::fs::read_to_string(&cli).unwrap();
        let (b, e) = ("<!-- settings:begin -->", "<!-- settings:end -->");
        let i = text.find(b).expect("docs/cli.md: settings:begin marker") + b.len();
        let j = text.find(e).expect("docs/cli.md: settings:end marker");
        let want = format!("\n{}\n", docs_tables().trim());
        if update {
            std::fs::write(&cli, format!("{}{want}{}", &text[..i], &text[j..])).unwrap();
        } else {
            assert!(
                text[i..j] == want,
                "docs/cli.md's settings tables are stale: LINUS_UPDATE_GENERATED=1 cargo test -p linus generated_files"
            );
        }
        let py = root.join("crates/linus-py/python/linus/settings.py");
        let want = python_stubs();
        if update {
            std::fs::write(&py, &want).unwrap();
        } else {
            assert!(
                std::fs::read_to_string(&py).ok().as_deref() == Some(want.as_str()),
                "linus-py's settings.py is stale: LINUS_UPDATE_GENERATED=1 cargo test -p linus generated_files"
            );
        }
    }
}
