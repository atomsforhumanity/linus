#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the parameter tables of linus-disp from the reference Fortran sources.

Usage::

    python3 -I crates/linus-disp/tools/gen_data.py --src DIR [--fetch]

DIR holds the extracted release tarballs listed in SOURCES (``--fetch`` downloads them
into DIR and checks their SHA-256). The script writes ``crates/linus-disp/src/data/*.rs``.

Every table is parsed from the Fortran, never copied by hand. Real literals are emitted
with exactly the decimal digits of the Fortran source (only the ``_wp`` kind suffix is
dropped), so rustc rounds them to the same IEEE double as gfortran. Whole-array
operations of the Fortran declarations (``aatoau *``, ``4/3 *``, ``sqrt(0.5*(r4r2*sqrt(Z)))``)
are *not* applied here: the Rust code applies them element by element with the same
single IEEE operations, which reproduces gfortran's correctly rounded constant folding.

Provenance of every table (path inside the tarball):

=====================  ====================================================================
D3 reference C6        simple-dftd3-1.7.0/src/dftd3/reference.f90 ``copy_c6`` (c6ab_view)
D3 reference CN        simple-dftd3-1.7.0/src/dftd3/reference.f90 ``reference_cn``
D3 number of refs      simple-dftd3-1.7.0/src/dftd3/reference.f90 ``number_of_references``
D3 <r4>/<r2>           simple-dftd3-1.7.0/src/dftd3/data/r4r2.f90 ``r4_over_r2``
D3 vdW radii (ATM r0)  simple-dftd3-1.7.0/src/dftd3/data/vdwrad.f90 ``vdwrad`` (Angstrom)
covalent radii         mctc-lib-0.7.0/src/mctc/data/covrad.f90 ``covalent_rad_2009`` (Angstrom)
CODATA 2018, pi        mctc-lib-0.7.0/src/mctc/io/codata2018.f90, io/constants.f90
D3(BJ) parameters      simple-dftd3-1.7.0/src/dftd3/param.f90 ``get_method_id``,
                       ``get_rational_damping``; DOIs from src/dftd3/citation.f90
D4 parameters          dftd4-4.3.1/src/dftd4/param.f90 ``get_functional_id``,
                       ``get_d4eeq_bjatm_parameter``, ``get_d4eeq_bj_parameter``
EEQ (2019) parameters  multicharge-0.6.0/src/multicharge/param/eeq2019.f90
=====================  ====================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "data"

SOURCES = {
    "s-dftd3": (
        "simple-dftd3-1.7.0",
        "https://github.com/dftd3/simple-dftd3/archive/refs/tags/v1.7.0.tar.gz",
        "4d733eb6dbe8266fcfad67cef31de932a079f7c911f24f4b6efc594cd3d28803",
    ),
    "dftd4": (
        "dftd4-4.3.1",
        "https://github.com/dftd4/dftd4/archive/refs/tags/v4.3.1.tar.gz",
        "01651dd3df06f0875ed4e710ee82f7dcb621edb262d6d22068dea818b915f5da",
    ),
    "multicharge": (
        "multicharge-0.6.0",
        "https://github.com/grimme-lab/multicharge/archive/refs/tags/v0.6.0.tar.gz",
        "129ab0325663e9155b5d91c4364199cec5fc0cbecdfe047cc24a37aa2f7a50e0",
    ),
    "mctc-lib": (
        "mctc-lib-0.7.0",
        "https://github.com/grimme-lab/mctc-lib/archive/refs/tags/v0.7.0.tar.gz",
        "83ddeea24082d25c62db375ad8bd7f53e49e6d81dc834613b1da77b1ca8a8813",
    ),
}

REAL_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?(_wp)?$")
NUM_TOKEN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?(?:_wp)?")


def rust_real(tok: str) -> str:
    """Fortran real literal -> Rust f64 literal with the same decimal digits."""
    tok = tok.strip()
    if not REAL_RE.match(tok):
        raise ValueError(f"not a real literal: {tok!r}")
    orig = tok.replace("_wp", "").replace("d", "e").replace("D", "e")
    tok = orig
    sign = ""
    if tok[0] in "+-":
        sign, tok = ("-" if tok[0] == "-" else ""), tok[1:]
    mant, _, exp = tok.partition("e")
    if mant.startswith("."):
        mant = "0" + mant
    if "." not in mant:
        mant += ".0"
    elif mant.endswith("."):
        mant += "0"
    out = sign + mant + (("e" + exp) if exp else "")
    assert float(out) == float(orig), (orig, out)
    return out


def strip_comments(text: str) -> str:
    """Remove Fortran ``!`` comments (no string literals in the parsed regions contain !)."""
    return "\n".join(re.sub(r"!.*", "", line) for line in text.split("\n"))


def array_body(src: str, name: str) -> str:
    """Text between the outer [ ] of ``name``'s parameter array constructor."""
    m = re.search(rf"::\s*{re.escape(name)}\s*(\([^)]*\))?\s*=", src)
    if m is None:
        raise KeyError(name)
    rest = strip_comments(src[m.end():])
    start = rest.index("[")
    depth = 0
    for k in range(start, len(rest)):
        if rest[k] == "[":
            depth += 1
        elif rest[k] == "]":
            depth -= 1
            if depth == 0:
                return rest[start + 1:k].replace("&", " ")
    raise ValueError(name)


def real_array(src: str, name: str, n: int | None = None) -> list[str]:
    body = array_body(src, name)
    toks = [t for t in re.split(r"[,\s]+", body) if t]
    vals = [rust_real(t) for t in toks]
    if n is not None and len(vals) != n:
        raise ValueError(f"{name}: {len(vals)} values, expected {n}")
    return vals


def int_array(src: str, name: str, n: int) -> list[int]:
    body = array_body(src, name)
    vals = [int(t) for t in re.split(r"[,\s]+", body) if t]
    if len(vals) != n:
        raise ValueError(f"{name}: {len(vals)} values, expected {n}")
    return vals


def scalar_param(src: str, name: str) -> str:
    # Fortran names are case-insensitive (codata2018.f90 spells it `Planck_constant`)
    m = re.search(rf"::\s*{re.escape(name)}\s*=\s*({NUM_TOKEN.pattern})", src, re.I)
    if m is None:
        raise KeyError(name)
    return rust_real(m.group(1))


def fmt_list(vals, per_line: int, indent: str = "    ") -> str:
    lines = []
    for k in range(0, len(vals), per_line):
        lines.append(indent + ", ".join(str(v) for v in vals[k:k + per_line]) + ",")
    return "\n".join(lines)


def header(sources: list[str], what: str) -> str:
    srcs = "\n".join(
        f"//   {SOURCES[s][0]}  ({SOURCES[s][1]}, sha256 {SOURCES[s][2][:16]}...)" for s in sources
    )
    return (
        "// SPDX-License-Identifier: LGPL-3.0-or-later\n"
        "// @generated by crates/linus-disp/tools/gen_data.py -- do not edit by hand.\n"
        f"// {what}\n"
        "// Sources:\n"
        f"{srcs}\n"
        "// Literals carry exactly the decimal digits of the Fortran source.\n"
        "#![allow(clippy::excessive_precision, clippy::unreadable_literal, clippy::approx_constant)]\n\n"
    )


# ----------------------------------------------------------------------------- D3 data
def gen_d3(root: Path) -> str:
    d3 = root / SOURCES["s-dftd3"][0] / "src" / "dftd3"
    ref = (d3 / "reference.f90").read_text()
    max_elem, max_ref = 103, 7
    nref = int_array(ref, "number_of_references", max_elem)
    body = array_body(ref, "reference_cn")
    toks = [t for t in re.split(r"[,\s]+", body.split("]")[0]) if t]
    refcn = [rust_real(t) for t in toks]
    assert len(refcn) == max_ref * max_elem, len(refcn)

    # reference C6: c6ab_view(a:b) = [ ... ] blocks inside copy_c6, in Fortran storage order
    sub = ref[ref.index("pure subroutine copy_c6"):ref.index("end subroutine copy_c6")]
    sub = strip_comments(sub)
    flat: list[str] = []
    for m in re.finditer(r"c6ab_view\((\d+):(\d+)\)\s*=\s*\[(.*?)\]", sub, re.S):
        lo, hi = int(m.group(1)), int(m.group(2))
        vals = [rust_real(t) for t in re.split(r"[,\s&]+", m.group(3)) if t]
        assert lo == len(flat) + 1 and hi - lo + 1 == len(vals), (lo, hi, len(vals))
        flat.extend(vals)
    npair = max_elem * (max_elem + 1) // 2
    assert len(flat) == max_ref * max_ref * npair, len(flat)

    # compact: for each element pair (i >= j, 1-based pair index ic = j + i(i-1)/2) keep the
    # nref(i) x nref(j) block; Fortran storage reference_c6(a, b, ic), a = reference of the
    # heavier element i (fastest), b = reference of the lighter element j.
    compact: list[str] = []
    offsets = [0]
    # (element, 0-based reference) of nonzero data beyond number_of_references, which
    # s-dftd3 never reads
    unused: set[tuple[int, int]] = set()
    for i in range(1, max_elem + 1):
        for j in range(1, i + 1):
            ic = j + i * (i - 1) // 2
            base = (ic - 1) * max_ref * max_ref
            for b in range(max_ref):
                for a in range(max_ref):
                    v = flat[base + a + max_ref * b]
                    if a < nref[i - 1] and b < nref[j - 1]:
                        compact.append(v)
                    elif float(v) != 0.0:
                        if a >= nref[i - 1]:
                            unused.add((i, a))
                        if b >= nref[j - 1]:
                            unused.add((j, b))
            offsets.append(len(compact))
    # s-dftd3 1.7.0 lists 7 reference CNs and C6 blocks for Cm but number_of_references(96) = 6,
    # so its 7th reference (CN 5.6866) is dead data; nothing else may be dropped.
    assert unused == {(96, 6)}, unused
    # offsets were appended in (i, j) loop order, which is ic order
    assert len(offsets) == npair + 1

    r4r2 = real_array((d3 / "data" / "r4r2.f90").read_text(), "r4_over_r2", 118)
    vdw = (d3 / "data" / "vdwrad.f90").read_text()
    m = re.search(r"::\s*vdwrad\(max_elem\*\(1\+max_elem\)/2\)\s*=\s*aatoau\s*\*\s*\[", vdw)
    assert m, "vdwrad declaration changed"
    vdw_vals = [rust_real(t) for t in re.split(r"[,\s&]+", strip_comments(vdw[m.end():]).split("]")[0]) if t]
    assert len(vdw_vals) == npair, len(vdw_vals)

    out = header(["s-dftd3"], "DFT-D3 reference data (s-dftd3 1.7.0).")
    out += f"pub const MAX_ELEM: usize = {max_elem};\npub const MAX_REF: usize = {max_ref};\n\n"
    out += "/// Number of reference systems per element (`number_of_references`).\n"
    out += f"pub static NUMBER_OF_REFERENCES: [u8; {max_elem}] = [\n{fmt_list(nref, 20)}\n];\n\n"
    out += "/// Reference coordination numbers (`reference_cn`), -1 pads unused references.\n"
    rows = ["    [" + ", ".join(refcn[k * max_ref:(k + 1) * max_ref]) + "]," for k in range(max_elem)]
    out += f"pub static REFERENCE_CN: [[f64; {max_ref}]; {max_elem}] = [\n" + "\n".join(rows) + "\n];\n\n"
    out += (
        "/// Offsets into [`REFERENCE_C6`] per element pair, pair index `j + i(i-1)/2 - 1`\n"
        "/// (1-based atomic numbers `i >= j`).\n"
    )
    out += f"pub static REFERENCE_C6_OFFSET: [u32; {npair + 1}] = [\n{fmt_list(offsets, 16)}\n];\n\n"
    out += (
        "/// Reference C6 coefficients (Eh bohr^6) of `copy_c6`, compacted: for every element pair\n"
        "/// `i >= j` the `nref(i) x nref(j)` block, value of (reference `a` of `i`, reference `b`\n"
        "/// of `j`) at `offset + b * nref(i) + a`. Dropped: zeros of the padding and the C6 of the\n"
        "/// 7th Cm reference, which s-dftd3 never reads (`number_of_references(96) = 6`).\n"
    )
    out += f"pub static REFERENCE_C6: [f64; {len(compact)}] = [\n{fmt_list(compact, 8)}\n];\n\n"
    out += "/// `<r4>/<r2>` expectation values (`r4_over_r2`, data/r4r2.f90).\n"
    out += f"pub static R4_OVER_R2: [f64; 118] = [\n{fmt_list(r4r2, 8)}\n];\n\n"
    out += (
        "/// Pair van-der-Waals radii in Angstrom (`vdwrad`, data/vdwrad.f90), pair index\n"
        "/// `j + i(i-1)/2 - 1` for `i >= j`; s-dftd3 multiplies each by `aatoau`.\n"
    )
    out += f"pub static VDW_RAD_AA: [f64; {npair}] = [\n{fmt_list(vdw_vals, 10)}\n];\n"
    return out


# ------------------------------------------------------------------------- mctc data
def gen_common(root: Path) -> str:
    mctc = root / SOURCES["mctc-lib"][0] / "src" / "mctc"
    codata = (mctc / "io" / "codata2018.f90").read_text()
    consts = (mctc / "io" / "constants.f90").read_text()
    cov = real_array((mctc / "data" / "covrad.f90").read_text(), "covalent_rad_2009", 118)
    out = header(["mctc-lib"], "mctc-lib 0.7.0 constants and covalent radii.")
    for rust, name, doc in [
        ("PLANCK", "planck_constant", "CODATA 2018 Planck constant (J s)"),
        ("ELECTRON_MASS", "electron_mass", "CODATA 2018 electron mass (kg)"),
        ("SPEED_OF_LIGHT", "speed_of_light_in_vacuum", "CODATA 2018 speed of light (m/s)"),
        ("FINE_STRUCTURE", "fine_structure_constant", "CODATA 2018 fine-structure constant"),
    ]:
        out += f"/// {doc}, mctc-lib io/codata2018.f90 `{name}`\n"
        out += f"pub const {rust}: f64 = {scalar_param(codata, name)};\n"
    out += "/// pi as written in mctc-lib io/constants.f90\n"
    out += f"pub const PI: f64 = {scalar_param(consts, 'pi')};\n\n"
    out += (
        "/// Covalent radii in Angstrom (Pyykko and Atsumi 2009, metals -10 %),\n"
        "/// mctc-lib data/covrad.f90 `covalent_rad_2009` (before `aatoau *` and `4/3 *`).\n"
    )
    out += f"pub static COVALENT_RAD_2009: [f64; 118] = [\n{fmt_list(cov, 10)}\n];\n"
    return out


# --------------------------------------------------------------------------- EEQ data
def gen_eeq(root: Path) -> str:
    p = root / SOURCES["multicharge"][0] / "src" / "multicharge" / "param" / "eeq2019.f90"
    src = p.read_text()
    out = header(["multicharge"], "EEQ 2019 parameters (Caldeweyher et al., JCP 150, 154122 (2019)).")
    for rust, name, doc in [
        ("EEQ_CHI", "eeq_chi", "electronegativity"),
        ("EEQ_ETA", "eeq_eta", "chemical hardness"),
        ("EEQ_KCNCHI", "eeq_kcnchi", "CN scaling of the electronegativity"),
        ("EEQ_RAD", "eeq_rad", "charge width"),
    ]:
        vals = real_array(src, name, 103)
        out += f"/// EEQ {doc} (`{name}`, param/eeq2019.f90)\n"
        out += f"pub static {rust}: [f64; 103] = [\n{fmt_list(vals, 8)}\n];\n\n"
    return out


# ------------------------------------------------------------------- parameter tables
def normalize(name: str) -> str:
    """linus-style key: lowercase, no '-', '_', '/', ' ' (ASCII lowercase only, like Fortran)."""
    s = "".join(c.lower() if c.isascii() else c for c in name)
    return re.sub(r"[-_/ ]", "", s)


def parse_kwargs(text: str) -> dict[str, str]:
    return {k: rust_real(v) for k, v in re.findall(r"(\w+)\s*=\s*(" + NUM_TOKEN.pattern + ")", text)}


def gen_params(root: Path) -> str:
    # ---- s-dftd3 rational damping
    d3 = root / SOURCES["s-dftd3"][0] / "src" / "dftd3"
    param = (d3 / "param.f90").read_text()
    cit = (d3 / "citation.f90").read_text()
    dois = dict(re.findall(r"(doi_\w+)\s*=\s*\"([^\"]+)\"", cit))
    f = param[param.index("function get_method_id"):param.index("end function get_method_id")]
    d3_alias: dict[str, list[str]] = {}
    for m in re.finditer(r"case\(([^)]*)\);\s*id\s*=\s*(p_\w+)", f):
        names = re.findall(r"\"([^\"]*)\"", m.group(1))
        d3_alias.setdefault(m.group(2), []).extend(names)
    sub = param[param.index("subroutine get_rational_damping("):param.index("end subroutine get_rational_damping")]
    defaults = {"s6": "1.0", "s8": "1.0", "s9": "0.0", "a1": "0.4", "a2": "5.0", "alp": "14.0"}
    td = param[param.index("type :: d3_param"):param.index("end type d3_param")]
    for k, v in re.findall(r"real\(wp\)\s*::\s*(\w+)\s*=\s*(" + NUM_TOKEN.pattern + ")", td):
        defaults[k] = rust_real(v)
    d3_entries = []
    for blk in re.split(r"\n\s*case\(", sub)[1:]:
        ids_txt, _, rest = blk.partition(")")
        ids = [t.strip() for t in ids_txt.split(",")]
        pm = re.search(r"param\s*=\s*d3_param\((.*?)\)\s*\n", rest, re.S)
        if pm is None:
            continue
        kw = parse_kwargs(pm.group(1).replace("&", " "))
        doi_m = re.search(r"doi\s*=\s*(doi_\w+)", rest)
        doi = dois[doi_m.group(1)] if doi_m else ""
        p = dict(defaults)
        p.update(kw)
        names = []
        for i in ids:
            names.extend(d3_alias.get(i, []))
        if not names:
            continue
        d3_entries.append((names, p, doi, ids[0]))

    # ---- dftd4 rational damping (D4)
    d4 = root / SOURCES["dftd4"][0] / "src" / "dftd4"
    param4 = (d4 / "param.f90").read_text()
    g = param4[param4.index("pure function get_functional_id"):param4.index("end function get_functional_id")]
    g = strip_comments(g).replace("&\n", " ").replace("&", " ")
    d4_alias: dict[str, list[str]] = {}
    for m in re.finditer(r"case\s*\(([^)]*)\)\s*num\s*=\s*(p_\w+)", g):
        d4_alias.setdefault(m.group(2), []).extend(re.findall(r"\"([^\"]*)\"", m.group(1)))

    def d4_table(subname: str, s9_default: str):
        s = param4[param4.index(f"subroutine {subname}("):param4.index(f"end subroutine {subname}")]
        s = s[:s.index("contains")]
        res = []
        for blk in re.split(r"\n\s*case\s*\(", s)[1:]:
            pid, _, rest = blk.partition(")")
            pid = pid.strip()
            if pid == "default" or not pid.startswith("p_"):
                continue
            # fit tag or DOI in the first "! (...)" comment of the block (on the case line or
            # after `dftd_param ( &`)
            note = re.search(r"!\s*\(([^)]*)\)", "case(" + blk)
            clean = strip_comments(rest).replace("&", " ")
            pm = re.search(r"param\s*=\s*dftd_param\s*\((.*?)\)", clean, re.S)
            if pm is None:
                continue
            kw = parse_kwargs(pm.group(1))
            p = {"s6": "1.0", "s9": s9_default, "alp": "16.0"}
            p.update(kw)
            res.append((pid, p, note.group(1) if note else ""))
        return res

    d4_entries = []
    seen = set()
    for pid, p, note in d4_table("get_d4eeq_bjatm_parameter", "1.0") + d4_table("get_d4eeq_bj_parameter", "0.0"):
        if pid in seen:
            continue  # bj-only entries never shadow bjatm ones (dftd4: get_rational_damping_id)
        seen.add(pid)
        names = d4_alias.get(pid, [])
        if names:
            d4_entries.append((names, p, note, pid))

    out = header(["s-dftd3", "dftd4"], "Functional parameter tables of DFT-D3(BJ) and DFT-D4.")
    out += (
        "/// One rational-damping (Becke-Johnson) parameter set of the reference program.\n"
        "#[derive(Clone, Copy, Debug, PartialEq)]\n"
        "pub struct BjEntry {\n"
        "    /// upstream enumerator (s-dftd3 `p_*_df`, dftd4 `p_*`)\n"
        "    pub id: &'static str,\n"
        "    /// names the upstream lookup accepts (verbatim)\n"
        "    pub names: &'static [&'static str],\n"
        "    pub s6: f64,\n    pub s8: f64,\n    pub a1: f64,\n    pub a2: f64,\n"
        "    /// three-body scaling the upstream library uses when the caller gives none\n"
        "    pub s9: f64,\n"
        "    /// exponent of the ATM zero damping as passed to the kernel minus nothing:\n"
        "    /// s-dftd3 stores 14 and adds 2 in the ATM kernel, dftd4 stores 16\n"
        "    pub alp: f64,\n"
        "    /// literature reference (DOI) or upstream fit tag\n"
        "    pub reference: &'static str,\n"
        "}\n\n"
    )

    def emit(var: str, entries, doc: str):
        nonlocal out
        out += f"/// {doc}\npub static {var}: &[BjEntry] = &[\n"
        for names, p, ref, pid in entries:
            nm = ", ".join('"' + n + '"' for n in names)
            out += (
                f"    BjEntry {{ id: \"{pid}\", names: &[{nm}], s6: {p['s6']}, s8: {p['s8']}, "
                f"a1: {p['a1']}, a2: {p['a2']}, s9: {p['s9']}, alp: {p['alp']}, "
                f"reference: \"{ref}\" }},\n"
            )
        out += "];\n\n"

    emit(
        "D3BJ_TABLE",
        d3_entries,
        "s-dftd3 1.7.0 `get_rational_damping` (src/dftd3/param.f90) with the names of\n"
        "/// `get_method_id`; s9 = 0 is the library default (the s-dftd3 CLI adds ATM only with `--atm`).",
    )
    emit(
        "D4_TABLE",
        d4_entries,
        "dftd4 4.3.1 `get_d4eeq_bjatm_parameter` (s9 = 1) and, for functionals without one,\n"
        "/// `get_d4eeq_bj_parameter` (s9 = 0), with the names of `get_functional_id`.",
    )
    # collision check of normalized names
    for var, entries in (("D3", d3_entries), ("D4", d4_entries)):
        keys: dict[str, str] = {}
        for names, _, _, pid in entries:
            for n in names:
                k = normalize(n)
                if k in keys and keys[k] != pid:
                    print(f"warning: {var} normalized name {k!r} maps to {keys[k]} and {pid}", file=sys.stderr)
                keys.setdefault(k, pid)
    print(f"D3(BJ): {len(d3_entries)} parameter sets; D4: {len(d4_entries)}", file=sys.stderr)
    return out


def fetch(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for key, (dirname, url, sha) in SOURCES.items():
        tgz = dst / f"{dirname}.tar.gz"
        if not tgz.exists():
            urllib.request.urlretrieve(url, tgz)
        h = hashlib.sha256(tgz.read_bytes()).hexdigest()
        if h != sha:
            raise SystemExit(f"{tgz}: sha256 {h} != {sha}")
        if not (dst / dirname).exists():
            with tarfile.open(tgz) as tf:
                tf.extractall(dst, filter="data")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="directory with the extracted tarballs")
    ap.add_argument("--fetch", action="store_true", help="download and verify the tarballs first")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.src)
    for key, (dirname, _, _) in SOURCES.items():
        if not (args.src / dirname).is_dir():
            raise SystemExit(f"missing {args.src / dirname} (run with --fetch)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d3.rs").write_text(gen_d3(args.src))
    (OUT / "common.rs").write_text(gen_common(args.src))
    (OUT / "eeq.rs").write_text(gen_eeq(args.src))
    (OUT / "params.rs").write_text(gen_params(args.src))
    for f in ("d3.rs", "common.rs", "eeq.rs", "params.rs"):
        print(f"wrote {OUT / f} ({(OUT / f).stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
