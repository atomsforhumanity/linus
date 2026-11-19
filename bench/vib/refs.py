"""Harmonic analysis and thermochemistry of a given Hessian by PySCF, Psi4 and ORCA.

Each function takes the same inputs (atomic numbers, positions in bohr, Hessian (3N x 3N,
Eh/bohr^2), masses in u, energy, multiplicity, optional dipole derivatives (3N x 3)) and
returns a dict in linus's units: `frequencies` (cm^-1, imaginary negative, vibrations only,
ascending), `modes` (unit Cartesian displacements, nvib x 3N), `reduced_masses` (u),
`force_constants` (mdyn/Å), `ir` (km/mol), `symmetry_number`, `rotational_constants`
(cm^-1), and `thermo` with `zpe`, `E`, `H`, `S`, `G` per term (`elec`, `trans`, `rot`, `vib`)
and totals (Eh, Eh/K).

Run with the reference env (.refenv/bin/python), which has PySCF 2.15 and Psi4 1.11; ORCA 6.0.0
from LINUS_ORCA_DIR (default ~/opt/orca-6.0.0; bench/bench_paths.py).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
    " Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
    " Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_paths import ORCA, REFENV  # noqa: E402  (env overrides LINUS_ORCA_DIR, LINUS_REFENV: bench/bench_paths.py)


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# -- PySCF ------------------------------------------------------------------------------------

def pyscf_analysis(z, pos, hess, masses, energy, mult, T=298.15, P=101325.0, apt=None, qrrho=False,
                   qrrho_enthalpy=False):
    from pyscf import gto
    from pyscf.hessian import thermo

    n = len(z)
    nelec = int(sum(z))
    spin = mult - 1
    charge = 0 if (nelec - spin) % 2 == 0 else 1
    # (the basis only has to exist for every element: the analysis uses masses and positions)
    mol = gto.M(atom=[(SYMBOLS[a], p) for a, p in zip(z, pos)], unit="Bohr",
                basis="sto-3g" if max(z) <= 36 else "def2-svp", charge=charge,
                spin=spin, verbose=0)
    mol.nucprop = {i + 1: {"mass": float(m)} for i, m in enumerate(masses)}
    h4 = np.asarray(hess).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    res = thermo.harmonic_analysis(mol, h4, mass=np.asarray(masses))
    fw = res["freq_wavenumber"]
    freqs = np.where(np.abs(fw.imag) > np.abs(fw.real), -np.abs(fw.imag), fw.real).astype(float)
    modes = _unit_rows(np.asarray(res["norm_mode"]).real.reshape(len(freqs), -1))
    out = {"frequencies": freqs, "modes": modes, "reduced_masses": np.asarray(res["reduced_mass"]).real,
           "force_constants": np.asarray(res["force_const_dyne"]).real}

    class Model:
        pass

    model = Model()
    model.mol, model.e_tot = mol, energy
    th = thermo.thermo(model, res["freq_au"], T, P, qrrho=qrrho, qrrho_enthalpy=qrrho_enthalpy)
    out["symmetry_number"] = th["sym_number"][0]
    out["rotational_constants"] = np.asarray(th["rot_const"][0]) * 1e9 / 2.99792458e10
    t = {"zpe": th["ZPE"][0]}
    for f in ("E", "H", "S", "G"):
        for k, kk in (("elec", "elec"), ("trans", "trans"), ("rot", "rot"), ("vib", "vib"), ("tot", "tot")):
            t[f"{f}_{kk}"] = th[f"{f}_{k}"][0]
    out["thermo"] = t
    return out


# -- Psi4 -------------------------------------------------------------------------------------

def _import_psi4():
    """Import Psi4 so that its exit handler (core.finalize) writes timer.dat, which names the
    host, into a temporary directory instead of the caller's working directory."""
    if "psi4" in sys.modules:
        return sys.modules["psi4"]
    import atexit
    import shutil

    td = tempfile.mkdtemp(prefix="psi4_")
    atexit.register(shutil.rmtree, td, True)  # registered before Psi4's handlers: runs after them
    import psi4

    atexit.register(os.chdir, td)  # registered after Psi4's handlers: runs before them
    return psi4


def psi4_analysis(z, pos, hess, masses, energy, mult, T=298.15, P=101325.0, apt=None):
    psi4 = _import_psi4()
    from psi4.driver.qcdb import vib as qvib

    psi4.core.be_quiet()
    n = len(z)
    # Psi4 1.11 computes rotational constants about the coordinate origin, not the centre of
    # mass (Molecule::rotational_constants); with no_com an off-centre molecule gets wrong
    # constants and rotational entropy. Centre it (the Hessian does not change).
    pos = np.asarray(pos) - np.average(pos, axis=0, weights=masses)
    nelec = int(sum(z))
    charge = 0 if (nelec - (mult - 1)) % 2 == 0 else 1
    lines = ["units bohr", "no_com", "no_reorient", "symmetry c1", f"{charge} {mult}"]
    lines += [f"{SYMBOLS[a]}@{m:.12f} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for a, m, p in zip(z, masses, pos)]
    mol = psi4.geometry("\n".join(lines))
    mol.update_geometry()
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "STO-3G" if max(z) <= 36 else "DEF2-SVP", quiet=True)
    m = np.array([mol.mass(i) for i in range(n)])
    geom = np.asarray(mol.geometry())
    dipder = None if apt is None else np.asarray(apt).T
    vi, _ = qvib.harmonic_analysis(np.asarray(hess), geom, m, basis, mol.irrep_labels(), dipder=dipder,
                                   project_trans=True, project_rot=True)
    keep = [i for i, t in enumerate(vi["TRV"].data) if t == "V"]
    om = vi["omega"].data[keep]
    freqs = np.where(np.abs(om.imag) > np.abs(om.real), -np.abs(om.imag), om.real).astype(float)
    modes = _unit_rows(np.asarray(vi["x"].data).T[keep])
    out = {"frequencies": freqs, "modes": modes, "reduced_masses": vi["mu"].data[keep],
           "force_constants": vi["k"].data[keep], "masses_used": m}
    if apt is not None:
        out["ir"] = vi["IR_intensity"].data[keep]
    sigma = mol.rotational_symmetry_number()
    rot = np.asarray(mol.rotational_constants())
    ti, _ = qvib.thermo(vi, T=T, P=P, multiplicity=mult, molecular_mass=float(m.sum()), E0=energy, sigma=sigma,
                        rot_const=rot, rotor_type=mol.rotor_type())
    out["symmetry_number"] = sigma
    out["point_group"] = mol.get_full_point_group()
    out["rotational_constants"] = rot
    t = {"zpe": ti["ZPE_vib"].data}
    for f in ("E", "H", "G"):
        for k in ("elec", "trans", "rot", "vib"):
            t[f"{f}_{k}"] = ti[f"{f}_{k}"].data + (energy if k == "elec" else 0.0)
        t[f"{f}_tot"] = ti[f"{f}_tot"].data
    for k in ("elec", "trans", "rot", "vib", "tot"):
        t[f"S_{k}"] = ti[f"S_{k}"].data * 1e-3  # mEh/K
    out["thermo"] = t
    return out


# -- ORCA -------------------------------------------------------------------------------------

def write_orca_hess(path, z, pos, hess, masses, energy, mult, apt=None):
    """An ORCA .hess file (input for orca_vib and PrintThermoChem)."""
    n3 = 3 * len(z)
    h = np.asarray(hess)
    out = ["", "$orca_hessian_file", "", "$act_atom", "  0", "", "$act_coord", "  0", "", "$act_energy",
           f"  {energy:.12f}", "", "$multiplicity", f"  {mult}", "", "$hessian", f"{n3}"]
    for c0 in range(0, n3, 5):
        cols = range(c0, min(c0 + 5, n3))
        out.append("      " + "".join(f"{c:19d}" for c in cols))
        for r in range(n3):
            out.append(f"{r:5d}  " + "".join(f"{h[r, c]:19.10E}" for c in cols))
    out += ["", "$atoms", f"{len(z)}"]
    out += [f" {SYMBOLS[a]:2s} {m:16.10f} {p[0]:20.12f} {p[1]:20.12f} {p[2]:20.12f}" for a, m, p in zip(z, masses, pos)]
    out += ["", "$actual_temperature", "  0.000000", "", "$frequency_scale_factor", "  1.000000", ""]
    if apt is not None:
        out += ["$dipole_derivatives", f"{n3}"]
        out += [" ".join(f"{x:20.12E}" for x in row) for row in np.asarray(apt)]
        out.append("")
    out += ["$end", ""]
    Path(path).write_text("\n".join(out))


def _orca_env():
    env = dict(os.environ)
    env["PATH"] = f"{ORCA}:{REFENV}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{ORCA}/lib:{REFENV}/lib"
    env.update(OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none",
               OMPI_MCA_hwloc_base_binding_policy="none")
    return env


def _hess_section(text, name):
    i = text.index(f"${name}")
    j = text.index("$", i + 1)
    return text[i:j]


def _hess_matrix(sec):
    lines = [ln for ln in sec.splitlines()[1:] if not ln.startswith("#")]
    n = int(lines[0].split()[0])
    ncol = int(lines[0].split()[1]) if len(lines[0].split()) > 1 else n
    m = np.zeros((n, ncol))
    i = 1
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        cols = [int(c) for c in lines[i].split()]
        for r in range(n):
            vals = lines[i + 1 + r].split()[1:]
            for c, v in zip(cols, vals):
                m[r, c] = float(v)
        i += n + 1
    return m


def orca_analysis(z, pos, hess, masses, energy, mult, T=298.15, P=101325.0, apt=None, qrrho=False,
                  workdir=None):
    """orca_vib (frequencies, modes, IR) and ORCA PrintThermoChem (thermochemistry, read at
    full precision from orca.property.txt). ORCA's thermochemistry always uses its quasi-RRHO
    entropy unless `QuasiRRHO false`; `qrrho` chooses."""
    wd = Path(workdir or tempfile.mkdtemp(prefix="orcavib_"))
    wd.mkdir(parents=True, exist_ok=True)
    write_orca_hess(wd / "mol.hess", z, pos, hess, masses, energy, mult, apt)
    env = _orca_env()
    vib = subprocess.run([str(ORCA / "orca_vib"), "mol.hess"], cwd=wd, env=env, capture_output=True, text=True)
    (wd / "orca_vib.out").write_text(vib.stdout + vib.stderr)
    text = (wd / "mol.hess").read_text()
    sec = _hess_section(text, "vibrational_frequencies").splitlines()
    nf = int(sec[1])
    allf = np.array([float(x.split()[1]) for x in sec[2:2 + nf]])
    modes = _hess_matrix(_hess_section(text, "normal_modes")).T
    keep = np.linalg.norm(modes, axis=1) > 0
    out = {"frequencies": allf[keep], "modes": _unit_rows(modes[keep])}
    if apt is not None:
        # $ir_spectrum of orca_vib: wavenumber, T**2 (a.u.; T = <0|mu|1> = dmu/dQ sqrt(hbar/2 omega)),
        # TX, TY, TZ; converted to km/mol by the caller
        rows = [r.split() for r in _hess_section(text, "ir_spectrum").splitlines()[2:] if r.strip() and not r.startswith("#")]
        out["ir_t2"] = np.array([float(r[1]) for r in rows])[keep]
    # thermochemistry
    inp = ["! PrintThermoChem Bohrs", '%geom inhessname "mol.hess" end',
           f"%freq Temp {T} QuasiRRHO {'true' if qrrho else 'false'} end", f"* xyz 0 {mult}"]
    nelec = int(sum(z))
    charge = 0 if (nelec - (mult - 1)) % 2 == 0 else 1
    inp[-1] = f"* xyz {charge} {mult}"
    inp += [f"{SYMBOLS[a]} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f} M = {m:.10f}" for a, m, p in zip(z, masses, pos)]
    inp += ["*", ""]
    (wd / "thermo.inp").write_text("\n".join(inp))
    r = subprocess.run([str(ORCA / "orca"), "thermo.inp"], cwd=wd, env=env, capture_output=True, text=True)
    (wd / "thermo.out").write_text(r.stdout)
    if "ORCA TERMINATED NORMALLY" not in r.stdout:
        raise RuntimeError(f"ORCA PrintThermoChem failed in {wd}")
    prop = (wd / "thermo.property.txt").read_text()
    g = lambda k: float(re.search(rf"&{k} \[&Type \"Double\"\]\s+(\S+)", prop).group(1))  # noqa: E731
    sn = re.search(r"Point Group:\s+(\S+), Symmetry Number:\s+(\d+)", r.stdout)
    rc = re.search(r"Rotational constants in cm-1:\s+(\S+)\s+(\S+)\s+(\S+)", r.stdout)
    out["point_group"], out["symmetry_number"] = sn.group(1), int(sn.group(2))
    out["rotational_constants"] = np.array([float(x) for x in rc.groups()])
    kt = 3.166811563e-6 * T  # only to split the PV term; ORCA's own kB*T is in enthalpyH - innerEnergyU
    t = {"zpe": g("zpe"), "E_trans": g("transEnergy"), "E_rot": g("rotEnergy"),
         "E_vib": g("vibEnergy") + g("zpe"), "TS_trans": g("qTrans"), "TS_rot": g("qRot"), "TS_vib": g("qVib"),
         "TS_elec": g("qEl"), "U_corr": g("innerEnergyU"), "H_corr": g("enthalpyH"), "TS_tot": g("entropyS"),
         "G_corr": g("freeEnergyG"), "kT": g("enthalpyH") - g("innerEnergyU")}
    for k in ("trans", "rot", "vib", "elec"):
        t[f"S_{k}"] = t[f"TS_{k}"] / T
    t["S_tot"] = t["TS_tot"] / T
    del kt
    out["thermo"] = t
    out["workdir"] = str(wd)
    return out
