"""linus against PySCF with exactly the same basis functions and DFT grid.

PySCF gets the basis from the same Basis Set Exchange JSON files linus reads (as
tests/python/compare_ints_pyscf.py does) and the molecule in bohr, so only the programs
differ. PySCF's default grid at level 3 is linus's `pyscf:3` (PySCF's small-density grid
pruning is switched off). Both SCFs are converged tightly.
"""

import json
import os
from pathlib import Path

import linus
import numpy as np
import pytest
from helpers import run
from linus.convert import to_pyscf, to_pyscf_mol

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto, scf

PYSCF_XC = {"pbe": "pbe,pbe", "b3lyp": "b3lyp"}


def pyscf_basis(name, symbols):
    """PySCF basis dict from linus's BSE JSON file (general contractions split, zeros dropped)."""
    data = json.loads(linus.basis_file(name).read_text())
    out = {}
    for sym in set(symbols):
        shells = []
        for sh in data["elements"][str(gto.charge(sym))]["electron_shells"]:
            exps = [float(e) for e in sh["exponents"]]
            ls, rows = sh["angular_momentum"], sh["coefficients"]
            if len(ls) == 1:
                ls = ls * len(rows)
            for l, row in zip(ls, rows):
                shells.append([l] + [[e, float(c)] for e, c in zip(exps, row) if float(c) != 0.0])
        out[sym] = shells
    return out


def pyscf_mol(structure, basis, cart=False):
    return gto.M(atom=[(s, tuple(x)) for s, x in zip(structure.symbols, structure.geometry)], unit="Bohr",
                 basis=pyscf_basis(basis, structure.symbols), charge=structure.charge,
                 spin=structure.multiplicity - 1, cart=cart, verbose=0)


def pyscf_scf(mol, method, restricted):
    if method == "hf":
        mf = scf.RHF(mol) if restricted else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if restricted else dft.UKS(mol)
        mf.xc = PYSCF_XC[method]
        mf.grids.level = 3
        mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.parametrize("method", ["hf", "pbe", "b3lyp"])
@pytest.mark.parametrize("charge,multiplicity", [(0, 1), (1, 2)], ids=["closed", "cation-doublet"])
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_energy_matches_pyscf(molecules, name, charge, multiplicity, method):
    s = molecules(name, charge, multiplicity)
    # orbital energies and the dipole are first order in the SCF residual: converge beyond 1e-6
    r = run(s, method=method, jk="exact", threads=4, scf={"convergence": "verytight"},
            outputs=("orbitals",)).results
    restricted = multiplicity == 1
    assert (r.wavefunction.orbitals.spin == "restricted") == restricted
    mf = pyscf_scf(pyscf_mol(s, "def2-svp"), method, restricted)
    assert r.energy == pytest.approx(mf.e_tot, abs=1e-9)
    eps_a = mf.mo_energy if restricted else mf.mo_energy[0]
    eps_b = mf.mo_energy if restricted else mf.mo_energy[1]
    orb = r.wavefunction.orbitals
    np.testing.assert_allclose(orb.energies_a, eps_a, atol=1e-6)
    np.testing.assert_allclose(orb.energies_a if restricted else orb.energies_b, eps_b, atol=1e-6)
    np.testing.assert_allclose(r.dipole_moment, mf.dip_moment(unit="au", verbose=0), atol=1e-6)
    assert r.nuclear_repulsion_energy == pytest.approx(mf.energy_nuc(), abs=1e-12)
    # occupations per spin as PySCF's
    occ = mf.mo_occ if not restricted else (mf.mo_occ / 2, mf.mo_occ / 2)
    np.testing.assert_array_equal(orb.occupations_a, occ[0])
    np.testing.assert_array_equal(orb.occupations_b, occ[1])


@pytest.mark.parametrize("pure", [True, False], ids=["spherical", "cartesian"])
def test_cartesian_and_spherical_energies(water, pure):
    """Cartesian d functions (cartesian=True) too: same energy as PySCF's cart=True."""
    r = run(water, basis="def2-tzvp", jk="exact", cartesian=not pure, threads=4).results
    mf = pyscf_scf(pyscf_mol(water, "def2-tzvp", cart=not pure), "hf", True)
    assert r.calcinfo_nbasis == mf.mol.nao
    assert r.energy == pytest.approx(mf.e_tot, abs=1e-9)


@pytest.mark.parametrize("pure", [True, False], ids=["spherical", "cartesian"])
def test_basis_set_model_reproduces_pyscf_basis(water, pure):
    """`to_pyscf_mol` of linus's basis set (the wavefunction's `qcdata.BasisSet`) is the same
    basis (same overlap, same AO order); linus's spherical AOs are PySCF's: its orbitals are
    orthonormal in PySCF's overlap."""
    wfn = run(water, basis="cc-pvtz", cartesian=not pure, outputs=("orbitals",)).results.wavefunction
    orb = wfn.orbitals
    ref = pyscf_mol(water, "cc-pvtz", cart=not pure)
    mol = to_pyscf_mol(water, wfn.basis)
    np.testing.assert_allclose(mol.intor("int1e_ovlp"), ref.intor("int1e_ovlp"), atol=1e-13)
    if pure:
        c = orb.coefficients_a
        np.testing.assert_allclose(c.T @ ref.intor("int1e_ovlp") @ c, np.eye(c.shape[1]), atol=1e-10)


def test_density_fitting_matches_pyscf(water):
    """DF-J/K with the same auxiliary basis as PySCF's density_fit."""
    r = run(water, method="b3lyp", jk={"kind": "df", "aux": "def2-universal-jkfit"}, threads=4).results
    mol = pyscf_mol(water, "def2-svp")
    mf = dft.RKS(mol).density_fit(auxbasis=pyscf_basis("def2-universal-jkfit", water.symbols))
    mf.xc, mf.small_rho_cutoff, mf.conv_tol = "b3lyp", 0.0, 1e-12
    mf.kernel()
    assert r.energy == pytest.approx(mf.e_tot, abs=1e-8)


def pyscf_ecp(name, symbols):
    """PySCF ECP dict from linus's BSE JSON file (the highest channel is the local one)."""
    data = json.loads(linus.basis_file(name).read_text())
    out = {}
    for sym in set(symbols):
        el = data["elements"][str(gto.charge(sym))]
        if "ecp_potentials" not in el:
            continue
        lmax = max(p["angular_momentum"][0] for p in el["ecp_potentials"])
        chans = []
        for p in el["ecp_potentials"]:
            rad = [[] for _ in range(8)]
            for n, e, c in zip(p["r_exponents"], p["gaussian_exponents"], p["coefficients"][0]):
                rad[n].append([float(e), float(c)])
            l = p["angular_momentum"][0]
            chans.append([-1 if l == lmax else l, rad])
        out[sym] = [el["ecp_electrons"], chans]
    return out


@pytest.mark.parametrize("method", ["hf", "b3lyp"])
def test_ecp_energy_and_gradient_match_pyscf(method):
    """def2 ECP on iodine (28 core electrons): electron count, nuclear repulsion with the
    effective charge, energy and analytic gradient (fixed grid: PySCF 2.15's grid-response
    gradient uses Z - ncore for the atomic-size adjustment of ECP atoms)."""
    from qcdata import Structure

    s = Structure.open(Path(__file__).resolve().parents[3] / "tests" / "molecules" / "ch3i.xyz")
    fixed = {} if method == "hf" else {"grid": {"response": False}}  # D41: HF has no grid
    r = run(s, "gradient", method=method, jk="exact", threads=4, scf={"convergence": "verytight"},
            outputs=("orbitals",), **fixed).results
    mol = gto.M(atom=[(sym, tuple(x)) for sym, x in zip(s.symbols, s.geometry)], unit="Bohr",
                basis=pyscf_basis("def2-svp", s.symbols), ecp=pyscf_ecp("def2-svp", s.symbols), verbose=0)
    assert r.calcinfo_nalpha + r.calcinfo_nbeta == mol.nelectron == 34  # I: 53 - 28, C, 3 H
    wfn = r.wavefunction
    orb = wfn.orbitals
    assert sorted(orb.core_electrons) == [0, 0, 0, 0, 28]
    # the ECP in qcdata's layout (decision 25): the iodine's, local and semi-local channels, with
    # the core electrons it replaces (Ecp.core_electrons, the orbitals' core_electrons there)
    (ecp,) = wfn.basis.ecp
    assert s.symbols[ecp.center] == "I" and ecp.core_electrons == orb.core_electrons[ecp.center] == 28
    assert len(ecp.semilocal) >= 3
    mf = pyscf_scf(mol, method, True)
    assert r.energy == pytest.approx(mf.e_tot, abs=1e-9)
    assert r.nuclear_repulsion_energy == pytest.approx(mf.energy_nuc(), abs=1e-11)
    g = mf.nuc_grad_method()
    if method != "hf":
        g.grid_response = False
    np.testing.assert_allclose(r.gradient, g.kernel(), atol=1e-7)
    # the PySCF export carries the ECPs
    m2 = to_pyscf_mol(s, wfn.basis)
    assert m2.nelectron == 34
    np.testing.assert_allclose(m2.intor("ECPscalar"), mol.intor("ECPscalar"), atol=1e-12)
    if method == "hf":
        # and PySCF's HF energy at linus's orbitals' density, with the exported ECP, is linus's
        m3, c, _, occ = to_pyscf(s, wfn)
        assert scf.RHF(m3).energy_tot(dm=scf.RHF(m3).make_rdm1(c, occ)) == pytest.approx(r.energy, abs=1e-8)


# range-separated hybrids: linus name -> (PySCF xc, VV10 of the PySCF functional switched off)
RSH = {"wb97x": ("wb97x", False), "wb97x-d3bj": ("wb97x-v", True), "cam-b3lyp": ("cam-b3lyp", False),
       "wb97m-d3bj": ("wb97m-v", True), "HYB_GGA_XC_LRC_WPBE": ("HYB_GGA_XC_LRC_WPBE", False)}


@pytest.mark.parametrize("method", list(RSH))
@pytest.mark.parametrize("charge,multiplicity", [(0, 1), (1, 2)], ids=["closed", "cation-doublet"])
def test_range_separated_hybrids_match_pyscf(molecules, method, charge, multiplicity):
    """Energies and analytic gradients (exact integrals, grid response) of range-separated
    hybrids; the -D3BJ variants without their dispersion term."""
    s = molecules("nh3", charge, multiplicity)
    r = run(s, "gradient", method=method, jk="exact", threads=4, scf={"convergence": "extreme"},
            dispersion="none").results
    mf = dft.RKS(pyscf_mol(s, "def2-svp")) if multiplicity == 1 else dft.UKS(pyscf_mol(s, "def2-svp"))
    mf.xc, no_nlc = RSH[method]
    if no_nlc:
        mf.nlc = False
    mf.grids.level, mf.small_rho_cutoff = 3, 0.0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
    mf.kernel()
    assert mf.converged
    gm = mf.nuc_grad_method()
    gm.grid_response = True
    assert r.energy == pytest.approx(mf.e_tot, abs=1e-9)
    np.testing.assert_allclose(np.asarray(r.gradient), gm.kernel(), atol=1e-8)


VV10_METHODS = pytest.mark.parametrize("method", ["wb97m-v", "wb97x-v", "b97m-v"])
VV10_SPINS = pytest.mark.parametrize("charge,multiplicity", [(0, 1), (1, 2)], ids=["closed", "cation-doublet"])
# PySCF's results for the VV10 cases, so that every push compares linus with them in seconds (PySCF's own VV10 SCF and
# grid-response gradient take minutes per case); the nightly's slow test recomputes and checks them, and
# LINUS_REFRESH_PYSCF_CACHE=1 pytest -m slow -k vv10_functionals rewrites them (e.g. after a PySCF update)
PYSCF_VV10 = Path(__file__).resolve().parent / "data" / "pyscf_vv10.json"


def vv10_inputs(s, method):
    """What PySCF's VV10 results depend on: the stored results are used only for these exact inputs."""
    return {"symbols": list(s.symbols), "geometry_bohr": np.asarray(s.geometry).tolist(), "charge": s.charge,
            "multiplicity": s.multiplicity, "xc": method, "basis": "def2-svp (linus's BSE file)", "grids.level": 3,
            "nlcgrids.level": "default (3)", "small_rho_cutoff": 0.0, "conv_tol": 1e-12, "conv_tol_grad": 1e-9,
            "gradient": "grid_response=True"}


def vv10_key(method, charge, multiplicity):
    return f"nh3/{charge}/{multiplicity}/{method}"


def linus_vv10(s, method):
    return run(s, "gradient", method=method, jk="exact", grid=3, threads=4, scf={"convergence": "extreme"})


def pyscf_vv10(s, method):
    """PySCF's energy, VV10 energy and grid-response gradient (the slow part: minutes per case)."""
    mol = pyscf_mol(s, "def2-svp")
    mf = dft.RKS(mol) if s.multiplicity == 1 else dft.UKS(mol)
    mf.xc = method
    mf.grids.level, mf.small_rho_cutoff = 3, 0.0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
    mf.kernel()
    assert mf.converged
    dm = mf.make_rdm1()
    enlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, method, dm if s.multiplicity == 1 else dm[0] + dm[1])[1]
    gm = mf.nuc_grad_method()
    gm.grid_response = True
    return {"e_tot": float(mf.e_tot), "e_nlc": float(enlc), "gradient": np.asarray(gm.kernel()).tolist()}


def check_vv10(f, ref):
    assert f.results.energy == pytest.approx(ref["e_tot"], abs=1e-9)
    assert f.record["vv10"]["energy"] == pytest.approx(ref["e_nlc"], abs=1e-10)
    np.testing.assert_allclose(f.results.gradient, np.asarray(ref["gradient"]), atol=1e-8)


@pytest.mark.slow
@VV10_METHODS
@VV10_SPINS
def test_vv10_functionals_match_pyscf(molecules, method, charge, multiplicity):
    """VV10 functionals: energies, the VV10 energy and analytic gradients with grid response
    (exact integrals; PySCF's XC grid level 3 and its default NLC grid, level 3, which linus's
    grid.nlc "auto" picks for PySCF XC grids). PySCF's own grid-response VV10 gradient is off by
    ~2e-9 Eh/bohr (linus's agrees with finite differences to 1e-12).
    Also checks the stored PySCF results the per-push test compares with (data/pyscf_vv10.json): the
    installed PySCF's version, the same inputs, the same numbers; LINUS_REFRESH_PYSCF_CACHE=1 rewrites them."""
    s = molecules("nh3", charge, multiplicity)
    ref = pyscf_vv10(s, method)
    check_vv10(linus_vv10(s, method), ref)
    key, inputs = vv10_key(method, charge, multiplicity), vv10_inputs(s, method)
    data = json.loads(PYSCF_VV10.read_text()) if PYSCF_VV10.exists() else {}
    if os.environ.get("LINUS_REFRESH_PYSCF_CACHE"):
        if data.get("pyscf") != pyscf.__version__:  # another PySCF: start over
            data = {"pyscf": pyscf.__version__, "cases": {}}
        data["about"] = ("PySCF's results for test_pyscf.py's VV10 cases (written by LINUS_REFRESH_PYSCF_CACHE=1 "
                         "pytest -m slow -k vv10_functionals; checked by the nightly)")
        data["cases"][key] = {"inputs": inputs, **ref}
        PYSCF_VV10.parent.mkdir(exist_ok=True)
        PYSCF_VV10.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        return
    refresh = "refresh them: LINUS_REFRESH_PYSCF_CACHE=1 pytest -m slow -k vv10_functionals"
    assert data.get("pyscf") == pyscf.__version__, (
        f"{PYSCF_VV10.name} holds PySCF {data.get('pyscf')}'s results but PySCF {pyscf.__version__} is installed: {refresh}")
    stored = data["cases"].get(key)
    assert stored is not None and stored["inputs"] == inputs, f"{PYSCF_VV10.name} has no results for {key} with these inputs: {refresh}"
    assert stored["e_tot"] == pytest.approx(ref["e_tot"], abs=1e-10)
    assert stored["e_nlc"] == pytest.approx(ref["e_nlc"], abs=1e-10)
    np.testing.assert_allclose(np.asarray(stored["gradient"]), np.asarray(ref["gradient"]), atol=1e-9)


@VV10_METHODS
@VV10_SPINS
def test_vv10_functionals_match_stored_pyscf(molecules, method, charge, multiplicity):
    """Per push: linus against PySCF's stored VV10 results (data/pyscf_vv10.json, recomputed and checked by the
    nightly's test_vv10_functionals_match_pyscf): the same comparison, in seconds."""
    data = json.loads(PYSCF_VV10.read_text())
    if data["pyscf"] != pyscf.__version__:
        pytest.skip(f"stored results are PySCF {data['pyscf']}'s, PySCF {pyscf.__version__} is installed "
                    "(the nightly's slow test fails until they are refreshed)")
    s = molecules("nh3", charge, multiplicity)
    stored = data["cases"][vv10_key(method, charge, multiplicity)]
    assert stored["inputs"] == vv10_inputs(s, method)
    check_vv10(linus_vv10(s, method), stored)


def test_range_separated_density_fitting_matches_pyscf_wk(water):
    """RI-K_LR in Psi4's wK form (Coulomb metric, erf 3-center integrals on one side): the
    reference is PySCF's DF with K_LR rebuilt from PySCF integrals in that form."""
    import scipy.linalg
    from pyscf import df

    r = run(water, method="wb97x", jk={"kind": "df", "aux": "def2-universal-jkfit"}, threads=4,
            scf={"convergence": "extreme"}).results
    mol = pyscf_mol(water, "def2-svp")
    aux = pyscf_basis("def2-universal-jkfit", water.symbols)
    mf = dft.RKS(mol).density_fit(auxbasis=aux)
    mf.xc, mf.small_rho_cutoff, mf.conv_tol = "wb97x", 0.0, 1e-12
    auxmol = df.addons.make_auxmol(mol, aux)
    nao, naux = mol.nao, auxmol.nao
    low = scipy.linalg.cholesky(auxmol.intor("int2c2e"), lower=True)

    def fitted(omega):
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            a3 = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1").reshape(nao * nao, naux)
        return scipy.linalg.solve_triangular(low, a3.T, lower=True).reshape(naux, nao, nao)

    b = fitted(0.0)
    orig = mf.with_df.get_jk

    def get_jk(dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13, omega=None):
        if not omega:
            return orig(dm, hermi, with_j, with_k, direct_scf_tol, omega)
        bw = fitted(omega)
        k = np.einsum("qml,ls,qns->mn", b, dm, bw, optimize=True)
        return None, 0.5 * (k + k.T)

    mf.with_df.get_jk = get_jk
    mf.kernel()
    assert r.energy == pytest.approx(mf.e_tot, abs=1e-9)



@pytest.mark.parametrize("case", ["water-spherical", "water-cartesian", "ch3i-ecp"])
def test_ao_matrices_are_pyscfs(molecules, case):
    """The exported overlap and core Hamiltonian (decision 25), converted to PySCF's AO
    convention with their variance (operators: 1 / (s_mu s_nu); Cartesian normalizations), are
    PySCF's own matrices of the exported basis set and ECPs; the Hamiltonian converts with them
    (its orbitals stay its eigenvectors), and the conversion round-trips."""
    from qcdata import PYSCF_CONVENTION, AOConvention, Structure

    from linus.convert import _pyscf_shell_order

    if case == "ch3i-ecp":
        s = Structure.open(Path(__file__).resolve().parents[3] / "tests" / "molecules" / "ch3i.xyz")
    else:
        s = molecules("water", 1, 2)
    cart = case == "water-cartesian"
    f = run(s, method="hf", basis="def2-svp", cartesian=cart, threads=4,
            outputs=("hamiltonian", "core_hamiltonian"))
    wf = f.results.wavefunction
    mol, order = _pyscf_shell_order(s, wf.basis)
    conv = wf.basis.transform
    S = conv(wf.overlap, PYSCF_CONVENTION, kind="operator", shell_order=order)
    h = conv(wf.core_hamiltonian, PYSCF_CONVENTION, kind="operator", shell_order=order)
    np.testing.assert_allclose(S, mol.intor("int1e_ovlp"), atol=1e-12)
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc") + (mol.intor("ECPscalar") if mol.has_ecp() else 0)
    np.testing.assert_allclose(h, hcore, atol=1e-10)
    # the whole wavefunction in PySCF's convention (same shell order here): still H C = S C eps
    if order == sorted(order):
        p = wf.with_convention(PYSCF_CONVENTION)
        c, e = np.asarray(p.orbitals.coefficients_a), p.orbitals.energies_a
        assert np.abs(p.hamiltonian_a @ c - p.overlap @ c * e).max() < 1e-10
        back = p.with_convention(AOConvention())
        for name in ("overlap", "hamiltonian_a", "core_hamiltonian"):
            np.testing.assert_allclose(getattr(back, name), getattr(wf, name), rtol=1e-14, atol=1e-14)
