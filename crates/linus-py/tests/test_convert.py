"""linus's orbitals as qcdata's models (the request output ``orbitals``: the coefficients, with the
wavefunction's basis set), and the converters on them: PySCF in both directions (`linus.convert`),
Molden files (qcinf's `qcinf.molden.write` of linus's orbitals, and the library's own
``output.molden``)."""

import linus
import numpy as np
import pytest
from helpers import run
from linus.convert import from_pyscf, from_pyscf_mol, to_pyscf, to_pyscf_mol
from qcdata import AOConvention, Orbitals, Structure, Wavefunction

pyscf = pytest.importorskip("pyscf")
qmolden = pytest.importorskip("qcinf.molden")
from pyscf import scf  # noqa: E402
from pyscf.tools import molden  # noqa: E402


def hf(structure, basis, pure=True, **kw):
    """An HF frame with its orbitals (exact J/K: PySCF's energy of the orbitals is compared
    with linus's), the structure and the wavefunction (with the coefficients and their basis
    set)."""
    kw.setdefault("jk", "exact")
    f = run(structure, basis=basis, cartesian=not pure, threads=16, outputs=("orbitals",), **kw)
    return f, f.structure, f.results.wavefunction


def pyscf_energy(mol, mo_coeff, mo_occ):
    """PySCF's HF energy of the density of given orbitals."""
    if np.ndim(mo_coeff) == 2:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.energy_tot(dm=dm)


@pytest.fixture(scope="module")
def uhf_nh3(molecules):
    return hf(molecules("nh3", 1, 2), "def2-tzvp")


def test_orbitals_are_qcdata_models(uhf_nh3):
    f, _, wfn = uhf_nh3
    orb = wfn.orbitals
    assert isinstance(orb, Orbitals) and orb.spin == "unrestricted" and orb.kind == "canonical"
    assert f.results.calcinfo_nbasis == wfn.basis.nao and orb.nmo == f.results.calcinfo_nmo
    assert wfn.basis.name == "def2-tzvp" and wfn.basis.ao_convention == AOConvention()
    text = wfn.model_dump_json()
    back_wfn = Wavefunction.model_validate_json(text)
    back = back_wfn.orbitals
    for a, b in zip(back_wfn.basis.shells, wfn.basis.shells):
        assert (a.center, a.angular_momentum, a.pure) == (b.center, b.angular_momentum, b.pure)
        np.testing.assert_array_equal(a.coefficients, b.coefficients)
    for name in ("coefficients_a", "coefficients_b", "energies_a", "energies_b", "occupations_a", "occupations_b"):
        np.testing.assert_array_equal(getattr(back, name), getattr(orb, name))


def test_levels_without_coefficients(molecules):
    """Every frame has the orbital levels; the coefficients and basis set come on request, and
    they are the same levels."""
    s = molecules("nh3", 1, 2)
    plain_wfn = run(s, basis="def2-tzvp").results.wavefunction
    plain = plain_wfn.orbitals
    full = run(s, basis="def2-tzvp", outputs=("orbitals",)).results.wavefunction.orbitals
    assert plain_wfn.basis is None and plain.coefficients_a is None and plain.spin == "unrestricted"
    for name in ("energies_a", "energies_b", "occupations_a", "occupations_b"):
        np.testing.assert_array_equal(getattr(plain, name), getattr(full, name))
    with pytest.raises(ValueError, match="coefficients"):
        plain.density()


def test_ao_labels(water):
    """linus's AO order is qcdata's default convention (PySCF's)."""
    basis = run(water, basis="def2-svp", outputs=("orbitals",)).results.wavefunction.basis
    labels = basis.ao_labels(water.symbols)
    assert len(labels) == basis.nao == 24
    # p shells are flagged Cartesian (x, y, z either way): labelled as such
    assert labels[:6] == ["0 O 1s", "0 O 2s", "0 O 3s", "0 O 2px", "0 O 2py", "0 O 2pz"]
    assert "0 O 3d-2" in labels and "0 O 3d+2" in labels
    cart = run(water, basis="def2-svp", cartesian=True, outputs=("orbitals",)).results.wavefunction.basis
    assert cart.ao_labels(water.symbols)[9:15] == ["0 O 3dxx", "0 O 3dxy", "0 O 3dxz", "0 O 3dyy", "0 O 3dyz",
                                                   "0 O 3dzz"]


@pytest.mark.parametrize("pure", [True, False], ids=["spherical", "cartesian"])
@pytest.mark.parametrize("case", ["rhf-water", "uhf-nh3+"])
def test_pyscf_round_trip(molecules, case, pure):
    """linus orbitals -> PySCF: orthonormal in PySCF's metric, same HF energy; and back."""
    s = molecules("water") if case == "rhf-water" else molecules("nh3", 1, 2)
    f, s, wfn = hf(s, "cc-pvtz", pure=pure)
    orb = wfn.orbitals
    mol, c, e, occ = to_pyscf(s, wfn)
    assert mol.cart == (not pure) and mol.nao == f.results.calcinfo_nbasis
    S = mol.intor("int1e_ovlp")
    for ci in ([c] if c.ndim == 2 else c):
        np.testing.assert_allclose(ci.T @ S @ ci, np.eye(ci.shape[1]), atol=1e-10)
    assert pyscf_energy(mol, c, occ) == pytest.approx(f.results.energy, abs=1e-8)
    s2, wfn2 = from_pyscf(mol, c, e, occ)
    assert s2.symbols == s.symbols and np.allclose(s2.geometry, s.geometry)
    assert (s2.charge, s2.multiplicity) == (s.charge, s.multiplicity)
    back_wfn = wfn2.with_convention(wfn.basis.ao_convention)
    back = back_wfn.orbitals
    np.testing.assert_allclose(back.coefficients_a, orb.coefficients_a, atol=1e-12)
    np.testing.assert_allclose(back.occupations_a, orb.occupations_a)
    np.testing.assert_allclose(back.occupations_b, orb.occupations_b)
    for a, b in zip(back_wfn.basis.shells, wfn.basis.shells):
        np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-12)


def test_from_pyscf_mol_general_contraction():
    """A PySCF shell with two contractions becomes two shells; the AOs are the same."""
    from pyscf import gto

    mol = gto.M(atom="Ne 0 0 0; He 0 0 1.5", basis={"Ne": [[0, [5.0, 0.5, 0.1], [1.0, 0.6, 0.9]], [2, [0.8, 1.0]]],
                                                     "He": "cc-pvdz"}, verbose=0)
    s, basis = from_pyscf_mol(mol)
    assert len(basis.shells) == mol.nbas + 1 and basis.nao == mol.nao and basis.ecp == []
    again = to_pyscf_mol(s, basis)
    np.testing.assert_allclose(again.intor("int1e_ovlp"), mol.intor("int1e_ovlp"), atol=1e-13)


@pytest.mark.parametrize("pure", [True, False], ids=["spherical", "cartesian"])
@pytest.mark.parametrize("case", ["rhf-water", "uhf-nh3+"])
def test_molden_readable_by_pyscf(tmp_path, molecules, case, pure):
    """qcinf's Molden file of linus's orbitals: PySCF's Molden reader gets the same molecule,
    orbitals and HF energy (cc-pVQZ: up to g)."""
    s = molecules("water") if case == "rhf-water" else molecules("nh3", 1, 2)
    f, s, wfn = hf(s, "cc-pvqz" if case == "rhf-water" else "def2-tzvp", pure=pure)
    orb = wfn.orbitals
    r = f.results
    path = tmp_path / "orbitals.molden"
    path.write_text(qmolden.write(wfn, s))
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(str(path))
    assert mol.nao == r.calcinfo_nbasis and mol.natm == len(s.symbols)
    np.testing.assert_allclose(mol.atom_coords(unit="Bohr"), s.geometry, atol=1e-12)
    S = mol.intor("int1e_ovlp")
    if orb.restricted:
        np.testing.assert_allclose(mo_energy, orb.energies_a, atol=1e-11)
        np.testing.assert_allclose(mo_occ, orb.occupations_a + orb.occupations_b)
        np.testing.assert_allclose(mo_coeff.T @ S @ mo_coeff, np.eye(r.calcinfo_nmo), atol=1e-9)
        e = pyscf_energy(mol, mo_coeff, mo_occ)
    else:
        mol.spin = r.calcinfo_nalpha - r.calcinfo_nbeta
        mol.charge = s.charge
        mol.build(0, 0)
        for c, eps, ref in zip(mo_coeff, mo_energy, (orb.energies_a, orb.energies_b)):
            np.testing.assert_allclose(eps, ref, atol=1e-11)
            np.testing.assert_allclose(c.T @ S @ c, np.eye(r.calcinfo_nmo), atol=1e-9)
        e = pyscf_energy(mol, np.asarray(mo_coeff), np.asarray(mo_occ))
    assert e == pytest.approx(r.energy, abs=1e-8)


def _same_orbitals(text: str, wfn: Wavefunction) -> None:
    """A Molden file of the wavefunction's own functions holds its orbitals: qcinf reads them back
    (to the digits the file prints)."""
    _, back = qmolden.read(text)
    back = back.with_convention(wfn.basis.ao_convention)
    a, b = back.orbitals, wfn.orbitals
    assert a.spin == b.spin and back.basis.nao == wfn.basis.nao
    for name in ("coefficients_a", "coefficients_b", "energies_a", "energies_b"):
        if getattr(b, name) is not None:
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=0, atol=1e-11)
    for name in ("occupations_a", "occupations_b"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("form", ["spherical", "cartesian"])
@pytest.mark.parametrize("pure", [True, False], ids=["spherical-basis", "cartesian-basis"])
@pytest.mark.parametrize("case", ["rhf-water", "uhf-nh3+"])
def test_molden_setting(tmp_path, molecules, case, pure, form):
    """output.molden: the library's file (``files["orbitals.molden"]``) holds the frame's orbitals
    (qcinf reads them back) for the basis set's own functions, and PySCF reads the same orbitals
    and HF energy from either form (a spherical basis set written with Cartesian functions
    included)."""
    s = molecules("water") if case == "rhf-water" else molecules("nh3", 1, 2)
    basis = "cc-pvqz" if case == "rhf-water" else "def2-tzvp"
    if form == "spherical" and not pure:
        with pytest.raises(linus.InputError, match="spherical basis functions"):
            hf(s, basis, pure=pure, output={"molden": form})
        return
    f, s, wfn = hf(s, basis, pure=pure, output={"molden": form})
    orb = wfn.orbitals
    r = f.results
    text = r.files["orbitals.molden"]
    if pure == (form == "spherical"):
        _same_orbitals(text, wfn)
    path = tmp_path / "orbitals.molden"
    path.write_text(text)
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(str(path))
    assert bool(mol.cart) == (form == "cartesian")
    np.testing.assert_allclose(mol.atom_coords(unit="Bohr"), s.geometry, atol=1e-12)
    S = mol.intor("int1e_ovlp")
    if orb.restricted:
        np.testing.assert_allclose(mo_energy, orb.energies_a, atol=1e-11)
        np.testing.assert_allclose(mo_coeff.T @ S @ mo_coeff, np.eye(r.calcinfo_nmo), atol=1e-9)
        e = pyscf_energy(mol, mo_coeff, mo_occ)
    else:
        mol.spin = r.calcinfo_nalpha - r.calcinfo_nbeta
        mol.charge = s.charge
        mol.build(0, 0)
        for c in mo_coeff:
            np.testing.assert_allclose(c.T @ S @ c, np.eye(r.calcinfo_nmo), atol=1e-9)
        e = pyscf_energy(mol, np.asarray(mo_coeff), np.asarray(mo_occ))
    assert e == pytest.approx(r.energy, abs=1e-8)


def test_molden_ecp_atoms_have_their_core():
    """An ECP atom's Molden entry: its core electrons in [core], in the library's file and in
    qcinf's, so PySCF reads it as an ECP atom with the electrons the orbitals hold (without [core]
    it read iodine as an all-electron atom), and qcinf reads the library's cores back; with
    PySCF's own def2 ECPs, PySCF's HF energy of the orbitals is linus's."""
    from pathlib import Path

    s = Structure.open(Path(__file__).resolve().parents[3] / "tests" / "molecules" / "ch3i.xyz")
    f, s, wfn = hf(s, "def2-svp", output={"molden": "spherical"})
    text = f.results.files["orbitals.molden"]
    _same_orbitals(text, wfn)
    (ecp,) = wfn.basis.ecp
    assert ecp.core_electrons == wfn.orbitals.core_electrons[ecp.center] == 28
    iodine = s.symbols.index("I")
    for t in (text, qmolden.write(wfn, s)):
        assert f"{iodine + 1} : 28" in t.split("[core]")[1].split("[MO]")[0]
        mol, _, mo_coeff, mo_occ, _, _ = _load(t)
        # PySCF's reader builds the molecule at [MO] and reads [core] after it into mol.ecp, as
        # for its own files: the next build counts the cores
        assert list(mol.ecp.values()) == [[28, []]]
        mol.build(0, 0)
        assert mol.atom_nelec_core(iodine) == 28 and sum(mol.atom_nelec_core(i) for i in range(mol.natm)) == 28
        assert mol.nelectron == round(float(np.sum(mo_occ))) == f.results.calcinfo_nalpha + f.results.calcinfo_nbeta
        # Molden carries the cores, not the potentials: with PySCF's def2-SVP ECP, the energy
        mol.ecp = {"I": "def2-svp"}
        mol.build(0, 0)
        assert pyscf_energy(mol, mo_coeff, mo_occ) == pytest.approx(f.results.energy, abs=1e-9)
    _, back = qmolden.read(text)
    assert list(back.orbitals.core_electrons) == list(wfn.orbitals.core_electrons)


def test_a_large_molden_file_is_saved_beside_the_output(tmp_path):
    """A text in ``files`` over 1 MiB goes to a file of its own beside the saved JSON (qcdata's
    save, qc* #47), with a pointer in its place, and comes back the same: caffeine's Molden file."""
    import json
    from pathlib import Path

    from qcdata import ProgramInput, ProgramOutput

    s = Structure.open(Path(__file__).resolve().parents[3] / "tests" / "molecules" / "caffeine.xyz")
    out = linus.compute(ProgramInput(program="linus", calctype="energy", structure=s,
                                     model={"method": "hf", "basis": "def2-svp"},
                                     keywords={"jk": "df", "threads": 8, "output": {"molden": "spherical"}}))
    text = out.results.files["orbitals.molden"]
    assert len(text.encode()) > 2**20
    out.save(tmp_path / "output.json")
    pointer = json.loads((tmp_path / "output.json").read_text())["results"]["files"]["orbitals.molden"]
    assert pointer["path"] == "output.results.files.orbitals%2Emolden"
    assert (tmp_path / pointer["path"]).read_text() == text
    assert ProgramOutput.open(tmp_path / "output.json").results.files["orbitals.molden"] == text


def _load(text):
    """PySCF's reading of a Molden text (through a temporary file)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".molden") as t:
        t.write(text)
        t.flush()
        return molden.load(t.name)
