"""Hessians with their vibrational analysis and thermochemistry (qcdata's models): linus.vib
against PySCF's analysis, the frame's fields and record, the freq settings, the CLI. The
unprojected-Hessian test runs on the device with ``LINUS_TEST_BACKEND=gpu`` in a GPU build."""

import json
import os
import subprocess

import linus
import numpy as np
import pytest
from helpers import failure, program_input, run
from linus.vib import harmonic_analysis, thermochemistry
from qcdata import Thermochemistry, VibrationalAnalysis


@pytest.fixture(scope="module")
def water_freq(water):
    return run(water, "hessian", threads=4, outputs=("ir",))


def test_hessian_result(water, water_freq):
    f = water_freq
    r = f.results
    v, t = r.vibrational_analysis, r.thermochemistry[0]
    assert isinstance(v, VibrationalAnalysis) and isinstance(t, Thermochemistry) and len(r.thermochemistry) == 1
    assert v.frequencies.shape == (3,) and v.normal_modes_cartesian.shape == (3, 3, 3)
    assert v.n_imaginary() == 0 and not any(v.external) and v.rotor_type == "nonlinear"
    assert (v.point_group, v.symmetry_number, v.symmetry_number_source) == ("Cs", 1, "detected")  # planar
    np.testing.assert_allclose(np.linalg.norm(v.normal_modes_cartesian.reshape(3, -1), axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(v.normal_modes_mass_weighted @ v.normal_modes_mass_weighted.T, np.eye(3), atol=1e-12)
    assert 1500 < v.frequencies[0] < 1900 and 3800 < v.frequencies[2] < 4300
    assert v.ir_intensities.shape == (3,) and (v.ir_intensities > 1.0).all()
    assert v.dipole_derivatives.shape == (9, 3) and r.polarizability.shape == (3, 3)
    assert f.record["cphf_iterations"] > 0 and f.record["polarizability_cphf_iterations"] > 0
    # linus's analysis beyond qcdata's model, in the record
    extra = f.record["vibrational_analysis"]
    assert extra["detected_symmetry_number"] == 1 and len(extra["eigenvalues"]) == 3 and extra["warnings"] == []
    assert extra["codata"] == "codata2022" and extra["hessian_asymmetry"] < 1e-8
    # acoustic sum rule of the atomic polar tensors (neutral molecule)
    np.testing.assert_allclose(v.dipole_derivatives.reshape(3, 3, 3).sum(axis=0), 0.0, atol=1e-6)
    # thermochemistry adds up
    T = t.temperature
    assert t.total.gibbs == pytest.approx(t.total.enthalpy - T * t.total.entropy, abs=1e-12)
    assert t.gibbs_correction == pytest.approx(t.total.gibbs - r.energy, abs=1e-12)
    modes = f.record["thermochemistry"][0]["modes"]
    assert t.zero_point_energy == pytest.approx(sum(m["zpe"] for m in modes), abs=1e-14)
    assert 0.020 < t.zero_point_energy < 0.024 and t.gibbs_correction > 0
    # the analysis of the returned Hessian is the same, also through a JSON round trip
    v2 = harmonic_analysis(water.symbols, water.geometry, r.hessian, dipole_derivatives=v.dipole_derivatives)
    np.testing.assert_allclose(v2.frequencies, v.frequencies, rtol=1e-12)
    np.testing.assert_allclose(v2.ir_intensities, v.ir_intensities, rtol=1e-12)
    t2 = thermochemistry(VibrationalAnalysis.model_validate_json(v2.model_dump_json()), r.energy, 1)[0]
    assert t2.total.gibbs == pytest.approx(t.total.gibbs, abs=1e-12)
    assert t2.zero_point_energy == pytest.approx(t.zero_point_energy, abs=1e-15)


def test_analysis_matches_pyscf(water, water_freq):
    """The same Hessian through PySCF's harmonic_analysis and thermo (CODATA 2014 as PySCF)."""
    gto = pytest.importorskip("pyscf.gto")
    from pyscf.hessian import thermo

    r = water_freq.results
    masses = [linus.vib.atomic_mass(s) for s in water.symbols]
    mol = gto.M(atom=[(s, p) for s, p in zip(water.symbols, water.geometry)], unit="Bohr", basis="sto-3g", verbose=0)
    mol.nucprop = {i + 1: {"mass": m} for i, m in enumerate(masses)}
    res = thermo.harmonic_analysis(mol, r.hessian.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3), mass=np.array(masses))
    v = harmonic_analysis(water.symbols, water.geometry, r.hessian, codata="codata2014")
    np.testing.assert_allclose(v.frequencies, res["freq_wavenumber"].real, atol=1e-6)
    np.testing.assert_allclose(v.reduced_masses, res["reduced_mass"], rtol=1e-10)
    np.testing.assert_allclose(v.force_constants, res["force_const_dyne"], rtol=1e-8)
    modes = res["norm_mode"].reshape(3, -1)
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    np.testing.assert_allclose(np.abs(np.sum(modes * v.normal_modes_cartesian.reshape(3, -1), axis=1)), 1.0, atol=1e-10)

    class Model:
        e_tot = r.energy

    Model.mol = mol
    for q in (False, True):
        ref = thermo.thermo(Model, res["freq_au"], 298.15, 101325, qrrho=q, qrrho_enthalpy=q)
        t = thermochemistry(v, r.energy, 1, thermo="qrrho-hg" if q else "rrho", codata="codata2014")[0]
        assert t.zero_point_energy == pytest.approx(ref["ZPE"][0], abs=1e-12)
        for key, term in (("trans", "translational"), ("rot", "rotational"), ("vib", "vibrational"), ("tot", "total")):
            assert getattr(t, term).entropy == pytest.approx(ref[f"S_{key}"][0], abs=1e-13), key
            assert getattr(t, term).gibbs == pytest.approx(ref[f"G_{key}"][0], abs=1e-10), key


def test_options_and_errors(water, water_freq):
    r = water_freq.results
    z, x, h = water.symbols, water.geometry, r.hessian
    freqs = r.vibrational_analysis.frequencies
    with pytest.raises(ValueError, match="mass"):
        harmonic_analysis(z, x, h, mass="isotope")
    with pytest.raises(ValueError, match="symmetry number"):
        harmonic_analysis(z, x, h, symmetry_number=0)
    with pytest.raises(ValueError, match="shape"):
        harmonic_analysis(z, x, h[:6, :6])
    # standard atomic weights lower the frequencies; deuterium more
    avg = harmonic_analysis(z, x, h, mass_table="average")
    d2o = harmonic_analysis(z, x, h, isotopes=[(1, 2.01410177812), (2, 2.01410177812)])
    assert (avg.frequencies < freqs).all() and (d2o.frequencies < 0.8 * freqs).all()
    # no projection: 9 modes, 6 of them external; this geometry is not a stationary point
    # (max |g| ~ 0.03 Eh/bohr), so the rotations are not zero modes (here 386-800 cm^-1) and
    # couple to the bend (~6 cm^-1); translations stay zero modes
    none = harmonic_analysis(z, x, h, projection="none")
    external = np.asarray(none.external)
    assert none.frequencies.shape == (9,) and external.sum() == 6
    np.testing.assert_allclose(none.vibrational_frequencies, freqs, rtol=5e-3)
    assert np.sort(np.abs(none.frequencies[external]))[:3].max() < 0.1
    # thermochemistry options
    v = r.vibrational_analysis
    t1 = thermochemistry(v, r.energy, temperature=400.0)[0]
    t2 = thermochemistry(v, r.energy, concentration=1.0)[0]
    assert t1.total.entropy > r.thermochemistry[0].total.entropy and t2.total.gibbs > r.thermochemistry[0].total.gibbs
    # an atom: no modes; translational entropy of argon (154.85 J/(mol K) at 298.15 K, 1 bar
    # in tables; 154.74 at 1 atm)
    ar = harmonic_analysis(["Ar"], [[0.1, 0.2, 0.3]], np.zeros((3, 3)))
    assert ar.frequencies.size == 0 and ar.rotor_type == "atom" and ar.point_group == "Kh"
    t_ar = thermochemistry(VibrationalAnalysis.model_validate_json(ar.model_dump_json()), -526.0)[0]
    j_per_mol_k = 4.3597447222060e-18 * 6.02214076e23
    assert t_ar.total.entropy * j_per_mol_k == pytest.approx(154.74, abs=0.01)
    with pytest.raises(ValueError, match="temprature"):
        thermochemistry(v, r.energy, temprature=300)
    with pytest.raises(ValueError, match="temperature"):
        thermochemistry(v, r.energy, temperature=-1.0)
    # an analysis without what thermochemistry reads
    with pytest.raises(linus.InputError, match="rotor_type"):
        thermochemistry(v.model_copy(update={"rotor_type": None}), r.energy)


def test_hessian_freq_settings(water, water_freq):
    """The session's freq settings: temperatures, thermo models; ones that cannot take effect
    are errors at opening; without ir no dipole derivatives."""
    r = run(water, "hessian", threads=4, freq={"temperature": [298.15, 310.0], "thermo": "qrrho"}).results
    assert r.vibrational_analysis.dipole_derivatives is None and r.vibrational_analysis.ir_intensities is None
    assert r.polarizability is None
    assert [t.temperature for t in r.thermochemistry] == [298.15, 310.0]
    assert r.thermochemistry[0].entropy_model == "qrrho_grimme" and r.thermochemistry[0].enthalpy_model == "rrho"
    assert r.thermochemistry[0].qrrho_frequency == 100.0 and r.thermochemistry[0].pressure == 101325.0
    assert r.thermochemistry[1].total.entropy > r.thermochemistry[0].total.entropy
    with pytest.raises(linus.InputError, match="rotor_cutoff"):
        run(water, "hessian", freq={"rotor_cutoff": 50.0})  # qrrho models only
    with pytest.raises(linus.InputError):
        linus.Session(program_input(water, "hessian", freq={"temprature": 300}))
    # ir as a setting: every Hessian of the session
    assert run(water, "hessian", freq={"ir": True}).results.vibrational_analysis.ir_intensities is not None
    off = run(water, "hessian", freq={"thermo": "off"}).results
    assert off.thermochemistry == [] and off.vibrational_analysis is not None
    assert run(water, "hessian", freq={"analysis": "off"}).results.vibrational_analysis is None
    # the frame's JSON round trip keeps the models
    back = type(water_freq).model_validate_json(water_freq.model_dump_json())
    assert np.asarray(back.results.vibrational_analysis.normal_modes_cartesian).shape == (3, 3, 3)
    assert len(back.results.thermochemistry) == 1
    assert np.asarray(back.results.vibrational_analysis.dipole_derivatives).shape == (9, 3)


def test_qcdata_hessian_has_frequencies(water, water_freq):
    from qcdata import CalcType, Model, ProgramInput

    inp = ProgramInput(program="linus", calctype=CalcType.hessian, structure=water,
                       model=Model(method="hf", basis="def2-svp"),
                       keywords={"threads": 4, "freq": {"temperature": 310.0, "ir": True, "thermo": "qrrho"}})
    out = linus.compute(inp)
    res = out.results
    va = res.vibrational_analysis
    np.testing.assert_allclose(va.frequencies, water_freq.results.vibrational_analysis.frequencies, atol=1e-6)
    assert np.asarray(va.normal_modes_cartesian).shape == (3, 3, 3)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(va.normal_modes_cartesian).reshape(3, -1), axis=1), 1.0,
                               atol=1e-6)
    th = res.thermochemistry
    assert len(th) == 1 and th[0].temperature == 310.0 and th[0].entropy_model == "qrrho_grimme"
    assert 0 < th[0].total.gibbs - res.energy < 0.05  # the energy plus the thermal free energy
    assert va.symmetry_number == 1 and len(va.ir_intensities) == 3 and va.symmetry_number_source == "detected"
    assert np.asarray(va.dipole_derivatives).shape == (9, 3)
    assert np.asarray(res.polarizability).shape == (3, 3)
    # freq.* is a setting like any other: unknown names, and a calculation without a Hessian, are input errors
    bad = failure(inp.model_copy(update={"keywords": {"freq": {"temprature": 1}}}))
    assert bad.error.kind == "input" and bad.error.code == "unknown_key"
    gradient = failure(inp.model_copy(update={"calctype": CalcType.gradient}))
    assert gradient.error.code == "not_applicable"


@pytest.mark.parametrize("method,jk,rtol,atol", [("hf", "exact", 1e-6, 1e-6), ("b3lyp", "df", 5e-3, 1e-3)])
def test_the_hessian_is_never_projected(water, method, jk, rtol, atol):
    """`results.hessian` is the full Cartesian second derivative at the input geometry (qc* #92):
    away from a stationary point, rotating the molecule turns the gradient as the Hessian says,
    H (L x) = L g for each rotation generator L (a Hessian with the rotations projected out sends
    L x to zero), and a translation is a zero mode (the DFT grid breaks both symmetries
    slightly). The harmonic analysis projects only its own copy: with it off, or with IR, the
    Hessian is bitwise the same."""
    backend = os.environ.get("LINUS_TEST_BACKEND", "cpu")
    x = np.asarray(water.geometry) + [[0.10, -0.05, 0.08], [-0.07, 0.12, -0.03], [0.04, 0.02, -0.11]]
    s = water.model_copy(update={"geometry": x})
    kw = dict(method=method, jk=jk, threads=4, backend=backend)
    f = run(s, "hessian", **kw)
    h, g = np.asarray(f.results.hessian).reshape(9, 9), np.asarray(f.results.gradient).reshape(3, 3)
    for axis in range(3):
        rotation = np.zeros((3, 3))
        i, j = [k for k in range(3) if k != axis]
        rotation[i, j], rotation[j, i] = -1.0, 1.0
        turned = (g @ rotation.T).ravel()
        assert np.linalg.norm(turned) > 0.02  # far from stationary
        np.testing.assert_allclose(h @ (x @ rotation.T).ravel(), turned, rtol=0, atol=rtol * np.linalg.norm(turned))
        shift = np.zeros((3, 3))
        shift[:, axis] = 1.0
        assert np.linalg.norm(h @ shift.ravel()) < atol
    off = run(s, "hessian", freq={"analysis": "off"}, **kw)
    assert off.results.vibrational_analysis is None
    np.testing.assert_array_equal(off.results.hessian, f.results.hessian)
    ir = run(s, "hessian", outputs=("ir",), **kw)
    np.testing.assert_array_equal(ir.results.hessian, f.results.hessian)


def test_cli_hessian_analysis(cli, water, water_freq):
    req = {"structure": {"symbols": list(water.symbols), "geometry": np.asarray(water.geometry).tolist()}, "method": "hf",
           "basis": "def2-svp", "calctype": "hessian", "threads": 4, "freq": {"ir": True}}
    out = json.loads(subprocess.run([str(cli), "--json"], input=json.dumps(req), capture_output=True, text=True,
                                    check=True).stdout)
    va, th = out["vibrational_analysis"], out["thermochemistry"]
    ref = water_freq.results
    np.testing.assert_allclose(va["frequencies"], ref.vibrational_analysis.frequencies, atol=1e-6)
    np.testing.assert_allclose(va["ir_intensities"], ref.vibrational_analysis.ir_intensities, rtol=1e-6)
    assert th[0]["total"]["gibbs"] == pytest.approx(ref.thermochemistry[0].total.gibbs, abs=1e-9)
    assert np.asarray(out["dipole_derivatives"]).shape == (9, 3)
