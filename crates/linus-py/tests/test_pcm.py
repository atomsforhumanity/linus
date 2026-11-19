"""Implicit solvent (PCM): the Python, qcdata and CLI interfaces, and exact agreement with
PySCF 2.15's pyscf.solvent.pcm (same basis functions, molecule in bohr)."""

import json
import subprocess

from conftest import MOLECULES
import linus
import numpy as np
import pytest
from helpers import program_input, run
from qcdata import Model, ProgramInput

EPS_THF = 7.4257


@pytest.fixture(scope="module")
def hf_cpcm(water):
    return linus.compute(program_input(water, "gradient", jk="exact", solvent={"eps": EPS_THF, "model": "cpcm"},
                                       threads=2, scf={"convergence": "extreme"}))


def test_solvent_result(hf_cpcm):
    out = hf_cpcm
    r = out.results
    assert out.resolved.settings["solvent"]["eps"] == EPS_THF and out.resolved.settings["solvent"]["model"] == "cpcm"
    assert sum(r.energy_components.values()) == pytest.approx(r.energy, abs=1e-10)
    e = r.energy_components["solvation_electrostatic"]
    assert -0.02 < e < -0.002  # a few kcal/mol for water in THF
    s = out.record["solvation"]
    assert s["model"] == "cpcm" and s["preset"] == "pyscf" and s["lebedev"] == [302] * 3 and s["discretization"] == "swig"
    assert s["electrostatic"] == e and s["npoints"] > 300 and 20 < s["area_angstrom2"] < 60
    # neutral solute: Gauss's law says zero total charge; the escaped density leaves a little
    assert s["gauss_charge"] == 0.0 and abs(s["surface_charge"]) < 0.02
    assert r.gradient.shape == (3, 3) and np.abs(r.gradient.sum(axis=0)).max() < 1e-8
    assert out.execution.timings["pcm_potential"] > 0 and out.execution.timings["pcm_setup"] > 0


def test_against_pyscf(water, hf_cpcm):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.solvent import pcm  # noqa: F401
    from test_pyscf import pyscf_basis

    r = hf_cpcm.results
    mol = gto.M(atom=[(s, tuple(x)) for s, x in zip(water.symbols, water.geometry)], unit="Bohr",
                basis=pyscf_basis("def2-svp", water.symbols), verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.eps = EPS_THF
    mf.conv_tol = 1e-12
    e = mf.kernel()
    assert r.energy == pytest.approx(e, abs=1e-9)
    assert r.energy_components["solvation_electrostatic"] == pytest.approx(mf.with_solvent.e, abs=1e-9)
    np.testing.assert_allclose(r.gradient, mf.nuc_grad_method().kernel(), atol=1e-8)


def test_models_and_options(water):
    gas = run(water).results.energy
    es = {}
    for model in ("cpcm", "cosmo", "iefpcm", "ssvpe"):
        r = run(water, solvent={"eps": 78.36, "model": model}).results
        es[model] = r.energy_components["solvation_electrostatic"]
        assert r.energy < gas
    # COSMO's f(eps) = (eps - 1)/(eps + 0.5) is below C-PCM's (eps - 1)/eps
    assert es["cosmo"] > es["cpcm"]
    # names come from the solvent database (qcref/CRC eps at 298.15 K)
    solvation = run(water, solvent="tetrahydrofuran").record["solvation"]
    assert solvation["eps"] == pytest.approx(7.43, abs=0.02) and solvation["solvent"] == "tetrahydrofuran"
    # the shorthands: a permittivity is {"eps": ..}, a name {"name": ..}; the model is in the group
    assert run(water, solvent=7.4257).results.energy == run(water, solvent={"eps": 7.4257}).results.energy
    assert linus.normalize_options({"solvent": 7.4257}) == {"solvent": {"eps": 7.4257}}
    assert linus.normalize_options({"solvent": "thf"}) == {"solvent": {"name": "thf"}}
    with pytest.raises(ValueError, match="model"):
        linus.Session(program_input(water, solvent={"eps": 2.0, "model": "dpcm"}))
    # the removed kwarg is an unknown key
    with pytest.raises(linus.InputError, match="pcm"):
        linus.Session(program_input(water, solvent=2.0, pcm="cpcm"))
    with pytest.raises(ValueError, match="lebedev"):
        run(water, solvent={"eps": 2.0, "lebedev": 74})


def test_qcdata_keyword(water):
    # (the orca preset's own test: test_orca_presets.py)
    inp = ProgramInput(program="linus", structure=water, calctype="energy", model=Model(method="hf", basis="def2-svp"),
                       keywords={"solvent": {"eps": EPS_THF, "preset": "pyscf", "lebedev": 194,
                                             "discretization": "iswig", "model": "cpcm"}})
    out = linus.compute(inp)
    solvation = out.record["solvation"]
    assert solvation["preset"] == "pyscf" and solvation["lebedev"] == [194, 194, 194]
    assert not solvation["point_potential"]
    assert solvation["discretization"] == "iswig"
    assert out.results.energy_components["solvation_electrostatic"] == solvation["electrostatic"] < 0


def test_cli_solvent_flags(cli, water):
    xyz = str(MOLECULES / "water.xyz")
    r = json.loads(subprocess.run([cli, xyz, "--eps", str(EPS_THF), "--solvent-model", "iefpcm", "--grad", "--json"],
                                  capture_output=True, text=True, check=True).stdout)
    assert r["solvation"]["model"] == "iefpcm" and r["components"]["solvation_electrostatic"] < 0
    assert sum(r["components"].values()) == pytest.approx(r["energy"], abs=1e-10)
    assert len(r["gradient_parts"]["solvent"]) == 3
    req = {"structure": {"symbols": list(water.symbols), "geometry": water.geometry.tolist()}, "method": "hf",
           "basis": "def2-svp",
           "solvent": {"eps": EPS_THF, "model": "cosmo"}}
    r2 = json.loads(subprocess.run([cli, "--json"], input=json.dumps(req), capture_output=True, text=True,
                                   check=True).stdout)
    assert r2["solvation"]["model"] == "cosmo"
    bad = subprocess.run([cli, "--json"], input=json.dumps({**req, "solvent": {"eps": 2, "modl": "x"}}),
                         capture_output=True, text=True, check=False)
    assert bad.returncode == 2 and "modl" in bad.stderr


def test_smd_against_pyscf(water):
    """SMD = IEF-PCM with SMD radii (590 points) + G_CDS, as PySCF's mf.SMD()."""
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.solvent import smd  # noqa: F401
    from test_pyscf import pyscf_basis

    mol = gto.M(atom=[(s, tuple(x)) for s, x in zip(water.symbols, water.geometry)], unit="Bohr",
                basis=pyscf_basis("def2-svp", water.symbols), verbose=0)
    mf = scf.RHF(mol).SMD()
    mf.with_solvent.solvent = "water"
    mf.conv_tol = 1e-12
    e = mf.kernel()
    g = mf.nuc_grad_method().kernel()
    f = run(water, "gradient", jk="exact", threads=2, scf={"convergence": "extreme"},
            solvent={"name": "water", "model": "smd", "eps": float(smd.solvent_db["water"][5]),
                     "cds_contact_smoothing": 0.0})
    r, solvation = f.results, f.record["solvation"]
    # PySCF's eps for water (78.3553) is passed explicitly; linus's default is the database's
    assert solvation["eps"] == smd.solvent_db["water"][5]
    assert solvation["model"] == "iefpcm" and solvation["lebedev"] == [590] * 3
    assert solvation["cds"] == pytest.approx(mf.with_solvent.e_cds, abs=1e-9)
    assert r.energy_components["solvation_cds"] == solvation["cds"]
    assert r.energy == pytest.approx(e, abs=1e-9)
    np.testing.assert_allclose(r.gradient, g, atol=1e-8)


def test_hessian_matches_finite_differences(water):
    """C-PCM analytic Hessian (RI-JK) against central differences of analytic gradients."""
    kw = dict(jk={"kind": "df", "aux": "def2-universal-jkfit"}, threads=2, scf={"convergence": "extreme"},
              solvent={"eps": EPS_THF, "lebedev": 110})
    x0 = np.asarray(water.geometry, dtype=float)
    step = 1e-3
    with linus.Session(program_input(water, "hessian", **kw)) as s:
        h = s.compute().results.hessian
        for k in (0, 4, 8):
            gs = []
            for sign in (1, -1):
                x = x0.copy()
                x.flat[k] += sign * step
                # the displacements start from the center and are not kept (a fan-out)
                gs.append(s.compute(x, calctype="gradient", commit=False).results.gradient.ravel())
            np.testing.assert_allclose((gs[0] - gs[1]) / (2 * step), h[k], atol=2e-6)
