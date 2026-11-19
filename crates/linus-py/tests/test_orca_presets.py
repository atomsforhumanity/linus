"""The orca PCM preset through qcdata keywords (its constants: data/orca/pcm.json), against the
sphere grids ORCA 6.0.0 picks."""

import linus
from qcdata import Model, ProgramInput

EPS_THF = 7.4257


def test_orca_preset_keyword(water):
    inp = ProgramInput(program="linus", structure=water, calctype="energy", model=Model(method="hf", basis="def2-svp"),
                       keywords={"solvent": {"eps": EPS_THF, "preset": "orca", "model": "cpcm"}})
    out = linus.compute(inp)
    solvation = out.record["solvation"]
    # ORCA's grids by point density (5 per Å^2): O 194, H 110; potential at the points
    assert solvation["preset"] == "orca" and solvation["lebedev"] == [194, 110, 110]
    assert solvation["point_potential"]
    assert solvation["discretization"] == "iswig"
    assert out.results.energy_components["solvation_electrostatic"] == solvation["electrostatic"] < 0
