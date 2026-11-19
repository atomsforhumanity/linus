"""Grid variants of the Hessian grid study (linus_grid::GridSpec as JSON, `{"kind": "spec", ...}`).

spec(): Treutler-Ahlrichs M4 radial grids (PySCF's formula, Treutler xi), per-row sizes (H-He,
Li-Ne, Na-Ar; later rows +10/+15/+20 radial shells), five-region pruning at NWChem's
Bragg-radius fractions, partitioning and size adjustment by name, weight cutoff 1e-15.
"""
NWCHEM = [[0.25, 0.5, 1.0, 4.5], [0.1667, 0.5, 0.9, 3.5], [0.1, 0.4, 0.8, 2.5]]
M4 = {"TreutlerAhlrichs": {"alpha": 0.6, "scale": "TreutlerXi"}}
M3 = {"GaussChebyshev": {"kind": "Second", "mapping": "TreutlerM3", "scale": "TreutlerXi"}}
PART = {"ssf": {"Ssf": {"a": 0.64}}, "becke": "Becke", "lko": {"Lko": {"r_cut": 5.0, "m_cut": 12}}}
ADJ = {"treutler": {"Treutler": "BraggPyscf"}, "becke": {"Becke": "BraggPyscf"}, "none": "None"}


def spec(nr=(40, 50, 55), ang=(194, 302, 302), regions=(26, 86, 194, 302, 194), part="ssf", adj="treutler", rad="m4", cutoff=1e-15):
    rows = [(nr[0], ang[0]), (nr[1], ang[1]), (nr[2], ang[2])]
    rows += [(nr[2] + 10, ang[2]), (nr[2] + 15, ang[2]), (nr[2] + 20, ang[2]), (nr[2] + 20, ang[2])]
    return {"kind": "spec", "radial": M4 if rad == "m4" else M3, "sizes": {"PerRow": [list(r) for r in rows]}, "overrides": [],
            "pruning": {"Regions": {"radii": "BraggPyscf", "alphas": NWCHEM, "grids": list(regions), "cap": True}},
            "partition": PART[part], "size_adjust": ADJ[adj], "weight_cutoff": cutoff,
            "orientation": "Identity", "batching": {"max_points": 256, "per_atom": False}}


def orca_like(changes=None):
    """GridSpec::orca_defgrid(2) with changes."""
    s = {"kind": "spec", "radial": M3, "sizes": {"OrcaLike": {"angular_grid": 4, "int_acc": 4.388, "b": 5.0, "h_reduced": True}},
         "overrides": [], "pruning": "OrcaRegions", "partition": "Becke", "size_adjust": {"Becke": "BraggPyscf"}, "weight_cutoff": 1e-14,
         "orientation": "Identity", "batching": {"max_points": 256, "per_atom": False}}
    s.update(changes or {})
    return s


VARIANTS = {
    # presets
    "orca2": {"kind": "orca", "defgrid": 2}, "orca3": {"kind": "orca", "defgrid": 3},
    "pyscf3": {"kind": "pyscf", "level": 3}, "pyscf5": {"kind": "pyscf", "level": 5}, "pyscf7": {"kind": "pyscf", "level": 7},
    "linus2": {"kind": "linus", "level": 2}, "linus3": {"kind": "linus", "level": 3},
    # one factor at a time around the old linus:2 (L2)
    "L2": spec(),
    "R30": spec(nr=(30, 35, 40)), "R35": spec(nr=(35, 40, 45)), "R45": spec(nr=(45, 60, 65)), "R50": spec(nr=(50, 65, 70)),
    "R60": spec(nr=(60, 80, 90)), "Rm3": spec(rad="m3"),
    "Pbecke_tr": spec(part="becke"), "Pbecke_be": spec(part="becke", adj="becke"), "Pbecke_no": spec(part="becke", adj="none"),
    "Plko_tr": spec(part="lko"), "Pssf_no": spec(adj="none"),
    "Aout302": spec(regions=(26, 86, 194, 302, 302)), "A50_110out": spec(regions=(50, 110, 194, 302, 302)),
    "AH302": spec(ang=(302, 302, 302)), "Aunpruned": spec(regions=(302, 302, 302, 302, 302)),
    "A434": spec(ang=(302, 434, 434), regions=(50, 194, 302, 434, 434)), "A434b": spec(ang=(302, 434, 434), regions=(26, 110, 302, 434, 302)),
    # orca:2 with one change
    "O2": orca_like(), "O2_m4": orca_like({"radial": M4}), "O2_tr": orca_like({"size_adjust": {"Treutler": "BraggPyscf"}}),
    "O2_ssf": orca_like({"partition": {"Ssf": {"a": 0.64}}}),
    "O2_out302": orca_like({"pruning": {"Regions": {"radii": "BraggPyscf", "alphas": NWCHEM, "grids": [26, 110, 194, 302, 302], "cap": True}}}),
    "O2_nr45": orca_like({"sizes": {"OrcaLike": {"angular_grid": 4, "int_acc": 5.05, "b": 5.0, "h_reduced": True}}}),
    # combinations
    "S1": spec(nr=(35, 40, 45), ang=(302, 434, 434), regions=(26, 110, 302, 434, 302), adj="none"),
    "S2": spec(nr=(35, 40, 45), ang=(302, 302, 302), regions=(26, 110, 194, 302, 194), adj="none"),
    "S3": spec(ang=(302, 302, 302), adj="none"),
    "S4": spec(ang=(302, 434, 434), regions=(26, 110, 302, 434, 302), adj="none"),
    "S5": spec(ang=(302, 434, 434), regions=(26, 110, 302, 434, 302), part="becke", adj="none"),
    "S6": spec(ang=(302, 434, 434), regions=(26, 110, 302, 434, 302), part="lko", adj="none"),
    "S7": spec(ang=(434, 590, 590), regions=(26, 110, 302, 590, 434), adj="none"),
    "S8": spec(nr=(35, 40, 45), ang=(194, 434, 434), regions=(26, 110, 302, 434, 302), adj="none"),
    "S9": spec(ang=(302, 302, 302), part="lko", adj="none"), "S10": spec(ang=(302, 302, 302), part="becke", adj="none"),
    "S11": spec(ang=(302, 434, 434), regions=(26, 110, 302, 434, 302), part="lko"),
    "S12": spec(ang=(434, 434, 434), regions=(26, 110, 302, 434, 434), part="lko", adj="none"),
    # around linus:3 (L3 = the old linus:3; L3p26 = the new one)
    "L3": spec(nr=(50, 65, 70), ang=(302, 434, 434), regions=(50, 194, 302, 434, 302)),
    "L3no": spec(nr=(50, 65, 70), ang=(302, 434, 434), regions=(50, 194, 302, 434, 302), adj="none"),
    "L3r45": spec(nr=(45, 55, 60), ang=(302, 434, 434), regions=(50, 194, 302, 434, 302)),
    "L3a302": spec(nr=(50, 65, 70), ang=(302, 302, 302), regions=(50, 194, 302, 302, 302)),
    "L3a302no": spec(nr=(50, 65, 70), ang=(302, 302, 302), regions=(50, 194, 302, 302, 302), adj="none"),
    "L3p26": spec(nr=(50, 65, 70), ang=(302, 434, 434), regions=(26, 110, 302, 434, 302)),
    # converged references with other partitionings
    "FINEssf": spec(nr=(80, 100, 110), ang=(590, 974, 974), regions=(110, 302, 590, 974, 974)),
    "FINEbno": spec(nr=(80, 100, 110), ang=(590, 974, 974), regions=(110, 302, 590, 974, 974), part="becke", adj="none"),
}
