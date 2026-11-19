# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""Tables of the workflow results (bench/results/2026-11-07-workflows/*.json) as Markdown.

    <venv>/bin/python bench/workflows/summarize.py > bench/results/2026-11-07-workflows/tables.md

Thermochemistry of both programs' Hessians is recomputed with linus.vib under one convention
(298.15 K, 1 atm, average atomic masses, symmetry number 1, imaginary modes left out; RRHO, and
quasi-RRHO = Grimme's entropy with a 100 cm^-1 cutoff), which reproduces ORCA's own G - E(el) to
a few µEh; ORCA's printed values are listed as well. Writes summary.json next to the tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402

KCAL = C.HARTREE_TO_KCAL
THERMO_OPTS = {"temperature": 298.15, "pressure": 101325.0}


def thermo(symbols, geometry, hessian, energy) -> dict | None:
    if hessian is None:
        return None
    from linus import vib

    va = vib.harmonic_analysis(list(symbols), np.asarray(geometry), C.hessian(hessian), mass_table="average",
                               symmetry_number=1)
    out = {"freqs": va.vibrational_frequencies.tolist(), "n_imag": int(sum(f < -20 for f in va.vibrational_frequencies))}
    for tag, opts in (("rrho", {}), ("qrrho", {"thermo": "qrrho"})):
        th = vib.thermochemistry(va, energy, multiplicity=1, **THERMO_OPTS, **opts)[0]  # one temperature
        out[tag] = {"zpe": th.zero_point_energy, "g": th.gibbs_free_energy, "h": th.enthalpy}
    return out


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def minima_table(summary: dict) -> list[str]:
    lines = ["| Molecule | Program / optimizer | Steps | Final E (Eh) | ΔE vs ORCA (kcal/mol) | ΔE vs lowest linus (µEh) | "
             "RMSD vs ORCA (Å) | Mean SCF iter. | Wall (s) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("la18", "u1an39"):
        o = C.load(f"orca_min_{name}")
        if not o:
            continue
        sym = [str(x) for x in C.structure(name).symbols]
        eo, go = o["opt"]["final_energy"], o["opt"]["final_geometry_bohr"]
        lines.append(f"| {name} | ORCA 6.0.0 (TightOpt) | {o['opt']['n_steps']} | {eo:.6f} | 0 | – | 0 | – | "
                     f"{o['opt']['wall_s']:.0f} |")
        lin = [C.load(f"linus_min_{name}_{opt}") for opt in ("geometric", "sella", "qcopt")]
        e_low = min((r["opt"]["final_energy"] for r in lin if r), default=None)
        summary.setdefault("minima", {})[f"{name}/orca"] = {"steps": o["opt"]["n_steps"], "energy": eo,
                                                             "wall_s": o["opt"]["wall_s"]}
        for opt in ("geometric", "sella", "qcopt"):
            r = C.load(f"linus_min_{name}_{opt}")
            if not r:
                continue
            p = r["opt"]
            e = p["final_energy"]
            its = [i for i in p["scf_iterations"] if i]
            rms = C.rmsd(p["final_geometry_bohr"], go) if p["final_geometry_bohr"] else None
            lines.append(f"| {name} | linus + {opt} | {p['n_steps']} | {e:.6f} | {(e - eo) * KCAL:+.3f} | "
                         f"{(e - e_low) * 1e6:.0f} | {fmt(rms, 3)} | {np.mean(its):.1f} | {p['wall_s']:.0f} |")
            summary["minima"][f"{name}/{opt}"] = {"steps": p["n_steps"], "energy": e, "de_kcal": (e - eo) * KCAL,
                                                  "rmsd_A": rms, "wall_s": p["wall_s"], "success": p["success"]}
    return lines


def freq_table(summary: dict) -> list[str]:
    lines = ["| Molecule | Program | Wall (s) | n_imag | Lowest 3 (cm⁻¹) | max abs Δν vs ORCA (cm⁻¹) | ZPE (kcal/mol) | "
             "G − E RRHO (kcal/mol) | G − E qRRHO (kcal/mol) | ORCA's own G − E (kcal/mol) |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name in ("la18", "u1an39"):
        o, l_ = C.load(f"orca_min_{name}"), C.load(f"linus_freq_{name}")
        rows = []
        sym = [str(x) for x in C.structure(name).symbols]
        if o and o.get("freq", {}).get("hessian"):
            f = o["freq"]
            t = thermo(sym, o["opt"]["final_geometry_bohr"], f["hessian"], f["energy"])
            rows.append(("ORCA 6.0.0", f["wall_s"], t, f["energy"], f["orca_thermo"].get("g_minus_eel")))
        if l_ and l_["freq"].get("hessian"):
            f = l_["freq"]
            m = C.load(f"linus_min_{name}_geometric")
            t = thermo(sym, m["opt"]["final_geometry_bohr"], f["hessian"], f["energy"])
            rows.append(("linus", f["wall_s"], t, f["energy"], None))
        ref = rows[0][2]["freqs"] if rows and rows[0][0].startswith("ORCA") else None
        for prog, wall, t, e, own in rows:
            dmax = None if ref is None or prog.startswith("ORCA") else float(np.max(np.abs(np.array(t["freqs"]) - ref)))
            lines.append(f"| {name} | {prog} | {wall:.0f} | {t['n_imag']} | {', '.join(f'{x:.1f}' for x in t['freqs'][:3])} | "
                         f"{fmt(dmax, 2)} | {t['rrho']['zpe'] * KCAL:.3f} | {(t['rrho']['g'] - e) * KCAL:.3f} | "
                         f"{(t['qrrho']['g'] - e) * KCAL:.3f} | {fmt(None if own is None else own * KCAL, 3)} |")
            summary.setdefault("freq", {})[f"{name}/{prog}"] = {"wall_s": wall, "thermo": {k: t[k] for k in ("rrho", "qrrho")},
                                                                  "max_dnu": dmax, "lowest": t["freqs"][:6]}
    return lines


def ts_tables(summary: dict, name: str = "ts61", level: str = "svp", forming: tuple[int, int] = (0, 1)) -> list[str]:
    """Stage and barrier tables of one TS workflow; `forming`: the atoms of the bond the reaction
    makes (the reactant end is where they are farther apart)."""
    suffix = "" if level == "svp" else f"_{level}"
    o, l_ = C.load(f"orca_ts_{name}{suffix}"), C.load(f"linus_ts_{name}{suffix}")
    sym = [str(x) for x in C.structure(name).symbols]
    out = []
    stages = ["| Stage | ORCA 6.0.0: steps / result | ORCA wall (s) | linus: steps / result | linus wall (s) |",
              "|---|---|---:|---|---:|"]

    def cell(rec, key, what):
        if rec and key == "tsopt" and key not in rec and "tsopt_partial" in rec:
            d = rec["tsopt_partial"]
            return (f"stopped after {d['n_steps']} steps (fmax {d['last_fmax_ev_per_A']:.3f} eV/Å, "
                    f"{d['energy_drop_kcal']:.1f} kcal/mol below the start)"), d["wall_s"]
        if rec and key == "optts" and rec.get(key, {}).get("stopped"):
            d = rec[key]
            drop = (d["energies"][0] - d["energies"][-1]) * KCAL
            return f"stopped after {d['n_steps']} steps ({drop:.1f} kcal/mol below the start)", d["wall_s"]
        if not rec or key not in rec:
            return "–", None
        d = rec[key]
        if what == "opt":
            return f"{d.get('n_steps')} steps{'' if d.get('success', True) else ' (failed)'}", d.get("wall_s")
        if what == "freq":
            fr = d.get("freqs_wavenumber") or []
            imag = [f for f in fr if f < -20]
            return f"{d.get('n_imaginary')} imaginary ({', '.join(f'{x:.0f}' for x in imag)})", d.get("wall_s")
        if what == "irc":
            n = d.get("n_frames_full") or d.get("n_steps")
            return f"{n} frames", d.get("wall_s")
        return "–", None

    for label, ok, lk, what in (("TS search", "optts", "tsopt", "opt"), ("TS search, continued from a fresh Hessian", "optts2", "tsopt2", "opt"), ("Frequencies at the TS", "freq_ts", "freq_ts", "freq"),
                                ("IRC (both directions)", "irc", "irc", "irc"),
                                ("Minimize end A", "opt_backward", "opt_reverse", "opt"),
                                ("Frequencies end A", "freq_backward", "freq_reverse", "freq"),
                                ("Minimize end B", "opt_forward", "opt_forward", "opt"),
                                ("Frequencies end B", "freq_forward", "freq_forward", "freq")):
        a, wa = cell(o, ok, what)
        b, wb = cell(l_, lk, what)
        stages.append(f"| {label} | {a} | {fmt(wa, 0)} | {b} | {fmt(wb, 0)} |")
    tot_o = o.get("total_wall_s") if o else None
    tot_l = l_.get("total_wall_s") if l_ else None
    stages.append(f"| **Whole TS workflow** | | **{fmt(tot_o, 0)}** | | **{fmt(tot_l, 0)}** |")
    out += stages + [""]

    # energies and barrier
    def species(rec, prog):
        if not rec:
            return {}
        keys = ({"ts": (rec.get("ts_from", "optts"), "freq_ts"), "A": ("opt_backward", "freq_backward"), "B": ("opt_forward", "freq_forward")}
                if prog == "orca" else
                {"ts": ("tsopt", "freq_ts"), "A": ("opt_reverse", "freq_reverse"), "B": ("opt_forward", "freq_forward")})
        res = {}
        for sp, (ko, kf) in keys.items():
            if ko not in rec or not rec[ko].get("final_geometry_bohr") or rec[ko].get("stopped"):
                continue
            geom = rec[ko]["final_geometry_bohr"]
            f = rec.get(kf) or {}
            e = f.get("energy") or rec[ko]["final_energy"]
            res[sp] = {"geometry": geom, "energy": e, "thermo": thermo(sym, geom, f.get("hessian"), e) if f.get("hessian") else None}
        return res

    so, sl = species(o, "orca"), species(l_, "linus")
    summary.setdefault("ts", {})
    if so or sl:
        out += ["| Program | E(TS) (Eh) | ΔE‡ (kcal/mol) | ΔE_rxn (kcal/mol) | ΔG‡ RRHO | ΔG‡ qRRHO | Imag. mode (cm⁻¹) | "
                "Reactant end | RMSD TS vs ORCA (Å) |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for prog, sp in (("ORCA 6.0.0", so), ("linus", sl)):
        if "ts" not in sp:
            continue
        ends = {k: v for k, v in sp.items() if k in ("A", "B")}
        if len(ends) < 2:
            out.append(f"| {prog} | {sp['ts']['energy']:.6f} | – | – | – | – | – | – | – |")
            continue
        # the reactant is the lower-energy end that is not the product: decided by forming bond below
        react, prod = reactant_product(sp, forming)
        ts = sp["ts"]
        de = (ts["energy"] - sp[react]["energy"]) * KCAL
        drxn = (sp[prod]["energy"] - sp[react]["energy"]) * KCAL
        dg = {}
        for tag in ("rrho", "qrrho"):
            if ts["thermo"] and sp[react]["thermo"]:
                dg[tag] = (ts["thermo"][tag]["g"] - sp[react]["thermo"][tag]["g"]) * KCAL
        imag = [f for f in (ts["thermo"] or {}).get("freqs", []) if f < -20]
        rms = C.rmsd(ts["geometry"], so["ts"]["geometry"]) if "ts" in so and prog == "linus" else 0.0
        out.append(f"| {prog} | {ts['energy']:.6f} | {de:.2f} | {drxn:+.2f} | {fmt(dg.get('rrho'), 2)} | "
                   f"{fmt(dg.get('qrrho'), 2)} | {', '.join(f'{x:.0f}' for x in imag)} | {react} | {fmt(rms, 3)} |")
        summary["ts"][f"{name}{suffix}/{prog}"] = {"dE": de, "dE_rxn": drxn, "dG": dg, "imag": imag, "reactant": react, "rmsd_ts_vs_orca": rms}
    return out




def reactant_product(sp: dict, forming: tuple[int, int]) -> tuple[str, str]:
    """Reactant end = the end where the bond made in the reaction is longer (not yet formed)."""
    i, j = forming
    d = {k: float(np.linalg.norm(np.asarray(sp[k]["geometry"])[i] - np.asarray(sp[k]["geometry"])[j]))
         for k in ("A", "B")}
    return ("A", "B") if d["A"] > d["B"] else ("B", "A")


def sp_table(summary: dict) -> list[str]:
    lines = ["| Structure (geometry from) | ORCA RIJCOSX E (Eh) | ORCA wall (s) | linus RI-JK E − ORCA (µEh) | wall (s) | "
             "linus RIJ+COSX E − ORCA (µEh) | wall (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    labels = ["la18_min", "u1an39_min", "ts61_ts", "ts61_backward", "ts61_forward"]
    for lab in labels:
        o = C.load(f"orca_sp_{lab}")
        if not o:
            continue
        cells = []
        for jk in ("df", "rij_snk"):
            r = C.load(f"linus_sp_orcageom_{lab}_{jk}")
            cells.append((r["energy"] - o["energy"]) * 1e6 if r and r.get("energy") else None)
            cells.append(r["wall_s"] if r else None)
        lines.append(f"| {lab} (ORCA) | {o['energy']:.6f} | {o['wall_s']:.0f} | {fmt(cells[0], 0)} | {fmt(cells[1], 0)} | "
                     f"{fmt(cells[2], 0)} | {fmt(cells[3], 0)} |")
        summary.setdefault("sp", {})[lab] = {"orca": o["energy"], "orca_wall": o["wall_s"], "df_minus_orca_uEh": cells[0],
                                             "df_wall": cells[1], "cosx_minus_orca_uEh": cells[2], "cosx_wall": cells[3]}
    return lines


def minima_workflow_table(summary: dict) -> list[str]:
    """Whole minimum workflow: optimization from the perturbed start + frequencies (+ TZVP single
    point on the optimized structure)."""
    lines = ["| Molecule | ORCA opt + freq (s) | ORCA TZVP SP (s) | ORCA total (s) | linus opt + freq (s) | "
             "linus TZVP SP RI-JK (s) | linus total (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("la18", "u1an39"):
        o, lo, lf = C.load(f"orca_min_{name}"), C.load(f"linus_min_{name}_geometric"), C.load(f"linus_freq_{name}")
        osp, lsp = C.load(f"orca_sp_{name}_min"), C.load(f"linus_sp_linusgeom_{name}_min_df")
        if not (o and lo and lf):
            continue
        ow = o["opt"]["wall_s"] + o["freq"]["wall_s"]
        lw = lo["opt"]["wall_s"] + lf["freq"]["wall_s"]
        osw = osp["wall_s"] if osp else 0.0
        lsw = lsp["wall_s"] if lsp else 0.0
        lines.append(f"| {name} | {ow:.0f} | {fmt(osp and osw, 0)} | {ow + osw:.0f} | {lw:.0f} | {fmt(lsp and lsw, 0)} | "
                     f"{lw + lsw:.0f} |")
        summary.setdefault("minima_workflow", {})[name] = {"orca_s": ow + osw, "linus_s": lw + lsw}
    return lines


def pcm_table(summary: dict) -> list[str]:
    """wB97X-D3BJ/def2-TZVP in THF (C-PCM) on ORCA's minima; the solvation shift is
    E(THF) - E(gas) of the same program and J/K method at the same geometry."""
    lines = ["| Structure | ORCA CPCM(THF) E (Eh) | ORCA ΔE_solv (kcal/mol) | ORCA wall (s) | linus J/K | linus E − ORCA (µEh) | "
             "linus ΔE_solv (kcal/mol) | linus wall (s) |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for lab in ("la18_min", "u1an39_min"):
        o, og = C.load(f"orca_sppcm_{lab}"), C.load(f"orca_sp_{lab}")
        if not o:
            continue
        o_solv = (o["energy"] - og["energy"]) * KCAL if og else None
        for jk, name in (("df", "RI-JK"), ("rij_snk", "RIJ+COSX")):
            r, rg = C.load(f"linus_sppcm_orcageom_{lab}_{jk}"), C.load(f"linus_sp_orcageom_{lab}_{jk}")
            if not r or not r.get("energy"):
                continue
            l_solv = (r["energy"] - rg["energy"]) * KCAL if rg else None
            lines.append(f"| {lab} | {o['energy']:.6f} | {fmt(o_solv, 3)} | {o['wall_s']:.0f} | {name} | "
                         f"{(r['energy'] - o['energy']) * 1e6:.0f} | {fmt(l_solv, 3)} | {r['wall_s']:.1f} |")
            summary.setdefault("pcm", {})[f"{lab}/{jk}"] = {"orca": o["energy"], "linus": r["energy"],
                                                            "orca_solv_kcal": o_solv, "linus_solv_kcal": l_solv}
    return lines


def guess_table(summary: dict) -> list[str]:
    lines = ["| Molecule | Path | Steps | Mean SCF iterations | Wall (s) |", "|---|---|---:|---:|---:|"]
    for name in ("la18", "u1an39"):
        g = C.load(f"guess_reuse_{name}")
        if not g:
            continue
        for k, lab in (("sad", "geomeTRIC via qccompute, SAD every step"),
                       ("propagate", "geomeTRIC via qccompute, propagate_wfn=True")):
            r = g["geometric"][k]
            its = [s["scf_iterations"] for s in r["steps"][1:]]
            lines.append(f"| {name} | {lab} | {r['n_steps']} | {np.mean(its):.2f} | {r['wall_s']:.1f} |")
        for k, lab in (("sad", "replay: SAD"), ("density", "replay: previous density as is"),
                       ("orbitals", "replay: previous orbitals, Löwdin-orthonormalized")):
            sm = g["replay_summary"][k]
            lines.append(f"| {name} | {lab} | {sm['steps']} | {sm['mean_iterations']:.2f} | {sm['total_wall_s']:.1f} |")
        summary.setdefault("guess", {})[name] = {k: g["replay_summary"][k] for k in g["replay_summary"]}
    return lines


def screen_table(summary: dict) -> list[str]:
    lines = ["| Mode | Structures | Wall (s) | s per structure | Sum of linus totals (s) |", "|---|---:|---:|---:|---:|"]
    for mode in ("inproc", "qc_t8", "qc_t4", "qc_t2", "qc_t1", "orca_t8", "orca_t1"):
        r = C.load(f"screen_{mode}")
        if not r:
            continue
        lines.append(f"| {mode} | {r['n']} | {r['wall_s']:.1f} | {r['per_structure_s']:.2f} | {fmt(r.get('linus_sum_s'), 1)} |")
        summary.setdefault("screen", {})[mode] = {"wall_s": r["wall_s"], "per_structure_s": r["per_structure_s"]}
    return lines


def main() -> None:



    summary: dict = {}
    parts = [("Guess reuse", guess_table), ("Minimizations", minima_table), ("Frequencies at the minima", freq_table),
             ("Whole minimum workflows (wall, s)", minima_workflow_table),
             ("Production single points in THF (C-PCM, ORCA geometries)", pcm_table),
             ("Transition state: ts61, B3LYP-D3(BJ)/def2-SVP", ts_tables),
             ("Transition state: HCN to HNC, wB97X-D3BJ/def2-SVP",
              lambda sm: ts_tables(sm, "hcn_ts", "wb97x-svp", forming=(2, 1))), ("Production single points (ORCA geometries)", sp_table),
             ("Conformer screen", screen_table)]
    for title, fn in parts:
        print(f"### {title}\n")
        print("\n".join(fn(summary)))
        print()
    C.save("summary", summary)


if __name__ == "__main__":
    main()
