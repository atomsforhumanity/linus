#!/usr/bin/env python3
"""The GPU backend against the CPU backend of the same linus build, end to end through the CLI.

Every case runs twice, ``--backend cpu`` and ``--backend gpu`` (a build with the ``gpu``
feature), and the results are compared at the tolerances of docs/gpu.md:

- energies within 1e-10 Eh, with the same SCF iteration counts;
- gradients within 1e-9 Eh/bohr;
- Hessians within 1e-10 Eh/bohr^2, harmonic frequencies within 1e-6 cm^-1, the same CPHF/CPKS
  iteration counts (one exception below); with ``--ir`` polarizabilities within 1e-8 au, dipole
  derivatives within 1e-8 au and IR intensities within 1e-8 of the largest: all three are
  stationary in the CPHF residual like the Hessian (the polarizability through the Galerkin
  solver, the dipole derivatives through their U^c . r^x term, decision D35), so the solves'
  round-off differences enter only quadratically (measured: dipole derivatives 4.7e-10 au, IR
  7.7e-10 of the largest, polarizabilities 2.6e-10 au at response.field_tol 1e-6; before D35 the
  dipole derivatives, linear in the residual, differed by up to 2.9e-7 au);
- the CPHF/CPKS iteration counts may differ by one when the side with fewer iterations stopped
  at the threshold (its final max residual within [tol/2, 2 tol]): the stopping test is
  discrete and the shared-subspace solver's residual is round-off sensitive (it drops
  near-dependent directions and converged columns; GPU and CPU final residuals differ by 1-25 %
  at the same iteration count), so a residual at the threshold can need one more iteration on
  either backend. Such a flip is shown in the case's line (``cphf_iterations=+1 (cpu 9 at
  6.8e-07, gpu 10; tol 1e-06)``) and counted in the summary; it never excuses an accuracy check
  (Hessian, frequencies, IR), and the SCF iteration counts stay strict (the default-precision
  cases below allow one either way);
- the -V functionals' cases (VV10 on the device) at a tighter bar, energies
  and gradients within 1e-11 (two slowly converging open-shell gradients within 1e-10, the
  reason at ``VV10_SLOW_SCF``);
- ``resolved.backend`` places each family of kernels as expected, fallbacks included;
- cases marked ``repeat`` run on the GPU a second time and must agree bit for bit;
- the ``*.dloop.*`` cases run the GPU side with ``backend.scf_loop: "device"`` (the SCF loop on
  the device). Open shells take the host loop's stability decisions there (the probe on the
  device; a saddle point's analysis continues on the host loop from the device's state): the
  ``stability`` check compares the decisions (each analysis' iteration, probe or full, step and
  Fock builds) with the CPU's;
- the closed-shell probe (D36): where both sides record ``resolved.scf.closed_shell_probe``, the
  ``closed_shell_probe`` check wants the same gate decision and, where it ran, the same verdict,
  kind and Fock builds and the Ritz value within ``TOL_PROBE``. At the default thresholds the
  probe's one-sided products carry the converged point's residual over h, which the backends'
  SCF paths leave differently (p-benzyne: 2e-6 Eh; 1e-11 at the extreme preset). The
  ``pbenzyne.*`` case (singlet p-benzyne, RKS gap 0.067 Eh, a broken-symmetry UKS solution 4e-2 Eh
  lower) runs the probe in the device loop and must flag it on both sides;
- the ``*.pcm*`` cases add an implicit solvent (THF; C-PCM, IEF-PCM or SMD, the ``pyscf`` or
  ``orca`` preset, the default ``predicted`` SCF mode or ``exact``): its surface terms run on
  the device (``pcm: gpu``), the PCM's Hessian terms on the CPU;
- the ``*.emu.*`` cases run the GPU side with ``precision.fp64: "emulated"`` (the exchange
  GEMMs in FP64 emulation on INT8 tensor cores) at the same tolerances, and
  ``resolved.backend.emulation`` must report the Ozaki scheme (7 slices, 55 bits) for the SCF's
  J/K and its density-fitting fit and, for Hessians, the explicit two-electron terms' and the
  response's J/K (the latter cuBLAS's 55-bit
  emulation when the response's exchange runs on the stored operator, which needs no Ozaki
  GEMM1 planes);
- Hessians check ``resolved.backend.response``: the response J/K's arithmetic (``fp64``,
  ``emulated``), the CPKS kernel's (``fp64``; ``fp32`` at the default precision where the CPU
  runs single precision), and the form of the Hessian's exchange (the stored operator: the
  matrix's Hessians all fit the device; at the default, auto's choice, below);
- the ``*.ir.freq`` cases cover Hessians, frequencies, IR intensities and polarizabilities of
  water, la18 and u1an39 with B3LYP and wB97X-D3BJ, closed and open shells, in FP64 and
  emulated;
- the ``*.default.freq`` cases keep linus's default ``precision.response`` (``auto``) on both
  backends. Where the CPU then runs single precision (la18 and u1an39: its exchange in FP32,
  two levels for large systems, its kernel in FP64), the GPU runs its CPKS kernel in FP32 and
  its exchange on the stored operator in FP64 or, on FP64-poor cards where its cost model
  estimates it cheaper (decision D37, ``estimate_s``; the response check expects that path), in
  FP32 through the tensor, so they are compared at the default-precision tolerances
  (``TOL_DEFAULT``: frequencies within 3e-3 cm^-1, IR intensities within 1e-3 of the largest,
  Hessians as measured) with the CPHF iterations within one. These cover the CPU default's own
  single-precision error, the larger of the two; where the same molecule's ``.ir.freq`` (or
  ``.freq``) case ran into the same ``--out``, the GPU's default is also checked against the
  CPU's FP64 run at the tighter ``TOL_DEFAULT_FP64`` (``TOL_DEFAULT_FP32`` after the FP32
  exchange). ``u1an39.b3lyp.rks.tzvp.default.freq`` (no IR, so no field solve that the operator
  would serve) exercises the FP32 exchange at the size where it is the faster path, and the two
  ``*.auto.default.freq`` cases (u1an39 with IR, its def2-TZVP twin without) repeat the choice at
  the card's own defaults, the operator's setup in FP64 emulation as users run it;
- the GPU side pins the arithmetic (``precision: "fp64"``, native FP64 with ``precision.fp32:
  "off"``; ``precision.fp64: "emulated"`` where the case names ``emulated``; the
  ``*.default.freq`` cases pin only ``precision.fp64: "native"``, since ``precision.fp32: "off"``
  would make their response FP64) and ``backend.scf_loop`` (``host`` unless ``device``): their
  defaults depend on the card and the shell (decision D27), and the matrix covers the same
  arithmetic and loop on every card. The ``*.auto.*`` cases pin nothing:
  ``resolved.settings.precision.fp64`` must be ``auto``'s choice for the card (``emulated`` on
  FP64-poor cards, ``native`` on FP64-rich ones: ``auto_precision``), reported in
  ``resolved.reasons["precision.fp64"]`` and ``resolved.record.backend.precision`` (``emulated``
  or ``double``), with the emulation tiers of the ``*.emu.*`` cases where it emulates, and
  ``scf_loop`` the device loop for closed and open shells (decisions D27, D31).

    cargo build --release -p linus-cli --features gpu
    python3 reference/validate_gpu.py --linus-bin target/release/linus           # the matrix
    python3 reference/validate_gpu.py --cases 'water.*' --threads 8
    python3 reference/validate_gpu.py --prefix 'taskset -c 0-7' --out gpu-validation
    python3 reference/validate_gpu.py --cpu-store            # CPU results from the reference store

The CPU side need not be recomputed when the CPU code's results have not changed:
``--cpu-store`` looks each CPU result up in bench/refstore.py's linus-cpu store (keyed by the
calculation's inputs, the glibc version and the commit's CPU numerics epoch, from
``$LINUS_CPU_EPOCHS``), computes and stores it on a miss, and reports which (``--store-misses``
lists what is missing, ``--cpu-only`` fills the store without a GPU, ``--import-cpu`` stores an
earlier run's outputs); ``--cpu-from`` takes them from an earlier run's ``--out``.

Results go to ``<out>/results.json`` (one record per case: the differences, their verdicts and
the timings of both runs) and a table on stdout; the exit code is the number of failed cases.
The response settings are ``precision.response: fp64`` for the Hessians (both backends then
compute their response J/K in FP64, one level), so the comparison measures the device
arithmetic, not the CPU's two-level single-precision default.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MOLS = {
    "water": REPO / "tests/molecules/water.xyz",
    "la18": REPO / "bench/molecules/la18.xyz",
    "u1an39": REPO / "bench/molecules/u1an39.xyz",
    "ts61": REPO / "bench/molecules/ts61.xyz",
}
# molecules of single cases, outside the per-molecule loops of the matrix
CASE_MOLS = {
    "pbenzyne": REPO / "tests/molecules/p_benzyne.xyz",
}
# (charge, multiplicity) of the closed shell and of the open shell (a cation; u1an39's neutral
# radical, the molecule being an anion)
SPINS = {
    "water": ((0, 1), (1, 2)),
    "la18": ((0, 1), (1, 2)),
    "u1an39": ((-1, 1), (0, 2)),
    "ts61": ((0, 1), (1, 2)),
}
# hessian, freq: the explicit terms on the device; the Hessian is stationary in the
# response. alpha, dipder, ir: stationary too (D35: the dipole derivatives' U^c . r^x
# term; alpha through the Galerkin solver), so the solves' round-off enters only quadratically:
# worst over water, la18 and u1an39 (B3LYP, wB97X-D3BJ, closed and open shells, FP64 and emulated)
# dipder 4.7e-10 au (la18 B3LYP UKS emulated), ir 7.7e-10 of the largest intensity (u1an39 B3LYP
# UKS), alpha 2.6e-10 au (la18 wB97X-D3BJ UKS, response.field_tol 1e-6); before D35 dipder and ir
# were linear in the residual and needed 1e-6 (u1an39 wB97X-D3BJ dipder 2.9e-7 au)
# the closed-shell probe's Ritz value, CPU against GPU (Eh; module docs)
TOL_PROBE = 1e-5
TOL = {"energy": 1e-10, "gradient": 1e-9, "hessian": 1e-10, "freq": 1e-6, "alpha": 1e-8, "dipder": 1e-8,
       "ir": 1e-8}
# default response precision, the GPU's default against the CPU's. It must cover the CPU
# default's own error, the larger one by far: its two-level single-precision response (la18
# wB97X-D3BJ UKS against the CPU's FP64 run: frequencies 1.97e-3 cm^-1 on the cation's 629i mode
# with glibc 2.43 and 1.59e-3 with glibc 2.36, whose erf rounding moves the single-precision
# solve; IR 4e-4 of the largest intensity; Hessian 6.7e-7). The GPU default there is within
# 8.9e-5 cm^-1 and 2.3e-8 of FP64 on an FP64-poor card (its FP32 CPKS kernel) and 6e-8 cm^-1 on
# an FP64-rich one, checked separately at TOL_DEFAULT_FP64 below. The bar is the same on every
# card (main's decision, 2026-11-11: 2e-3 -> 3e-3; 2e-3 came from a 1.1e-3 measurement before the
# BLAS-3 CPHF solver and response.field_tol 1e-6 moved the CPU default's error)
TOL_DEFAULT = {"hessian": 1e-6, "freq": 3e-3, "alpha": 1e-8, "ir": 1e-3}
# the GPU's default against the CPU's FP64 run of the same molecule (the ``.ir.freq`` case): the
# device's own default-precision error, 3-5x the worst of the five measured cases (
# Hessians 2.3e-8 and frequencies 8.3e-5 cm^-1 at la18 wB97X-D3BJ UKS, IR 1.3e-6 of the largest
# intensity at la18 B3LYP; the others 1.5e-9..4.7e-9, 2.9e-6..7.3e-6 cm^-1, 5.6e-8..4.4e-7)
TOL_DEFAULT_FP64 = {"hessian": 1e-7, "freq": 3e-4, "ir": 5e-6}
# the same, where the GPU's auto ran the response exchange in FP32 through the tensor (decision D37:
# on FP64-poor cards, when its cost model finds that path cheaper than the stored FP64 operator;
# ``resolved.backend.response``'s ``exchange: "tensor"``, ``jk: "fp32"``): the FP32 exchange's
# error against the operator, measured on the RTX PRO 6000 at Hessians 2.6e-8..4.6e-8 Eh/bohr^2
# and frequencies 2.1e-5..1.05e-4 cm^-1 with def2-SVP (la18, u1an39, ts61; IR 3.1e-7 of the
# largest intensity at u1an39) and 1.8e-7 Eh/bohr^2 and 6.9e-4 cm^-1 at u1an39/def2-TZVP; against the
# CPU's FP64 run, the AFH RTX gate (b1ac8f3) found u1an39/def2-TZVP at 2.2e-7 and 5.8e-4 cm^-1 and
# u1an39's UKS radical at IR 5.4e-6 of the largest intensity (520 km/mol); the bars 2-5x that, still at
# or below the CPU default's own single-precision error (TOL_DEFAULT)
TOL_DEFAULT_FP32 = {"hessian": 1e-6, "freq": 1.5e-3, "ir": 2e-5}

# where each family of kernels must run on the GPU backend (jk df; None: not applicable)
GLOBAL = {"jk": "gpu", "xc": "gpu", "gradient_jk": "gpu", "gradient_xc": "gpu",
          "response_jk": "gpu", "response_xc": "gpu", "hessian_jk": "gpu",
          "hessian_xc": "gpu"}
# range-separated hybrids: the long-range exchange on the device too
RSH = dict(GLOBAL)
HF = dict(GLOBAL, xc=None, gradient_xc=None, response_xc=None, hessian_xc=None)
# -V functionals: VV10's energy, SCF potential and gradient on the device
VV10 = {"jk": "gpu", "xc": "gpu", "vv10": "gpu", "gradient_jk": "gpu", "gradient_xc": "gpu"}
# their Hessians, frequencies and polarizabilities: every family on the device,
# VV10's response pair sums too (its NLC-grid passes on the CPU)
VV10_HESS = dict(GLOBAL, vv10="gpu")
VV10_METHODS = ("wb97m-v", "wb97x-v", "b97m-v")
# meta-GGAs: every family on the device, the Hessian's explicit XC terms (tau)
# included; r2SCAN and the range-separated meta-GGA hybrid wB97M-D3BJ
MGGA_METHODS = ("r2scan", "wb97m-d3bj")
# r2SCAN's Hessians follow the CPHF residual far more than the other functionals': at the default
# response.tol 1e-6 the shared-subspace solver stopped water r2SCAN RKS after the same 6 iterations at
# max residual 7.9e-7 on the CPU and 1.3e-7 on the GPU (round-off moves it), and the Hessians differ
# by 3.4e-9 Eh/bohr^2 (dipole derivatives 1.9e-6 au) whether the Hessian's explicit XC terms run on
# the device or on the CPU; at 1e-9 they agree within 3.2e-12. The device's explicit XC terms and
# CPKS kernel match the CPU's to round-off at equal inputs (3e-13 and 2e-15), so the r2SCAN cases
# converge the response to 1e-9 on both backends: the comparison then measures the device's
# arithmetic, not where the solver stopped (as precision.response fp64 does for the J/K)
MGGA_CPHF_TOL = {"r2scan": ["--set", "response.tol=1e-9"]}
# an implicit solvent: its surface terms on the device
PCM_MODELS = {"cpcm": [], "iefpcm": ["--solvent-model", "iefpcm"], "smd": ["--solvent-model", "smd"]}
PCM_PRESETS = {"pyscf": [], "orca": ["--set", "solvent.preset=orca"]}


def auto_precision(cc: str) -> str:
    """``precision.fp64: auto``'s choice for a compute capability (decision D27; linus-gpu's
    ``DeviceInfo::fp64_rich``), in the words of ``resolved.record.backend.precision``: double
    (``precision.fp64`` native) on FP64-rich cards (no INT8 tensor cores below 7.2, or A100 8.0
    and H100/H200 9.0), emulated on the others."""
    major, minor = (int(x) for x in cc.split("."))
    rich = (major, minor) < (7, 2) or (major, minor) in ((8, 0), (9, 0))
    return "double" if rich else "emulated"


def pcm_case(mol: str, method: str, shell: int, calc: str, model: str = "cpcm", preset: str = "pyscf",
             scf: str = "", loop: str = "", repeat: bool = False, precision: str = "") -> dict:
    charge, mult = SPINS[mol][shell]
    spin = "uks" if shell else "rks"
    place = dict({"b3lyp": GLOBAL, "wb97x-d3bj": RSH, "hf": HF, **{m: VV10 for m in VV10_METHODS}}[method],
                 pcm="gpu")
    tag = "pcm" + ("" if model == "cpcm" else f"-{model}") + ("" if preset == "pyscf" else f"-{preset}") \
        + (f"-{scf}" if scf else "")
    extra = ["--solvent", "thf"] + PCM_MODELS[model] + PCM_PRESETS[preset] + (["--set", f"solvent.scf_terms={scf}"] if scf else [])
    emu = ".emu" if precision == "emulated" else ""
    c = dict(name=f"{mol}.{method}.{spin}.{tag}{emu}{'.dloop' if loop else ''}.{calc}", mol=mol, charge=charge,
             mult=mult, method=method, basis="def2-svp", calc=calc, place=place, extra=extra, repeat=repeat)
    if precision:
        c["precision"] = precision
    if loop:
        c["place"] = dict(place, scf_loop=loop)
        c["scf_loop"] = loop
    return c


def pcm_cases() -> list[dict]:
    out = []
    # C-PCM (pyscf preset, predicted): every molecule, method and shell; energies and gradients
    for mol in ("water", "la18", "u1an39"):
        for method in ("b3lyp", "wb97x-d3bj"):
            for shell in (0, 1):
                out.append(pcm_case(mol, method, shell, "energy", repeat=(mol, method, shell) == ("la18", "b3lyp", 0)))
                out.append(pcm_case(mol, method, shell, "gradient",
                                    repeat=(mol, method, shell) == ("la18", "b3lyp", 0)))
    # the models at both presets (water and la18, B3LYP, closed and open shells)
    for mol in ("water", "la18"):
        for model in PCM_MODELS:
            for preset in PCM_PRESETS:
                if (model, preset) == ("cpcm", "pyscf"):
                    continue
                for shell in (0, 1):
                    out.append(pcm_case(mol, "b3lyp", shell, "energy", model, preset))
                out.append(pcm_case(mol, "b3lyp", 0, "gradient", model, preset))
    # exact surface terms (two passes per iteration)
    for mol in ("water", "la18"):
        for shell in (0, 1):
            out.append(pcm_case(mol, "b3lyp", shell, "energy", scf="exact"))
    out.append(pcm_case("la18", "b3lyp", 0, "gradient", scf="exact"))
    # the device SCF loop with the reaction field on the device
    out.append(pcm_case("water", "b3lyp", 0, "energy", loop="device"))
    out.append(pcm_case("water", "b3lyp", 1, "energy", loop="device"))
    out.append(pcm_case("la18", "wb97x-d3bj", 0, "energy", loop="device", repeat=True))
    out.append(pcm_case("la18", "b3lyp", 0, "energy", "iefpcm", loop="device"))
    out.append(pcm_case("u1an39", "b3lyp", 0, "energy", loop="device"))
    out.append(pcm_case("la18", "b3lyp", 0, "gradient", loop="device"))
    # a Hessian: the PCM's second-derivative and response terms on the CPU
    out.append(pcm_case("la18", "b3lyp", 0, "freq"))
    # with the exchange in FP64 emulation (precision.fp64 "emulated"): the reaction field and
    # the Ozaki exchange in one Fock build
    for method in ("b3lyp", "wb97x-d3bj"):
        for shell in (0, 1):
            out.append(pcm_case("la18", method, shell, "energy", precision="emulated",
                                repeat=(method, shell) == ("b3lyp", 0)))
    out.append(pcm_case("u1an39", "b3lyp", 0, "energy", precision="emulated"))
    out.append(pcm_case("u1an39", "wb97x-d3bj", 0, "energy", precision="emulated"))
    out.append(pcm_case("la18", "b3lyp", 0, "gradient", precision="emulated"))
    out.append(pcm_case("la18", "b3lyp", 0, "energy", "iefpcm", precision="emulated"))
    out.append(pcm_case("la18", "b3lyp", 0, "energy", "smd", precision="emulated"))
    out.append(pcm_case("la18", "b3lyp", 0, "energy", scf="exact", precision="emulated"))
    out.append(pcm_case("la18", "wb97x-d3bj", 0, "energy", loop="device", precision="emulated"))
    out.append(pcm_case("la18", "b3lyp", 0, "freq", precision="emulated"))
    # -V functionals with C-PCM: VV10 on the device beside the reaction field (energies and
    # gradients, closed and open shells; the device loop)
    for mol in ("la18", "u1an39"):
        for shell in (0, 1):
            for calc in ("energy", "gradient"):
                out.append(pcm_case(mol, "wb97m-v", shell, calc))
    out.append(pcm_case("la18", "b97m-v", 0, "gradient"))
    out.append(pcm_case("la18", "wb97x-v", 1, "energy"))
    out.append(pcm_case("la18", "wb97m-v", 0, "energy", loop="device"))
    return out


def vv10_cases() -> list[dict]:
    """-V functionals: VV10's energy, SCF potential and gradient on the device,
    against the CPU; water, la18 and u1an39, RKS and UKS, the three functionals; the device loop;
    Hessians with VV10's response pair sums on the device."""
    out = []
    for mol in ("water", "la18", "u1an39"):
        for method in VV10_METHODS:
            for shell in (0, 1):
                charge, mult = SPINS[mol][shell]
                spin = "uks" if shell else "rks"
                for calc in ("energy", "gradient"):
                    out.append(dict(name=f"{mol}.{method}.{spin}.{calc}", mol=mol, charge=charge, mult=mult,
                                    method=method, basis="def2-svp", calc=calc, place=VV10,
                                    repeat=(mol, method, shell) == ("la18", "wb97m-v", 0)))
    # the device loop (VV10 in its Fock terms)
    for mol in ("water", "la18", "u1an39"):
        for method in ("wb97m-v", "b97m-v"):
            charge, mult = SPINS[mol][0]
            out.append(dict(name=f"{mol}.{method}.rks.dloop.energy", mol=mol, charge=charge, mult=mult,
                            method=method, basis="def2-svp", calc="energy", place=dict(VV10, scf_loop="device"),
                            scf_loop="device", repeat=(mol, method) == ("la18", "wb97m-v")))
    out.append(dict(name="la18.wb97m-v.rks.dloop.gradient", mol="la18", charge=0, mult=1, method="wb97m-v",
                    basis="def2-svp", calc="gradient", place=dict(VV10, scf_loop="device"), scf_loop="device"))
    # open shells in the device loop (the probe on the device, a saddle point's analysis on the
    # host loop; VV10 staying on the device)
    for mol in ("water", "la18"):
        charge, mult = SPINS[mol][1]
        for method in ("b97m-v", "wb97m-v"):
            out.append(dict(name=f"{mol}.{method}.uks.dloop.energy", mol=mol, charge=charge, mult=mult,
                            method=method, basis="def2-svp", calc="energy", place=dict(VV10, scf_loop="device"),
                            scf_loop="device"))
    # Hessians, frequencies, IR intensities and polarizabilities: water and la18,
    # RKS and UKS
    for mol, method, shell, ir in (("water", "wb97m-v", 0, True), ("water", "b97m-v", 1, True),
                                   ("la18", "wb97m-v", 0, False), ("la18", "wb97x-v", 1, False)):
        charge, mult = SPINS[mol][shell]
        spin = "uks" if shell else "rks"
        c = dict(name=f"{mol}.{method}.{spin}.freq", mol=mol, charge=charge, mult=mult, method=method,
                 basis="def2-svp", calc="freq", ir=ir, place=VV10_HESS)
        if method == "b97m-v":
            # no exact exchange, so its response reports none (as r2SCAN's)
            c["exchange"] = None
        out.append(c)
    return out


# The bar for the -V cases: VV10 on the device (double-float pair sums by default) must be
# FP64-equal, energies and gradients within 1e-11 of the CPU (tighter than the global TOL)
TOL_VV10 = {"energy": 1e-11, "gradient": 1e-11}
# except the la18 cation's ωB97M-V and ωB97X-V UKS gradients: their SCF takes 58-61 iterations at
# the gradient's convergence through a flat region, and the GPU and CPU runs stop at slightly
# different densities; main 324bfbe's GPU backend, VV10 still on the CPU, is as far from the CPU
# (4.3e-11 and 4.4e-11 Eh/bohr; device VV10 FP64 4.3e-11 and 5.6e-11, double-float 4.0e-11 and
# 4.2e-11). main's decision: a per-case bound, as for la18 HF UKS Hessians
VV10_SLOW_SCF = {"la18.wb97m-v.uks.gradient": 1e-10, "la18.wb97x-v.uks.gradient": 1e-10}


def mgga_cases() -> list[dict]:
    """Meta-GGA frequencies: the Hessian's explicit XC terms with their tau parts
    on the device, against the CPU; water and la18, r2SCAN (its response to 1e-9, ``MGGA_CPHF_TOL``)
    and wB97M-D3BJ, RKS and UKS (water r2SCAN RKS with IR intensities). r2SCAN has no exact
    exchange, so its response reports none."""
    out = []
    for mol in ("water", "la18"):
        for method in MGGA_METHODS:
            for shell in (0, 1):
                charge, mult = SPINS[mol][shell]
                spin = "uks" if shell else "rks"
                c = dict(name=f"{mol}.{method}.{spin}.freq", mol=mol, charge=charge, mult=mult, method=method,
                         basis="def2-svp", calc="freq", place=GLOBAL, extra=MGGA_CPHF_TOL.get(method, []),
                         ir=(mol, method, shell) == ("water", "r2scan", 0),
                         repeat=(mol, method, shell) == ("la18", "r2scan", 0))
                if method == "r2scan":
                    c["exchange"] = None
                out.append(c)
    return out


def with_vv10_bar(cases: list[dict]) -> list[dict]:
    for c in cases:
        if c["method"] in VV10_METHODS:
            c.setdefault("energy_tol", TOL_VV10["energy"])
            c.setdefault("gradient_tol", VV10_SLOW_SCF.get(c["name"], TOL_VV10["gradient"]))
    return cases


def cases() -> list[dict]:
    out = []
    # the device SCF loop (scf_loop "device")
    for mol in ("water", "la18", "u1an39"):
        for method, place in [("hf", HF), ("b3lyp", GLOBAL), ("wb97x-d3bj", RSH)]:
            for shell in (0, 1):
                charge, mult = SPINS[mol][shell]
                spin = "uks" if shell else "rks"
                loop = "device"
                out.append(dict(name=f"{mol}.{method}.{spin}.dloop.energy", mol=mol, charge=charge, mult=mult,
                                method=method, basis="def2-svp", calc="energy", place=dict(place, scf_loop=loop),
                                scf_loop="device", repeat=(mol == "la18" and method in ("b3lyp", "hf"))))
    out.append(dict(name="la18.b3lyp.rks.dloop.gradient", mol="la18", charge=0, mult=1, method="b3lyp",
                    basis="def2-svp", calc="gradient", place=dict(GLOBAL, scf_loop="device"), scf_loop="device"))
    out.append(dict(name="la18.b3lyp.rks.dloop.freq", mol="la18", charge=0, mult=1, method="b3lyp",
                    basis="def2-svp", calc="freq", place=dict(GLOBAL, scf_loop="device"), scf_loop="device"))
    # an open-shell gradient and frequencies (the zero-mode check of the Hessian on the device)
    out.append(dict(name="la18.b3lyp.uks.dloop.gradient", mol="la18", charge=1, mult=2, method="b3lyp",
                    basis="def2-svp", calc="gradient", place=dict(GLOBAL, scf_loop="device"), scf_loop="device"))
    out.append(dict(name="water.b3lyp.uks.dloop.freq", mol="water", charge=1, mult=2, method="b3lyp",
                    basis="def2-svp", calc="freq", place=dict(GLOBAL, scf_loop="device"), scf_loop="device"))
    # the escalation at max_iter (scf.escalate): the device loop hands over at max_iter 14 and the
    # host loop escalates from the lowest iterate the device kept (converged in 9 more iterations)
    out.append(dict(name="la18.b3lyp.uks.maxiter.dloop.energy", mol="la18", charge=1, mult=2,
                    method="b3lyp", basis="def2-svp", calc="energy", place=dict(GLOBAL, scf_loop="device"),
                    scf_loop="device", extra=["--max-iter", "14"]))
    # the closed-shell probe (D36) on the device, a singlet diradical below the gap gate: flagged
    # on both backends, the device result bitwise run to run
    out.append(dict(name="pbenzyne.b3lyp.rks.probe.dloop.energy", mol="pbenzyne", charge=0, mult=1,
                    method="b3lyp", basis="def2-svp", calc="energy", place=dict(GLOBAL, scf_loop="device"),
                    scf_loop="device", probe=True, repeat=True))
    # an implicit solvent with exact surface terms on the device, in the device loop
    out.append(dict(name="water.b3lyp.rks.pcm-water-exact.dloop.energy", mol="water", charge=0, mult=1,
                    method="b3lyp", basis="def2-svp", calc="energy",
                    place=dict(GLOBAL, scf_loop="device", pcm="gpu"), scf_loop="device",
                    extra=["--solvent", "water", "--set", "solvent.scf_terms=exact"]))
    methods = [("hf", HF), ("b3lyp", GLOBAL), ("wb97x-d3bj", RSH), ("cam-b3lyp", RSH)]
    for mol in MOLS:
        for method, place in methods:
            for shell in (0, 1):
                charge, mult = SPINS[mol][shell]
                spin = "uks" if shell else "rks"
                out.append(dict(name=f"{mol}.{method}.{spin}.energy", mol=mol, charge=charge, mult=mult,
                                method=method, basis="def2-svp", calc="energy", place=place,
                                repeat=(mol == "la18" and method == "b3lyp")))
    for mol in ("water", "la18", "u1an39"):
        for method, place in methods:
            for shell in (0, 1):
                if method in ("hf", "cam-b3lyp") and shell == 1:
                    continue
                charge, mult = SPINS[mol][shell]
                spin = "uks" if shell else "rks"
                out.append(dict(name=f"{mol}.{method}.{spin}.gradient", mol=mol, charge=charge, mult=mult,
                                method=method, basis="def2-svp", calc="gradient", place=place,
                                repeat=(mol == "la18" and method == "b3lyp" and shell == 0)))
    hess = [("water", "b3lyp", 0, False), ("water", "hf", 1, False), ("water", "wb97x-d3bj", 1, True),
            ("la18", "b3lyp", 0, True), ("la18", "b3lyp", 1, False), ("la18", "wb97x-d3bj", 0, False),
            ("la18", "wb97x-d3bj", 1, False), ("la18", "cam-b3lyp", 0, False), ("la18", "hf", 0, False),
            ("la18", "hf", 1, False), ("u1an39", "b3lyp", 0, False), ("u1an39", "b3lyp", 1, False),
            ("u1an39", "hf", 0, False), ("u1an39", "wb97x-d3bj", 0, True)]
    for mol, method, shell, ir in hess:
        charge, mult = SPINS[mol][shell]
        spin = "uks" if shell else "rks"
        place = {"hf": HF, "b3lyp": GLOBAL, "wb97x-d3bj": RSH, "cam-b3lyp": RSH}[method]
        c = dict(name=f"{mol}.{method}.{spin}.freq", mol=mol, charge=charge, mult=mult, method=method,
                 basis="def2-svp", calc="freq", ir=ir, place=place,
                 repeat=(mol == "la18" and method == "b3lyp" and shell == 0))
        if (mol, method, shell) == ("la18", "hf", 1):
            # a 126-iteration UHF SCF whose converged orbitals carry the device setup's round-off:
            # |dH| 4.9e-11 before the density-fitting setup moved to the device,
            # 1.02e-10 after (energy 2e-12, gradient 5.5e-11); main's decision, the global 1e-10 stays
            c["hessian_tol"] = 2e-10
        out.append(c)
    # emulated FP64 exchange (precision.fp64 "emulated"): the Ozaki scheme on INT8 tensor cores
    for mol in ("water", "la18", "u1an39"):
        for method, place in [("b3lyp", GLOBAL), ("wb97x-d3bj", RSH)]:
            for shell in (0, 1):
                charge, mult = SPINS[mol][shell]
                spin = "uks" if shell else "rks"
                out.append(dict(name=f"{mol}.{method}.{spin}.emu.energy", mol=mol, charge=charge, mult=mult,
                                method=method, basis="def2-svp", calc="energy", place=place,
                                precision="emulated", repeat=(mol == "la18" and shell == 0)))
    out.append(dict(name="la18.b3lyp.rks.emu.dloop.energy", mol="la18", charge=0, mult=1, method="b3lyp",
                    basis="def2-svp", calc="energy", place=dict(GLOBAL, scf_loop="device"), scf_loop="device",
                    precision="emulated", repeat=True))
    out.append(dict(name="u1an39.b3lyp.rks.emu.gradient", mol="u1an39", charge=-1, mult=1, method="b3lyp",
                    basis="def2-svp", calc="gradient", place=GLOBAL, precision="emulated"))
    # the device-dependent defaults (decisions D27, D31): precision.fp64 auto (emulated on
    # FP64-poor cards, native on FP64-rich ones) and scf_loop auto (the device loop, closed and
    # open shells), nothing pinned
    for mol, method, shell, calc in [("la18", "b3lyp", 0, "energy"), ("la18", "b3lyp", 1, "energy"),
                                     ("u1an39", "wb97x-d3bj", 0, "energy"), ("u1an39", "wb97x-d3bj", 1, "energy"),
                                     ("la18", "b3lyp", 0, "gradient"), ("la18", "b3lyp", 1, "gradient"),
                                     ("la18", "b3lyp", 0, "freq"), ("water", "b3lyp", 1, "freq")]:
        charge, mult = SPINS[mol][shell]
        spin = "uks" if shell else "rks"
        loop = "device"
        place = {"b3lyp": GLOBAL, "wb97x-d3bj": RSH}[method]
        out.append(dict(name=f"{mol}.{method}.{spin}.auto.{calc}", mol=mol, charge=charge, mult=mult, method=method,
                        basis="def2-svp", calc=calc, place=dict(place, scf_loop=loop), auto=True,
                        repeat=(mol, shell, calc) == ("la18", 0, "energy")))
    for mol, method, place in [("la18", "b3lyp", GLOBAL), ("la18", "wb97x-d3bj", RSH), ("u1an39", "b3lyp", GLOBAL)]:
        charge, mult = SPINS[mol][0]
        out.append(dict(name=f"{mol}.{method}.rks.emu.freq", mol=mol, charge=charge, mult=mult, method=method,
                        basis="def2-svp", calc="freq", place=place, precision="emulated"))
    # the response matrix: Hessians with IR intensities and polarizabilities, the
    # exchange on the stored operator, FP64 and emulated, closed and open shells
    for mol in ("water", "la18", "u1an39"):
        for method, place in [("b3lyp", GLOBAL), ("wb97x-d3bj", RSH)]:
            for shell in (0, 1):
                charge, mult = SPINS[mol][shell]
                spin = "uks" if shell else "rks"
                for precision in ("", "emulated"):
                    emu = ".emu" if precision else ""
                    c = dict(name=f"{mol}.{method}.{spin}{emu}.ir.freq", mol=mol, charge=charge, mult=mult,
                             method=method, basis="def2-svp", calc="freq", ir=True, place=place,
                             repeat=(mol, method, shell, precision) == ("la18", "wb97x-d3bj", 1, ""))
                    if precision:
                        c["precision"] = precision
                    out.append(c)
    # linus's default response precision (auto): single precision and the FP32 CPKS kernel
    for mol, method, shell in [("la18", "b3lyp", 0), ("la18", "wb97x-d3bj", 1), ("u1an39", "b3lyp", 0),
                               ("u1an39", "wb97x-d3bj", 0), ("u1an39", "b3lyp", 1)]:
        charge, mult = SPINS[mol][shell]
        spin = "uks" if shell else "rks"
        out.append(dict(name=f"{mol}.{method}.{spin}.default.freq", mol=mol, charge=charge, mult=mult,
                        method=method, basis="def2-svp", calc="freq", ir=True,
                        place={"b3lyp": GLOBAL, "wb97x-d3bj": RSH}[method], response="default",
                        repeat=(mol, method) == ("la18", "b3lyp")))
    # u1an39/def2-TZVP at the default (no IR), where on FP64-poor cards auto's cost model takes the FP32
    # exchange through the tensor (decision D37: 10 % faster than the stored operator on the RTX PRO 6000),
    # and its FP64 twin (``.freq``, the reference of the default's ``*_vs_fp64`` checks)
    charge, mult = SPINS["u1an39"][0]
    # (against the CPU's default, whose two-level single-precision response lies further from FP64 with
    # def2-TZVP than with def2-SVP: the GPU default 7.7e-3 cm^-1 and 9.7e-7 Eh/bohr^2 from it at b1ac8f3, and
    # 5.8e-4 cm^-1 and 2.2e-7 Eh/bohr^2 from the CPU's FP64 run, so the CPU default's own error is ~7e-3
    # cm^-1: its bars 2-3x that; the GPU's own error is held to TOL_DEFAULT_FP32 against the FP64 twin)
    tzvp_default = dict(freq_tol=1.5e-2, hessian_tol=3e-6)
    for name, response in (("default.freq", "default"), ("freq", None)):
        c = dict(name=f"u1an39.b3lyp.rks.tzvp.{name}", mol="u1an39", charge=charge, mult=mult, method="b3lyp",
                 basis="def2-tzvp", calc="freq", place=GLOBAL)
        if response:
            c.update(response=response, **tzvp_default)
        out.append(c)
    # the card's own defaults at the default response precision (``*.auto.*``: nothing pinned; decision D37 on
    # FP64-poor cards with the operator's setup in FP64 emulation, as users run it): u1an39/def2-SVP with IR (the
    # operator, which serves the field solve too) and u1an39/def2-TZVP without (the FP32 exchange); against the
    # CPU's FP64 run of the named twin
    for name, basis, ir, ref in (("u1an39.b3lyp.rks.auto.default.freq", "def2-svp", True, "u1an39.b3lyp.rks.ir.freq"),
                                 ("u1an39.b3lyp.rks.tzvp.auto.default.freq", "def2-tzvp", False,
                                  "u1an39.b3lyp.rks.tzvp.freq")):
        c = dict(name=name, mol="u1an39", charge=charge, mult=mult, method="b3lyp", basis=basis, calc="freq",
                 ir=ir, place=dict(GLOBAL, scf_loop="device"), auto=True, response="default", fp64_ref=ref)
        if basis == "def2-tzvp":
            c.update(tzvp_default)
        out.append(c)
    # a -V functional at the default response precision: on FP64-poor cards the FP32 CPKS kernel
    # and VV10's FP32 CPKS pair sums, on FP64-rich ones FP64 (checked against the CPU's FP64
    # la18.wb97m-v.rks.freq)
    out.append(dict(name="la18.wb97m-v.rks.default.freq", mol="la18", charge=0, mult=1, method="wb97m-v",
                    basis="def2-svp", calc="freq", ir=True, place=VV10_HESS, response="default"))
    return with_vv10_bar(out + pcm_cases() + vv10_cases() + mgga_cases())


# CPU results by command line: a case's ``.emu`` twin runs the same CPU calculation (only the
# GPU side pins the arithmetic), and the CPU backend is deterministic, so it runs once
_CPU_RUNS: dict[str, tuple[str, float]] = {}


def command(args, case: dict, backend: str) -> list[str]:
    """The linus command of a case on a backend (without the --prefix)."""
    mol = MOLS.get(case["mol"]) or CASE_MOLS[case["mol"]]
    cmd = [args.linus_bin, str(mol), "--method", case["method"], "--basis", case["basis"],
           "--jk", "df", "--charge", str(case["charge"]), "--mult", str(case["mult"]),
           "--threads", str(args.threads), "--backend", backend, "--json"]
    cmd += case.get("extra", [])
    # the GPU side pins the arithmetic and the loop it names (decision D27 made both defaults
    # device-dependent: emulated on FP64-poor cards, the device loop for closed shells), so
    # that every card covers FP64 and the host loop; the ``*.auto.*`` cases keep the defaults.
    # FP64 is ``precision: fp64`` (native, precision.fp32 off: VV10's pair sums in FP64 too),
    # except at the default response precision, which precision.fp32 off would make FP64
    if backend == "gpu" and not case.get("auto"):
        cmd += ["--set", f"backend.scf_loop={case.get('scf_loop') or 'host'}"]
        if case.get("precision") == "emulated":
            cmd += ["--set", "precision.fp64=emulated"]
        elif case.get("response") == "default":
            cmd += ["--set", "precision.fp64=native"]
        else:
            cmd += ["--precision", "fp64"]
    if case["calc"] == "gradient":
        cmd.append("--grad")
    if case["calc"] == "freq":
        cmd.append("--hess")
        if case.get("response") != "default":
            cmd += ["--set", "precision.response=fp64"]
        if case.get("ir"):
            cmd.append("--ir")
    return cmd


def cpu_spec(args, case: dict, glibc: str) -> dict:
    """The linus-cpu store's spec of a case's CPU result (bench/refstore.py)."""
    cmd = command(args, case, "cpu")
    return refstore().linus_cpu_spec(args.commit, glibc, case["calc"], cmd[1], cmd[2:])


def run(args, case: dict, backend: str, tag: str) -> dict:
    out = Path(args.out) / f"{case['name']}.{tag}.json"
    prefix = args.gpu_prefix if backend == "gpu" and args.gpu_prefix else args.prefix
    cmd = shlex.split(prefix) + command(args, case, backend)
    key = shlex.join(cmd)
    spec = stored = None
    if backend == "cpu" and args.cpu_store:
        # the linus-cpu reference store (bench/refstore.py): reuse a CPU result of the same inputs, numerics epoch
        # and glibc, else compute it and store it
        spec = cpu_spec(args, case, refstore().glibc_here())
        stored = refstore().linus_cpu_get(spec)
    if stored is not None:
        stdout, wall, stderr = json.dumps(stored["run"]), float("nan"), ""
    elif backend == "cpu" and key in _CPU_RUNS:
        stdout, wall = _CPU_RUNS[key]
        stderr = ""
    else:
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t
        stdout, stderr = p.stdout, p.stderr
        if backend == "cpu" and p.returncode == 0:
            _CPU_RUNS[key] = (stdout, wall)
    out.write_text(stdout)
    try:
        r = json.loads(stdout)
    except json.JSONDecodeError:
        r = {"success": False, "error": stderr[-2000:]}
    put = {}
    if spec is not None and stored is None and (r.get("success") or "error_kind" in r):
        put = refstore().linus_cpu_put([refstore().linus_cpu_record(spec, r, "reference", provenance(args, wall))],
                                       f"linus-cpu:validate_gpu:{args.commit[:7]}")
    r["_wall"] = wall
    if spec is not None:  # refused: the store's epoch guard (bench/refstore.py linus_cpu_guard)
        r["_cpu_store"] = "hit" if stored is not None else ("refused" if put.get("refused") else "stored")
    return r


def refstore():
    """bench/refstore.py, the reference store (imported on first use)."""
    sys.path.insert(0, str(REPO / "bench"))
    import refstore as rs  # noqa: E402
    return rs


def provenance(args, wall: float) -> dict:
    """A stored CPU result's provenance: program and build, machine (the node as an opaque id, bench/node_id.py;
    no host names), threads, wall time and date."""
    import platform
    sys.path.insert(0, str(REPO / "bench"))
    from node_id import node_id  # noqa: E402
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), None)
    return dict(source="validate_gpu", linus_commit=args.commit, linus_bin=str(args.linus_bin), threads=args.threads,
                wall_s=wall, cpu_model=cpu, glibc=os.confstr("CS_GNU_LIBC_VERSION"), kernel=platform.release(),
                node_id=node_id(), image=os.environ.get("LINUS_IMAGE"), image_digest=os.environ.get("LINUS_IMAGE_DIGEST"),
                date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def import_cpu(args, todo: list[dict]) -> int:
    """Store the CPU outputs an earlier run left in --import-cpu (<case>.cpu.json) for the listed cases."""
    recs = []
    for case in todo:
        f = Path(args.import_cpu) / f"{case['name']}.cpu.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if not (r.get("success") or "error_kind" in r):
            continue
        spec = cpu_spec(args, case, args.glibc)
        prov = dict(source=f"import:{args.import_cpu}", linus_commit=args.commit, glibc=args.glibc,
                    cpu_model=args.cpu_model or None, threads=None, wall_s=None)
        recs.append(refstore().linus_cpu_record(spec, r, "reference", prov))
    n = refstore().linus_cpu_put(recs, f"linus-cpu:import:{Path(args.import_cpu).name}")
    print(f"{args.import_cpu}: {len(recs)} CPU outputs of {len(todo)} cases; {n}")
    return 0


def load(args, case: dict, tag: str) -> dict | None:
    """The output of an earlier run (`--rescore`), None when it is missing."""
    p = Path(args.out) / f"{case['name']}.{tag}.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text())
    except json.JSONDecodeError:
        r = {"success": False, "error": "unreadable output"}
    r["_wall"] = float("nan")
    return r


def maxdiff(a, b) -> float:
    if isinstance(a, list):
        return max((maxdiff(x, y) for x, y in zip(a, b)), default=0.0)
    return abs(a - b)


def bitwise(a: dict, b: dict) -> bool:
    keys = ("energy", "gradient", "hessian")
    return all(json.dumps(a.get(k)) == json.dumps(b.get(k)) for k in keys)


def compare(case: dict, c: dict, g: dict, g2: dict | None, ref64: dict | None = None) -> dict:
    rec: dict = {"case": case["name"], "checks": {}, "wall": {"cpu": c["_wall"], "gpu": g["_wall"]}}
    chk = rec["checks"]

    def check(name, value, ok):
        chk[name] = {"value": value, "ok": bool(ok)}

    for b, r in (("cpu", c), ("gpu", g)):
        if not r.get("success"):
            check(f"{b}_ran", r.get("error", "failed"), False)
            return rec
    # the GPU's arithmetic: as pinned, or (``*.auto.*``) auto's choice for the card, resolved in
    # resolved.settings.precision.fp64 with the reason in resolved.reasons["precision.fp64"]
    prec = case.get("precision") or "double"
    if case.get("auto"):
        rb0 = g["resolved"]["record"]["backend"]
        prec = auto_precision(rb0["device"]["compute_capability"])
        fp64 = {"double": "native", "emulated": "emulated"}[prec]
        got_fp64 = ((g["resolved"].get("settings") or {}).get("precision") or {}).get("fp64")
        note = (g["resolved"].get("reasons") or {}).get("precision.fp64") or ""
        check("precision_auto", [got_fp64, rb0.get("precision"), note],
              got_fp64 == fp64 and rb0.get("precision") == prec and note.startswith(f"auto: {fp64} ("))
    de = g["energy"] - c["energy"]
    check("energy", de, abs(de) <= case.get("energy_tol", TOL["energy"]))
    check("scf_iterations", [c["iterations"], g["iterations"]], c["iterations"] == g["iterations"])
    if case.get("scf_loop") == "device" and not c.get("restricted", True):
        # the stability decisions of the device loop (the probe on the device) are the CPU's
        def decisions(r):
            return [(x["iter"], x.get("probe", False), x["step"], x["fock_builds"])
                    for x in (r.get("scf") or {}).get("stability", [])]
        dc, dg = decisions(c), decisions(g)
        check("stability", dg, dc == dg)
    # the closed-shell probe (D36): the same decision on both sides (records from before it, as
    # the CPU store may return, have none: no check, except for the cases that must probe)
    pc, pg = ((((r.get("resolved") or {}).get("record") or {}).get("scf") or {}).get("closed_shell_probe")
              for r in (c, g))
    if (pc and pg) or case.get("probe"):
        same = bool(pc and pg) and all(pc.get(k) == pg.get(k) for k in ("skipped", "unstable", "kind", "fock_builds"))
        de = abs(pc["eigenvalue"] - pg["eigenvalue"]) if same and pc.get("eigenvalue") is not None else 0.0
        flagged = not case.get("probe") or (same and pg.get("unstable") is True)
        check("closed_shell_probe", [(pg or {}).get("skipped") or (pg or {}).get("eigenvalue"), de],
              same and de <= TOL_PROBE and flagged)
    if c.get("gradient"):
        d = maxdiff(c["gradient"], g["gradient"])
        check("gradient", d, d <= case.get("gradient_tol", TOL["gradient"]))
    if c.get("hessian"):
        default = case.get("response") == "default"
        tol = TOL_DEFAULT if default else TOL
        d = maxdiff(c["hessian"], g["hessian"])
        check("hessian", d, d <= case.get("hessian_tol", tol["hessian"]))
        ci, gi = c.get("cphf_iterations"), g.get("cphf_iterations")
        if ci == gi or (default and abs(ci - gi) <= 1):
            check("cphf_iterations", [ci, gi], True)
        else:
            # one iteration more on one side is allowed when the side with fewer iterations
            # stopped at the threshold (final max residual within [tol/2, 2 tol]): the stopping
            # test is discrete and the residual round-off sensitive (module docs); shown, counted
            ctol = c["resolved"]["settings"]["response"]["tol"]
            lo, lo_r, hi = ("cpu", c, "gpu") if ci < gi else ("gpu", g, "cpu")
            res = lo_r["resolved"]["record"]["response"]["max_residual"]
            flip = abs(ci - gi) == 1 and ctol / 2 <= res <= 2 * ctol
            check("cphf_iterations", f"+1 ({lo} {min(ci, gi)} at {res:.1e}, {hi} {max(ci, gi)}; tol {ctol:.0e})"
                  if flip else [ci, gi], flip)
            rec["cphf_flip"] = flip
        fc, fg = c["vibrational_analysis"], g["vibrational_analysis"]
        d = maxdiff(fc["frequencies"], fg["frequencies"])
        check("frequencies", d, d <= case.get("freq_tol", tol["freq"]))
        if fc.get("ir_intensities") is not None:
            d = maxdiff(fc["ir_intensities"], fg.get("ir_intensities") or [])
            check("ir_intensities", d, d <= tol["ir"] * max(map(abs, fc["ir_intensities"]), default=0.0))
        if c.get("polarizability"):
            d = maxdiff(c["polarizability"], g["polarizability"])
            check("polarizability", d, d <= tol["alpha"])
        if c.get("dipole_derivatives") and "dipder" in tol:
            d = maxdiff(c["dipole_derivatives"], g["dipole_derivatives"])
            check("dipole_derivatives", d, d <= tol["dipder"])
        resp = {r["purpose"]: r for r in g["resolved"]["record"]["backend"].get("response", [])}
        h = resp.get("hessian", {})
        if default and ref64 is not None and ref64.get("success"):
            # the GPU's default against the CPU in FP64 (the same molecule's ``.ir.freq`` case):
            # the device's own error at the default, without the CPU's single-precision one; by
            # the exchange auto ran (TOL_DEFAULT_FP32 for the FP32 exchange through the tensor)
            bar = TOL_DEFAULT_FP32 if h.get("jk") == "fp32" else TOL_DEFAULT_FP64
            f64 = ref64["vibrational_analysis"]
            d = maxdiff(ref64["hessian"], g["hessian"])
            check("hessian_vs_fp64", d, d <= bar["hessian"])
            d = maxdiff(f64["frequencies"], fg["frequencies"])
            check("frequencies_vs_fp64", d, d <= bar["freq"])
            if f64.get("ir_intensities") is not None:
                d = maxdiff(f64["ir_intensities"], fg.get("ir_intensities") or [])
                big = max(map(abs, f64["ir_intensities"]), default=0.0)
                check("ir_intensities_vs_fp64", d, d <= bar["ir"] * big)
        # the response's device arithmetic and the form of its exchange: FP64 on the stored
        # operator (every Hessian of the matrix fits); at the default precision where the CPU
        # runs single precision, the GPU's auto runs the CPKS kernel in FP32 and, on FP64-poor
        # cards, its exchange on the path its cost model estimated cheaper (decision D37: the
        # stored operator in FP64, or FP32 through the tensor; ``estimate_s``); with `emulated` an
        # FP64 CPKS kernel runs its large GEMMs in the Ozaki scheme
        auto_single = default and c["resolved"]["record"]["response"]["single_precision"]
        emulated = prec == "emulated"
        want = {"jk": "emulated" if emulated else "fp64", "exchange": case.get("exchange", "operator")}
        # an FP64-rich card's precision.response auto stays FP64 (decision D27)
        rich = auto_precision(g["resolved"]["record"]["backend"]["device"]["compute_capability"]) == "double"
        est = h.get("estimate_s")
        if auto_single and not rich and est:
            if est["tensor"] < est["operator"]:
                want.update(jk="fp32", exchange="tensor")
            else:
                want["exchange"] = "operator"
            got_est = True
        else:
            # the estimates only where auto weighed the two paths
            got_est = est is None or (auto_single and not rich)
        if case["place"].get("response_xc"):
            want["xc"] = "fp32" if auto_single and not rich else "emulated" if emulated else "fp64"
        got = {k: h.get(k) for k in want}
        if h.get("exchange") != want["exchange"]:
            # why the operator was not built (e.g. under device_memory_gib: it did not fit)
            got["operator_note"] = h.get("operator_note") or h.get("reason")
        if est:
            got["estimate_s"] = est
        check("response", got, got_est and all(got.get(k) == v for k, v in want.items()))
    rb = g["resolved"]["record"]["backend"]
    want = {k: v for k, v in case["place"].items()}
    if case["calc"] == "energy":
        want = {k: want[k] for k in ("jk", "xc", "vv10", "scf_loop", "pcm") if k in want}
    got = {k: rb.get(k) for k in want}
    reasons = {f["kernel"] for f in rb["fallbacks"]}
    missing = [k for k, v in want.items() if v == "cpu" and k not in reasons]
    check("placement", got, got == want and not missing and rb.get("device") is not None)
    if prec == "emulated":
        emu = {e["kernel"]: e for e in rb.get("emulation", [])}
        kernels = ["jk"] + (["response_jk", "hessian_jk"] if case["calc"] == "freq" else [])
        got_emu = {k: (emu.get(k, {}).get("scheme"), emu.get(k, {}).get("bits")) for k in kernels}
        # the SCF's density-fitting fit in the Ozaki scheme as well
        fit = emu.get("jk", {}).get("fit") or {}
        got_emu["jk_fit"] = (fit.get("scheme"), fit.get("bits"))
        # an exchange on the stored operator: its GEMMs in cuBLAS's emulation (the family's last
        # setup is reported: with --ir the polarizability's); auto's FP32 exchange through the
        # tensor (decision D37): no emulation
        last = (rb.get("response") or [{}])[-1]
        op = last.get("exchange") == "operator"
        want_emu = {k: ("cublas", 55) if k == "response_jk" and op else ("ozaki", 55) for k in got_emu}
        if last.get("jk") == "fp32":
            want_emu["response_jk"] = (None, None)
        check("emulation", got_emu, got_emu == want_emu)
    cb = c["resolved"]["record"]["backend"]
    check("cpu_backend", c["resolved"]["settings"]["backend"]["kind"], cb["device"] is None and not cb["fallbacks"]
          and all(cb.get(k) in (None, "cpu", "host") for k in want))
    if g2 is not None:
        check("bitwise_run_to_run", g2.get("success"), g2.get("success") and bitwise(g, g2))
    rec["timings"] = {b: {k: v for k, v in r["timings"].items()} for b, r in (("cpu", c), ("gpu", g))}
    rec["fallbacks"] = rb["fallbacks"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--linus-bin", default=str(REPO / "target/release/linus"))
    ap.add_argument("--cases", default="*", help="glob over case names (comma-separated)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prefix", default="", help="command prefix for every run (pinning, memory caps)")
    ap.add_argument("--gpu-prefix", default="", help="command prefix of the GPU runs instead of --prefix "
                    "(e.g. one that takes the node's GPU lock)")
    ap.add_argument("--out", default=str(REPO / "reference/out/gpu"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="compare the outputs a previous run left in --out again (no calculations)")
    ap.add_argument("--cpu-store", action="store_true",
                    help="take each CPU result from the linus-cpu reference store (bench/refstore.py) when it has one "
                         "of the same inputs, numerics epoch and glibc, else compute it and store it")
    ap.add_argument("--commit", default=os.environ.get("LINUS_COMMIT", ""),
                    help="the linus commit of --linus-bin (for --cpu-store; default $LINUS_COMMIT, else HEAD)")
    ap.add_argument("--import-cpu", default="", help="store the CPU outputs of a previous run's --out (no "
                    "calculations; with --commit, --glibc and --cpu-model of that run)")
    ap.add_argument("--cpu-only", action="store_true",
                    help="only the CPU side (with --cpu-store: fill the store for this build's numerics epoch and "
                         "glibc, no GPU needed); no comparisons")
    ap.add_argument("--store-misses", action="store_true",
                    help="list the cases whose CPU result the linus-cpu store lacks (for --commit and --glibc, "
                         "default this machine's), with how many distinct CPU calculations they need; no runs")
    ap.add_argument("--distinct", action="store_true",
                    help="with --store-misses: one case per missing CPU calculation (cases sharing CPU inputs, "
                         "e.g. GPU-only variants, listed once)")
    ap.add_argument("--glibc", default="", help="the glibc of the --import-cpu run, e.g. 'glibc 2.43'")
    ap.add_argument("--cpu-model", default="", help="the CPU of the --import-cpu run")
    ap.add_argument("--cpu-from", default="",
                    help="take the CPU results from a previous run's --out where it has them (same cases; "
                         "the CPU backend of a build whose CPU results are bitwise those of that run's)")
    args = ap.parse_args()
    if (args.cpu_store or args.import_cpu) and not args.commit:
        args.commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()
    pats = args.cases.split(",")
    todo = [c for c in cases() if any(fnmatch.fnmatch(c["name"], p) for p in pats)]
    if args.list:
        print("\n".join(c["name"] for c in todo))
        return 0
    if args.store_misses:
        glibc = args.glibc or refstore().glibc_here()
        miss = [c for c in todo if refstore().linus_cpu_get(cpu_spec(args, c, glibc)) is None]
        distinct = {refstore().key(cpu_spec(args, c, glibc)) for c in miss}
        if args.distinct:
            seen, one = set(), []
            for c in miss:
                k = refstore().key(cpu_spec(args, c, glibc))
                if k not in seen:
                    seen.add(k)
                    one.append(c)
            miss = one
        print("\n".join(c["name"] for c in miss))
        print(f"{len(miss)} of {len(todo)} cases lack a stored CPU result ({len(distinct)} distinct calculations; "
              f"{glibc}, epoch {refstore().linus_cpu_epoch(args.commit)})", file=sys.stderr)
        return 0
    if args.import_cpu:
        if not args.glibc:
            ap.error("--import-cpu needs --glibc (the glibc of the run that computed them)")
        return import_cpu(args, todo)
    if args.cpu_only:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        src, bad = {}, 0
        for case in todo:
            c = run(args, case, "cpu", "cpu")
            how = c.get("_cpu_store", "computed")
            src[how] = src.get(how, 0) + 1
            bad += not c.get("success")
            print(f"{'ok  ' if c.get('success') else 'FAIL'} {case['name']:34s} cpu {c['_wall']:7.1f} s ({how})", flush=True)
        print(f"{len(todo) - bad} of {len(todo)} CPU results ok; {src}")
        return bad
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results = []
    failed = 0
    for case in todo:
        if args.rescore:
            c, g, g2 = (load(args, case, t) for t in ("cpu", "gpu", "gpu2"))
            if c is None or g is None:
                continue
            g2 = g2 if case.get("repeat") else None
        else:
            c = load(argparse.Namespace(out=args.cpu_from), case, "cpu") if args.cpu_from else None
            if c is not None and c.get("success"):
                (Path(args.out) / f"{case['name']}.cpu.json").write_text(json.dumps(c))
                cpu_src = "from"
            else:
                c = run(args, case, "cpu", "cpu")
                cpu_src = c.get("_cpu_store", "computed")
            g = run(args, case, "gpu", "gpu")
            g2 = run(args, case, "gpu", "gpu2") if case.get("repeat") else None
        ref64 = None
        if case.get("response") == "default":
            # the CPU's FP64 run of the same molecule (its ``.ir.freq`` case, else its ``.freq`` one),
            # when its case ran into the same --out, else from the store
            names = ([case["fp64_ref"]] if case.get("fp64_ref") else
                     [case["name"].replace(".default.freq", s) for s in (".ir.freq", ".freq")])
            ref64 = next((r for n in names if (r := load(args, dict(name=n), "cpu")) is not None), None)
            if ref64 is None and args.cpu_store and not args.rescore:
                irc = next((x for n in names for x in cases() if x["name"] == n), None)
                ref64 = run(args, irc, "cpu", "cpu") if irc is not None else None
        rec = compare(case, c, g, g2, ref64)
        if not args.rescore:
            rec["cpu_source"] = cpu_src  # from (--cpu-from), hit or stored (--cpu-store), computed
        ok = all(v["ok"] for v in rec["checks"].values())
        failed += not ok
        rec["ok"] = ok
        results.append(rec)
        # floats as numbers; an allowed CPHF iteration flip always shown
        vals = " ".join(
            f"{k}={v['value']:.1e}" if isinstance(v["value"], float)
            else f"{k}={v['value']}" if k == "cphf_iterations" and isinstance(v["value"], str)
            else f"{k}={'ok' if v['ok'] else v['value']}"
            for k, v in rec["checks"].items())
        print(f"{'ok  ' if ok else 'FAIL'} {case['name']:34s} cpu {rec['wall']['cpu']:7.1f} s"
              f"{'' if rec.get('cpu_source', 'computed') == 'computed' else ' (' + rec['cpu_source'] + ')'}  "
              f"gpu {rec['wall']['gpu']:7.1f} s  {vals}", flush=True)
        (Path(args.out) / "results.json").write_text(json.dumps(results, indent=1))
    flips = sum(1 for r in results if r.get("cphf_flip"))
    src = {}
    for r in results:
        src[r.get("cpu_source", "computed")] = src.get(r.get("cpu_source", "computed"), 0) + 1
    print(f"{len(results) - failed} of {len(results)} cases passed ({flips} with an allowed CPHF iteration flip); "
          f"CPU results: {src}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
