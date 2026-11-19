# Robustness reproducers

Minimal reproducers for what the sweep found, one directory each, with a linus JSON request
(`linus --json < request.json`, positions in bohr) and what to look for. `make_request.py` writes
a request for any molecule of the sets. linus a2fea72 (the sweep's image), unless noted.

## scf_level_shift_pi_flip: UHF CN radical stuck in the π-hole configuration

`request.json`: CN (1.172 Å, doublet), HF/def2-SVP, RI-JK, convergence `extreme`.

- linus: no convergence in 300 iterations (128 with default settings). The energy is
  constant to 1e-11 Eh while max |F_ai| stays at 7.2e-3 (RI-JK) or 1.6e-2 (exact J/K) from
  about iteration 20. The energy is 0.046 Eh above the ground state: the ²Π (π-hole)
  configuration. The two degenerate π-hole determinants alternate from iteration to iteration,
  with the same energy and a large density change.
- PySCF 2.15 (minao guess, DIIS from cycle 1, no level shift): converges in 54 iterations to
  -92.1287327498 (X²Σ⁺). Psi4 1.11: 19 iterations. ORCA 6.0.0: 29. Same energy.
- linus with `scf.level_shift: 0` converges in 27 iterations to -92.1287327498 (PySCF -1e-11).
  With `damping 0.5` it takes 40, `guess core` 23, `adiis` 32. With `soscf` it does not converge:
  max |F_ai| never drops below `scf.soscf_from` (3.3e-3). Neither does a level shift of 0.5 kept
  to 1e-4. `stability` never runs because the SCF doesn't converge.
- Cause: the default level shift (0.25 Eh until max |F_ai| < 1e-2) steers the early iterations
  into the π-hole region. There, aufbau occupation flips between πx and πy every iteration, and
  nothing detects the two-cycle.
- Seen in the sweep: GMTKN55 W4-11/cn, HF only (every J/K, def2-SVP and def2-TZVP, C-PCM).
  The DFT functionals converge.

## scf_c2_stagnation: singlet C2 (PBE) oscillates, only SOSCF converges

`request.json`: C2 (1.243 Å, singlet), PBE/def2-SVP, RI-JK, convergence `extreme`.

- linus: no convergence. max |F_ai| hovers around 1e-2 while the energy wanders by about 1e-3.
  The last energy (-75.7227) is 0.011 Eh above the converged solution.
- linus with `scf.soscf: true` converges in 15 iterations to -75.7335739764. Damping, the core
  guess, ADIIS and dropping the shift do not converge.
- PySCF 2.15 fails too (not converged). ORCA 6.0.0 converges in 14 iterations to -75.7335729
  (DefGrid3, its TRAH/SOSCF machinery).
- Seen in the sweep: GMTKN55 W4-11/c2 and hard/C2_singlet, PBE (def2-SVP and def2-TZVP).
- Together with the CN case: each case has some setting that converges, but a different one,
  and the SCF never escalates on stagnation (DIIS stuck with max |F_ai| at 1e-2 for hundreds of
  iterations). ORCA's automatic switch to TRAH handles both.

## pcm_memory_budget: the PCM matrices are not in the memory budget

`request.json`: ethanol, B3LYP/def2-SVP, C-PCM with 5810 Lebedev points per sphere
(18,817 surface points after switching), `memory_gib: 1`.

- linus: killed by the out-of-memory killer at a 3 GB scope cap (exit 137, RSS 3.1 GB at the
  cap) instead of a typed `resource` error. One dense surface matrix is 18,817² x 8 B = 2.6 GiB,
  and C-PCM holds S twice while building it (5.3 GiB); IEF-PCM and SMD hold about six such
  matrices. Every other budget derives from `memory_gib` (docs/usage.md).
- Prepared fix (`solvent::scf::check_dense_memory`, `pcm::dense_peak_bytes`, test
  `pcm_matrices_count_against_memory_gb`): the surface is built first (cheap) and the peak
  bytes of the dense matrices are checked against `memory_gib` (else the machine's or the
  cgroup's memory) before anything is allocated. The reproducer then stops with exit 1,
  `error_kind: "resource"`: "the cpcm surface has 18817 points: its dense matrices need 5.3 GiB
  while they are built, more than the 1.0 GiB of the memory_gib allow; use fewer points per sphere
  (solvent.lebedev, now up to 5810), give linus more memory ...".
- Seen in the sweep: Phase 2's PCM reference runs (5810 points per sphere) killed their pods.
- Production relevance: at the default 302 points per sphere, 100 atoms give about 30k points
  (7.3 GB dense) and 200 atoms 60k (29 GB). A large molecule in PCM can be killed rather than
  refused.

## pyscf_smd_ecp_cds: PySCF's SMD G_CDS treats ECP atoms as other elements (reference bug)

`check.py`: G_CDS of linus and of PySCF 2.15, with PySCF's `mol.atom_charges()` and with the
proton numbers, for XeF2, SnCl4, I2, BiCl3 and PtCl4²⁻ in water and toluene.

- PySCF's `smd.get_cds_legacy` passes `mol.atom_charges()` to the MNSOL routine as the atomic
  numbers. For an ECP atom that is Z minus the core electrons: Xe (54-28) becomes Fe, Sn Ti, I Mn,
  Bi V, Pt Ar. Their radii and surface tensions then enter G_CDS.
- Differences: XeF2 water +3.9e-4 Eh, SnCl4 toluene -8.4e-4 Eh (0.53 kcal/mol), BiCl3 toluene
  -4.3e-4, PtCl4²⁻ toluene -3.5e-4, I2 toluene -1.3e-5 (I has no CDS terms in water). With the
  proton numbers PySCF agrees with linus to 2e-12 Eh. linus is right.
- The sweep patches PySCF before its runs, so SMD with ECP
  atoms is an exact comparison. Same pattern as the known `grids_response_becke` issue
  (`run_ref.pyscf_ecp_grid_response_fix`). Worth reporting upstream (PySCF, `pyscf/solvent/smd.py`).

## pyscf_ecp_screening: libcint drops ECP integrals across the ECP atom (reference limitation)

`check.py ID BASIS ...`: PySCF HF (or DFT with `METHOD=pbe` etc., on PySCF's level-3 grid) with
its own ECP matrix and with linus's full one (`crates/linus-ints/examples/ecp_dump`), against
linus.

- XeF2 (linear F-Xe-F): the elements PySCF screens reach 6.1e-5 (def2-SVP) and 2.9e-5
  (def2-TZVP). HF energy offsets are 2.9e-9 Eh (SVP) and -1.6e-6 Eh (TZVP), gradients 2-3e-6
  Eh/bohr, exact or RI J/K. DFT offsets differ from HF's: PBE 1.79e-8, B3LYP 1.64e-8 Eh at
  def2-SVP (the HF offset there is small by cancellation). With linus's ECP matrix PySCF agrees
  with linus to 5e-13 Eh (HF) and 3e-13 Eh (PBE, B3LYP). Elsewhere the matrices agree to 1.5e-12.
  compare.py bounds the label by 5x the HF offset, so XeF2's def2-SVP DFT comparisons (1.6-2.1e-8)
  stay unexplained in the tables; this check is their explanation.
- reference/validation/README.md documents the screening, but only up to 2.2e-7 per element
  (HgH2/def2-QZVP) and energies below 1e-11 Eh. A linear X-M-X molecule with ligands on
  opposite sides of the ECP atom is far worse.
- `PTR_EXPCUTOFF` does not change it (the cutoff in `check_3c_overlap` is fixed), so the sweep
  verifies affected molecules with this check and labels their PySCF comparisons
  `pyscf-ecp-screening`.

## cucl4_d9_states: tetrahedral CuCl4²⁻ (d⁹ doublet): several UKS solutions, a saddle accepted

Exactly tetrahedral CuCl4²⁻ (GFN2-xTB kept Td), def2-SVP, RI-JK, convergence `extreme`. A
Jahn-Teller-degenerate ground state, so UHF/UKS solutions break symmetry in several ways.

- **r2SCAN, symmetry images (not a bug).** linus and PySCF converge to the same energy (7e-11 Eh)
  with different densities: dipoles of equal length (0.2417 e bohr, equal to 7e-10) along
  (-1,1,-1) and (1,1,1), and the same sorted per-atom gradient norms. The triage labels this
  `symmetry-equivalent-solution`.
- **B3LYP: the default path reports a higher solution as converged.** `b3lyp_diis.json` (the
  sweep's settings) converged on the cluster (Xeon 8470) in 67 iterations to -3480.7438131 Eh,
  and its stability check found nothing. `b3lyp_soscf.json` (`soscf: true`) converges in 147
  iterations to -3480.7443685 Eh, 5.6e-4 Eh lower. ORCA 6.0.0 (DefGrid3) gives -3480.7443649.
  The same DIIS request on this node's Zen 5 with `scf.stability: "always"` does not converge: max |F_ai|
  stays at 1.1e-5 for 300 iterations with |dE| ~ 1e-12. Which of the two happens depends on
  round-off, a sign of a flat, multi-minimum surface.
- **HF: a saddle point is returned as converged.** `hf_noshift_saddle.json` (`level_shift: 0`)
  converges in 93 iterations to -3476.3094604 Eh. The saddle check triggers a stability analysis
  at iteration 92: lowest orbital-Hessian eigenvalue -2.2e-4. The three trial steps along the mode
  (0.25, 0.5, 1.0; `scf/mod.rs`) lower nothing by more than 1e-9, so the result is accepted with no
  warning. ORCA, and linus with `soscf` or `guess core`, approach -3476.3095635 Eh, 1.0e-4 Eh
  lower (they stall at max |F_ai| 1e-8 / 1e-7 before g_tol 1e-10). For a shallow negative mode
  the minimum along it lies at a much smaller rotation than 0.25, so all three steps overshoot.
  With the default shift, HF does not converge at all (max |F_ai| 1.2e-3 after 300 iterations).
- **PBE / wB97X-D3BJ:** no convergence with the defaults. For PBE only `soscf` converges (81
  iterations, -3479.6312151 Eh; ORCA -3479.6312012 on DefGrid3). For wB97X-D3BJ (RI-JK, psi4 grid)
  linus's last energy is 2.7e-2 Eh above Psi4's converged one.
- PySCF fails to converge for B3LYP, TPSS, HF and PBE here; ORCA converges every functional.

## pyscf_smd_cds_gradient: PySCF's analytic G_CDS gradient is wrong for ethane (reference bug)

`check.py [ID [SOLVENT]]`: PySCF's analytic G_CDS gradient against 5-point central differences of
its own G_CDS energy (geometry only, no SCF).

- Idealized staggered ethane (`hard.C2H6_rCC1.54`), water: 1.26e-3 Eh/bohr off (the gradient is at
  most 1.65e-3). GMTKN55 DIPCS10 ethane: 1.09e-3. Aspirin 9.8e-10, caffeine (toluene) 4.7e-10.
- Not translation invariant: the analytic gradient sums to 2.2e-3 Eh/bohr over the atoms. This is
  the CH3SH bug already in the engineering log (the Fortran SASA-gradient routine), now seen on
  more molecules.
- linus's SMD gradient agrees with central differences of its own energy to 3e-10 there, and the
  G_CDS energies agree to 2e-12. The sweep's PySCF reference uses differences of PySCF's G_CDS
  energy (`refrun.py`). Issue draft: `bench/robust/upstream/pyscf_smd_cds_gradient.md`.

## stability_saddle_no_step: converged on a saddle, the stability step never taken

`request.json`: HF⁺ (GMTKN55 G21IP/IP_63, doublet), r²SCAN/def2-SVP, RI-JK, `pyscf:3` grid,
convergence `extreme`. Positions converted with the CODATA 2010 bohr (0.52917721092 Å). The
outcome is a knife edge: with the CODATA 2022 bohr (a 7e-10 relative change, as in the sweep)
the default path lands on the minimum directly.

- linus (this branch: main a4a1e71 plus the PCM check): converged in 22 iterations at
  -99.7356786955 Eh. The saddle check triggers a stability analysis (converged 7.1e-7 Eh above
  the lowest energy visited), which finds the orbital-Hessian eigenvalue -1.69e-3. The three
  trial rotations along the mode (0.25, 0.5 and 1.0) all raise the energy, so no step is taken
  (`step 0`), and the result reports convergence with no warning.
- linus with `scf.soscf: true`: -99.7356899124 Eh in 28 iterations, 1.12e-5 Eh lower; its
  stability analysis finds +8.6e-4 (a minimum).
- Why no step: with the curvature -1.69e-3 and a quartic term that puts the minimum 1.1e-5 Eh
  down, the minimum along the mode lies near a 0.1 rad rotation, and at 0.25 the quartic term
  already wins (model estimate: E(0.25) about +1.4e-4 Eh).
- PySCF 2.15 (DIIS, minao guess, the same grid and auxiliary basis, conv_tol 1e-12): not
  converged in 5 of 6 identical runs at 2 threads, end energies spread over 5e-6 Eh.
- Seen in the sweep (interim tier 1): 611 linus jobs
  ran a stability analysis; 159 found a mode below -1e-4; in 98 the result says converged while
  the last analysis still has a negative mode, and in 39 no step was ever taken (eigenvalues
  -1e-4 to -6.3e-3, median -3.6e-4). Same pattern as `cucl4_d9_states` (HF, -2.2e-4), on a
  two-atom molecule that runs in under a second.

## linus3_heavy_hydrides: linus:3 gradients off by up to 9e-4 Eh/bohr for X-H bonds beyond the 2nd row

`grid3_hydrides.py`: idealized hydrides (CH4 ... SnH4, NH3/PH3/AsH3, H2O/H2S/H2Se, HCl, HBr,
GaH3), B3LYP/def2-SVP, RI-JK, tight SCF; gradients on pyscf:3, linus:2, linus:3 and linus:4
against pyscf:8 at the same geometry. `grid3_freqs.py`: frequencies with the Hessian defaults
against linus:4.

- linus:3, the grid of analytic Hessians and frequencies by default, has max-abs gradient errors
  of 8.7e-4 Eh/bohr (H2Se), 5.4e-4 (HBr), 4.3e-4 (AsH3), 2.5e-4 (GaH3), 2.3e-4 (HCl) and 1.1e-4
  (PH3), with energy errors of 1-4e-5 Eh. The coarser linus:2 stays at 0.6-4.3e-5, linus:4 and
  pyscf:3 at 0.1-6e-6. For C, N and O hydrides linus:3 is fine (1.4-1.5e-5); tetrahedral X keeps
  a zero gradient by symmetry, so GeH4 and SnH4 show it on H only (4-6e-5).
- Phase 2 (cluster) shows the same on GMTKN55 HEAVYSB11/geh3 for all six functionals: linus:3
  2.4-3.5e-4 Eh/bohr, linus:2 1.4-4.6e-5, the energy/gradient default (pyscf:3) 0.9-2.7e-6.
- Frequencies with the defaults (linus:3, CPKS on linus:1) stay within 0.05-0.38 cm^-1 of
  linus:4 on the same molecules: the error is in the gradient (and the energy) that a frequency
  calculation reports, not in the Hessian.
- Cause, confirmed with a scratch build: level 3 drops the Treutler size adjustment
  (`GridSpec::linus`, `size_adjust: if level == 3 { SizeAdjust::None }`), so the cell boundary sits
  at the bond midpoint and H's coarse outer shells integrate the heavier atom's valence region.
  The choice was tuned on C/H/N/O molecules (lactides, u1an39). With the Treutler adjustment at
  level 3 too (one-line change, not committed), the same scan gives linus:3 gradient errors of
  1.6e-6 to 1.3e-5 Eh/bohr (H2Se 8.7e-4 → 3.4e-6, HBr 5.4e-4 → 2.7e-6, AsH3 4.3e-4 → 1.3e-5, HCl
  2.3e-4 → 1.0e-5, GaH3 2.5e-4 → 1.1e-5; CH4 4.9e-6 → 1.6e-6, H2O 1.4e-5 → 5.0e-6) and energy errors
  of at most 1.6e-6 Eh, and the default frequencies move closer to linus:4 (max 0.02-0.10 cm^-1
  instead of 0.05-0.38). The trade-off recorded in presets.rs is the u1an39 soft modes (0.39
  cm^-1 without the adjustment, 0.74 with it); keeping no adjustment only for H bonded to C, N or
  O is one option.
- Fix (commits 38d9c21, narrowed in 453fbba; decided: keep molecules of H to Ne atoms
  bitwise): Treutler's adjustment for the pairs of H with an atom beyond neon only. The hydride
  scan then gives 2.7e-6 to 1.3e-5 Eh/bohr (as the full adjustment) and the frequencies stay
  within 0.1 cm^-1 of linus:4; CH4, NH3, H2O are bitwise unchanged. Adjusting every pair with
  an atom beyond neon (38d9c21) also fixed C-I and C-Br pairs (MeI.OCH2 gradient 9.3e-5 ->
  1.4e-5) but worsened molecules without X-H bonds (IF5 frequencies 0.36 -> 1.06 cm^-1, BeCl2
  gradient 3.1e-6 -> 1.1e-5, Al2Cl6 frequencies 0.64 -> 0.81), which the H-X rule leaves
  bitwise as before (`grid3_variants.py`-style comparison against linus:4, B3LYP/def2-SVP).
- Soft modes: the torsion of an X-H group moves away from the fine grid (CH3SH B3LYP/def2-SVP:
  134.58 -> 134.38 cm^-1 against 134.64 on linus:4), the trade-off recorded in presets.rs for H
  next to C/N/O. Stiff modes and gradients improve.
- Cost: the same grid points (only the partition weights change); frequency jobs took the same
  time (HCl 0.18 / 0.20 s, AsH3 1.05 / 1.04, HBr 0.81 / 0.79, GaH3 1.03 / 1.03 s before / after;
  one run each, 4 threads).

## pcm_soft_mode_plateau: degenerate open shells in a PCM stall at max |F_ai| ~1e-8

`request.json`: F atom (²P), HF/def2-SVP, RI-JK, C-PCM acetonitrile (302 points), convergence
`extreme`. `lebedev_scan.py`: the same with 50 to 5810 points per sphere.

- linus: the energy is stationary to 1e-11 Eh from iteration 10, max |F_ai| stays at
  1.78e-8 (g_tol 1e-10), and the SCF ends unconverged at `max_iter`. In iterations 10-17 the
  density changes by less than 1e-8 while max |F_ai| stays at 1.779e-8; later DIIS wanders
  (density changes up to 3e-4) at the same energy. SOSCF stays at exactly 1.78e-8. In the gas
  phase the same SCF converges in 13 iterations (5.6e-11).
- The residual follows the surface's anisotropy: max |F_ai| over the last 20 iterations is
  0.4-29e-6 with 50 points per sphere, 2.8e-7 with 110, 1.7-4.2e-8 with 302, 5.8e-9 with 590,
  2.5e-9 with 1202 and 3.2e-9 with 5810.
- Mechanism: in the gas phase, turning the p hole among the three p orbitals is an exact
  symmetry (zero gradient, zero curvature). A Lebedev surface is not spherical, so in a PCM that
  rotation carries a tiny gradient and a tiny curvature. DIIS and SOSCF with the diagonal
  preconditioner (orbital energy differences of order 0.5 Eh) step along it by g / 0.5 per
  iteration, which against a curvature many orders smaller makes no progress. The energy is
  converged; only the tight gradient criterion is not met.
- Seen in the sweep (interim tier 1, `nonconv_extreme_only`: unconverged with max |F_ai| below
  1e-7 at the end): 117 PCM cases on 55 molecules, all atoms with a partly filled degenerate
  shell (S, Cl⁺, P⁺, N⁺, Co, Fe, V, ...) or linear radicals with a partly filled π shell (NO, OH,
  SH, ClO, OF, SiH, SiF, HI⁺, ...). H2O⁺, Na, O2 and F⁻ converge in the same PCM. In the gas
  phase, 262 cases on 70 molecules of the same two kinds, 248 of them DFT: the XC grid, not
  spherical either, plays the surface's part (the HF ones end at 1-3e-10, just above the
  threshold, except MnO4²⁻ at 5.7e-8). A Newton step with exact
  Hessian-vector products (as in the stability analysis) or accepting the plateau when the
  energy is stationary and the residual lies along a near-zero mode would end these SCFs.

## pyscf_smd_ecp_gradient: the sweep's patched PySCF SMD gradient with ECP atoms (reference harness)

`linus_check.py` and `pyscf_check.py`: CsF (Cs with the def2 ECP), B3LYP/def2-SVP, density
fitting, grid level 3, analytic gradient against central differences (h = 1e-3 bohr) of the same
program's energy. PySCF with the sweep's patches (refrun.py: G_CDS with proton numbers and by
differences of its energy; run_ref's ECP grid-response fix) and the grid response on.

- PySCF 2.15 with those patches, SMD water: analytic 0.0346550 Eh/bohr, differences of its energy
  0.0343346 (3.2e-4 apart). IEF-PCM in the same setup: 4.6e-8. PySCF as shipped (G_CDS with
  Z - ncore, pyscf#3617): self-consistent to 4.4e-8.
- linus, SMD water (PySCF preset): analytic 0.0343346, 4.4e-8 from its own differences and 4e-8
  from PySCF's differences; the energies agree to 1e-13 Eh. linus is right.
- G_CDS's gradient by differences matches the differences of G_CDS (2e-11) and PySCF's solvent part
  matches linus's (8e-7); the inconsistency is elsewhere and comes with the proton-number G_CDS.
  Not localized further: a harness artifact, labelled `pyscf-smd-ecp-gradient` by compare.py.

## uhf_hessian_zero_mode: analytic UHF Hessians at a zero or negative orbital-Hessian mode

`check.py ID [METHOD]`: the lowest eigenvalues of the UHF orbital Hessian, linus's and PySCF's
analytic Hessians against central differences of linus's analytic gradients, and the second
derivative of the energy along the direction where they differ (def2-SVP, exact J/K, tight SCF).

- Triplet water at the BH76 RKT14 geometry (`gmtkn55.BH76.RKT14`), HF: the orbital Hessian has
  an exact zero eigenvalue (next 0.068). linus and PySCF agree on the energy (1e-14 Eh) and the
  gradient, and both analytic Hessians differ from the differences: max 7.2e-3 (linus) and
  about 1e-2 (PySCF, which also changes from run to run by a few 1e-3). Along the worst
  direction the energy's second derivative is 0.01129 (differences of energies and of gradients,
  h = 1e-3 and 2.5e-4 bohr), linus 0.02638, PySCF 0.02556. linus's CPHF reports converged
  (max residual 8.8e-10).
- C2H2⁺ (`gmtkn55.G21IP.IP_71`, linear, π hole), HF: zero mode; energies 0.1308, linus 0.1521,
  PySCF 0.1354; max |H - H_FD| 1.1e-2 (linus), 6.8e-3 (PySCF).
- The other open-shell Hessian outliers of tier 1 are the same kind: SiH (zero mode), BH76
  RKT11 (1.9e-5), and two unstable solutions, NO2 (-7.2e-3) and C4H4⁺ (RC21 10p1, -1.8e-2).
- At a zero mode the solution is one of a degenerate family (rotating the hole among degenerate
  orbitals, as in pcm_soft_mode_plateau). A displacement lifts the degeneracy at second order and
  the SCF picks the lowest member, which first-order response from one member does not describe.
  At a negative mode the solution is a saddle. Either way the analytic Hessian, and the
  frequencies, are not those of the energy surface, and neither program says so.

## pyscf_overlap_threshold: PySCF drops overlap eigenvectors below 1e-6, linus below 1e-8 (reference setting)

`check.py [ID] [BASIS]`: cubane (`gmtkn55.DC13.ISO_C8H8_41`, def2-TZVP), HF RI-JK.

- The smallest overlap eigenvalue is 6.8e-7. PySCF 2.15 removes eigenvectors of S below
  `scf.hf.overlap_zero_eigenvalue_threshold` = 1e-6 by default (canonical orthogonalization, a
  warning at verbose 4), linus below `scf.lindep` = 1e-8, so PySCF works in 295 functions and
  linus in 296: linus is lower by 2.2e-6 Eh (HF), 3.4-6.5e-5 Eh (PBE, B3LYP, ωB97X-D3BJ).
- With the threshold set to 1e-8 PySCF agrees with linus to 4e-12 Eh. linus's density
  evaluated in PySCF gives linus's energy to 4e-12 and is stationary there (max |F_ia| 1.5e-10).
- reference/run_ref.py does not set the threshold; in tier 1 only cubane falls between the two.
