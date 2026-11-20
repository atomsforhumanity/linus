# linus

A from-scratch Hartree-Fock and Kohn-Sham DFT program: Rust core, Python front end that speaks
[qcdata](https://github.com/atomsforhumanity/qcdata). Fast analytic energies, gradients and
Hessians, validated against PySCF, Psi4 and ORCA 6.0.0. Built at Atoms for Humanity.

## What it computes

Methods by derivative and by Coulomb/exchange scheme (`jk`: `df` = RI-JK, the default; `exact`
integral-direct, `incore` (exact, held in memory), `rij_snk` = RI-J + seminumerical (COSX)
exchange):

| Method | Energy | Gradient | Hessian, frequencies, IR intensities, polarizability |
|---|---|---|---|
| HF (RHF, UHF) | exact, incore, df, rij_snk | exact, incore, df, rij_snk | exact, incore, df |
| LDA, GGA, meta-GGA (SVWN, PBE, BLYP, TPSS, r2SCAN, any libxc functional) | all four | all four | exact, incore, df |
| Global hybrids (B3LYP, B3LYP5, PBE0, TPSSh, mixtures) | all four | all four | exact, incore, df |
| Range-separated hybrids (ωB97X, ωB97X-D3BJ, ωB97M-D3BJ, CAM-B3LYP, ωPBE) | all four (RI: Psi4's wK scheme) | all four | exact, incore, df |
| VV10 functionals (ωB97M-V, ωB97X-V, B97M-V) | all four | all four | exact, incore, df (VV10's kernel and explicit terms on the NLC grid) |

Across all of them:

| Feature | Energy | Gradient | Hessian |
|---|---|---|---|
| Open shells (UHF/UKS) | yes | yes | yes |
| Effective core potentials (def2, Rb-Rn; automatic) | yes | yes | yes |
| PCM: C-PCM, COSMO, IEF-PCM, SS(V)PE (Gaussian surface charges, PySCF's conventions) | yes | yes | yes |
| SMD (IEF-PCM + CDS) | yes | yes | yes |
| Dispersion: D3(BJ), D3(BJ)+ATM, D4 (crates/linus-disp) | yes | yes | yes |
| Cartesian basis functions | yes | yes | yes |
| NVIDIA GPU (`backend: "gpu"`, a build with the `gpu` feature, `jk: "df"`; [docs/gpu.md](docs/gpu.md)) | J/K, XC, VV10, the PCM's surface terms and the SCF loop | two-electron, XC, VV10 and PCM terms | explicit two-electron and XC terms, CPHF/CPKS J/K and XC kernel (VV10's pair sums too); VV10's NLC-grid passes and the PCM's terms on the CPU |

Also: DFT grids that reproduce PySCF (`pyscf:N`), Psi4 (`psi4`) and ORCA 6.0.0's DefGrids
(`orca:N`) point for point, plus linus's own (`linus:N`); harmonic analysis, point groups and RRHO/quasi-RRHO
thermochemistry for any program's Hessian (`linus.vib`); orbital reuse between geometries;
deterministic results, bitwise identical at any thread count and across CPUs with the same C math
library ([docs/usage.md](docs/usage.md#10-reproducibility)). Not available: ROHF, excited
states, post-HF methods, periodic systems ([Limitations](#limitations)).

## Limitations

- **ROHF/ROKS** is not implemented: open shells run UHF/UKS (`reference: "restricted"` on an open
  shell is refused as `unsupported`).
- **Effective core potentials:** def2-type scalar ECPs only; LANL2-type ECPs (r^-2 local terms)
  and spin-orbit potentials are refused as `unsupported`.
- **RIJ + COSX (`jk: "rij_snk"`)** gives energies and gradients only: no seminumerical exchange
  in the CPHF/CPKS response and no COSX Hessians. Hessians, frequencies, IR intensities and
  polarizabilities need `jk` exact or df.
- **VV10 Hessians** spend most of their response time in VV10's kernel. With `backend: "gpu"`
  its pair sums run on the device; the NLC grid's passes (first-order densities, contractions)
  and the potential's own terms stay on the CPU.
- **Closed-shell stability is a probe, not an analysis.** About 3% of closed-shell molecules
  have a lower broken-symmetry UKS solution with B3LYP. The closed-shell probe
  (`scf.stability_closed_probe`) looks for one only when the RKS HOMO-LUMO gap is below 0.15 Eh,
  warns and records it (`resolved.record.scf.closed_shell_probe`), and never switches to UKS. It is
  off by default for Hartree-Fock, whose triplet instabilities are common (opt-in with
  `scf.stability_closed_probe: 2`), and its two products can miss an instability. Internal
  instabilities (a lower RHF/RKS solution) are not probed. Run `reference: "unrestricted"` with
  `scf.stability: "always"` for the full analysis and descent.
- **Symmetric saddle points.** The stability probe sees the symmetry block of its start
  vector, so an SCF that converges straight onto a symmetric saddle point, without a negative
  probe, can stay there. Compare with `scf.stability: "always"`, or with `reference: "unrestricted"` and a
  symmetry-broken start, when a symmetric open shell's state matters.
- **The GPU backend** runs on NVIDIA GPUs with CUDA 13, one device per calculation, and has
  J/K kernels for density fitting only (`jk: "df"`): exact and seminumerical J/K, one-electron
  and ECP integrals, dispersion, the cavity and SMD's CDS term, the PCM's Hessian terms and the
  density-fitted PCM run on the CPU ([docs/gpu.md](docs/gpu.md)).
- **Non-x86-64 builds** (macOS arm64) run the portable kernels and a libxc compiled with the
  platform compiler's floating-point contraction: their results agree with x86-64 to round-off,
  not bit for bit, and two tests (a meta-GGA functional reference at density-threshold points,
  the triplet-O3 stability path) differ there.
- **IR intensities and polarizabilities** are validated against finite differences and finite
  fields, not against another program
  ([reference/validation/README.md](reference/validation/README.md)).

## Build and install

```bash
cargo build --release                       # target/release/linus, the command-line program
cd crates/linus-py && maturin develop --release   # Python package `linus` (+ qcdata) into the active venv
```

Release wheels are on PyPI as two distributions of the same version, both imported as `linus`
(install one of them, not both):

```bash
pip install afh-linus             # the CPU build
pip install afh-linus-cuda13      # the same with the GPU backend (NVIDIA GPUs, CUDA 13)
pip install qccompute             # to run many calculations (lists, a result store, workers)
```

Pick `afh-linus-cuda13` to run calculations on an NVIDIA GPU (`backend: "gpu"`); it runs
everything on the CPU too. It needs, at run time only, an NVIDIA driver with CUDA 13 support
(580 or newer) and the CUDA 13 libraries NVRTC, cuBLAS and cuSOLVER on the dynamic loader's path
(a CUDA 13 toolkit's `lib64`, or `LD_LIBRARY_PATH`); without them, `backend: "cpu"` (the
default) and `"auto"` still run on the CPU. `linus.GPU_BUILD` tells the two apart at run time
([docs/gpu.md](docs/gpu.md)). The wheels are manylinux_2_28 x86-64 (glibc 2.28 or newer) for
CPython 3.11 or newer, with the basis sets inside.

Requirements:
- Rust >= 1.89 (`rust-version`), a C compiler, CMake, curl and tar.
- Network access at build time, unless you prepare it: crates/linus-xc downloads libxc 7.0.0
  from gitlab.com (68 MB, SHA-256 checked) and builds it once per machine (cached in
  `~/.cache/linus-xc`, or `LINUS_XC_CACHE_DIR`); `LIBXC_TARBALL=<file>` or
  `LIBXC_DIR=<installed libxc>` build offline. Cargo fetches the axtb-d4 dependency (D4
  dispersion) from github.com; `cargo fetch` or `cargo vendor` beforehand for offline builds.
- Python >= 3.11 and maturin for the Python package (a wheel: `maturin build --release`).

No BLAS or LAPACK: linear algebra is faer. Builds are portable across x86-64 CPUs (the AVX2 and
AVX-512 kernels are chosen at run time; `LINUS_ISA=generic|avx2|avx512` overrides); other
architectures use the portable kernels (a weekly CI job builds and tests on macOS arm64). Release artifacts: `cargo build --profile dist`
and `maturin build --profile dist` (no debug info, stripped). `--no-default-features` builds a
CLI or wheel without D3/D4 dispersion (an MIT-only build; see License).

Container images (the CLI and the Python package, with and without the GPU backend):
`docker/build.sh all <commit>` builds them from a commit ([docker/README.md](docker/README.md)).

Basis sets: the Python package ships the sets of `data/basis` (Basis Set Exchange JSON) and
reads them from its own copy. The `linus` binary reads `--basis-dir`, else `$LINUS_BASIS_DIR`,
else the source tree's `data/basis` as it was compiled: a binary installed elsewhere needs
`LINUS_BASIS_DIR` pointing at a copy of `data/basis` ([docs/cli.md](docs/cli.md#basis-sets)).

## Quick start

B3LYP-D3(BJ)/def2-SVP gradient with RI-JK on 8 threads:

```bash
linus mol.xyz --method b3lyp --dispersion d3bj --basis def2-svp --jk df --grad --threads 8 --json > out.json
```

```python
import linus
from qcdata import ProgramInput, Structure

inp = ProgramInput(program="linus", calctype="gradient", structure=Structure.open("mol.xyz"),
                   model={"method": "b3lyp-d3bj", "basis": "def2-svp"}, keywords={"jk": "df", "threads": 8})
out = linus.compute(inp)                     # one calculation in this process: a qcdata ProgramOutput
out.results.energy, out.results.gradient, out.results.energy_components    # Eh, Eh/bohr (natoms x 3)

with linus.Session(inp) as s:                # one molecule at many geometries
    for x in geometries:                     # bohr, natoms x 3: each SCF starts from the last one's orbitals
        f = s.compute(x)                     # a qcdata Frame
```

Both forms take the same settings under the same names (`linus --help`, `--set KEY=VALUE`,
`--list-functionals`/`--list-basis`/`--list-solvents`; `memory_gib` bounds the memory), every
result reports what was actually used in `resolved`, and errors are typed (Python `InputError`,
`UnsupportedError`, `ResourceError`, `ConvergenceError`, all `qcdata.ProgramError`s; CLI exit code
2 for bad input and an `error_kind` in the JSON; a failed calculation raises in Python, with what
it reached). `qccompute.compute([inp, ...])` runs many linus calculations, each on its share of the
machine, and qcopt runs linus as its subprograms (sessions) for optimizations, transition states
and IRCs ([docs/usage.md](docs/usage.md#4-transition-states-and-geometry-optimizations)). The
Python package: [docs/python.md](docs/python.md).
The command line, every setting, the request and the result: [docs/cli.md](docs/cli.md).
Recipes (implicit solvent, frequencies and thermochemistry, optimizations, TS searches and IRCs
through qcopt, ECPs, VV10, choosing `jk` and the grid, hard SCFs), each in CLI and Python form:
[docs/usage.md](docs/usage.md). Sessions, lineages and the worker: [docs/python.md](docs/python.md).

Design rules:
- **Structured results, not logs.** Every result is a typed object (Rust struct -> Python object ->
  qcdata `ProgramOutput`); the CLI emits JSON. Human-readable progress goes to stderr and is never
  the source of results. Anything written to disk is structured (JSON plus binary arrays, or
  TREXIO for wavefunctions); Molden is an export for viewers only.
- **Explicit threading.** One thread pool per calculation, sized by the caller.
- **Correct by default, reproducible on request.** Presets reproduce other programs' grids and
  conventions exactly for validation (see `reference/`).
- **Backend boundary at the J/K, XC and response builders.** The SCF's J/K builds
  (`JkBuilder`), the J/K of the CPHF/CPKS responses (`ResponseJk`), the XC potential
  (`XcPotential`) and the CPKS XC kernel (`ResponseKernel`) are traits; the GPU backend
  (`backend: "gpu"`, [docs/gpu.md](docs/gpu.md)) implements them, and the gradient's
  two-electron and XC terms, beside the CPU's. The backend is resolved per family of kernels,
  each falling back to the CPU with its reason in `resolved.record.backend`.

## Testing

```bash
cargo test --profile test-fast --workspace  # unit and integration tests (incl. feature combinations)
cargo build --release --workspace           # crates/linus-py/tests/test_vib.py calls target/release/linus
(cd crates/linus-py && maturin develop --release --extras test && PYTHONPATH=python pytest tests -q)   # Python package
.refenv/bin/python reference/validate.py --quick --refs none        # ~4 min: linus against stored reference results
```

The tests build under the `test-fast` profile (Cargo.toml): the release profile without thin
LTO, in 16 codegen units. It computes the same numbers as release, bit for bit (Rust has no
fast-math, so inlining and code layout do not change floating-point results), and it builds
much faster, because thin LTO compiles the whole program again for each of the ~90 test and
example binaries. On 8 cores a cold test build takes 5 minutes instead of 21, and a rebuild
after a one-line edit in linus-scf takes 21 seconds instead of 12 minutes; the tests run as
long (under 4 minutes). Timings and the binaries users get are built with `--release`
(`--profile dist` for release artifacts).

`validate.py --quick` compares the current binary with stored reference results and writes to
`reference/out/` (git-ignored; `--store` writes into `reference/validation/`); exit code 0 means
no unexplained differences. The reference environment (`.refenv`: PySCF 2.15, Psi4 1.11) is
set up as in [reference/README.md](reference/README.md). Deeper per-feature checks against
PySCF and Psi4 are in [tests/python/README.md](tests/python/README.md); what is validated
where: the validation map in [reference/validation/README.md](reference/validation/README.md).
The full gate before a merge, the determinism rule and how to regenerate generated code:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

```
crates/linus-num    deterministic accumulation and the run-time dispatched SIMD layer
crates/linus-core   the shared vocabulary: units, elements, Structure, spin channels, error kinds,
                    memory budgets, arithmetic, timings and the run record
crates/linus-ints   Gaussian integrals: the basis set library, the McMurchie-Davidson engine
                    (SHARK form, core/), one-electron and ECP integrals (one_e/), four- and
                    three-center ERIs (eri4c/, eri3c/), integrals on points (points/), and
                    their nuclear derivatives
crates/linus-grid   molecular integration grids (linus grids; PySCF/Psi4-compatible grid presets)
crates/linus-xc     exchange-correlation functionals (libxc 7.0.0 bindings)
crates/linus-disp   DFT-D3(BJ) and D4 dispersion with analytic Hessians (LGPL-3.0-or-later)
crates/linus-gpu    the GPU engines (CUDA loaded at run time; feature gpu)
crates/linus-base   errors, deterministic dense linear algebra, memory limits, and the method
                    families' terms (the traits the SCF and the responses call them through)
crates/linus-jk     J and K: exact, density fitting (RI-JK, RI-J), COSX; their derivatives
crates/linus-dft    the XC quadrature (energies, potentials, gradients, Hessians, CPKS kernels,
                    VV10)
crates/linus-scf    the SCF, linear response (CPHF/CPKS) and the reference determinant
crates/linus-solvent  continuum solvation (PCM, SMD) and the solvent database
crates/linus-vib    harmonic analysis and thermochemistry of any program's Hessian
crates/linus        the facade (see ARCHITECTURE.md): driver/ (Session, settings, plan,
                    report), backend/ (CPU and GPU providers), derivatives/ (gradients,
                    Hessians), properties/ (dipoles, polarizability), the J/K and solvent
                    dispatch, dispersion
crates/linus-py     Python package `linus` (Session and compute on qcdata, qccompute's program protocol)
crates/linus-cli    `linus` command-line program (JSON in/out)
crates/linus-dev    study and benchmark programs of development (examples/; not published)
data/basis          basis sets (Basis Set Exchange JSON)
docker/             container images (Dockerfile.gpu, Dockerfile.cpu, build.sh)
docs/               usage recipes, CLI reference, grids, solvation, qcdata proposal
reference/          reference harness (PySCF, Psi4), validation matrix, program quirks
tests/              molecules and the Python deep checks against PySCF and Psi4
bench/              benchmark molecules and levels, campaign scripts (bench/README.md)
tools/              generators: basis export, integral kernels (intgen), Rys tables, solvent tables
```

## Documentation

- [docs/usage.md](docs/usage.md): recipes for the CLI and Python.
- [docs/python.md](docs/python.md): the Python package: one calculation, sessions and lineages, many calculations through qccompute (grants, a result store, workers), converters.
- [docs/cli.md](docs/cli.md): the command line, every setting, the JSON request and result.
- [docs/grids.md](docs/grids.md): choosing a DFT grid (defaults, `linus:2` for speed, VV10 grids).
- [docs/solvation.md](docs/solvation.md): implicit solvent (PCM models, presets, SMD) and its validation.
- [docs/gpu.md](docs/gpu.md): the GPU backend (building, what runs where, precision, validation).
- [reference/validation/README.md](reference/validation/README.md): the validation matrix and map.
- [reference/README.md](reference/README.md): reference programs, settings and quirks.
- [tests/python/README.md](tests/python/README.md): per-feature deep checks.
- [bench/README.md](bench/README.md): benchmark scripts.
- [ARCHITECTURE.md](ARCHITECTURE.md): the crates, the module map, layering and the backend boundary.
- [CHANGELOG.md](CHANGELOG.md): changes to the command line, the JSON and the Python package.
- [CONTRIBUTING.md](CONTRIBUTING.md): building, the test gate, determinism, generated code.

## Citing

[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository"). Cite also the methods,
functionals, basis sets (Basis Set Exchange) and libxc you use.

## License

MIT ([LICENSE](LICENSE)). crates/linus-disp (D3/D4 dispersion) is LGPL-3.0-or-later and on by
default (cargo feature `dispersion` of `linus`, `linus-cli` and `linus-py`), so default builds
of the CLI and the Python package contain it; `--no-default-features` leaves it out. libxc
(MPL-2.0) is linked statically, and the basis set data are BSD-3-Clause.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists every third-party part, its license and
provenance; the license texts are in [LICENSES/](LICENSES).

ORCA: the comparisons with ORCA, the ORCA reference values in the validation records and tests,
and the ORCA-compatible presets (the `orca:N` grids, which reproduce ORCA's DefGrid layouts as
determined from ORCA's printed output; the `orca` PCM preset; ORCA's solvent table) are included
with permission of FAccTs GmbH. They come from ORCA 6.0.0; each comparison states its settings.
linus contains no ORCA code or binaries.
