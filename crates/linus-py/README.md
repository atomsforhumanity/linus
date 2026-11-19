# linus (Python package, distributed as afh-linus and afh-linus-cuda13)

Hartree-Fock and Kohn-Sham DFT in-process: energies, analytic gradients, analytic Hessians,
frequencies and thermochemistry, for closed and open shells, with global, range-separated and
VV10 functionals, D3(BJ)/D4 dispersion, ECPs and implicit solvent (PCM, SMD), on the CPU or an
NVIDIA GPU (the `afh-linus-cuda13` build). The core is Rust; results are bitwise
identical at any thread count. The package speaks
[qcdata](https://github.com/atomsforhumanity/qcdata): `linus.compute(ProgramInput)` for one
calculation, `linus.Session(ProgramInput)` for one molecule at many geometries (each step starts
from its predecessor's orbitals), and qccompute's program protocol (`linus.qc.program`,
registered as the `qccompute.programs` entry point `linus`).

```sh
pip install afh-linus            # the CPU build, with numpy and qcdata
pip install afh-linus-cuda13     # or: the same with the GPU backend (NVIDIA GPUs, CUDA 13)
pip install qccompute            # to run many calculations (lists, a result store, workers)
```

Both distributions have the same version and install the package `linus`: install one of them,
not both. `afh-linus-cuda13` runs everything `afh-linus` does, and with `backend: "gpu"` on an
NVIDIA GPU; at run time it needs an NVIDIA driver with CUDA 13 support (580 or newer) and the CUDA
13 libraries NVRTC, cuBLAS and cuSOLVER on the dynamic loader's path (a CUDA 13 toolkit's `lib64`,
or `LD_LIBRARY_PATH`). Without them it still runs on the CPU. `linus.GPU_BUILD` is True in the GPU
build. Linux x86-64 (glibc 2.28 or newer), Python >= 3.11. The basis sets (Basis Set Exchange
JSON) ship in the package; `LINUS_BASIS_DIR` points linus at another directory of them.

```python
import linus
from qcdata import ProgramInput, Structure

inp = ProgramInput(program="linus", calctype="gradient", structure=Structure.open("mol.xyz"),
                   model={"method": "b3lyp-d3bj", "basis": "def2-svp"}, keywords={"threads": 8})
out = linus.compute(inp)                            # a qcdata.ProgramOutput; raises if it fails
out.results.energy, out.results.gradient, out.resolved.settings    # Eh, Eh/bohr; what it used

with linus.Session(inp) as s:
    f = s.compute(x)                                # positions in bohr: a qcdata.Frame
    h = s.compute(x, calctype="hessian", outputs=["ir"])   # Hessian, harmonic analysis, thermochemistry
    h.results.vibrational_analysis.frequencies, h.results.thermochemistry[0].gibbs_correction
```

A failed calculation raises a `linus.LinusError`, a `qcdata.ProgramError` carrying what it
reached (`results`; a session's failed `frame`, `linus.compute`'s failed `output`): `InputError`
(a `ValueError`), `UnsupportedError` (a `NotImplementedError`), `ResourceError` (a `MemoryError`),
`ConvergenceError`, `BackendError` (the GPU backend failed) and `NumericalError` (an
`ArithmeticError`), with qcdata's kind and code in `.error`.

Documentation: sessions and the Python API in
[docs/python.md](https://github.com/atomsforhumanity/linus/blob/main/docs/python.md), recipes
in [docs/usage.md](https://github.com/atomsforhumanity/linus/blob/main/docs/usage.md); every
setting and the result fields in
[docs/cli.md](https://github.com/atomsforhumanity/linus/blob/main/docs/cli.md).

License: MIT. The default build links linus-disp (D3/D4 dispersion, LGPL-3.0-or-later) and
libxc (MPL-2.0) and ships basis-set data under BSD-3-Clause; the wheel carries the notices and
license texts (`THIRD_PARTY_NOTICES.md`, `LICENSES/`).
