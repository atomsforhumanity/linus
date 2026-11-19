# Reference program versions

Recorded 2026-11-06 on AFH's benchmark node (AMD Threadripper PRO 9985WX, 64 cores / 128 threads,
AVX-512, 246 GB RAM, Linux 7.0.0-38-generic). Full lock files: `reference/env/`
(`refenv.yml`, `refenv-explicit.txt` for `micromamba create --file`, `refenv-pip.txt`).

## Tools

| What | Version | Where |
|---|---|---|
| micromamba | 2.10.0 (static binary from `https://micro.mamba.pm/api/micromamba/linux-64/latest`) | `~/.local/bin/micromamba`; package cache `MAMBA_ROOT_PREFIX=~/micromamba` |
| reference env | conda-forge only | `.refenv` in the main checkout (git-ignored) |

## `.refenv` (conda-forge, plus pip where noted)

| Package | Version / build | Notes |
|---|---|---|
| python | 3.12.16 (`h5f976f7_0_cpython`) | |
| **psi4** | **1.11** (`py312h39b40a1_4`) | latest conda-forge build for py3.12 on 2026-11-06 |
| **pyscf** | **2.15.0** (PyPI wheel, released 2026-11-05) | conda-forge has only 2.14.0, so the conda build was removed and the wheel installed. The wheel bundles its own libcint 6.2.0 and libxc 7.1.2 (`pyscf/lib/deps`) |
| basis_set_exchange | 0.12 | |
| numpy / scipy | 2.5.4 / 1.18.1 | OpenBLAS 0.3.34 (`libblas 3.11.0 openblas`); mkl 2025.3.1 is also installed as a Psi4 dependency |
| **openmpi** | **4.1.6** (`hc5af2df_101`) | matches ORCA 6.0.0's build; provides `mpirun` and `libmpi.so.40` for ORCA |
| libxc-c | 7.1.2 (`cpu_hed2233d_2`) | used by Psi4 |
| libint | 2.13.1 | used by Psi4 |
| gau2grid / gauxc | 2.0.9 / 1.1 | Psi4 DFT collocation / grids |
| qcelemental / qcengine / optking | 0.51.4 / 0.52.0 / 0.5.0 | Psi4 dependencies (Psi4 uses CODATA 2014 internally; irrelevant here because geometries go in as bohr) |
| h5py | 3.16.0 (pip, pulled in by the pyscf wheel) | |
| **qcdata** | **0.23.1** (PyPI) | same version as `~/dev/afh/qcdata` |
| **qccompute** | **0.22.0** (PyPI) | same version as `~/dev/afh/qccompute` (that checkout has uncommitted changes, so PyPI was used) |
| **qccodec** | **0.21.0** (PyPI) | same version as `~/dev/afh/qccodec` |
| qcref | 0.5.1 (PyPI) | CODATA 2022: 0.529177210544 Å/bohr |

Created with:

```bash
export MAMBA_ROOT_PREFIX=$HOME/micromamba
micromamba create -y -p .refenv -c conda-forge --override-channels \
    python=3.12 'psi4=1.11=*_4' pyscf basis_set_exchange numpy scipy openmpi=4.1.6 pip
micromamba remove -y -p .refenv pyscf            # conda-forge pyscf is 2.14.0
.refenv/bin/python -m pip install pyscf==2.15.0 qcdata==0.23.1 qccompute==0.22.0 qccodec==0.21.0
```

## `.refenv-disp` (dispersion overlay, 2026-11-07)

PySCF 2.15 and Psi4 1.11 compute D3/D4 only through extra packages, which `.refenv` lacks.
Installed from PyPI wheels with `pip install --no-deps --target .refenv-disp` (so nothing in
`.refenv` changes); `run_ref.py --disp` puts it on `sys.path`/`PYSCF_EXT_PATH` (PySCF) and
`PYTHONPATH` (the Psi4 subprocess). Commands in `validation/README.md`.

| Package | Version | Used by |
|---|---|---|
| pyscf-dispersion | 1.7.0 (bundles libs-dftd3, libdftd4) | PySCF `mf.disp` |
| dftd3 | 1.7.0 (s-dftd3 Python API) | Psi4 `-d3bj` via qcengine's s-dftd3 harness |
| dftd4 | 4.3.1 | Psi4 `-d4` via qcengine's dftd4 harness |
| cffi / pycparser | 2.1.1 / 3.12 | dftd3, dftd4 |

## ORCA

| What | Value |
|---|---|
| Version | **ORCA 6.0.0** (`Program Version 6.0.0 - RELEASE -`), AVX2 shared build against Open MPI 4.1.6, without the autoci modules; built with libxc 6.2.2 |
| Installed at | `~/opt/orca-6.0.0`, or wherever `LINUS_ORCA_DIR` points. **Licensed: never copy it into the repo, commit or push it.** |


## MIRP

See `reference/MIRP.md` (commit, build env `.refenv-mirp`, data record).
