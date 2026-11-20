# Contributing to linus

Build requirements are in the [README](README.md#build-and-install). This file has the checks
a change must pass, the rules the code keeps, and how the generated code is made.

## Rules

- **Determinism.** Results are bitwise identical at any thread count (check 1, 3 and 8
  threads). Parallel code writes into per-task buffers and reduces them in a fixed order; no
  floating-point atomics, no reductions whose order depends on scheduling. A change that moves
  a result by round-off says so in its commit message and quantifies it (against the previous
  commit and against PySCF).
- **Dense linear algebra goes through `linus_base::linalg`.** faer computes a product with one
  result row or column (a matrix-vector product, a triangular solve with one right-hand side)
  as `par.degree()` partial sums, so with the calculator's parallelism its rounding would
  follow the thread count. `linus_base::linalg` runs those shapes sequentially (`matmul`,
  `gemm`, `mm`, `solve_lower_in_place`, `solve_upper_in_place`), the eigensolver with a fixed
  degree (`eigh`), and factorizations with an explicit parallelism (`cholesky`,
  `cholesky_solve`, `lu_solve`, `svd`). Library code does not call faer's products, triangular solves,
  convenience solvers (`Solve::solve`) or factorizations (`Mat::llt`, `Mat::svd`) directly:
  clippy.toml lists them as disallowed methods, and clippy fails on them. Nor does it use faer's
  matrix operators (`&a * &b`), which, like the convenience methods, run with faer's global
  parallelism (the size of the current thread pool); clippy cannot see operators, so
  `crates/linus/tests/faer_parallelism.rs` runs calculations with that parallelism disabled. An
  exception (a fixed degree that does not depend on the thread count, as the PCM's dense
  solves) takes `#[allow(clippy::disallowed_methods)]` on the function, with the reason.
- **Numbers do not change by accident.** Refactorings keep results bitwise identical; compare
  the energies, gradients and Hessians of the reference cases before and after.
- **Structured results, not logs.** Results are typed (Rust structs, Python objects, the CLI's
  JSON); progress goes to stderr and is never parsed.
- **Errors are typed.** Bad input is `LinusError::Input` (CLI exit code 2), a valid request
  linus does not implement is `Unsupported`, and so on (`error_kind` in
  [docs/cli.md](docs/cli.md)); library code returns errors instead of panicking.
- **Interfaces.** The command line, the JSON request and result, and the Python package are the
  supported interfaces (before 1.0 a minor release may change them). Any change to them gets an
  entry under "Unreleased" in
  [CHANGELOG.md](CHANGELOG.md); a breaking change to the result or failure JSON also raises
  `SCHEMA_VERSION` (`schema_version` in the JSON). The Rust crates are unstable (0.x).
- **Third-party code and data.** Record where every table or algorithm port comes from, with
  its license, in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (license texts in
  `LICENSES/`). No code from GPL-only sources.
- **Commits.** Small, logical commits; formatting changes in their own commit.

## The gate

Run these before a change is merged. On a shared machine, cap the memory of each job (for
example `systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 <command>`).

1. Rust tests, under the `test-fast` profile (the release profile without thin LTO: the same
   numbers, builds in a fraction of the time; [README.md](README.md#testing)):
   ```bash
   cargo test --profile test-fast --workspace
   ```
   This covers `crates/linus-gpu` with its `gpu` feature off: the kernel source generators,
   the harness and the deterministic sums, no device. Changes to GPU code also need the
   kernel tests on a machine with a GPU and the CUDA 13 libraries (CI has no GPU):
   ```bash
   LINUS_GPU_REQUIRED=1 cargo test --profile test-fast -p linus-gpu --features gpu
   ```
2. The Python package, pinned to this tree:
   ```bash
   cargo build --release --workspace            # the CLI tests call target/release/linus
   (cd crates/linus-py && maturin develop --release --extras test)
   PYTHONPATH=crates/linus-py/python pytest crates/linus-py/tests -q
   ```
3. The validation matrix against the stored reference results (needs the reference
   environment, [reference/README.md](reference/README.md)); exit code 0 and no unexplained
   cells:
   ```bash
   .refenv/bin/python reference/validate.py --quick --refs none
   ```
4. The PySCF checks ([tests/python/README.md](tests/python/README.md#merge-gate)), with the
   example binaries of step 1's build; each exits 1 (`FAIL` on its line) when a difference is
   above its tolerance; with `LINUS_PYSCF_CACHE=<dir>` PySCF's results are reused between runs:
   ```bash
   export LINUS_EXAMPLES=target/test-fast/examples
   .refenv/bin/python tests/python/compare_grad_pyscf.py tests/molecules/ch3sh.xyz def2-tzvp hf --tol-e 1e-11 --tol-g 2e-9
   .refenv/bin/python tests/python/compare_hess_pyscf.py tests/molecules/nh3.xyz def2-tzvp hf 8 --tol-e 1e-11 --tol-h 3e-9
   ```
5. For changes to the numerics: the energies, gradients and Hessians of the reference cases are
   bitwise identical to the previous commit at 1 and 8 threads. For changes to hot code: Hessian
   timings before and after, alternating runs on the same cores, of `--release` builds (the
   binaries users run).

Formatting and lints: `cargo fmt --all` (CI checks `cargo fmt --all --check`),
`cargo clippy --workspace --all-targets -- -D warnings` and
`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`; CI's lint job fails on any
warning, with the toolchain pinned in ci.yml (raise it on purpose). A lint that is wrong for
numerical code (`needless_range_loop` where the index addresses several arrays,
`too_many_arguments` on a kernel entry point, `type_complexity`) is allowed on the function,
with the reason (`#[allow(clippy::..., reason = "...")]`); a hot kernel's code is never
changed to satisfy a lint. CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the
lints, the Rust tests and pytest against the wheel (with the CLI) on every push to main and every
pull request, on the CPU and without the slow tests.

## Generated code

Generated files start with an `@generated` header naming their generator; never edit them by
hand. Change the generator, rerun it, and commit both.

| Files | Generator | Needs |
|---|---|---|
| `crates/linus-ints/src/generated/*.rs` (integral kernels) | `python3 tools/intgen/intgen.py` | standard library |
| `crates/linus-gpu/src/kernels/eri3c_gen/` (GPU three-center kernels) | `python3 tools/intgen/intgen.py cuda` (`--check` compares) | standard library |
| `crates/linus-ints/src/core/rys_tables.rs`, `rys_ref.txt` | `python tools/gen_rys_tables.py` (about a minute) | mpmath |
| `crates/linus-grid/src/element_data.rs` | `crates/linus-grid/tools/gen_element_tables.py` | PySCF 2.15 |
| `crates/linus-grid/src/lebedev_data.rs` | `crates/linus-grid/tools/gen_lebedev_table.py` | SciPy >= 1.15 (PySCF optional, a cross-check) |
| `crates/linus-disp/src/data/*.rs` | `python3 -I crates/linus-disp/tools/gen_data.py --src DIR [--fetch]` | the release tarballs it lists (SHA-256 checked) |
| `crates/linus-solvent/src/data/crc.*` | `tools/sync_qcref_solvents.py` (`--check` compares; the shipped rows are listed there) | qcref |
| `crates/linus-solvent/src/data/{smd_minnesota,program_eps,aliases}.csv` | `tools/gen_solvent_db.py` | its inputs (docstring) |
| `data/basis/*.json` | `tools/fetch_basis.py [names]` | basis_set_exchange |

A new basis set in `data/basis` also needs a symlink in `crates/linus-py/python/linus/basis`
so that the wheel ships it (`crates/linus-py/pyproject.toml`).

## Releasing

1. Bump `version` in `[workspace.package]` (Cargo.toml) and update Cargo.lock; the Python
   package takes it from there. Retake the golden CLI outputs, whose `version` moves.
2. Move the "Unreleased" entries of CHANGELOG.md under the new version; check that
   `SCHEMA_VERSION` was raised if the result changed incompatibly.
3. The gate above on the release commit, and green CI.
4. Build the two wheels from the release commit, each in the manylinux_2_28 image
   (`quay.io/pypa/manylinux_2_28_x86_64`), with `LINUS_DIST=1` in the environment (release
   builds then carry no compile-time paths into the source tree, such as the `data/basis`
   fallback; tests and development builds leave it unset):
   - `afh-linus`: `cd crates/linus-py && maturin build --profile dist --compatibility manylinux_2_28`;
   - `afh-linus-cuda13`: in a copy of the tree, `python3 tools/cuda13_distribution.py`, then
     the same with `--features gpu`.

   Both are `<name>-X.Y.Z-cp311-abi3-manylinux_2_28_x86_64.whl` with the same version. Check
   them with `twine check`, and that each wheel's `.dist-info/licenses/` holds the license files
   (`unzip -l`).
5. Install checks, each wheel in a fresh virtual environment with the qc* releases from PyPI:
   `import linus` (`linus.GPU_BUILD` False and True), `qccompute.testing.check_program` on
   `linus.qc.program`, an energy and gradient on water through `qccompute` (the GPU wheel's with
   `backend: "gpu"` on a GPU), and `linus.available_basis_sets()`.
6. Tag `vX.Y.Z` and publish a GitHub release of that tag with both wheels attached.
   [.github/workflows/publish.yml](.github/workflows/publish.yml) then uploads them to PyPI by
   trusted publishing: `afh_linus-X.Y.Z-*.whl` to the project `afh-linus` (GitHub environment
   `pypi`) and `afh_linus_cuda13-X.Y.Z-*.whl` to `afh-linus-cuda13` (environment
   `pypi-cuda13`). Nothing is built in that workflow. A GitHub release marked as a pre-release
   (wheels for testers, `X.Y.ZaN`) uploads nothing.
