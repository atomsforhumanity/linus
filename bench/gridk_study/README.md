# Compensated grid-K's accuracy study

The fixed-density study behind `jk: "rij_cgk"`'s opt-in status (docs/internal/h2/gridk.md,
"Accuracy study"): for every species of the 16 GMTKN55 subsets of Hohenstein et al. (and,
optionally, the repository's molecules), the SCF density converged once with RI-JK, then at
that density the exchange and Coulomb energies (and, with `--grad`, their gradients) of exact
four-center J and K, of density fitting with each auxiliary basis, and of grid-K in its four
variants (raw, dSCF, overlap fit, compensated) on each grid. Their errors against the exact
ones are the first-order errors of the SCF energy and the two-electron gradient.

- `crates/linus-dev/examples/gridk_study.rs`: one species (prints one JSON object).
- `make_study.py`: a run directory of tasks (one species and basis each) for CPU pods
  (`run.yaml`: image, shards, threads; the GMTKN55 subsets from their `<subset>.tar` files).
- `study_task.py`: one task in a pod (runs `gridk_study`, writes `result.json`).
- `analyze_study.py`: the tables (absolute errors per atom, reaction energies, gradients) from
  the pulled run directories.

```bash
cargo build --release -p linus-dev --example gridk_study
python3 bench/gridk_study/make_study.py runs/svp target/release/examples/gridk_study \
    def2-svp,def2-tzvp --image <linus image> --gmtkn-tars <dir of GMTKN55 tars> \
    --subsets paper16 --grad --max-atoms 1000
# submit runs/svp as a CPU run, pull its output, then
python3 bench/gridk_study/analyze_study.py runs/svp --json svp.json
```
