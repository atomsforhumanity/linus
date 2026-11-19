# Draft issue for PySCF: the analytic SMD G_CDS gradient is wrong for some molecules (area derivatives)


**Title:** `solvent.smd`: analytic G_CDS gradient off by ~1e-3 Eh/bohr and not translation invariant (ethane, CH3SH): the area derivatives in `mnsol.F` (`DAREAL`)

**Version:** PySCF 2.15.0 (PyPI wheel). `pyscf/lib/solvent/mnsol.F` and `mnsol_interface.f90` are
byte-identical in v2.15.0 and on master (checked 2026-11-10), so master has the same behaviour.

**What happens.** `smd.get_cds_legacy` returns G_CDS and its gradient from the MNSOL routine
(`mnsol_interface_` → `CDSSET` → `CDS_EG`). `pyscf/solvent/grad/smd.py` (`get_cds`) uses that
gradient. For some molecules it is not the derivative of the G_CDS that the same call returns:

| molecule (water) | max \|analytic − FD\| (Eh/bohr) | \|Σ_atoms analytic\| | \|Σ_atoms FD\| | max \|gradient\| |
|---|---|---|---|---|
| ethane (GMTKN55 DIPCS10 geometry) | 1.09e-3 | 8.9e-4 | 1e-13 | 1.58e-3 |
| ethane (idealized staggered, C-C 1.54 Å) | 1.26e-3 | 2.2e-3 | 1e-13 | 1.65e-3 |
| CH3SH (slightly distorted, below) | 1.27e-3 | 1.15e-3 | 6e-14 | 1.50e-3 |
| aspirin | 9.8e-10 | 3e-18 | | 3.1e-3 |
| caffeine (toluene) | 4.7e-10 | | | 2.3e-3 |

FD: 5-point central differences, step 1e-4 bohr, of the G_CDS energy from `get_cds_legacy`. The
error is as large as the gradient itself. The analytic gradient is **not translation invariant**
(its sum over the atoms is ~1e-3, the same after moving the molecule), while the differences sum
to 1e-13. G_CDS depends on the geometry alone (no basis, no SCF). An independent SMD implementation
(linus, AFH's HF/DFT code) gives the same G_CDS energies to 2e-12 Eh, and its analytic SMD
gradient agrees with differences of its own energy to 3e-10 on these molecules: the energy is
right and smooth, only PySCF's analytic gradient is wrong.

**Where.** It is in the analytic **area derivatives**, not in SMD's surface-tension switching
functions. With custom `solvent_descriptors` that leave only the macroscopic γ term (n = α = β =
φ = ψ = 0, γ = 50; non-aqueous branch), every atomic surface tension is a constant, so the
gradient is σ · dA/dx. It is still wrong: max \|analytic − FD\| 2.2e-4 Eh/bohr for ethane, where
the true gradient is at most 1.2e-4, and the sum over atoms is 1.8e-4. Each of the n, α and β terms
alone shows the same (2-4e-4). The area derivatives come from `SUBROUTINE DAREAL` in
`pyscf/lib/solvent/mnsol.F` (called from `CDS_EG`). Our guess is a case of the analytic
circle-intersection derivatives that these geometries hit (several symmetry-equivalent sphere
pairs; all six atoms are needed for CH3SH to show it). Not traced further.

**Reproducer** (public PySCF only):

```python
import numpy as np
from pyscf import gto, scf
from pyscf.solvent import smd

ETHANE = [("C", 0.0, 0.0, 0.766009), ("C", 0.0, 0.0, -0.766009),            # GMTKN55 DIPCS10, angstrom
          ("H", -0.884402, 0.51061, 1.162263), ("H", 0.0, -1.021219, 1.162263),
          ("H", 0.884402, 0.51061, 1.162263), ("H", 0.884402, -0.51061, -1.162263),
          ("H", -0.884402, -0.51061, -1.162263), ("H", 0.0, 1.021219, -1.162263)]
CH3SH = [("C", -0.047, 1.149, 0.010), ("S", -0.042, -0.669, -0.005), ("H", 1.288, -0.937, 0.020),
         ("H", -1.087, 1.485, -0.010), ("H", 0.451, 1.530, 0.893), ("H", 0.431, 1.545, -0.883)]

def check(atoms, solvent="water", descriptors=None, h=1e-4):
    syms = [a[0] for a in atoms]
    x0 = np.array([a[1:] for a in atoms]) / 0.529177210544   # bohr
    def cds(x):
        mol = gto.M(atom=list(zip(syms, x.tolist())), unit="Bohr", basis="sto-3g", verbose=0)
        so = scf.RHF(mol).SMD().with_solvent
        so.solvent = solvent
        if descriptors is not None:
            so.solvent_descriptors = descriptors
        return smd.get_cds_legacy(so)
    e0, g0 = cds(x0)
    g0 = np.asarray(g0)
    fd = np.zeros_like(x0)
    for a in range(len(x0)):
        for k in range(3):
            e = []
            for s in (-2, -1, 1, 2):
                x = x0.copy(); x[a, k] += s * h; e.append(cds(x)[0])
            fd[a, k] = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * h)
    print(f"max|analytic - FD| {abs(g0 - fd).max():.2e}   |sum analytic| {abs(g0.sum(0)).max():.2e}   "
          f"|sum FD| {abs(fd.sum(0)).max():.2e}")

check(ETHANE)        # max|analytic - FD| 1.09e-03   |sum analytic| 8.89e-04   |sum FD| 1.0e-13
check(CH3SH)         # max|analytic - FD| 1.27e-03   |sum analytic| 1.15e-03   |sum FD| 6.1e-14
# only the macroscopic gamma term: constant surface tensions, so the gradient is sigma * dA/dx
check(ETHANE, solvent="toluene", descriptors=[0.0, 0.0, 0.0, 0.0, 50.0, 2.0, 0.0, 0.0])   # 2.2e-04, sum 1.8e-04
```

(The CH3SH geometry is linus's validation molecule, `tests/molecules/ch3sh.xyz`.)

**Workaround used by AFH:** our PySCF reference takes the G_CDS gradient as central differences of
`get_cds_legacy`'s energy.
