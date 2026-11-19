# Draft issue for PySCF: SMD's CDS term uses ECP-reduced charges as atomic numbers


**Title:** `solvent.smd`: G_CDS treats atoms with an ECP as other elements (`mol.atom_charges()` passed as atomic numbers)

**Version:** PySCF 2.15.0 (PyPI wheel), `pyscf/solvent/smd.py`.

**What happens.** `get_cds_legacy(smdobj)` passes `mol.atom_charges()` to the MNSOL routine
(`mnsol_interface_`), which uses it as atomic numbers to look up SMD's CDS radii and atomic
surface tensions. For an atom with an ECP, `atom_charges()` is Z minus the core electrons. So
Xe (54 - 28 = 26) is treated as Fe, Sn (22) as Ti, I (25) as Mn, Bi (23) as V and Pt (18) as Ar.
The energy and the gradient (`pyscf/solvent/grad/smd.py`, which takes `dcds` from the same call)
are both affected. The electrostatic part is not: its cavity radii are right.

**Size** (HF/def2-SVP, SMD, 590 points per sphere; G_CDS in Eh):

| molecule | solvent | G_CDS with `atom_charges()` | with proton numbers | difference |
|---|---|---|---|---|
| XeF2 | water | 0.0033748160 | 0.0029856149 | +3.9e-4 |
| SnCl4 | toluene | -0.0067981690 | -0.0059537135 | -8.4e-4 (0.53 kcal/mol) |
| BiCl3 | toluene | -0.0047280157 | -0.0042985291 | -4.3e-4 |
| PtCl4²⁻ | toluene | -0.0060535259 | -0.0057072829 | -3.5e-4 |
| I2 | toluene | 0.0019313751 | 0.0019447636 | -1.3e-5 |

With the proton numbers, PySCF's G_CDS agrees to 2e-12 Eh with an independent implementation
(linus, AFH's HF/DFT code). Every def2 basis puts ECPs on Rb-Rn, so every SMD calculation with
those elements and a def2 basis is affected.

**Reproducer:**

```python
import numpy as np
from pyscf import gto, scf
from pyscf.solvent import smd

mol = gto.M(atom="Xe 0 0 0; F 0 0 1.98; F 0 0 -1.98", basis="def2-svp", ecp="def2-svp")
so = scf.RHF(mol).SMD().with_solvent
so.solvent = "water"
print("atom_charges():", mol.atom_charges())            # [26  9  9]: Xe as Fe
e_now = smd.get_cds_legacy(so)[0]
z = np.asarray([gto.charge(s) for s in mol.elements])   # [54  9  9]
mol.atom_charges = lambda: z
e_fixed = smd.get_cds_legacy(so)[0]
print(e_now, e_fixed, e_now - e_fixed)
```

**Fix** (one line in `get_cds_legacy`): pass the proton numbers, not the ECP-reduced charges:

```python
charges = np.asarray([gto.charge(s) for s in mol.elements], dtype=np.int32, order='F')
```

(`mol._atm` holds the ECP-reduced charges too, so it is no alternative). The module says it
was copied from GPU4PySCF, which may have the same line (not checked).

The table's numbers are for the sweep's GFN2-xTB geometries (bench/robust/molecules); the
reproducer's idealized XeF2 gives slightly different values of the same size.

**Related:** `grad.rks.grids_response_becke` has the same pattern for the Becke-Treutler radii
(known; AFH patches it in its reference runs).
