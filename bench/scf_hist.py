"""Print the per-iteration SCF history of a `linus --json` result (stdin or file).

usage: linus mol.xyz ... --json | python bench/scf_hist.py [-]
"""
import json, sys

d = json.load(open(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else sys.stdin)
h = d["scf"]["history"]
print(f"{'it':>3} {'energy':>22} {'dE':>10} {'gmax':>9} {'grms':>9} {'|dD|':>9} {'fock':>5} {'step':>9} {'t_jk':>7} {'t_xc':>7} {'t_step':>7}")
for r in h:
    print(f"{r['iter']:3d} {r['energy']:22.12f} {r['de']:10.2e} {r['gmax']:9.2e} {r['grms']:9.2e} {r['ddmax']:9.2e} "
          f"{r['fock'][:5]:>5} {r['step']:>9} {r['t_jk']:7.3f} {r['t_xc']:7.3f} {r['t_step']:7.3f}")
t = d["timings"]
print(f"E = {d['energy']:.12f}  converged {d['converged']}  fock builds {d['scf']['fock_builds']} "
      f"(incremental {d['scf']['incremental_builds']})  conv {d['scf']['convergence']}")
print("timings: " + ", ".join(f"{k} {v:.3f}" for k, v in t.items()))
