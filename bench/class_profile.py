"""Per-class CPU time of a J+K build, generic reference path vs FastJk.

python bench/class_profile.py <generic.json> <fast.json> [n]

<generic.json>: output of `jk_profile_generic` (old DirectJk loop: eri_quartet + digestion);
<fast.json>: output of `jk_bench --builder new --profile`. Prints a table of the n most
expensive classes (by the generic path) with CPU seconds summed over threads.
"""
import json
import sys


def main():
    g = json.load(open(sys.argv[1]))
    f = json.load(open(sys.argv[2]))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    fast = {c[0]: c for c in f["stats"]["classes"]}
    tg = g["cpu_total"]
    tf = sum(c[4] for c in f["stats"]["classes"])
    print(f"nbf {g['nbf']}: CPU total generic {tg:.2f} s, fast {tf:.2f} s ({tg / tf:.1f}x); "
          f"wall generic {g['wall']:.3f} s, fast {f['median']:.3f} s at {f['threads']} threads")
    print(f"{'class':8s} {'quartets':>11s} {'generic s':>9s} {'ns/q':>7s} {'fast s':>8s} {'ns/q':>7s} {'speedup':>7s}")
    for c in g["classes"][:n]:
        name = c["class"]
        t_g = c["t_int"] + c["t_digest"]
        fc = fast.get(name)
        if fc:
            t_f = fc[4]
            print(f"{name:8s} {c['quartets']:11d} {t_g:9.3f} {c['ns_per_quartet']:7.0f} {t_f:8.3f} {t_f / fc[2] * 1e9:7.0f} {t_g / t_f:6.1f}x")
        else:
            print(f"{name:8s} {c['quartets']:11d} {t_g:9.3f} {c['ns_per_quartet']:7.0f}")


if __name__ == "__main__":
    main()
