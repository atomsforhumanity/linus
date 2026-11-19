"""Export basis sets from the Basis Set Exchange (python package) to data/basis/*.json.

Usage: python tools/fetch_basis.py [names...]   (default: the curated list below)
Records the BSE version in data/basis/VERSION.
"""
import json
import pathlib
import sys

import basis_set_exchange as bse

CURATED = [
    "sto-3g", "3-21g", "6-31g", "6-31g*", "6-31g**", "6-31+g*", "6-311g**", "6-311+g(2d,p)",
    "def2-svp", "def2-svpd", "def2-tzvp", "def2-tzvpp", "def2-tzvpd", "def2-tzvppd", "def2-qzvp", "def2-qzvpp",
    "def2-universal-jfit", "def2-universal-jkfit", "def2-svp-rifit", "def2-tzvp-rifit", "def2-tzvpp-rifit", "def2-qzvpp-rifit",
    "cc-pvdz", "cc-pvtz", "cc-pvqz", "aug-cc-pvdz", "aug-cc-pvtz",
    "cc-pvdz-jkfit", "cc-pvtz-jkfit", "cc-pvqz-jkfit", "cc-pvdz-rifit", "cc-pvtz-rifit",
    "aug-cc-pvdz-jkfit", "aug-cc-pvtz-jkfit",
    "pcseg-1", "pcseg-2", "ma-def2-svp", "ma-def2-tzvp",
]

# characters of basis set names that file names avoid (as linus_ints::basis::FILE_ESCAPES)
FILE_ESCAPES = (("*", "_st_"), ("+", "_pl_"), ("/", "_sl_"), ("(", "_lp_"), (")", "_rp_"), (",", "_cm_"))


def stem(name: str) -> str:
    s = name.lower()
    for c, e in FILE_ESCAPES:
        s = s.replace(c, e)
    return s

def main(names):
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "basis"
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        try:
            text = bse.get_basis(n, fmt="json")
        except Exception as e:  # noqa: BLE001
            print(f"skip {n}: {e}")
            continue
        d = json.loads(text)
        keep = {"name": d.get("name", n), "description": d.get("description", ""), "elements": d["elements"]}
        (out / f"{stem(n)}.json").write_text(json.dumps(keep, separators=(",", ":")))
        print(f"{n}: {len(d['elements'])} elements")
    (out / "VERSION").write_text(f"basis_set_exchange {bse.version()}\n")

if __name__ == "__main__":
    main(sys.argv[1:] or CURATED)
