#!/usr/bin/env python3
"""The layering of the workspace's crates: every dependency between linus crates is one the
documented graph allows.

The graph is the table `LAYERS` below and the table in ARCHITECTURE.md's "Crates" section (between
the `layering:begin` and `layering:end` comments). This script checks that the two agree and that
`cargo metadata` shows exactly the normal dependencies they list (optional ones included), each to a
lower layer, and only the dev-dependencies `DEV` allows. A new edge between linus crates is a
design decision: change both tables in the same commit. CI's lint job and the gate's chain L run it.

    python3 tools/check_layering.py        (exit status 1 on any violation)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# crate: (layer, its normal dependencies among the linus crates)
LAYERS = {
    "linus-num": (0, set()),
    "linus-core": (0, set()),
    "linus-ints": (1, {"linus-num", "linus-core"}),
    "linus-grid": (1, {"linus-num", "linus-core"}),
    "linus-xc": (1, set()),
    "linus-disp": (1, set()),
    "linus-base": (2, {"linus-core", "linus-ints", "linus-grid", "linus-xc"}),
    "linus-gpu": (2, {"linus-num", "linus-core", "linus-ints", "linus-grid", "linus-xc"}),
    "linus-jk": (3, {"linus-num", "linus-core", "linus-ints", "linus-grid", "linus-base"}),
    "linus-dft": (3, {"linus-num", "linus-core", "linus-ints", "linus-grid", "linus-xc", "linus-base"}),
    "linus-scf": (3, {"linus-num", "linus-core", "linus-ints", "linus-base"}),
    "linus-solvent": (3, {"linus-num", "linus-core", "linus-ints", "linus-grid", "linus-base"}),
    "linus-vib": (3, {"linus-core", "linus-base"}),
    "linus": (
        4,
        {
            "linus-num", "linus-core", "linus-ints", "linus-grid", "linus-xc", "linus-disp", "linus-base",
            "linus-gpu", "linus-jk", "linus-dft", "linus-scf", "linus-solvent", "linus-vib",
        },
    ),
    "linus-cli": (5, {"linus"}),
    "linus-py": (5, {"linus"}),
    "linus-dev": (
        5,
        {
            "linus-core", "linus-ints", "linus-grid", "linus-xc", "linus-base", "linus-jk", "linus-dft",
            "linus-scf", "linus-solvent", "linus-vib", "linus",
        },
    ),
}

# Dev-dependencies beyond a crate's own normal ones: the crate itself with a test feature, and a
# sibling's helper (linus-dft's tests take their SAD densities from linus-scf; no cycle).
DEV = {
    "linus-ints": {"linus-ints"},
    "linus-gpu": {"linus-gpu"},
    "linus-dft": {"linus-scf"},
}

BEGIN = "<!-- layering:begin -->"
END = "<!-- layering:end -->"


def documented():
    """ARCHITECTURE.md's table: crate -> (layer, dependencies)."""
    text = (ROOT / "ARCHITECTURE.md").read_text()
    if BEGIN not in text or END not in text:
        sys.exit(f"ARCHITECTURE.md: no {BEGIN} ... {END} table")
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    table = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].startswith("`"):
            continue
        table[cells[0].strip("`")] = (int(cells[1]), set(re.findall(r"`([a-z-]+)`", cells[2])))
    return table


def fmt(entry):
    if entry is None:
        return "nothing"
    layer, deps = entry
    return f"layer {layer} on {{{', '.join(sorted(deps)) or 'none'}}}"


def main():
    errors = []
    doc = documented()
    for name in sorted(set(doc) | set(LAYERS)):
        if doc.get(name) != LAYERS.get(name):
            errors.append(
                f"{name}: ARCHITECTURE.md says {fmt(doc.get(name))}, tools/check_layering.py {fmt(LAYERS.get(name))}"
            )
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    workspace = {p["name"] for p in meta["packages"]}
    for name in sorted(workspace - set(LAYERS)):
        errors.append(f"{name}: a workspace crate missing from the layering table")
    for p in meta["packages"]:
        name = p["name"]
        if name not in LAYERS:
            continue
        layer, allowed = LAYERS[name]
        deps = [d for d in p["dependencies"] if d["name"] in workspace]
        normal = {d["name"] for d in deps if d["kind"] is None}
        dev = {d["name"] for d in deps if d["kind"] == "dev"}
        build = {d["name"] for d in deps if d["kind"] == "build"}
        for d in sorted(normal - allowed):
            errors.append(f"{name} depends on {d}: not in the documented graph")
        for d in sorted(allowed - normal):
            errors.append(f"{name}: the documented graph lists {d}, which it does not depend on")
        for d in sorted(normal):
            if d in LAYERS and LAYERS[d][0] >= layer:
                errors.append(f"{name} (layer {layer}) depends on {d} (layer {LAYERS[d][0]}): not a lower layer")
        for d in sorted(dev - allowed - DEV.get(name, set())):
            errors.append(f"{name} has the dev-dependency {d}: not allowed (DEV in tools/check_layering.py)")
        for d in sorted(build):
            errors.append(f"{name} has the build-dependency {d}: none allowed")
    if errors:
        print(f"layering: {len(errors)} violation(s)")
        for e in errors:
            print("  " + e)
        return 1
    print(f"layering: {len(LAYERS)} crates, every edge as documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
