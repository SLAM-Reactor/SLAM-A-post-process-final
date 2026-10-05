#!/usr/bin/env python3
import re
import sys
from pathlib import Path

INIT = "init_average_axis = true"
PHI = "PHIEDGE = 0.01"

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file.toml>")
    sys.exit(1)

p = Path(sys.argv[1])
if not p.is_file():
    print(f"File not found: {p}")
    sys.exit(1)

init_re = re.compile(r"^\s*init_average_axis\s*=")
phi_re = re.compile(r"^\s*PHIEDGE\s*=")

lines = p.read_text().splitlines()

# Remove existing PHIEDGE lines so we can place exactly one in the right spot
lines = [ln for ln in lines if not phi_re.match(ln)]

# Find init_average_axis in first 20 lines
init_idx = next((i for i, ln in enumerate(lines[:20]) if init_re.match(ln)), None)

if init_idx is None:
    while len(lines) < 18:
        lines.append("")
    if lines[17].strip() == "":
        lines[17] = INIT
    else:
        lines.insert(17, INIT)
    init_idx = 17
else:
    lines[init_idx] = INIT  # normalize format

# Insert PHIEDGE immediately after init_average_axis
lines.insert(init_idx + 1, PHI)

p.write_text("\n".join(lines) + "\n")
print(f"Updated: {p}")




