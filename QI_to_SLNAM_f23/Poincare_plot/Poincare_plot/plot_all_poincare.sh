#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NTRANSIT="${POINCARE_NTRANSIT:-400}"
CASES=(1p0 2p0)  # the two cases in Figure 3

cd "$HERE"

for case in "${CASES[@]}"; do
    data_file="$HERE/SLNAMified_L_over_R_${case}_poincare_data_ntransit${NTRANSIT}.npz"
    output_stem="$HERE/Poincare_plot_L_over_R_${case}_ntransit${NTRANSIT}"
    if [[ ! -f "$data_file" ]]; then
        echo "Missing data, skipping: $data_file"
        continue
    fi
    POINCARE_DATA_FILENAME="$data_file" \
    POINCARE_OUTPUT_STEM="$output_stem" \
    POINCARE_Y_LIMITS="${POINCARE_Y_LIMITS:-}" \
    python3 plot_poincare_trace.py
done
