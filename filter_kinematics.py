"""
filter_kinematics.py
====================
Low-pass filter an OpenSim kinematics .sto file and write a new filtered .sto.

Usage:
    python filter_kinematics.py

Edit the CONFIG block below before running.
"""

import numpy as np

from b3d_io      import read_mot_sto, write_sto
from b3d_filters import apply_lpf_columns

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "/mnt/d/output/P010_split0/walk_fast_1_segment_0/ik.mot"
OUTPUT_FILE = "/mnt/d/output/P010_split0/walk_fast_1_segment_0/ik_filtered.mot"
CUTOFF_HZ   = 15.0   # low-pass cutoff frequency in Hz
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Reading:  {INPUT_FILE}")
    col_names, data = read_mot_sto(INPUT_FILE)

    # Time column (index 0): compute sampling rate from median dt
    time = data[:, 0]
    dt   = float(np.median(np.diff(time)))
    fs   = 1.0 / dt
    print(f"  Frames:          {len(time)}")
    print(f"  Duration:        {time[0]:.4f} s  to  {time[-1]:.4f} s")
    print(f"  Median sample rate: {fs:.2f} Hz")
    print(f"  Applying {CUTOFF_HZ} Hz low-pass filter...")

    # apply_lpf_columns skips column 0 (time) by default
    filtered = apply_lpf_columns(data, cutoff_hz=CUTOFF_HZ, fs=fs, skip_col0=True)

    # Reuse the shared .sto writer; fs is embedded in the time column already,
    # so we pass fs=1.0 and reconstruct time from the existing column.
    # Instead, write manually to preserve the original time vector exactly.
    from pathlib import Path
    out = Path(OUTPUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        # Reproduce the original header (all lines up to and including endheader)
        with open(INPUT_FILE) as src:
            for line in src:
                f.write(line)
                if line.strip().lower() == "endheader":
                    break
        f.write("\t".join(col_names) + "\n")
        for row in filtered:
            f.write("\t".join(f"{v:.10f}" for v in row) + "\n")

    print(f"Written:  {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
