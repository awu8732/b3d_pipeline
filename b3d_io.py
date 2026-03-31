"""
b3d_io.py
=========
Shared file I/O utilities for the b3d_pipeline package.

Covers:
  - OpenSim file writers: .trc, .mot, .sto, body .json
  - OpenSim file readers: .mot/.sto, .trc  (used by validate_b3d_export)

Both process_b3d.py and validate_b3d_export.py previously contained
overlapping or mirrored logic for these formats; it now lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ── WRITERS ──────────────────────────────────────────────────────────────────

def write_trc(path: Path, marker_names: list, data: np.ndarray, fs: float) -> None:
    """
    Write an OpenSim .trc file.

    Parameters
    ----------
    path         : destination file path
    marker_names : list of marker label strings
    data         : (n_frames, n_markers * 3)  column order: X0 Y0 Z0 X1 Y1 Z1 ...
    fs           : sampling rate in Hz
    """
    n_frames  = data.shape[0]
    n_markers = len(marker_names)
    assert data.shape[1] == n_markers * 3, (
        f"Expected {n_markers * 3} columns, got {data.shape[1]}"
    )
    t = np.arange(n_frames) / fs
    with open(path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        f.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )
        f.write(
            f"{fs:.3f}\t{fs:.3f}\t{n_frames}\t{n_markers}\t"
            f"m\t{fs:.3f}\t1\t{n_frames}\n"
        )
        header1 = ["Frame#", "Time"] + [m for m in marker_names for _ in range(3)]
        header2 = ["", ""] + [ax for _ in marker_names for ax in ("X", "Y", "Z")]
        f.write("\t".join(header1) + "\n")
        f.write("\t".join(header2) + "\n")
        f.write("\n")
        for i in range(n_frames):
            row = [str(i + 1), f"{t[i]:.6f}"] + [f"{v:.6f}" for v in data[i]]
            f.write("\t".join(row) + "\n")
    print(f"    [trc] {n_frames} frames x {n_markers} markers -> {path.name}")


def write_mot(
    path: Path,
    col_names: list,
    data: np.ndarray,
    fs: float,
    header_name: str = "motion",
    in_degrees: bool = True,
) -> None:
    """Write an OpenSim .mot file (IK angles or GRF)."""
    n_frames = data.shape[0]
    t        = np.arange(n_frames) / fs
    all_cols = ["time"] + col_names
    all_data = np.column_stack([t, data])
    with open(path, "w") as f:
        f.write(header_name + "\n")
        f.write("version=1\n")
        f.write(f"nRows={n_frames}\n")
        f.write(f"nColumns={len(all_cols)}\n")
        f.write("inDegrees=" + ("yes" if in_degrees else "no") + "\n")
        f.write("endheader\n")
        f.write("\t".join(all_cols) + "\n")
        for row in all_data:
            f.write("\t".join(f"{v:.8f}" for v in row) + "\n")
    print(f"    [mot] {n_frames} frames x {len(col_names)} cols -> {path.name}")


def write_sto(
    path: Path,
    col_names: list,
    data: np.ndarray,
    fs: float,
    header_name: str = "inverse_dynamics",
) -> None:
    """Write an OpenSim .sto file (ID moments)."""
    n_frames = data.shape[0]
    t        = np.arange(n_frames) / fs
    all_cols = ["time"] + col_names
    all_data = np.column_stack([t, data])
    with open(path, "w") as f:
        f.write(header_name + "\n")
        f.write("version=1\n")
        f.write(f"nRows={n_frames}\n")
        f.write(f"nColumns={len(all_cols)}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("\t".join(all_cols) + "\n")
        for row in all_data:
            f.write("\t".join(f"{v:.8f}" for v in row) + "\n")
    print(f"    [sto] {n_frames} frames x {len(col_names)} cols -> {path.name}")


def write_body_json(path: Path, params: dict) -> None:
    """Write subject-level body parameters to a JSON file."""
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"    [json] body params -> {path.name}")


# ── READERS (used by validate_b3d_export) ─────────────────────────────────────

def read_mot_sto(path: Path) -> tuple[list[str], np.ndarray]:
    """
    Read an OpenSim .mot or .sto file.

    Returns
    -------
    col_names : list of column header strings (including 'time')
    data      : (n_frames, n_cols) float array
    """
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "endheader":
            header_end = i
            break
    else:
        raise ValueError(f"No 'endheader' found in {path}")
    col_names = lines[header_end + 1].strip().split("\t")
    data = np.loadtxt(path, skiprows=header_end + 2)
    return col_names, data


def read_trc(path: Path) -> tuple[list[str], np.ndarray]:
    """
    Read an OpenSim .trc file.

    Returns
    -------
    marker_names : list of marker label strings
    data         : (n_frames, 2 + n_markers*3) float array
                   cols: [frame#, time, X0, Y0, Z0, X1, Y1, Z1, ...]
    """
    with open(path) as f:
        lines = f.readlines()
    # Row index 3 is the marker-name row; names repeat 3x for X/Y/Z
    name_row     = lines[3].strip().split("\t")
    raw_names    = name_row[2:]          # strip 'Frame#' and 'Time'
    marker_names = raw_names[::3]        # every 3rd entry is the marker name
    data = np.loadtxt(path, skiprows=6)
    return marker_names, data
