"""
b3d_filters.py
==============
Zero-phase Butterworth low-pass filter helpers for the b3d_pipeline package.

Previously duplicated across process_b3d.py (make_lpf / apply_lpf) and
filter_kinematics.py (butter_lowpass). Both sets of logic are consolidated
here with a unified interface.

Usage
-----
    from b3d_filters import make_lpf, apply_lpf, apply_lpf_columns

# Design a filter once, apply many times (process_b3d style):
    sos = make_lpf(cutoff_hz=6.0, order=4, fs=100.0)
    filtered = apply_lpf(data, sos)          # (n_frames, n_cols) or (n_frames,)

# One-shot convenience for .sto/.mot column arrays (filter_kinematics style):
    filtered = apply_lpf_columns(data, cutoff_hz=15.0, fs=200.0)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def make_lpf(cutoff_hz: float, order: int, fs: float):
    """
    Design a zero-phase Butterworth low-pass SOS filter.

    Parameters
    ----------
    cutoff_hz : cutoff frequency in Hz
    order     : filter order (zero-phase doubling means effective order = 2*order)
    fs        : sampling rate in Hz

    Returns
    -------
    sos : second-order sections array suitable for sosfiltfilt()

    Raises
    ------
    ValueError if cutoff_hz >= Nyquist frequency
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        raise ValueError(
            f"Cutoff {cutoff_hz} Hz >= Nyquist {nyq:.1f} Hz (fs={fs} Hz). "
            "Lower the cutoff or raise the sampling rate."
        )
    return butter(order, cutoff_hz / nyq, btype="low", output="sos")


def apply_lpf(data: np.ndarray, sos) -> np.ndarray:
    """
    Apply a zero-phase SOS filter to data.

    Parameters
    ----------
    data : (n_frames,) or (n_frames, n_channels)
    sos  : SOS filter from make_lpf()

    Returns
    -------
    filtered array with same shape as data
    """
    if data.ndim == 1:
        return sosfiltfilt(sos, data)
    return sosfiltfilt(sos, data, axis=0)


def apply_lpf_columns(
    data: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
    skip_col0: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: design and apply a low-pass filter to every column of
    a 2-D array, optionally skipping column 0 (typically a time column).

    This replaces the per-column loop previously in filter_kinematics.py.

    Parameters
    ----------
    data       : (n_frames, n_cols) array; column 0 is usually time
    cutoff_hz  : low-pass cutoff in Hz
    fs         : sampling rate in Hz
    order      : Butterworth order
    skip_col0  : if True, column 0 is copied unchanged (time passthrough)

    Returns
    -------
    filtered : same shape as data
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        print(
            f"  Warning: cutoff {cutoff_hz} Hz >= Nyquist {nyq:.1f} Hz; "
            "skipping filter."
        )
        return data.copy()

    sos      = make_lpf(cutoff_hz, order, fs)
    filtered = data.copy()
    start    = 1 if skip_col0 else 0
    for i in range(start, data.shape[1]):
        filtered[:, i] = sosfiltfilt(sos, data[:, i])
    return filtered
