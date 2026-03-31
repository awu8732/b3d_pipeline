# b3d_pipeline

A modular Python toolkit for extracting and validating OpenSim-compatible outputs
from AddBiomechanics `.b3d` files via `nimblephysics`.

---

## Package structure

```
b3d_pipeline/
  b3d_io.py              # shared file writers (trc, mot, sto, json) and readers
  b3d_filters.py         # zero-phase Butterworth LPF helpers
  b3d_extract.py         # frame-level data extraction from SubjectOnDisk
  process_b3d.py         # main processing pipeline (subject + trial loop)
  validate_b3d_export.py # post-hoc output validation (PASS/WARN/FAIL)
  filter_kinematics.py   # standalone .sto low-pass filter utility
  check_abd.py           # nimblephysics API smoke test (OpenSimParser wrappers)
  probe_b3d_api.py       # introspects the installed nimblephysics API
```

---

## Dependency graph

```
process_b3d.py
  └── b3d_io.py        (write_trc, write_mot, write_sto, write_body_json)
  └── b3d_filters.py   (make_lpf, apply_lpf)
  └── b3d_extract.py   (read_all_frames, extract_markers, extract_ik,
                         extract_id, extract_grf, GRF_COLUMNS)

validate_b3d_export.py
  └── b3d_io.py        (read_mot_sto, read_trc)

filter_kinematics.py
  └── b3d_io.py        (read_mot_sto)
  └── b3d_filters.py   (apply_lpf_columns)

check_abd.py           (no shared deps; uses nimblephysics.OpenSimParser directly)
probe_b3d_api.py       (no shared deps; pure introspection)
```

---

## Consolidations made

### 1. Package name: `b3d_pipeline`
Groups the five original scripts under a common identity reflecting their shared
purpose: reading, processing, filtering, and validating biomechanical data from
the AddBiomechanics `.b3d` format.

### 2. Shared I/O module: `b3d_io.py`
`process_b3d.py` contained five writer functions (`write_trc`, `write_mot`,
`write_sto`, `write_body_json`). `validate_b3d_export.py` contained two reader
functions (`load_mot_sto`, `load_trc`) that parsed the exact same formats.
These are now unified in `b3d_io.py` under consistent names (`read_mot_sto`,
`read_trc`), so both scripts stay in sync if the format ever changes.

### 3. Shared filter module: `b3d_filters.py`
`process_b3d.py` had `make_lpf` / `apply_lpf` using SOS design.
`filter_kinematics.py` had its own `butter_lowpass` using `ba` design.
These are merged into `b3d_filters.py`; both scripts now use the same SOS
path (strictly better numerically). A new `apply_lpf_columns` convenience
wrapper covers the column-loop pattern from `filter_kinematics.py`.

### 4. Shared extraction module: `b3d_extract.py`
The four extraction functions (`read_all_frames`, `extract_markers`,
`extract_ik`, `extract_id`, `extract_grf`) and the `GRF_COLUMNS` constant
were inlined inside `process_b3d.py`. They are now in `b3d_extract.py`,
importable by any future script that needs raw frame data without running the
full pipeline.

### 5. `process_b3d.py` split into three logical sections
The original monolith mixed I/O, filtering, extraction, IK post-processing,
and the trial loop all in one function. It is now split as:

| Function | Responsibility |
|---|---|
| `_load_subject_metadata()` | One-time subject-level setup (skeleton, DOF/marker names, body params) |
| `_rad_to_deg_ik()` | Convert rotational DOFs to degrees, leave translations in metres |
| `_append_beta_coords()` | Append LaiArnold patellofemoral beta coordinates |
| `process_trial()` | Per-trial extraction + write (Stages 1-3) |
| `process_subject()` | Outer loop: load subject, iterate trials, write .osim |

---

## Usage

### Run the full pipeline
```bash
python process_b3d.py
```
Edit the `CONFIG` block at the top of `process_b3d.py` to set paths and filter
parameters.

### Validate a processed trial
```bash
python validate_b3d_export.py \
    --trial_dir output/subject10/run200_segment_0 \
    --body      output/subject10/run200_segment_0/body.json
```

### Filter an existing .sto kinematics file
```bash
python filter_kinematics.py   # edit CONFIG block inside the file
```

### Probe the installed nimblephysics API
```bash
python probe_b3d_api.py path/to/subject.b3d
```

### Quick API smoke test (OpenSimParser wrappers)
```bash
python check_abd.py
```

---

## Dependencies
```
nimblephysics
numpy
scipy
```
