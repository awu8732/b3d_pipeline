# b3d_pipeline

Modular toolkit for extracting OpenSim outputs from AddBiomechanics .b3d files.

## Module layout
- b3d_io.py       : file writers/readers (.trc, .mot, .sto, .json)
- b3d_filters.py  : zero-phase Butterworth LPF helpers
- b3d_extract.py  : frame extraction from nimblephysics SubjectOnDisk
- process_b3d.py  : main subject/trial pipeline
- validate_b3d_export.py : output validation (PASS/WARN/FAIL)
- filter_kinematics.py   : standalone .sto filter utility

## Key dependencies
nimblephysics, numpy, scipy

## Data paths
B3D files live under AddBiomechanicsDataset/. Outputs go to output/<subject>/<trial>/.

## Notes
- Pass indices: 0=KINEMATICS, 1=LOW_PASS_FILTER, 2=DYNAMICS
- GRF index 0 = calcn_r (right), 1 = calcn_l (left)
- Pelvis translations stay in metres; rotational DOFs convert to degrees