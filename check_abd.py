"""
check_abd.py
============
Quick smoke-test: reads a .b3d file via nimblephysics and writes standard
OpenSim output files to confirm the API is working correctly.

This script uses nimblephysics's built-in OpenSimParser writers directly
(unlike process_b3d.py, which uses b3d_io custom writers) and is intended
as a lightweight API sanity check, not a production pipeline.
"""

import nimblephysics as nimble
import numpy as np

FILE_PATH = "AddBiomechanicsDataset/test/With_Arm/Hammer2013_Formatted_With_Arm/subject10/subject10.b3d"

subject    = nimble.biomechanics.SubjectOnDisk(FILE_PATH)
skeleton   = subject.readSkel(0)
all_frames = subject.readFrames(0, 0, subject.getTrialLength(0))

trial_fps  = subject.getTrialTimestep(0)
n_frames   = len(all_frames)
timestamps = [i * trial_fps for i in range(n_frames)]

# Stack into 2-D array and transpose to (dofs, frames)
joint_angles = np.array([f.processingPasses[0].pos for f in all_frames]).T
joint_taus   = np.array([f.processingPasses[0].tau for f in all_frames]).T

print(f"joint_angles shape: {joint_angles.shape}")
print(f"joint_taus shape:   {joint_taus.shape}")

print("\n[1] saveMot...")
nimble.biomechanics.OpenSimParser.saveMot(
    skeleton, "subject10_ik.mot", timestamps, joint_angles
)
print("    OK")

print("[2] saveIDMot...")
nimble.biomechanics.OpenSimParser.saveIDMot(
    skeleton, "subject10_id.mot", timestamps, joint_taus
)
print("    OK")

print("[3] saveTRC...")
marker_names = subject.getMarkerNameList()
marker_dicts = [dict(zip(marker_names, f.markerObservations)) for f in all_frames]
nimble.biomechanics.OpenSimParser.saveTRC(
    "subject10_markers.trc", timestamps, marker_dicts
)
print("    OK")

print("[4] saveRawGRFMot...")
nimble.biomechanics.OpenSimParser.saveRawGRFMot(
    "subject10_grf.mot",
    timestamps,
    [f.rawForcePlateForces for f in all_frames],
    [f.rawForcePlateCenterOfPressures for f in all_frames],
    [f.rawForcePlateTorques for f in all_frames],
)
print("    OK")

print("\nAll done.")
