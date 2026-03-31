"""
probe_b3d_api.py
================
Run once to print the exact nimblephysics API available on your install.

Usage:
    python probe_b3d_api.py path/to/subject.b3d
"""

import sys
import nimblephysics as nimble

B3D_PATH = sys.argv[1] if len(sys.argv) > 1 else "subject10.b3d"

SEP = "=" * 60
print(f"\n{SEP}\n")

# ── SubjectOnDisk class-level methods ─────────────────────────────────────────
print("=== SubjectOnDisk class methods ===")
for a in sorted(a for a in dir(nimble.biomechanics.SubjectOnDisk) if not a.startswith("_")):
    print(f"  {a}")

# ── Instance methods ──────────────────────────────────────────────────────────
print(f"\n=== Loading: {B3D_PATH} ===")
subject = nimble.biomechanics.SubjectOnDisk(B3D_PATH)

print("\n=== SubjectOnDisk instance attributes ===")
for a in sorted(a for a in dir(subject) if not a.startswith("_")):
    print(f"  {a}")

# ── Trial attributes ──────────────────────────────────────────────────────────
print("\n=== Reading trial 0 ===")
try:
    trial = subject.readTrial(0)
    print("\n=== Trial instance attributes ===")
    for a in sorted(a for a in dir(trial) if not a.startswith("_")):
        print(f"  {a}")
except Exception as e:
    print(f"  Could not read trial 0: {e}")

# ── Candidate method probe (subject) ─────────────────────────────────────────
print("\n=== Candidate method probe (subject) ===")
for name in [
    "getSubjectName", "getName", "getSubjectTag", "getHref",
    "getMassKg", "getMass", "getHeightM", "getHeight",
    "getNumTrials", "getTrialName", "getTrialNames",
    "getDofNames", "getDofs", "getNumDofs",
    "getMarkerNames", "getMarkers",
    "getBodyScales", "getBodyMasses", "getBodyMassesMap",
    "getGroundForceBodies", "getContactBodies",
]:
    print(f"  {'OK  ' if hasattr(subject, name) else 'MISS'} subject.{name}()")

# ── Candidate method probe (trial) ────────────────────────────────────────────
print("\n=== Candidate method probe (trial) ===")
try:
    for name in [
        "getNumTimesteps", "getNumFrames", "getTimesteps",
        "getMarkerObservations", "getMarkerPositions", "getMarkersMap",
        "getJointAnglesMatrix", "getPoses", "getPos",
        "getJointMomentsMatrix", "getTaus", "getJointTorques",
        "getGroundBodyForces", "getGroundReactionForces", "getGRF",
        "getResiduals",
    ]:
        print(f"  {'OK  ' if hasattr(trial, name) else 'MISS'} trial.{name}()")
except Exception:
    print("  (trial object unavailable)")

print(f"\n{SEP}")
print("Paste this output to diagnose any API mismatches in process_b3d.py.")
print(f"{SEP}\n")
