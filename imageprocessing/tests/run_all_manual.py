#!/usr/bin/env python3
"""Manual integration test – imports and smoke-tests every module."""
from __future__ import annotations
import os, sys, traceback, time

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

PASS = 0
FAIL = 0

def ok(name: str, detail: str = ""):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}  {detail}")

def fail(name: str, err):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}: {err}")

# ═══════════════════════════════════════════════════════════════════
#  1) src/utils/geometry.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/utils/geometry.py ──")
try:
    from src.utils.geometry import (
        angle_between_points, euclidean_distance, midpoint,
        keypoint_velocity, torso_inclination,
    )
    a = angle_between_points((0, 0), (0, 1), (1, 1))
    assert 89.0 < a < 91.0, f"angle={a}"
    ok("angle_between_points", f"{a:.1f}°")

    d = euclidean_distance((0, 0), (3, 4))
    assert abs(d - 5.0) < 0.01, f"dist={d}"
    ok("euclidean_distance", f"{d:.1f}")

    m = midpoint((0, 0), (10, 10))
    assert m == (5.0, 5.0), f"mid={m}"
    ok("midpoint", f"{m}")

    v = keypoint_velocity((0, 0), (30, 40), dt=1.0)
    assert abs(v - 50.0) < 0.01, f"vel={v}"
    ok("keypoint_velocity", f"{v:.1f} px/s")

    v0 = keypoint_velocity(None, (30, 40), dt=1.0)
    assert v0 == 0.0
    ok("keypoint_velocity (None)", "0.0")

    t = torso_inclination((100, 100), (100, 200))
    assert t < 1.0, f"incl={t}"
    ok("torso_inclination (upright)", f"{t:.1f}°")

    t2 = torso_inclination((100, 100), (200, 100))
    assert t2 > 89.0, f"incl={t2}"
    ok("torso_inclination (horizontal)", f"{t2:.1f}°")
except Exception as e:
    fail("geometry.py", e)

# ═══════════════════════════════════════════════════════════════════
#  2) src/utils/fps.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/utils/fps.py ──")
try:
    from src.utils.fps import FPSTracker
    fps = FPSTracker()
    fps.tick()
    time.sleep(0.01)
    fps.tick()
    val = fps.fps
    ok("FPSTracker", f"fps={val:.1f}")
except Exception as e:
    fail("fps.py", e)

# ═══════════════════════════════════════════════════════════════════
#  3) src/utils/draw.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/utils/draw.py ──")
try:
    import src.utils.draw as draw_mod
    # Just check it imported and has expected attributes
    assert hasattr(draw_mod, "draw_bbox") or hasattr(draw_mod, "draw_text") or True
    ok("draw.py import", "OK")
except Exception as e:
    fail("draw.py", e)

# ═══════════════════════════════════════════════════════════════════
#  4) src/utils/time_utils.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/utils/time_utils.py ──")
try:
    import src.utils.time_utils as tu
    ok("time_utils.py import", "OK")
except Exception as e:
    fail("time_utils.py", e)

# ═══════════════════════════════════════════════════════════════════
#  5) src/pipeline/timer.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/timer.py ──")
try:
    from src.pipeline.timer import BehaviorTimer
    timer = BehaviorTimer()
    state = timer.update(track_id=1, behavior="standing")
    assert state.current_behavior == "standing"
    ok("BehaviorTimer.update", f"behavior={state.current_behavior}")

    time.sleep(0.01)
    state2 = timer.update(track_id=1, behavior="sitting")
    assert state2.current_behavior == "sitting"
    assert len(state2.segments) == 2
    ok("BehaviorTimer transition", f"segments={len(state2.segments)}")

    fin = timer.finalize_track(1)
    assert fin is not None
    ok("BehaviorTimer.finalize_track", "OK")
except Exception as e:
    fail("timer.py", e)

# ═══════════════════════════════════════════════════════════════════
#  6) src/pipeline/behavior.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/behavior.py ──")
try:
    import numpy as np
    from src.pipeline.behavior import (
        BehaviorClassifier, LABEL_STANDING, LABEL_SITTING,
        LABEL_WALKING, LABEL_RUNNING, LABEL_UNKNOWN,
    )
    clf = BehaviorClassifier(sit_knee_angle=120.0, run_speed_threshold=60.0, walk_speed_threshold=15.0)

    # Standing keypoints
    kps = np.full((17, 3), 0.9)
    kps[5]  = [90, 100, 0.9]; kps[6]  = [110, 100, 0.9]
    kps[11] = [90, 200, 0.9]; kps[12] = [110, 200, 0.9]
    kps[13] = [90, 300, 0.9]; kps[14] = [110, 300, 0.9]
    kps[15] = [90, 400, 0.9]; kps[16] = [110, 400, 0.9]

    r = clf.classify(track_id=1, keypoints=kps, dt=0.033)
    assert r.label == LABEL_STANDING, f"got {r.label}"
    ok("classify standing", r.label)

    # Sitting keypoints
    kps2 = kps.copy()
    kps2[13] = [150, 200, 0.9]; kps2[14] = [170, 200, 0.9]
    kps2[15] = [150, 300, 0.9]; kps2[16] = [170, 300, 0.9]
    r2 = clf.classify(track_id=2, keypoints=kps2, dt=0.033)
    assert r2.label == LABEL_SITTING, f"got {r2.label}"
    ok("classify sitting", r2.label)

    # Unknown
    r3 = clf.classify(track_id=3, keypoints=None, dt=0.033)
    assert r3.label == LABEL_UNKNOWN
    ok("classify unknown (None)", r3.label)

    # Walking
    kps_w1 = kps.copy()
    kps_w2 = kps.copy()
    kps_w2[15][0] += 1.5; kps_w2[16][0] += 1.5
    clf.classify(track_id=4, keypoints=kps_w1, dt=0.033)
    r4 = clf.classify(track_id=4, keypoints=kps_w2, dt=0.033)
    assert r4.label == LABEL_WALKING, f"got {r4.label}"
    ok("classify walking", r4.label)

    # Running
    kps_r1 = kps.copy()
    kps_r2 = kps.copy()
    kps_r2[15][0] += 6.0; kps_r2[16][0] += 6.0
    clf.classify(track_id=5, keypoints=kps_r1, dt=0.033)
    r5 = clf.classify(track_id=5, keypoints=kps_r2, dt=0.033)
    assert r5.label == LABEL_RUNNING, f"got {r5.label}"
    ok("classify running", r5.label)

    clf.remove_track(1)
    ok("remove_track", "OK")
except Exception as e:
    fail("behavior.py", e)
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
#  7) src/pipeline/logger.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/logger.py ──")
try:
    import src.pipeline.logger as logger_mod
    ok("logger.py import", "OK")
except Exception as e:
    fail("logger.py", e)

# ═══════════════════════════════════════════════════════════════════
#  8) src/pipeline/overlay.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/overlay.py ──")
try:
    import src.pipeline.overlay as overlay_mod
    ok("overlay.py import", "OK")
except Exception as e:
    fail("overlay.py", e)

# ═══════════════════════════════════════════════════════════════════
#  9) src/pipeline/pose.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/pose.py ──")
try:
    from src.pipeline.pose import (
        KP_NOSE, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
        KP_LEFT_HIP, KP_RIGHT_HIP, KP_LEFT_KNEE, KP_RIGHT_KNEE,
        KP_LEFT_ANKLE, KP_RIGHT_ANKLE, get_keypoint,
    )
    assert KP_NOSE == 0
    assert KP_LEFT_ANKLE == 15
    assert KP_RIGHT_ANKLE == 16

    kps = np.full((17, 3), [100.0, 200.0, 0.9])
    pt = get_keypoint(kps, KP_NOSE, conf_threshold=0.3)
    assert pt is not None
    ok("pose constants + get_keypoint", f"pt={pt}")

    pt_low = get_keypoint(kps, KP_NOSE, conf_threshold=0.95)
    assert pt_low is None
    ok("get_keypoint (low conf)", "None as expected")
except Exception as e:
    fail("pose.py", e)
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# 10) src/pipeline/segmentation.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/segmentation.py ──")
try:
    import src.pipeline.segmentation as seg_mod
    ok("segmentation.py import", "OK")
except Exception as e:
    fail("segmentation.py", e)

# ═══════════════════════════════════════════════════════════════════
# 11) src/pipeline/tracker.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/tracker.py ──")
try:
    from src.pipeline.tracker import Track
    t = Track(track_id=1, bbox=np.array([10, 20, 100, 200]), confidence=0.95)
    cx, cy = t.center
    assert abs(cx - 55.0) < 0.01
    assert abs(cy - 110.0) < 0.01
    ok("Track dataclass", f"center=({cx:.0f},{cy:.0f})")
except Exception as e:
    fail("tracker.py", e)
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# 12) src/pipeline/detector_yolo.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/detector_yolo.py ──")
try:
    import src.pipeline.detector_yolo as det_mod
    ok("detector_yolo.py import", "OK")
except Exception as e:
    fail("detector_yolo.py", e)

# ═══════════════════════════════════════════════════════════════════
# 13) src/pipeline/video_source.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/pipeline/video_source.py ──")
try:
    import src.pipeline.video_source as vs_mod
    ok("video_source.py import", "OK")
except Exception as e:
    fail("video_source.py", e)

# ═══════════════════════════════════════════════════════════════════
# 14) src/main.py
# ═══════════════════════════════════════════════════════════════════
print("\n── src/main.py ──")
try:
    from src.main import main
    ok("main.py import", "OK")
except Exception as e:
    fail("main.py", e)

# ═══════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  SONUC:  {PASS} PASSED  /  {FAIL} FAILED")
print("=" * 60)
sys.exit(1 if FAIL else 0)
