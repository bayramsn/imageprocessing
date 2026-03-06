#!/usr/bin/env python3
"""
macOS Camera Permission Fixer
==============================
Run this script ONCE to trigger the macOS camera permission dialog for your
terminal application. After granting permission, run the main pipeline.

Usage:
    python fix_camera_permission.py
"""
from __future__ import annotations

import sys
import time

if sys.platform != "darwin":
    print("This script is only needed on macOS.")
    sys.exit(0)

print("=" * 60)
print("  macOS Camera Permission Request")
print("=" * 60)
print()
print("Attempting to access the camera via AVFoundation...")
print("A system permission dialog should appear.")
print("Click 'OK' to grant access, then re-run your pipeline.")
print()

import cv2  # noqa: E402

# Using CAP_AVFOUNDATION explicitly so macOS triggers the permission dialog
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
time.sleep(1)  # give macOS time to show the dialog

if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        print("✅  Camera access GRANTED and working!")
        print()
        print("You can now run the pipeline:")
        print("  python -m src.main --source 0 --enable_tracking --enable_pose \\")
        print("         --enable_behavior --enable_overlay --enable_logging")
    else:
        print("⚠️  Camera opened but could not read a frame.")
        print("   Try granting permission and running again.")
else:
    cap.release()
    print("❌  Camera could NOT be opened.")
    print()
    print("Please grant camera access manually:")
    print()
    print("  1. Open:  System Settings → Privacy & Security → Camera")
    print("  2. Enable camera for:  Terminal  (or iTerm2 / Warp / VS Code)")
    print("  3. Restart your terminal app completely")
    print("  4. Run this script again to verify")
    print()
    print("  Or run this in your terminal to open System Preferences directly:")
    print('  open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"')
    sys.exit(1)
