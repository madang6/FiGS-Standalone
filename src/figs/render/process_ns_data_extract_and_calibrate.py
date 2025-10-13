#!/usr/bin/env python3
"""
Combined utility to extract frames from a video and run OpenCV checkerboard
camera calibration on the extracted frames.

Usage (example):
    python /home/admin/StanfordMSL/SousVide-Semantic/FiGS-Semantic/src/figs/render/process_ns_data_extract_and_calibrate.py \
        --video /home/admin/StanfordMSL/SousVide-Semantic/gsplats/capture/calibration_video.MOV \
        --out /home/admin/StanfordMSL/SousVide-Semantic/configs/capture/rgbs \
        --subsample 5 \
        --checkerboard 8 6 \
        --square 25.0 --square-units mm \  
        --out-json /home/admin/StanfordMSL/SousVide-Semantic/configs/capture/camera_intrinsics.json

This is built from two helper scripts: extract_frames and capture_calibration.
"""

import argparse
import os
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


def extract_frames(video_path, output_dir, subsample_rate=1):
    """Extract frames from a video and save as RGB PNGs.

    Returns the number of saved frames and the output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % subsample_rate == 0:
            # Convert BGR to RGB and save
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_path = output_dir / f"{saved_count:04d}.png"
            Image.fromarray(frame_rgb).save(out_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count, str(output_dir)


def calibrate_from_images(images_glob, checkerboard_dimensions=(8, 13), square_size=4.0, progress_enabled=True):
    """Run camera calibration using images matched by images_glob (glob pattern or list).

    Returns: camera_info dict and mean_reprojection_error (float).
    """
    # Prepare object points
    cb_x, cb_y = checkerboard_dimensions
    objp = np.zeros((cb_x * cb_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_x, 0:cb_y].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints = []
    imgpoints = []

    if isinstance(images_glob, (list, tuple)):
        images = list(images_glob)
    else:
        images = sorted(glob.glob(images_glob))

    total_images = len(images)
    if total_images == 0:
        raise RuntimeError(f"No images found for pattern: {images_glob}")

    success_count = 0
    fail_count = 0
    image_size = None

    progress = None
    if progress_enabled:
        progress = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("[green]{task.fields[success_count]}/{task.total} successes"), TextColumn("[red]{task.fields[fail_count]} failures"), TimeRemainingColumn())
        progress.start()
        task = progress.add_task("Processing images", total=total_images, success_count=0, fail_count=0)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            fail_count += 1
            if progress:
                progress.update(task, advance=1, fail_count=fail_count)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (cb_x, cb_y), None)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            success_count += 1
        else:
            fail_count += 1

        if progress:
            progress.update(task, advance=1, success_count=success_count, fail_count=fail_count)

    if progress:
        progress.stop()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise RuntimeError("No checkerboard corners were found in any image. Calibration cannot proceed.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    camera_info = {
        "camera_model": "OPENCV",
        "fl_x": float(camera_matrix[0, 0]),
        "fl_y": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "w": int(image_size[0]),
        "h": int(image_size[1]),
    }

    dist = dist_coeffs.flatten()
    camera_info.update({
        "k1": float(dist[0]) if len(dist) > 0 else 0.0,
        "k2": float(dist[1]) if len(dist) > 1 else 0.0,
        "p1": float(dist[2]) if len(dist) > 2 else 0.0,
        "p2": float(dist[3]) if len(dist) > 3 else 0.0,
        "k3": float(dist[4]) if len(dist) > 4 else 0.0,
    })

    # Reprojection error
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        mean_error += error
    mean_error /= len(objpoints)

    return camera_info, float(mean_error)


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from video and calibrate camera using checkerboard images.")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--out", required=True, help="Output directory for extracted frames")
    p.add_argument("--subsample", type=int, default=1, help="Save every nth frame")
    p.add_argument("--checkerboard", type=int, nargs=2, default=[8, 13], help="Checkerboard inner corners: cols rows (e.g. 8 13)")
    p.add_argument("--square", type=float, default=40.0, help="Square size (numeric). Units set by --square-units")
    p.add_argument("--square-units", choices=["mm", "cm", "m"], default="mm", help="Units for --square (mm, cm, m). Output is normalized to mm")
    p.add_argument("--sensor-width-mm", type=float, default=None, help="Optional: sensor physical width in millimeters. If provided, focal lengths in mm will be computed and saved in the JSON")
    p.add_argument("--out-json", default="camera_intrinsics.json", help="Path to save camera intrinsics JSON")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Extracting frames from {args.video} into {args.out} (every {args.subsample} frames)")
    saved, out_dir = extract_frames(args.video, args.out, args.subsample)
    print(f"Saved {saved} images to {out_dir}")

    images_pattern = os.path.join(out_dir, "*.png")
    print(f"Running calibration on images: {images_pattern}")

    # Normalize square size to mm
    units = args.square_units
    multiplier = {"mm": 1.0, "cm": 10.0, "m": 1000.0}[units]
    square_mm = float(args.square) * multiplier

    camera_info, reproj_err = calibrate_from_images(images_pattern, tuple(args.checkerboard), square_mm)

    # Add unit metadata and optionally compute focal length in mm if sensor width provided
    camera_info["square_size_mm"] = float(square_mm)
    camera_info["square_units"] = "mm"
    if args.sensor_width_mm is not None:
        try:
            sensor_w = float(args.sensor_width_mm)
            camera_info["sensor_width_mm"] = sensor_w
            # convert focal length from pixels to mm: f_mm = f_px * sensor_w_mm / image_w_px
            camera_info["fl_x_mm"] = float(camera_info["fl_x"]) * sensor_w / float(camera_info["w"])
            camera_info["fl_y_mm"] = float(camera_info["fl_y"]) * sensor_w / float(camera_info["w"])
        except Exception:
            # don't fail the whole run if conversion fails; just skip mm fields
            pass

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(camera_info, f, indent=4)

    print("Camera intrinsic parameters saved to:", out_json)
    print("Camera intrinsic parameters:")
    print(json.dumps(camera_info, indent=4))
    print(f"Mean reprojection error: {reproj_err}")


if __name__ == "__main__":
    main()
