# Developed from: https://github.com/madang6/flightroom_ns_process/tree/feature/video_process

import os
import numpy as np
from PIL import Image
import cv2
import json
import figs.render.capture_generation as cg

from pathlib import Path
from typing import Tuple

def camera_calibration(calibration_file_name:str,camera_name:str,
                       gsplats_path:Path=None,config_path:Path=None,
                       checkerboard_size:tuple=(9, 6),square_size:float=3.0,max_images:int=100) -> None:
    """
    Camera calibration using a checkerboard pattern.

    Args:
        - calibration_file_name:    Name of the calibration video file.
        - camera_name:              Name of the camera to calibrate.
        - capture_path:             Path to the video file.
        - config_path:              Path to the camera configuration file.
        - checkerboard_size:        Number of inner corners in the checkerboard (rows, cols).
        - square_size:              Size of the squares in the checkerboard (in mm).
        - max_images:               Maximum number of images to use for calibration.
    """

    # Normalize and determine the relevant paths to use
    if gsplats_path is None:
        gsplats_path = Path(__file__).parent.parent.parent / 'gsplats'
    else:
        gsplats_path = Path(gsplats_path).expanduser()

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs'
    else:
        config_path = Path(config_path).expanduser()

    gsplat_capture_path = gsplats_path / 'capture'

    # If user passed a file (e.g. a camera json), write to its parent; otherwise use config_path/'camera'
    if config_path.is_file() or str(config_path).lower().endswith('.json'):
        config_camera_path = config_path.parent
    else:
        config_camera_path = config_path / 'camera'

    config_camera_path.mkdir(parents=True, exist_ok=True)

    # Extract frames from the video. Return frames as BGR numpy arrays and a list of saved paths.
    images, saved_paths = extract_frames(calibration_file_name, gsplat_capture_path, max_images=max_images)

    # Detect checkerboard corners
    object_points, image_points = process_checkerboard(images, checkerboard_size, square_size)

    # Calibrate the camera
    camera_parameters = None
    if len(object_points) > 0 and len(images) > 0:
        camera_parameters = estimate_camera_parameters(object_points, image_points, images[0].shape[:2])
    else:
        print("No valid checkerboard detections for calibration.")

    # Save the camera parameters to a file if calibration succeeded
    if camera_parameters is not None:
        json_file = config_camera_path / f"{camera_name}.json"
        with open(json_file, "w") as file:
            json.dump(camera_parameters, file, indent=4)

        print(f"Camera parameters saved to {json_file}")
    else:
        print("Skipping writing camera parameters because calibration failed.")

def process_checkerboard(frames:np.ndarray, checkerboard_size:Tuple[int,int],square_size:float) -> Tuple[np.ndarray,np.ndarray]:
    """
    Detects the checkerboard corners in a list of frames.
    
    Args:
        frames (list): List of video frames (numpy arrays).
        checkerboard_size (tuple): Number of inner corners in the checkerboard (rows, cols).

    Returns:
        tuple: Object points, image points
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[1], 0:checkerboard_size[0]].T.reshape(-1, 2)
    objp *= square_size

    object_points = []
    image_points = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(objp)
            image_points.append(corners2)

            # Draw and save visualization
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, checkerboard_size, corners2, ret)

    print(f"Checkerboard detected in {len(image_points)} frames.")

    return object_points, image_points
 
def estimate_camera_parameters(object_points, image_points, image_shape):
    """
    Calibrates the camera using detected object and image points and maps the results
    to a dictionary containing camera intrinsics and distortion coefficients.

    Args:
        object_points (list): 3D object points in real-world space.
        image_points (list): 2D points in image plane.
        image_shape (tuple): Shape of the calibration image (height, width).

    Returns:
        camera_parameters: Camera information including intrinsics and distortion coefficients.
    """

    # Calibrate the camera
    ret, intrinsic_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_shape[::-1], None, None
    )
    
    if not ret:
        raise RuntimeError("Calibration failed.")

    # Map results to the specified dictionary format
    camera_parameters = {
        "model": "OPENCV",
        "height": image_shape[0],
        "width": image_shape[1],
        "intrinsics_matrix": intrinsic_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
    }

    # Compute reprojection error
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints_projected, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], intrinsic_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        mean_error += error
    mean_error /= len(object_points)

    # Print results
    print("="*80)
    print("Estimated Camera Parameters (rounded for presentation):")
    print(f"Intrinsic Matrix:\n{intrinsic_matrix}")
    print(f"Distortion Coefficients:\n{dist_coeffs}")
    print(f"Mean Reprojection Error: {mean_error}")
    print("-"*80)
    print("="*80)

    return camera_parameters

def extract_frames(video_path, output_dir, max_images=None, subsample_rate=1):
    """
    Extracts frames from a video file, saves them as PNGs under output_dir, and returns frames.

    Args:
        video_path (str): Path to the input video file (relative names are resolved against output_dir when necessary).
        output_dir (str or Path): Directory where the extracted images will be saved.
        max_images (int or None): Maximum number of frames to extract. None means no limit.
        subsample_rate (int): Save every nth frame (default 1 saves all frames).

    Returns:
        frames (list of np.ndarray): List of frames as BGR numpy arrays (OpenCV format).
        saved_paths (list of str): List of file paths where frames were saved.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory at: {output_dir}")

    # If video_path is not absolute and exists relative to output_dir, try resolving
    video_path = str(video_path)
    resolved_video_path = None
    if os.path.isabs(video_path) and os.path.exists(video_path):
        resolved_video_path = video_path
    else:
        # try relative to output_dir
        candidate = output_dir / video_path
        if candidate.exists():
            resolved_video_path = str(candidate)
        else:
            # As last resort, try video_path as given
            if os.path.exists(video_path):
                resolved_video_path = video_path

    if resolved_video_path is None:
        print(f"Error: Cannot find video file {video_path}")
        return [], []

    print(f"Opening video file: {resolved_video_path}")
    cap = cv2.VideoCapture(resolved_video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {resolved_video_path}")
        return [], []

    frames = []
    saved_paths = []
    frame_count = 0
    saved_count = 0

    print("Starting frame extraction...")

    while True:
        ret, frame = cap.read()
        if not ret:
            # finished
            break

        if frame_count % subsample_rate == 0:
            if max_images is not None and saved_count >= int(max_images):
                break

            # Keep frame in BGR for downstream OpenCV processing
            frames.append(frame)

            # Save as PNG (convert BGR->RGB for PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_path = output_dir / f"{saved_count:04d}.png"
            image = Image.fromarray(frame_rgb)
            image.save(str(output_path))
            saved_paths.append(str(output_path))
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"{saved_count} frames saved...")

        frame_count += 1

    cap.release()
    print(f"Extraction completed. Saved {saved_count} frames out of {frame_count} total frames.")
    return frames, saved_paths

