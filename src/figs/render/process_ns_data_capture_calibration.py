import numpy as np
import cv2
import glob
import json

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Termination criteria for refining corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Checkerboard dimensions
checkerboard_dimensions = (8, 13)  # Number of inner corners per a chessboard row and column
square_size = 4.0  # Size of a square in your defined unit (centimeters)

# Prepare object points based on the real world coordinates
objp = np.zeros((checkerboard_dimensions[0] * checkerboard_dimensions[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dimensions[0], 0:checkerboard_dimensions[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D real-world points
imgpoints = []  # 2D points in image plane

# Load images

images = glob.glob('/home/admin/StanfordMSL/flightroom_ns_process/results/calib/rgbs/*.png')  # Update the path and extension if necessary
print(f"Found {len(images)} images for calibration.")

total_images = len(images)
success_count = 0
fail_count = 0

# Initialize image size
image_size = None

with Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    TextColumn("[green]{task.fields[success_count]}/{task.total} successes"),
    TextColumn("[red]{task.fields[fail_count]} failures"),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Processing images", total=total_images, success_count=0, fail_count=0)
 
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Image {fname} could not be loaded.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Set the image size if not set before
        if image_size is None:
            image_size = gray.shape[::-1]

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dimensions, None)

        if ret:
            objpoints.append(objp)

            # Refine corner detection for better accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

            success_count += 1

            # Optional: Draw and display the corners
            # cv2.drawChessboardCorners(img, checkerboard_dimensions, corners_refined, ret)
            # cv2.imshow('Checkerboard', img)
            # cv2.waitKey(500)
        else:
            # print(f"Checkerboard couldn't be detected in image: {fname}")
            fail_count += 1

        # Update the progress bar
        progress.update(
            task, 
            advance=1, 
            success_count=success_count, 
            fail_count=fail_count
        )

cv2.destroyAllWindows()

# Check if at least one image was processed
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("No checkerboard corners were found in any image. Calibration cannot proceed.")
else:
    # Perform camera calibration to get the camera matrix and distortion coefficients
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Output the camera calibration results
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)

        # Extract intrinsic parameters
    int_mat = camera_matrix  # For clarity, we can alias camera_matrix as int_mat
    img_W, img_H = image_size  # Unpack image width and height

    # Insert camera parameters into the dictionary
    camera_info = {}
    camera_info["camera_model"] = "OPENCV"
    camera_info["fl_x"] = float(int_mat[0, 0])
    camera_info["fl_y"] = float(int_mat[1, 1])
    camera_info["cx"] = float(int_mat[0, 2])
    camera_info["cy"] = float(int_mat[1, 2])
    camera_info["w"] = img_W
    camera_info["h"] = img_H

    # Distortion coefficients may have different lengths depending on the model
    dist_coeffs = dist_coeffs.flatten()
    camera_info["k1"] = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else 0.0
    camera_info["k2"] = float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.0
    camera_info["p1"] = float(dist_coeffs[2]) if len(dist_coeffs) > 2 else 0.0
    camera_info["p2"] = float(dist_coeffs[3]) if len(dist_coeffs) > 3 else 0.0
    camera_info["k3"] = float(dist_coeffs[4]) if len(dist_coeffs) > 4 else 0.0

    # Output the camera calibration results
    print("Camera intrinsic parameters:")
    print(json.dumps(camera_info, indent=4))

    # Save the dictionary to a JSON file
    with open('/home/admin/StanfordMSL/flightroom_ns_process/results/calib/camera_intrinsics.json', 'w') as f:
        json.dump(camera_info, f, indent=4)

    # Optional: Compute and print the reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        mean_error += error
    mean_error /= len(objpoints)
    print(f"\nMean reprojection error: {mean_error}")