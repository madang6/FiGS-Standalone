# Importing the necessary libraries
from pathlib import Path
import figs.render.capture_generation as pg
import figs.render.capture_calibration as cc
import figs.visualize.plot_trajectories as pt
import figs.visualize.generate_videos as gv

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC

#%%
# - calibration_file_name:    Name of the calibration video file.
# - camera_name:              Name of the camera to calibrate.
# - capture_path:             Path to the video file.
# - config_path:              Path to the camera configuration file.
# - checkerboard_size:        Number of inner corners in the checkerboard (rows, cols).
# - square_size:              Size of the squares in the checkerboard (in mm).
# - max_images:               Maximum number of images to use for calibration.
# Set paths relative to this notebook's parent directory (repo root)
repo_root = Path(__file__).parent.parent

# Note: gsplats_path and config_path default to the repo root structure if not provided
# This ensures they stay in sync with the main repository structure
cc.camera_calibration(
    calibration_file_name=str(repo_root / "3dgs/capture/calibration_video.MOV"),
    camera_name="camera_front",
    # Paths default to repo root, no need to specify them explicitly
    # config_path defaults to repo_root / 'configs'
    # gsplats_path defaults to repo_root / '3dgs'
    checkerboard_size=(8, 6),
    square_size=25,
    max_images=20
)