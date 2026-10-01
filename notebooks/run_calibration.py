"""Wrapper for camera calibration library.

Run inside the FiGS Docker container:

    > docker compose -f docker-compose.yml run --rm figs
    ... (once inside docker)
    > python notebooks/run_calibration.py

Caution
- May need to run `chmod -R 777 configs/` for folder permissions to write JSON output
- Does not seem to work in oneline call (see below). Likely due to docker initialization issues.
    
    docker compose -f docker-compose.yml run --rm figs \
      python notebooks/run_calibration.py
"""

from pathlib import Path
from figs.render.capture_calibration import camera_calibration

REPO = Path(__file__).resolve().parent.parent

camera_calibration(
    calibration_file_name="iphone11pm_monitor.MOV",
    camera_name="iphone11pm_monitor",
    gsplats_path=REPO / "3dgs",
    config_path=REPO / "configs",
    checkerboard_size=(9, 6),
    square_size=25.0,
    max_images=120,
)