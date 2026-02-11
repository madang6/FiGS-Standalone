# Importing the necessary libraries
from pathlib import Path
import figs.render.capture_generation as pg
import figs.visualize.plot_trajectories as pt
import figs.visualize.generate_videos as gv

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC


#%%
# FiGS Capture Examples (scene_name, capture_name)
capture_examples = [
    'flightroom_ssv_exp'
]

# Set repo root for consistent path handling
# Note: gsplats_path and config_path default to repo root structure if not provided
repo_root = Path(__file__).parent.parent

#%%
# Generate the FiGS environment
for scene_name in capture_examples:
    print("=============================================================")
    print(f"Generating 3DGS for [{scene_name}]")
    print("-------------------------------------------------------------")

    # Paths default to repo root, no need to specify them explicitly
    # gsplats_path defaults to repo_root / '3dgs'
    # config_path defaults to repo_root / 'configs'
    pg.generate_gsplat(scene_name, capture_cfg_name='iphone15pro_142', force_recompute=True)
    # pg.generate_gsplat(scene_name, force_recompute=True)