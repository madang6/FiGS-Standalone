# notebooks/figs_3dgs_oneliner.py

# Importing the necessary libraries
from pathlib import Path
import subprocess
import figs.render.capture_generation as pg


#%%
# FiGS Capture Examples (scene_name, capture_name)
capture_examples = [
    # 'flightroom_ssv_exp'
    "carine-central-1"
]

# Set repo root for consistent path handling
# Note: gsplats_path and config_path default to repo root structure if not provided
repo_root = Path(__file__).parent.parent

#%%
# Generate the FiGS environment
for scene_name in capture_examples:
    print("=============================================================")
    print(f"Exporting 3DGS for [{scene_name}]")
    print("-------------------------------------------------------------")

    # Export point cloud
    print(f"Exporting Gaussian splat point cloud for {scene_name}...")
    workspace_path = repo_root / '3dgs' / 'workspace'

    # Find latest config.yml from trained model
    config_path = sorted(
        workspace_path.glob(f"outputs/{scene_name}/splatfacto/*/config.yml"),
        key=lambda p: p.stat().st_mtime
    )[-1]  # Most recent

    # Create export directory
    export_dir = workspace_path / "exports" / scene_name
    export_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(export_dir)
    ], check=True, cwd=str(workspace_path))
    print(f"✓ Point cloud exported to: {export_dir}")