# src/figs/scripts/run_all_carine_3dgs.py
from __future__ import annotations

import subprocess
from pathlib import Path
import time
import wave
import contextlib

import figs.render.capture_generation as pg


def _repo_root() -> Path:
    # notebooks/run_all_carine_3dgs.py -> repo root
    return Path(__file__).resolve().parent.parent


def main() -> None:
    repo_root = _repo_root()
    project_root = repo_root.parent

    capture_dir = project_root / "3dgs" / "capture"
    workspace_dir = project_root / "3dgs" / "workspace"

    mp4s = sorted(capture_dir.glob("*.mp4"))
    print(f"Found {len(mp4s)} capture mp4 files in {capture_dir}")

    for mp4 in mp4s:
    
        # Start a timer
        start_time = time.time()

        scene_name = mp4.stem

        if not scene_name.startswith("carine"):
            continue

        print("=============================================================")
        print(f"Generating 3DGS for [{scene_name}]")
        print("-------------------------------------------------------------")

        # Gaussian splatting
        pg.generate_gsplat(scene_name, force_recompute=True)

        # Find latest config.yml
        config_path = sorted(
            workspace_dir.glob(f"outputs/{scene_name}/splatfacto/*/config.yml"),
            key=lambda p: p.stat().st_mtime,
        )[-1]

        # Export directory
        export_dir = workspace_dir / "exports" / scene_name
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export Gaussian splat point cloud
        subprocess.run(
            [
                "ns-export",
                "gaussian-splat",
                "--load-config",
                str(config_path),
                "--output-dir",
                str(export_dir),
            ],
            check=True,
            cwd=str(workspace_dir),
        )

        print(f"✓ Point cloud exported to: {export_dir}")

        # End timer and print duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"✓ Completed in {duration:.2f} seconds\n")
        # Report the size of the mp4
        mp4_size = mp4.stat().st_size / (1024 * 1024)  # in MB
        print(f"✓ Capture mp4 size: {mp4_size:.2f} MB\n")
        # Report the length of the mp4 (don't use cv2 or moviepy, use default libraries)
        with contextlib.closing(wave.open(str(mp4), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            print(f"✓ Capture mp4 duration: {duration:.2f} seconds\n")
        # Report the time per seconds of capture
        if duration > 0:
            time_per_second = (end_time - start_time) / duration
            print(f"✓ Processing time per second of capture: {time_per_second:.2f} seconds\n")
        # Write this to a log file
        log_file = export_dir / "processing_log.txt"
        with open(log_file, "w") as f:
            f.write(f"Capture mp4 size: {mp4_size:.2f} MB\n")
            f.write(f"Capture mp4 duration: {duration:.2f} seconds\n")
            f.write(f"Total processing time: {end_time - start_time:.2f} seconds\n")
            if duration > 0:
                f.write(f"Processing time per second of capture: {time_per_second:.2f} seconds\n")



if __name__ == "__main__":
    """
    Run with the following from StanfordMSL/FiGS-Standalone/
    conda activate FiGS
    python src/scripts/run_all_carine_3dgs.py
    """
    main()
