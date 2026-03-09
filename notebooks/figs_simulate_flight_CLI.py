#!/usr/bin/env python3
"""Simulate a flight in a GSplat scene and save video + GT camera poses.

Outputs:
    {output-dir}/video.mp4         — rendered video at control rate
    {output-dir}/images/*.png      — individual frames (RGB)
    {output-dir}/transforms.json   — GT poses in nerfstudio format

Usage:
    python figs_simulate_flight_CLI.py \
        --scene flightroom_ssv_exp/gemsplat/2026-02-27_025654 \
        --course circle_toward_center \
        --output-dir /media/admin/data/StanfordMSL/GOGGLES/experiments/circle_toward_center
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
from figs.visualize import generate_videos as gv
from figs.utilities import trajectory_helper as th


def build_transforms_json(sim, Xro, Imgs):
    """Build a nerfstudio-format transforms.json from simulation output.

    The FiGS dynamics world frame is NED (Z-down). With the T_c2b from
    carl.json, T_c2w = T_b2w @ T_c2b produces a c2w in OpenGL camera
    convention (Y-up, Z-backward) — the same convention nerfstudio uses
    in transforms.json. No axis flip needed.
    """
    T_c2b = sim.conFiG["drone"]["T_c2b"]
    cam = sim.conFiG["drone"]["camera"]
    num_frames = Imgs["rgb"].shape[0]

    frames = []
    for k in range(num_frames):
        T_b2w = th.xv_to_T(Xro[:, k])
        T_c2w = T_b2w @ T_c2b  # c2w in OpenGL convention (NED world)

        frames.append({
            "file_path": f"images/frame_{k:05d}.png",
            "transform_matrix": T_c2w.tolist(),
        })

    transforms = {
        "fl_x": float(cam["fx"]),
        "fl_y": float(cam["fy"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
        "w": int(cam["width"]),
        "h": int(cam["height"]),
        "camera_model": "OPENCV",
        "frames": frames,
    }
    return transforms


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a flight in a GSplat scene, save video and GT poses.",
    )
    parser.add_argument(
        "--scene", required=True,
        help="GSplat model path relative to 3dgs/workspace/outputs/.",
    )
    parser.add_argument(
        "--course", required=True,
        help="Course config name (from configs/course/).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for outputs (video.mp4, images/, transforms.json).",
    )
    parser.add_argument(
        "--rollout", default="baseline",
        help="Rollout config name (default: baseline).",
    )
    parser.add_argument(
        "--frame", default="carl",
        help="Frame/drone config name (default: carl).",
    )
    parser.add_argument(
        "--policy", default="vrmpc_rrt",
        help="MPC policy config name (default: vrmpc_rrt).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load simulator and controller
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Scene:  {args.scene}")
    print(f"Course: {args.course}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    print("Loading simulator...")
    sim = Simulator(args.scene, args.rollout, args.frame)

    print("Loading controller...")
    ctl = VehicleRateMPC(args.course, args.policy, args.frame)

    t0 = ctl.tXUd[0, 0]
    tf = ctl.tXUd[0, -1]
    x0 = ctl.tXUd[1:11, 0]
    duration = tf - t0

    print(f"Trajectory: {duration:.1f}s at {ctl.hz} Hz control rate")

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    print("Simulating...")
    t_start = time.time()
    Tro, Xro, Uro, Imgs, _, _ = sim.simulate(ctl, t0, tf, x0)
    t_sim = time.time() - t_start

    num_frames = Imgs["rgb"].shape[0]
    print(f"Simulation complete: {num_frames} frames in {t_sim:.1f}s")

    # ------------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------------
    video_path = str(output_dir / "video.mp4")
    gv.images_to_mp4(Imgs["rgb"], video_path, ctl.hz)
    print(f"Saved video: {video_path}")

    # ------------------------------------------------------------------
    # Save individual frames
    # ------------------------------------------------------------------
    print(f"Saving {num_frames} frames...")
    for k in range(num_frames):
        frame_path = str(output_dir / f"images/frame_{k:05d}.png")
        # Imgs["rgb"] is (N, H, W, 3) in RGB; cv2 expects BGR
        cv2.imwrite(frame_path, Imgs["rgb"][k][..., ::-1])
    print(f"Saved frames to {output_dir / 'images/'}")

    # ------------------------------------------------------------------
    # Save GT poses as transforms.json
    # ------------------------------------------------------------------
    transforms = build_transforms_json(sim, Xro, Imgs)
    transforms_path = output_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"Saved GT poses: {transforms_path} ({num_frames} frames)")

    # ------------------------------------------------------------------
    # Save scene point cloud in NED frame (for trajectory visualization)
    # ------------------------------------------------------------------
    print("Extracting scene point cloud in NED frame...")
    try:
        import open3d as o3d
        from figs.scene_editing import rescale_point_cloud

        epcds, _, _, _, _, _ = rescale_point_cloud(sim.gsplat, verbose=False)
        pc_path = str(output_dir / "sparse_pc_ned.ply")
        o3d.io.write_point_cloud(pc_path, epcds)
        print(f"Saved NED point cloud: {pc_path} ({len(epcds.points)} pts)")
    except Exception as e:
        print(f"Warning: could not save NED point cloud: {e}")

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    del ctl  # Release ACADOS resources

    print("=" * 60)
    print("Done.")
    print(f"  Video:      {output_dir / 'video.mp4'}")
    print(f"  Frames:     {output_dir / 'images/'} ({num_frames} PNGs)")
    print(f"  GT poses:   {output_dir / 'transforms.json'}")
    print(f"  Point cloud: {output_dir / 'sparse_pc_ned.ply'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
