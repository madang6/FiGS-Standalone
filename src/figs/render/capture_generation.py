# Developed from: https://github.com/madang6/flightroom_ns_process/tree/feature/video_process

from pathlib import Path
import json
from typing import List, Tuple, Dict, Union
import time
import shutil

from nerfstudio.process_data.images_to_nerfstudio_dataset import (
    ImagesToNerfstudioDataset,
)

from rich.console import Console
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

import figs.utilities.capture_helper as ch
import cv2
import numpy as np
import open3d as o3d
import subprocess

# Initialize console for rich output management
console = Console()

# Default processing mode when no config provided
DEFAULT_MODE = "rgb"  # Standard splatfacto training

def _resolve_repo_root() -> Path:
    """
    Resolve the absolute path to the FiGS-Standalone repository root.
    This ensures outputs always go to the correct location regardless of where
    the function is called from.
    """
    # Current file is at: src/figs/render/capture_generation.py
    # Go up 4 levels to reach the repo root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root

def _stage_complete(stage_path: Path) -> bool:
    """
    Check if a processing stage has already been completed by verifying
    if key output files exist.

    Args:
        stage_path: The directory containing stage outputs

    Returns:
        True if stage appears to be complete, False otherwise
    """
    if not stage_path.exists():
        return False

    # Check if directory has any relevant files
    if isinstance(stage_path, Path) and stage_path.is_dir():
        files = list(stage_path.glob("*"))
        return len(files) > 0

    return False

def _print_stage(stage_name: str, status: str = "STARTED", elapsed_time: float = None) -> None:
    """
    Print a formatted stage status message using rich formatting.

    Args:
        stage_name: Name of the processing stage
        status: Status of the stage (STARTED, IN PROGRESS, COMPLETED, SKIPPED, FAILED)
        elapsed_time: Optional elapsed time in seconds for the stage
    """
    # Define status colors
    status_colors = {
        "STARTED": "yellow",
        "IN PROGRESS": "cyan",
        "COMPLETED": "green",
        "SKIPPED": "blue",
        "FAILED": "red"
    }

    color = status_colors.get(status, "white")

    # Build the status text
    stage_text = Text(f"▶ {stage_name}", style="bold cyan")
    status_text = Text(f" {status}", style=f"bold {color}")

    # Add timing information if provided
    if elapsed_time is not None:
        timing_text = Text(f" ({elapsed_time:.1f}s)", style="dim white")
        console.print(stage_text + status_text + timing_text)
    else:
        console.print(stage_text + status_text)

def _load_capture_config(
    capture_cfg_path: Path,
    capture_cfg_name: str | None,
    console: Console
) -> dict | None:
    """
    Load capture configuration or return None for nerfstudio video mode.

    Returns None when no config is available, signaling that nerfstudio's
    VideoToNerfstudioDataset should be used instead of custom ArUco extraction.

    Args:
        capture_cfg_path: Path to config directory
        capture_cfg_name: Config name or None for nerfstudio video mode
        console: Rich console for output

    Returns:
        dict: Config with keys {"camera", "extractor", "mode"} for ArUco path
        None: Use nerfstudio video mode (no ArUco markers expected)
    """
    # If no config name provided, use nerfstudio video mode
    if not capture_cfg_name:
        console.print("[yellow]No capture config specified, using nerfstudio video mode[/yellow]")
        return None

    # Try to load config file
    config_file = capture_cfg_path / f"{capture_cfg_name}.json"
    try:
        with open(config_file, "r") as f:
            loaded_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[yellow]Config file '{capture_cfg_name}.json' not found, using nerfstudio video mode[/yellow]")
        return None
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Invalid JSON in '{capture_cfg_name}.json': {e}, using nerfstudio video mode[/yellow]")
        return None

    # If no extractor config, can't do ArUco processing → use nerfstudio video mode
    if "extractor" not in loaded_config:
        console.print("[yellow]No 'extractor' in config, using nerfstudio video mode[/yellow]")
        return None

    # Build result with loaded config + defaults for missing keys
    result = {
        "camera": loaded_config.get("camera", None),  # Can be None - extracted from SfM
        "extractor": loaded_config["extractor"],      # Required for ArUco processing
        "mode": loaded_config.get("mode", DEFAULT_MODE)  # Default to "rgb" if missing
    }

    # Warn if mode is missing
    if "mode" not in loaded_config:
        console.print(f"[yellow]'mode' not found in config, using default: {DEFAULT_MODE}[/yellow]")

    return result

def _process_video_without_config(
    video_path: Path | str,
    sfm_path: Path,
    force_recompute: bool,
    console: Console
) -> None:
    """
    Process video using nerfstudio's VideoToNerfstudioDataset.

    This path is used when no capture config exists (no ArUco markers).
    Handles frame extraction and SfM automatically via nerfstudio.

    Args:
        video_path: Path to video file
        sfm_path: Output path for SfM results
        force_recompute: Whether to recompute SfM results
        console: Rich console for output
    """
    from nerfstudio.process_data.video_to_nerfstudio_dataset import VideoToNerfstudioDataset

    # Check if already processed
    sfm_tfm_path = sfm_path / "transforms.json"
    if sfm_tfm_path.exists() and not force_recompute:
        _print_stage("Video Processing (nerfstudio)", "SKIPPED", None)
        console.print(f"[blue]  Found existing SfM results, skipping video processing[/blue]")
        return

    # Clear existing results if force_recompute
    if force_recompute and sfm_path.exists():
        shutil.rmtree(sfm_path)

    _print_stage("Video Processing (nerfstudio)", "IN PROGRESS")
    video_start = time.time()

    # Use nerfstudio's video processor
    ns_obj = VideoToNerfstudioDataset(
        data=Path(video_path),
        output_dir=sfm_path,
        num_frames_target=300,           # Default frame count
        matching_method="sequential",    # Better for video sequences
        sfm_tool="hloc",                 # Use HLOC like existing code
        gpu=True
    )
    ns_obj.main()

    video_elapsed = time.time() - video_start
    _print_stage("Video Processing (nerfstudio)", "COMPLETED", video_elapsed)

def generate_gsplat(scene_file_name: str,
                    capture_cfg_name: str | None = None,
                    gsplats_path: Path = None,
                    config_path: Path = None,
                    force_recompute: bool = False) -> None:
    """
    Generate 3D Gaussian Splats from a video with checkpointing support.

    Camera calibration is optional. If not provided, camera intrinsics are automatically
    extracted from the video using Structure from Motion (SfM). This allows processing
    videos without pre-calibrated camera parameters.

    Args:
        scene_file_name: Name of the scene to process
        capture_cfg_name: Name of the camera capture configuration. Can be:
            - None (default): Use nerfstudio video mode (no ArUco markers, safest option)
            - Config name like 'iphone15pro': Load from configs/capture/{name}.json
            - 'default': Load default config (only if it has extractor settings)
        gsplats_path: Path to the 3dgs directory (defaults to repo_root/3dgs)
        config_path: Path to the configs directory (defaults to repo_root/configs)
        force_recompute: If True, recompute all stages. If False, skip completed stages.

    Processing Modes (determined by config or default):
        - "rgb": Standard RGB training using splatfacto (default, most general-purpose)
        - "semantic": Semantic-aware training using gemsplat with RANSAC alignment
        - "no_ransac": Standard training without RANSAC marker alignment

    Examples:
        # Minimal usage - just video file (uses nerfstudio video mode by default)
        generate_gsplat("my_scene")

        # Explicit video mode (same as above)
        generate_gsplat("my_scene", capture_cfg_name=None)

        # With ArUco markers and specific camera configuration
        generate_gsplat("my_scene", capture_cfg_name="iphone15pro")

        # With custom paths
        generate_gsplat("my_scene",
                       capture_cfg_name="pixel8pro",
                       gsplats_path=Path("/custom/3dgs"),
                       config_path=Path("/custom/configs"))
    """
    # Resolve repository root to ensure outputs always go to the correct location
    repo_root = _resolve_repo_root()

    # Initialize base paths with absolute paths
    # Default to the repository root
    if gsplats_path is None:
        gsplats_path = (repo_root / '3dgs').resolve()

    if config_path is None:
        config_path = (repo_root / 'configs').resolve()

    # Ensure gsplats_path is in the repo root (prevent creating /3dgs in /src)
    gsplats_path = gsplats_path.resolve()
    if not str(gsplats_path).startswith(str(repo_root)):
        console.print(f"[yellow]Warning: gsplats_path {gsplats_path} is outside repo root. Using repo root instead.[/yellow]")
        gsplats_path = repo_root / '3dgs'

    capture_cfg_path = config_path/'capture'
    capture_path = gsplats_path/'capture'
    workspace_path = gsplats_path/'workspace'
    
    # Find the correct video path
    video_files = list(capture_path.glob(f"*{scene_file_name}*"))
    # Just files, no folders
    video_files = [f for f in video_files if f.is_file()]
    if len(video_files) == 0:
        raise FileNotFoundError(f"No file found with name containing '{scene_file_name}' in {capture_path}")
    elif len(video_files) > 1:
        raise ValueError(f"Multiple files found with name containing '{scene_file_name}' in {capture_path}")
    else:
        video_path = str(video_files[0])

    # Initialize process paths
    process_path = workspace_path / scene_file_name

    images_path = process_path / "images"
    spc_path = process_path / "sparse_pc.ply"
    tfm_path = process_path / "transforms.json"

    sfm_path = process_path / "sfm"
    sfm_spc_path = sfm_path / "sparse_pc.ply"
    sfm_tfm_path = sfm_path / "transforms.json"
    
    process_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    # Initialize output paths
    outputs_path = workspace_path/'outputs'
    output_path = outputs_path/scene_file_name

    output_path.mkdir(parents=True, exist_ok=True)

    # Load the capture config
    _print_stage("Loading Configuration", "STARTED")
    config_dict = _load_capture_config(capture_cfg_path, capture_cfg_name, console)
    _print_stage("Loading Configuration", "COMPLETED")

    if config_dict is None:
        # ===== PATH B: No config → Nerfstudio video mode (no ArUco markers) =====
        console.print("[cyan]Using nerfstudio video mode (no ArUco markers)[/cyan]")

        _process_video_without_config(video_path, sfm_path, force_recompute, console)

        # Load SfM results
        sfm_tfm_path = sfm_path / "transforms.json"
        sfm_spc_path = sfm_path / "sparse_pc.ply"
        with open(sfm_tfm_path, "r") as f:
            tfm_data = json.load(f)

        # No RANSAC transform - use SfM coordinates directly
        # Copy to main process paths for training
        shutil.copy(sfm_tfm_path, tfm_path)
        shutil.copy(sfm_spc_path, spc_path)

        # Symlink images directory to match ArUco mode structure
        sfm_images_path = sfm_path / "images"
        if images_path.exists() and not images_path.is_symlink():
            # Remove the empty directory created earlier
            images_path.rmdir()
        if not images_path.exists():
            images_path.symlink_to(sfm_images_path, target_is_directory=True)

        # Training command (standard splatfacto, no coordinate transforms)
        command = [
            "ns-train",
            "splatfacto",
            "--data", scene_file_name,
            "--viewer.quit-on-train-completion", "True",
            "--output-dir", 'outputs',
            "--pipeline.model.camera-optimizer.mode", "SO3xR3",
            "nerfstudio-data",
            "--orientation-method", "none",
            "--center-method", "none",
        ]

    else:
        # ===== PATH A: With config → Custom ArUco marker-based processing =====
        console.print("[cyan]Using custom ArUco marker-based processing[/cyan]")

        camera_config = config_dict["camera"]
        extractor_config = config_dict["extractor"]
        embedding_config = config_dict["mode"]

        # Extract the frame data (with checkpointing)
        if _stage_complete(images_path) and not force_recompute:
            frame_count = len(list(images_path.glob("*.png")))
            _print_stage("Extracting Frames", "SKIPPED", None)
            console.print(f"[blue]  Found {frame_count} existing frames, skipping extraction[/blue]")
        else:
            _print_stage("Extracting Frames", "IN PROGRESS")
            frame_start = time.time()
            # Clear existing frames if force_recompute
            if force_recompute and images_path.exists():
                shutil.rmtree(images_path)
            extract_frames(video_path, images_path, extractor_config)
            frame_elapsed = time.time() - frame_start
            _print_stage("Extracting Frames", "COMPLETED", frame_elapsed)
    
        # ns_process data step (with checkpointing)
        if sfm_tfm_path.exists() and not force_recompute:
            _print_stage("Structure from Motion", "SKIPPED", None)
            console.print(f"[blue]  Found existing SfM results, skipping SfM processing[/blue]")
        else:
            _print_stage("Structure from Motion", "IN PROGRESS")
            sfm_start = time.time()
            # Clear existing SfM results if force_recompute
            if force_recompute and sfm_path.exists():
                shutil.rmtree(sfm_path)
            ns_obj = ImagesToNerfstudioDataset(
                data=images_path, output_dir=sfm_path,
                camera_type="perspective", matching_method="exhaustive", sfm_tool="hloc", gpu=True
            )
            ns_obj.main()
            sfm_elapsed = time.time() - sfm_start
            _print_stage("Structure from Motion", "COMPLETED", sfm_elapsed)

        # Load the resulting transforms.json and sparse_points.ply
        with open(sfm_tfm_path, "r") as f:
            tfm_data = json.load(f)

        sparse_pcloud = o3d.io.read_point_cloud(sfm_spc_path.as_posix())
    
        # Check if frame count matches
        if len(tfm_data["frames"]) != extractor_config["num_images"]:
            if len(tfm_data["frames"]) < round(0.99 * extractor_config["num_images"]):
                raise ValueError(f"Frame count mismatch: {len(tfm_data['frames'])} frames in SfM data. Expected {extractor_config['num_images']} images.")
            else:
                print(f"Warning: Frame count mismatch: {len(tfm_data['frames'])} frames in SfM data. Expected {extractor_config['num_images']} images.")
            # raise ValueError(f"Frame count mismatch: {len(tfm_data['frames'])} frames in SfM data. Expected {extractor_config['num_images']} images.")

        # Use sfm config if camera config is not provided
        if camera_config is None:
            fx,fy = tfm_data["fl_x"],tfm_data["fl_y"]
            cx,cy = tfm_data["cx"],tfm_data["cy"]
            k1,k2 = tfm_data["k1"],tfm_data["k2"]
            p1,p2 = tfm_data["p1"],tfm_data["p2"]

            camera_config = {
                "model": tfm_data["camera_model"],
                "height": tfm_data["h"],
                "width": tfm_data["w"],
                "intrinsics_matrix": [
                    [ fx, 0.0,  cx],
                    [0.0,  fy,  cy],
                    [0.0, 0.0, 1.0]
                ],
                "distortion_coefficients": [k1,k2,p1,p2]
            }
        
        # Compute the transform using aruco markers (with checkpointing)
        # Check if transforms have already been computed
        transforms_computed = tfm_path.exists() and (tfm_path.stat().st_size > 1000)  # Check file is substantial

        if transforms_computed and not force_recompute:
            _print_stage("Computing Transforms", "SKIPPED", None)
            console.print(f"[blue]  Found existing transform results, skipping transform computation[/blue]")
            # Still need to load the computed transforms for later use
            with open(tfm_path, "r") as f:
                tfm_data_updated = json.load(f)
            # Extract scale, rotation, translation from saved data if available
            if "sfm_to_mocap_T" in tfm_data_updated and len(tfm_data_updated["sfm_to_mocap_T"]) > 0:
                T_data = np.array(tfm_data_updated["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])
                cs, Rs, ts = 1.0, T_data[:3, :3], T_data[:3, 3]
            else:
                # Fallback: recompute if not found
                _print_stage("Computing Transforms", "IN PROGRESS")
                transform_start = time.time()
                Psfm, Parc = extract_positions(sfm_path, extractor_config, camera_config)
                cs, Rs, ts = ch.compute_ransac_transform(Psfm, Parc)
                transform_elapsed = time.time() - transform_start
                _print_stage("Computing Transforms", "COMPLETED", transform_elapsed)
        else:
            _print_stage("Computing Transforms", "IN PROGRESS")
            transform_start = time.time()
            Psfm, Parc = extract_positions(sfm_path, extractor_config, camera_config)
            cs, Rs, ts = ch.compute_ransac_transform(Psfm, Parc)
            transform_elapsed = time.time() - transform_start
            _print_stage("Computing Transforms", "COMPLETED", transform_elapsed)

        if embedding_config == "semantic":
            # sfm_to_world_T = np.eye(4)
            # sfm_to_world_T[:3,:3],sfm_to_world_T[:3,3] = cs*Rs,ts

            # # Save the updated files
            # tfm_data["sfm_to_mocap_T"] = []
            # sfm_T = {}
            # sfm_T["sfm_to_mocap_T"] = sfm_to_world_T.tolist()
            # tfm_data["sfm_to_mocap_T"].append(sfm_T)
            # with open(tfm_path, "w", encoding="utf8") as f:
            #     json.dump(tfm_data, f, indent=4)
        
            # o3d.io.write_point_cloud(spc_path.as_posix(),sparse_pcloud)
        
            # # Run the gsplat generation
            # command = [
            #     "ns-train",
            #     "gemsplat",
            #     "--data", scene_file_name,
            #     "--viewer.quit-on-train-completion", "True",
            #     "--output-dir", 'outputs',
            #     "--pipeline.model.camera-optimizer.mode", "SO3xR3",
            #     # "--pipeline.model.rasterize-mode antialiased",
            #     "nerfstudio-data",
            #     "--orientation-method", "none",
            #     "--center-method", "none"
            # ]
            sfm_to_world_T = np.eye(4)
            sfm_to_world_T[:3,:3],sfm_to_world_T[:3,3] = cs*Rs,ts

            # Save the updated files
            tfm_data["sfm_to_mocap_T"] = []
            sfm_T = {}
            sfm_T["sfm_to_mocap_T"] = sfm_to_world_T.tolist()
            tfm_data["sfm_to_mocap_T"].append(sfm_T)
            with open(tfm_path, "w", encoding="utf8") as f:
                json.dump(tfm_data, f, indent=4)
        
            # Generate the sparse point cloud and transform files
            for frame in tfm_data["frames"]:
                Tc2s = np.array(frame["transform_matrix"])

                Tc2w = np.eye(4)
                Tc2w[:3,:3],Tc2w[:3,3] = Rs@Tc2s[:3,:3],cs*Rs@Tc2s[:3,3] + ts

                frame["transform_matrix"] = Tc2w.tolist()

            sparse_points = np.asarray(sparse_pcloud.points)
            for idx, point in enumerate(sparse_points):
                sparse_points[idx,:] = cs*Rs@point + ts

            sparse_pcloud.points = o3d.utility.Vector3dVector(sparse_points)

            # Save the updated files
            with open(tfm_path, "w", encoding="utf8") as f:
                json.dump(tfm_data, f, indent=4)

            o3d.io.write_point_cloud(spc_path.as_posix(),sparse_pcloud)

            # Run the gsplat generation
            print("Command to run:")
            command = [
                "ns-train",
                "gemsplat",
                "--data", scene_file_name,
                "--viewer.quit-on-train-completion", "True",
                "--output-dir", 'outputs',
                "--pipeline.model.camera-optimizer.mode", "SO3xR3",
                "nerfstudio-data",
                "--orientation-method", "none",
                "--center-method", "none",
            ]
            print(" ".join(command))
        elif embedding_config == "no_ransac":

            sfm_to_world_T = np.eye(4)
            sfm_to_world_T[:3,:3],sfm_to_world_T[:3,3] = cs*Rs,ts

            # Save the updated files
            tfm_data["sfm_to_mocap_T"] = []
            sfm_T = {}
            sfm_T["sfm_to_mocap_T"] = sfm_to_world_T.tolist()
            tfm_data["sfm_to_mocap_T"].append(sfm_T)
            with open(tfm_path, "w", encoding="utf8") as f:
                json.dump(tfm_data, f, indent=4)
        
            # Generate the sparse point cloud and transform files
            for frame in tfm_data["frames"]:
                Tc2s = np.array(frame["transform_matrix"])

                Tc2w = np.eye(4)
                Tc2w[:3,:3],Tc2w[:3,3] = Rs@Tc2s[:3,:3],cs*Rs@Tc2s[:3,3] + ts
                # Tc2w[:3,:3],Tc2w[:3,3] = Tc2s[:3,:3],Tc2s[:3,3]

                frame["transform_matrix"] = Tc2w.tolist()

            sparse_points = np.asarray(sparse_pcloud.points)
            for idx, point in enumerate(sparse_points):
                sparse_points[idx,:] = cs*Rs@point + ts
                # sparse_points[idx,:] = point

            sparse_pcloud.points = o3d.utility.Vector3dVector(sparse_points)

            # Save the updated files
            with open(tfm_path, "w", encoding="utf8") as f:
                json.dump(tfm_data, f, indent=4)

            o3d.io.write_point_cloud(spc_path.as_posix(),sparse_pcloud)

            # Run the gsplat generation
            print("Command to run:")
            command = [
                "ns-train",
                "splatfacto",
                "--data", scene_file_name,
                "--viewer.quit-on-train-completion", "True",
                "--output-dir", 'outputs',
                "--pipeline.model.camera-optimizer.mode", "SO3xR3",
                "nerfstudio-data",
                "--orientation-method", "none",
                "--center-method", "none",
            ]
            print(" ".join(command))
        else:
            # Generate the sparse point cloud and transform files
            for frame in tfm_data["frames"]:
                Tc2s = np.array(frame["transform_matrix"])

                Tc2w = np.eye(4)
                Tc2w[:3,:3],Tc2w[:3,3] = Rs@Tc2s[:3,:3],cs*Rs@Tc2s[:3,3] + ts

                frame["transform_matrix"] = Tc2w.tolist()

            sparse_points = np.asarray(sparse_pcloud.points)
            for idx, point in enumerate(sparse_points):
                sparse_points[idx,:] = cs*Rs@point + ts

            sparse_pcloud.points = o3d.utility.Vector3dVector(sparse_points)

            # Save the updated files
            with open(tfm_path, "w", encoding="utf8") as f:
                json.dump(tfm_data, f, indent=4)

            o3d.io.write_point_cloud(spc_path.as_posix(),sparse_pcloud)

            # Run the gsplat generation
            command = [
                "ns-train",
                "splatfacto",
                # "gemsplat",
                "--data", scene_file_name,
                "--viewer.quit-on-train-completion", "True",
                "--output-dir", 'outputs',
                "--pipeline.model.camera-optimizer.mode", "SO3xR3",
                "nerfstudio-data",
                "--orientation-method", "none",
                "--center-method", "none",
                "--auto-scale-poses", "False"
            ]

    # Run the command (with checkpointing)
    # Check if training output already exists by searching for nerfstudio_models folder
    nerfstudio_models_paths = list(output_path.glob("**/nerfstudio_models"))
    training_output_exists = len(nerfstudio_models_paths) > 0

    if training_output_exists and not force_recompute:
        _print_stage("Training 3D Gaussian Splats", "SKIPPED", None)
        console.print(f"[blue]  Found existing training outputs, skipping training[/blue]")
    else:
        _print_stage("Training 3D Gaussian Splats", "IN PROGRESS")
        training_start = time.time()
        result = subprocess.run(command, cwd=workspace_path.as_posix(), capture_output=False, text=True)
        training_elapsed = time.time() - training_start

        # Check the result
        if result.returncode == 0:
            _print_stage("Training 3D Gaussian Splats", "COMPLETED", training_elapsed)
            console.print(result.stdout)  # Output of the command
        else:
            _print_stage("Training 3D Gaussian Splats", "FAILED")
            console.print(result.stderr)  # Error output

def extract_frames(video_path:Path,rgbs_path:Path,
                   extractor_config:Dict['str',Union[int,float]]) -> List[np.ndarray]:
    """
    Extracts frame data from video into a folder of images.

    """

    # Unpack the extractor configs
    Nimg = extractor_config["num_images"]
    Narc = extractor_config["num_marked"]
    mkr_id = extractor_config["marker_id"]

    # Initialize the aruco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Cannot open the video file.")

    # Survey frames for aruco markers with progress bar
    Ntot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Tarc,Temp = [],[]

    progress = Progress(
        TextColumn("[bold cyan]Scanning for markers"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        task = progress.add_task("", total=Ntot)
        for _ in range(Ntot):
            ret, frame = cap.read()
            if not ret:
                break

            # Check if the frame has an aruco marker
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, ids, _ = detector.detectMarkers(gray)

            # Bin the frame by the marker detection
            if ids is not None and len(ids) == 1 and ids[0] == mkr_id:
                Tarc.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            else:
                Temp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            progress.update(task, advance=1)

    # Check if enough aruco markers were found
    if len(Tarc) < Narc:
        Tout = Tarc + ch.distribute_values(Temp,Nimg-len(Tarc))
        console.print(f"[yellow]Warning: Only {len(Tarc)} aruco markers found. Using {Narc-len(Tarc)} empty frames to fill the gap.[/yellow]")
    else:
        Tout = ch.distribute_values(Tarc,Narc) + ch.distribute_values(Temp,Nimg-Narc)

    Tout.sort()

    # Extract the selected frames with progress bar
    progress = Progress(
        TextColumn("[bold cyan]Extracting frames"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        task = progress.add_task("", total=len(Tout))
        for idx, tout in enumerate(Tout):
            cap.set(cv2.CAP_PROP_POS_MSEC,tout)
            ret, frame = cap.read()
            if not ret:
                break

            # Save the image
            rgb_path = rgbs_path / f"frame_{idx+1:05d}.png"
            cv2.imwrite(str(rgb_path),frame)

            progress.update(task, advance=1)

    # Release the video capture object
    cap.release()

def extract_positions(sfm_path:Path,
                      extractor_config:Dict['str',Union[int,float]],
                      camera_config:Dict['str',Union[int,float]]=None) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    
    # Unpack the extractor configs
    Narc = extractor_config["num_marked"]
    marker_length = extractor_config["marker_length"]
    marker_id = extractor_config["marker_id"]

    # Unpack the camera configs
    if camera_config is None:
        # TODO: Add option to use SfM camera parameters
        raise ValueError("Camera configuration is not provided.")
    else:
        camera_matrix = np.array(camera_config["intrinsics_matrix"])
        dist_coeffs = np.array(camera_config["distortion_coefficients"])

    # Initialize the aruco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ])
    
    # Open the transforms.json file
    with open(sfm_path / "transforms.json", "r") as f:
        transforms = json.load(f)
    frames = transforms["frames"]

    TTarc,TTsfm = [],[]
    for frame in frames:
        # Open the image file
        image_path = sfm_path.parent / frame["file_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Error: Cannot open the image file {image_path}")
        
        # Detect the aruco marker in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 1 and ids[0] == marker_id:            
            # Compute the Aruco transform
            ret, rvec, tvec = cv2.solvePnP(
                marker_points, corners[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            # Compute the transforms
            if ret:
                Tw2c_arc = np.eye(4)
                Tw2c_arc[:3, :3],Tw2c_arc[:3, 3] = cv2.Rodrigues(rvec)[0],tvec.flatten()  # world to camera

                # Compute the camera to world transforms
                Tarc = np.linalg.inv(Tw2c_arc)
                Tsfm = np.array(frame["transform_matrix"])

                TTarc.append(Tarc)
                TTsfm.append(Tsfm)
    
    # Check if the number of transforms match our expectations
    if len(TTarc) < Narc-1:
        raise ValueError(f"Error: Mismatched number of aruco and sfm transforms. Only found {len(TTarc)} aruco markers. Expected {Narc}.")

    # Extract the positions
    Parc = np.array([T[:3,3] for T in TTarc]).T
    Psfm = np.array([T[:3,3] for T in TTsfm]).T

    return Psfm,Parc