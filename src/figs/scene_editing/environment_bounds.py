"""
Environment bounds detection and collision checking for FiGS-Standalone.

This module provides dynamic environment bounds detection from point clouds,
eliminating the need for manual bound specification.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.stats import zscore


class EnvironmentBounds:
    """
    Dynamically detects and manages environment bounds from point clouds.

    Supports multiple detection methods:
    - percentile: Percentile-based bounds (robust to outliers)
    - aabb: Axis-aligned bounding box (min/max)
    - convex_hull: Convex hull vertices
    - adaptive: Statistical outlier removal then AABB

    Provides collision checking via KD-tree queries.
    """

    def __init__(self,
                 method: str = "percentile",
                 padding: float = 0.5,
                 percentile: Tuple[int, int] = (5, 95)):
        """
        Initialize EnvironmentBounds.

        Args:
            method: Detection method ("percentile", "aabb", "convex_hull", "adaptive")
            padding: Safety margin to shrink navigable space (meters)
            percentile: (low, high) percentile for bounds (default 5th-95th)
        """
        self.method = method
        self.padding = padding
        self.percentile = percentile
        self.bounds = None  # {"minbound": (x,y,z), "maxbound": (x,y,z)}
        self.pcd = None  # Original point cloud
        self.kdtree = None  # For collision checking

    @classmethod
    def from_point_cloud(cls,
                        pcd_arr: np.ndarray,
                        method: str = "percentile",
                        z_range: Optional[Tuple[float, float]] = None,
                        padding: float = 0.5,
                        percentile: Tuple[int, int] = (5, 95)) -> 'EnvironmentBounds':
        """
        Dynamically compute bounds from Nx3 point cloud array.

        Args:
            pcd_arr: Nx3 array of (x,y,z) points
            method: Detection strategy
            z_range: Optional (z_min, z_max) to filter altitude
            padding: Safety margin (meters)
            percentile: (low, high) percentile for bounds

        Returns:
            EnvironmentBounds instance with computed bounds
        """
        env = cls(method=method, padding=padding, percentile=percentile)

        # Filter by altitude if specified
        if z_range is not None:
            mask = (pcd_arr[:, 2] >= z_range[0]) & (pcd_arr[:, 2] <= z_range[1])
            pcd_arr = pcd_arr[mask]

        if len(pcd_arr) == 0:
            raise ValueError("Point cloud is empty after filtering")

        # Compute bounds based on method
        if method == "percentile":
            minbound = np.percentile(pcd_arr, percentile[0], axis=0)
            maxbound = np.percentile(pcd_arr, percentile[1], axis=0)

        elif method == "aabb":
            minbound = np.min(pcd_arr, axis=0)
            maxbound = np.max(pcd_arr, axis=0)

        elif method == "convex_hull":
            # Uses scipy.spatial ConvexHull for tighter bounds
            hull = ConvexHull(pcd_arr)
            minbound = np.min(pcd_arr[hull.vertices], axis=0)
            maxbound = np.max(pcd_arr[hull.vertices], axis=0)

        elif method == "adaptive":
            # Statistical outlier removal then AABB
            z_scores = np.abs(zscore(pcd_arr, axis=0))
            mask = np.all(z_scores < 3, axis=1)  # 3-sigma filtering
            pcd_filtered = pcd_arr[mask]
            if len(pcd_filtered) == 0:
                print("Warning: Adaptive filtering removed all points. Using original bounds.")
                pcd_filtered = pcd_arr
            minbound = np.min(pcd_filtered, axis=0)
            maxbound = np.max(pcd_filtered, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'percentile', 'aabb', 'convex_hull', or 'adaptive'")

        # Apply padding (shrink navigable space for safety)
        env.bounds = {
            "minbound": tuple(minbound + padding),
            "maxbound": tuple(maxbound - padding)
        }
        env.pcd = pcd_arr

        # Build KDTree for collision checking
        env.kdtree = cKDTree(pcd_arr)

        return env

    @classmethod
    def from_gsplat_scene(cls,
                         scene_name: str,
                         gsplat_dir: str = "data/gsplat",
                         method: str = "percentile",
                         z_range: Optional[Tuple[float, float]] = None,
                         padding: float = 0.5,
                         percentile: Tuple[int, int] = (5, 95)) -> 'EnvironmentBounds':
        """
        Load point cloud from GSplat scene and compute bounds.

        Args:
            scene_name: Scene name (e.g., "flightroom_ssv_exp")
            gsplat_dir: Path to GSplat data directory
            method: Detection method
            z_range: Optional altitude filter
            padding: Safety margin (meters)
            percentile: (low, high) percentile for bounds

        Returns:
            EnvironmentBounds instance
        """
        from figs.scene_editing.scene_editing_utils import rescale_point_cloud
        from figs.render.gsplat_semantic import GSplat

        # Load GSplat scene
        gsplat_path = Path(gsplat_dir) / scene_name
        gsplat = GSplat(scene_name, device="cuda")

        # Generate and rescale point cloud
        epcds, pcd_arr, epcds_bounds, pcd, pcd_mask, pcd_attr = rescale_point_cloud(
            gsplat, viz=False, cull=False, verbose=False
        )

        # pcd_arr is already 3xN, transpose to Nx3
        if pcd_arr.shape[0] == 3:
            pcd_arr = pcd_arr.T

        return cls.from_point_cloud(pcd_arr, method=method, z_range=z_range,
                                    padding=padding, percentile=percentile)

    @classmethod
    def from_bounds_dict(cls, bounds_dict: Dict[str, Tuple[float, float, float]]) -> 'EnvironmentBounds':
        """
        Create EnvironmentBounds from pre-computed bounds dictionary.

        Args:
            bounds_dict: {"minbound": (x,y,z), "maxbound": (x,y,z)}

        Returns:
            EnvironmentBounds instance (without KDTree)
        """
        env = cls()
        env.bounds = bounds_dict
        return env

    def is_within_bounds(self, point: np.ndarray) -> bool:
        """
        Check if 3D point is within environment bounds.

        Args:
            point: (x,y,z) position or (N,3) array of positions

        Returns:
            Boolean or boolean array
        """
        if self.bounds is None:
            raise ValueError("Bounds not initialized. Call from_point_cloud() first.")

        point = np.atleast_2d(point)
        minbound = np.array(self.bounds["minbound"])
        maxbound = np.array(self.bounds["maxbound"])

        within = np.all((point >= minbound) & (point <= maxbound), axis=1)

        return within[0] if len(within) == 1 else within

    def check_collision(self, point: np.ndarray, radius: float = 0.3) -> bool:
        """
        KDTree-based collision detection.

        Args:
            point: (x,y,z) position or (N,3) array
            radius: Collision radius (meters)

        Returns:
            Boolean or boolean array indicating collision
        """
        if self.kdtree is None:
            print("Warning: KDTree not initialized. Cannot check collision.")
            return False

        point = np.atleast_2d(point)
        distances, _ = self.kdtree.query(point, k=1)

        collisions = distances < radius

        return collisions[0] if len(collisions) == 1 else collisions

    def get_safe_clearance(self, point: np.ndarray) -> Union[float, np.ndarray]:
        """
        Return minimum distance to nearest obstacle.

        Args:
            point: (x,y,z) position or (N,3) array

        Returns:
            Distance or array of distances
        """
        if self.kdtree is None:
            print("Warning: KDTree not initialized. Returning infinity.")
            return float('inf')

        point = np.atleast_2d(point)
        distances, _ = self.kdtree.query(point, k=1)

        return distances[0] if len(distances) == 1 else distances

    def get_bounds_extent(self) -> Tuple[float, float, float]:
        """
        Get the extent of the bounds in each dimension.

        Returns:
            (dx, dy, dz) dimensions
        """
        if self.bounds is None:
            raise ValueError("Bounds not initialized.")

        minbound = np.array(self.bounds["minbound"])
        maxbound = np.array(self.bounds["maxbound"])
        extent = maxbound - minbound

        return tuple(extent)

    def to_dict(self) -> Dict[str, Union[Dict, Tuple]]:
        """
        Export bounds for RRT/trajectory planning.

        Returns:
            Dictionary with bounds and range information
        """
        if self.bounds is None:
            raise ValueError("Bounds not initialized.")

        return {
            "bounds": self.bounds,
            "x_range": (self.bounds["minbound"][0], self.bounds["maxbound"][0]),
            "y_range": (self.bounds["minbound"][1], self.bounds["maxbound"][1]),
            "z_range": (self.bounds["minbound"][2], self.bounds["maxbound"][2]),
            "extent": self.get_bounds_extent()
        }

    def to_rrt_bounds(self) -> list:
        """
        Convert to RRT-compatible bounds format.

        Returns:
            [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        """
        bounds_dict = self.to_dict()
        return [
            bounds_dict["x_range"],
            bounds_dict["y_range"],
            bounds_dict["z_range"]
        ]

    def __repr__(self) -> str:
        if self.bounds is None:
            return "EnvironmentBounds(uninitialized)"

        extent = self.get_bounds_extent()
        return (f"EnvironmentBounds(method='{self.method}', "
                f"minbound={self.bounds['minbound']}, "
                f"maxbound={self.bounds['maxbound']}, "
                f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})m)")
