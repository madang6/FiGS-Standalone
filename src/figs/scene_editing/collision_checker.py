"""
Real-time collision detection for FiGS-Standalone.

This module provides efficient collision checking during trajectory execution
and rollout generation.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


class CollisionChecker:
    """
    Efficient collision detection using KD-tree spatial indexing.

    Provides per-timestep clearance computation and collision detection
    for trajectory validation and safety monitoring.
    """

    def __init__(self, point_cloud: np.ndarray, collision_radius: float = 0.3):
        """
        Initialize CollisionChecker with obstacle point cloud.

        Args:
            point_cloud: Nx3 or 3xN array of obstacle points
            collision_radius: Collision threshold distance (meters)
        """
        # Ensure point cloud is Nx3
        if point_cloud.shape[1] != 3:
            if point_cloud.shape[0] == 3:
                point_cloud = point_cloud.T
            else:
                raise ValueError(f"Point cloud must be Nx3 or 3xN, got {point_cloud.shape}")

        self.point_cloud = point_cloud
        self.collision_radius = collision_radius

        # Build KD-tree for efficient nearest neighbor queries
        self.kdtree = cKDTree(point_cloud)

        print(f"CollisionChecker initialized with {len(point_cloud)} obstacle points, "
              f"collision radius={collision_radius}m")

    def compute_clearances(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance to obstacles at each position.

        Args:
            positions: Nx3 array of (x,y,z) positions or single (3,) position

        Returns:
            clearances: (N,) array of minimum distances to nearest obstacle
        """
        positions = np.atleast_2d(positions)

        if positions.shape[1] != 3:
            raise ValueError(f"Positions must be Nx3, got {positions.shape}")

        # Query KD-tree for nearest obstacle
        distances, _ = self.kdtree.query(positions, k=1)

        return distances

    def detect_collision(self, positions: np.ndarray) -> Tuple[bool, Optional[int], np.ndarray]:
        """
        Detect collision along trajectory.

        Args:
            positions: Nx3 array of trajectory positions

        Returns:
            collision_detected: Boolean indicating if collision occurred
            collision_index: Index of first collision (None if no collision)
            clearances: (N,) array of clearances at each timestep
        """
        clearances = self.compute_clearances(positions)

        # Check for collision
        collision_mask = clearances < self.collision_radius
        collision_detected = np.any(collision_mask)

        collision_index = None
        if collision_detected:
            collision_index = int(np.argmax(collision_mask))

        return collision_detected, collision_index, clearances

    def check_segment_collision(self,
                               start: np.ndarray,
                               end: np.ndarray,
                               num_samples: int = 10) -> bool:
        """
        Check if line segment collides with obstacles.

        Args:
            start: (3,) start position
            end: (3,) end position
            num_samples: Number of points to sample along segment

        Returns:
            True if collision detected
        """
        # Sample points along segment
        t = np.linspace(0, 1, num_samples)[:, np.newaxis]
        segment_points = start + t * (end - start)

        collision_detected, _, _ = self.detect_collision(segment_points)

        return collision_detected

    def get_safe_distance_field(self, query_points: np.ndarray) -> np.ndarray:
        """
        Compute signed distance field at query points.

        Args:
            query_points: Nx3 array of query positions

        Returns:
            distances: (N,) array of distances (positive = free space)
        """
        return self.compute_clearances(query_points)

    def validate_trajectory(self,
                           tXUd: np.ndarray,
                           min_clearance: float = 0.5,
                           return_violations: bool = False) -> Tuple[bool, dict]:
        """
        Validate trajectory for collision safety.

        Args:
            tXUd: (11+, N) trajectory array with positions in rows 1-3
            min_clearance: Minimum required clearance (meters)
            return_violations: If True, return detailed violation information

        Returns:
            is_safe: Boolean indicating if trajectory is collision-free
            info: Dictionary with validation statistics
        """
        # Extract positions (rows 1-3 are x, y, z)
        positions = tXUd[1:4, :].T  # Convert to Nx3

        # Compute clearances
        clearances = self.compute_clearances(positions)

        # Check violations
        collision_detected = np.any(clearances < self.collision_radius)
        min_clearance_violated = np.any(clearances < min_clearance)

        is_safe = not (collision_detected or min_clearance_violated)

        # Build info dictionary
        info = {
            "is_safe": is_safe,
            "collision_detected": collision_detected,
            "min_clearance_violated": min_clearance_violated,
            "min_clearance_value": float(np.min(clearances)),
            "mean_clearance": float(np.mean(clearances)),
            "max_clearance": float(np.max(clearances)),
            "num_timesteps": len(positions)
        }

        if return_violations:
            collision_indices = np.where(clearances < self.collision_radius)[0]
            violation_indices = np.where(clearances < min_clearance)[0]

            info["collision_indices"] = collision_indices.tolist()
            info["violation_indices"] = violation_indices.tolist()
            info["clearances"] = clearances

        return is_safe, info

    def compute_collision_rewards(self,
                                 clearances: np.ndarray,
                                 collision_detected: bool,
                                 collision_penalty: float = -10.0,
                                 clearance_threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Compute collision-based rewards for RL training.

        Ports functionality from SousVide-Semantic collision_detector.py.

        Args:
            clearances: (N,) array of clearances
            collision_detected: Boolean indicating collision
            collision_penalty: Penalty value for collision
            clearance_threshold: Threshold for clearance penalty

        Returns:
            per_step_rewards: (N,) array of per-timestep rewards
            terminal_reward: Final reward (large penalty if collision)
        """
        N = len(clearances)
        per_step_rewards = np.zeros(N)

        # Apply penalty for being too close to obstacles
        for i in range(N):
            if clearances[i] < clearance_threshold:
                per_step_rewards[i] = collision_penalty * (clearance_threshold - clearances[i])

        # Terminal penalty for collision
        terminal_reward = collision_penalty if collision_detected else 0.0

        return per_step_rewards, terminal_reward

    @classmethod
    def from_environment_bounds(cls,
                                env_bounds,
                                collision_radius: float = 0.3) -> 'CollisionChecker':
        """
        Create CollisionChecker from EnvironmentBounds instance.

        Args:
            env_bounds: EnvironmentBounds with point cloud
            collision_radius: Collision threshold (meters)

        Returns:
            CollisionChecker instance
        """
        if env_bounds.pcd is None:
            raise ValueError("EnvironmentBounds has no point cloud data")

        return cls(env_bounds.pcd, collision_radius=collision_radius)

    def update_point_cloud(self, point_cloud: np.ndarray):
        """
        Update obstacle point cloud and rebuild KD-tree.

        Args:
            point_cloud: Nx3 or 3xN array of new obstacle points
        """
        # Ensure Nx3
        if point_cloud.shape[1] != 3:
            if point_cloud.shape[0] == 3:
                point_cloud = point_cloud.T
            else:
                raise ValueError(f"Point cloud must be Nx3 or 3xN, got {point_cloud.shape}")

        self.point_cloud = point_cloud
        self.kdtree = cKDTree(point_cloud)

        print(f"CollisionChecker updated with {len(point_cloud)} obstacle points")

    def get_statistics(self) -> dict:
        """
        Get statistics about the collision checker.

        Returns:
            Dictionary with statistics
        """
        return {
            "num_obstacles": len(self.point_cloud),
            "collision_radius": self.collision_radius,
            "obstacle_bounds": {
                "min": self.point_cloud.min(axis=0).tolist(),
                "max": self.point_cloud.max(axis=0).tolist()
            }
        }

    def __repr__(self) -> str:
        return (f"CollisionChecker(obstacles={len(self.point_cloud)}, "
                f"radius={self.collision_radius}m)")
