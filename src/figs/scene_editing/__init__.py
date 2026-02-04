"""
Scene editing module for FiGS-Standalone.

Provides utilities for:
- Point cloud processing and visualization
- Environment bounds detection
- Obstacle detection and clustering
- Collision checking
"""

from figs.scene_editing.scene_editing_utils import (
    rescale_point_cloud,
    get_points,
    get_centroid,
    in_convex_hull,
    spherical_filter,
    get_interpolated_gaussians,
    plot_point_cloud,
)

from figs.scene_editing.environment_bounds import EnvironmentBounds
from figs.scene_editing.obstacle_detector import ObstacleDetector
from figs.scene_editing.collision_checker import CollisionChecker

__all__ = [
    # Existing utilities
    "rescale_point_cloud",
    "get_points",
    "get_centroid",
    "in_convex_hull",
    "spherical_filter",
    "get_interpolated_gaussians",
    "plot_point_cloud",
    # New classes
    "EnvironmentBounds",
    "ObstacleDetector",
    "CollisionChecker",
]
