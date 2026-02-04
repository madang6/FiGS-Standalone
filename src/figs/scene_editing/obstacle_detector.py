"""
Obstacle detection and safe loiter orbit generation for FiGS-Standalone.

This module provides DBSCAN clustering of obstacles and generation of
safe circular orbits for loiter maneuvers.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from figs.scene_editing.environment_bounds import EnvironmentBounds


class ObstacleDetector:
    """
    Detects obstacle clusters and generates safe loiter orbits.

    Uses DBSCAN clustering to group obstacle points, then generates
    circular waypoint rings around each cluster at a safe clearance distance.
    """

    def __init__(self,
                 env_bounds: Optional[EnvironmentBounds] = None,
                 cluster_eps: float = 0.5,
                 min_samples: int = 10,
                 clearance: float = 0.2):
        """
        Initialize ObstacleDetector.

        Args:
            env_bounds: EnvironmentBounds instance for spatial filtering
            cluster_eps: DBSCAN epsilon (neighborhood radius in meters)
            min_samples: Minimum points to form a cluster
            clearance: Safety margin around obstacles (meters)
        """
        self.env_bounds = env_bounds
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.clearance = clearance

        self.clusters = []  # List of cluster point arrays
        self.rings = []  # List of loiter orbit arrays
        self.centroids = []  # List of cluster centroids
        self.radii = []  # List of orbit radii

    def detect_clusters(self,
                       point_cloud: np.ndarray,
                       z_range: Optional[Tuple[float, float]] = None) -> List[np.ndarray]:
        """
        Apply DBSCAN clustering to find obstacle groups.

        Args:
            point_cloud: Nx3 array of obstacle points (from semantic segmentation)
            z_range: Optional altitude filtering (z_min, z_max)

        Returns:
            List of cluster point arrays
        """
        # Ensure point cloud is Nx3
        if point_cloud.shape[1] != 3:
            if point_cloud.shape[0] == 3:
                point_cloud = point_cloud.T
            else:
                raise ValueError(f"Point cloud must be Nx3 or 3xN, got {point_cloud.shape}")

        # Filter by altitude
        if z_range is not None:
            mask = (point_cloud[:, 2] >= z_range[0]) & (point_cloud[:, 2] <= z_range[1])
            point_cloud = point_cloud[mask]

        # Filter by environment bounds if available
        if self.env_bounds is not None:
            in_bounds = self.env_bounds.is_within_bounds(point_cloud)
            point_cloud = point_cloud[in_bounds]

        if len(point_cloud) == 0:
            print("Warning: No points remaining after filtering")
            self.clusters = []
            return []

        # DBSCAN clustering
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(point_cloud)

        # Extract clusters (ignore noise label=-1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        self.clusters = []
        for label in unique_labels:
            cluster_points = point_cloud[labels == label]
            self.clusters.append(cluster_points)

        print(f"Detected {len(self.clusters)} obstacle clusters from {len(point_cloud)} points")

        return self.clusters

    def generate_loiter_rings(self,
                             sample_size: int = 10,
                             bidirectional: bool = True,
                             min_ring_points: int = 3) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate safe circular orbits around each cluster.

        Args:
            sample_size: Number of points on each orbit circle
            bidirectional: If True, generate clockwise + counter-clockwise rings
            min_ring_points: Minimum valid points required to keep a ring

        Returns:
            (rings, centroids)
            rings: List of (M, 3) arrays - loiter waypoints (M <= sample_size)
            centroids: List of (3,) arrays - cluster centers
        """
        self.rings = []
        self.centroids = []
        self.radii = []

        if len(self.clusters) == 0:
            print("Warning: No clusters to generate rings for. Call detect_clusters() first.")
            return [], []

        for cluster_idx, cluster in enumerate(self.clusters):
            # Compute cluster properties
            centroid = np.mean(cluster, axis=0)
            distances = np.linalg.norm(cluster - centroid, axis=1)
            max_radius = np.max(distances)
            orbit_radius = max_radius + self.clearance

            # Generate circle points at cluster altitude
            angles = np.linspace(0, 2*np.pi, sample_size, endpoint=False)
            ring_points = []

            for angle in angles:
                # Sample point on circle
                point = centroid.copy()
                point[0] += orbit_radius * np.cos(angle)
                point[1] += orbit_radius * np.sin(angle)
                # Keep Z coordinate from centroid

                # Validate: within bounds and safe clearance
                if self.env_bounds is not None:
                    if not self.env_bounds.is_within_bounds(point):
                        continue
                    clearance = self.env_bounds.get_safe_clearance(point)
                    if clearance < self.clearance:
                        continue

                ring_points.append(point)

            # Only keep rings with sufficient valid points
            if len(ring_points) >= min_ring_points:
                ring = np.array(ring_points)
                self.rings.append(ring)
                self.centroids.append(centroid)
                self.radii.append(orbit_radius)

                if bidirectional:
                    # Add reversed ring for opposite orbit direction
                    self.rings.append(ring[::-1])
                    self.centroids.append(centroid)
                    self.radii.append(orbit_radius)

                print(f"  Cluster {cluster_idx}: centroid={centroid}, radius={orbit_radius:.3f}m, "
                      f"points={len(ring_points)}/{sample_size}")
            else:
                print(f"  Cluster {cluster_idx}: Insufficient valid ring points ({len(ring_points)}/{min_ring_points}), skipping")

        print(f"Generated {len(self.rings)} loiter rings ({len(self.rings)//2 if bidirectional else len(self.rings)} unique)")

        return self.rings, self.centroids

    def get_nearest_loiter_target(self, current_pos: np.ndarray) -> Optional[Tuple[np.ndarray, int, float]]:
        """
        Find nearest loiter orbit to current position.

        Args:
            current_pos: (x,y,z) current position

        Returns:
            (centroid, cluster_index, distance) or None
        """
        if len(self.centroids) == 0:
            return None

        current_pos = np.atleast_1d(current_pos)
        distances = [np.linalg.norm(c - current_pos) for c in self.centroids]
        nearest_idx = np.argmin(distances)

        return self.centroids[nearest_idx], nearest_idx, distances[nearest_idx]

    def get_ring_by_index(self, index: int) -> Optional[np.ndarray]:
        """
        Get loiter ring by index.

        Args:
            index: Ring index

        Returns:
            (M, 3) array of waypoints or None
        """
        if 0 <= index < len(self.rings):
            return self.rings[index]
        return None

    @classmethod
    def from_semantic_segmentation(cls,
                                   semantic_image: np.ndarray,
                                   depth_image: np.ndarray,
                                   camera_intrinsics: Dict[str, float],
                                   obstacle_label: int,
                                   env_bounds: Optional[EnvironmentBounds] = None,
                                   **kwargs) -> 'ObstacleDetector':
        """
        Create detector from semantic segmentation output.

        Args:
            semantic_image: HxW semantic label map
            depth_image: HxW depth map (meters)
            camera_intrinsics: {"fx", "fy", "cx", "cy", "width", "height"}
            obstacle_label: Semantic class ID for obstacles (e.g., 1 for "obstacle")
            env_bounds: EnvironmentBounds instance
            **kwargs: Additional arguments for ObstacleDetector constructor

        Returns:
            ObstacleDetector with detected clusters
        """
        # Extract obstacle pixels
        mask = (semantic_image == obstacle_label)
        v, u = np.where(mask)  # Note: v=row, u=col

        if len(u) == 0:
            print("Warning: No obstacle pixels found in semantic image")
            detector = cls(env_bounds, **kwargs)
            return detector

        # Backproject to 3D using pinhole camera model
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]

        z = depth_image[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        point_cloud = np.stack([x, y, z], axis=1)

        # Create detector and process
        detector = cls(env_bounds, **kwargs)
        detector.detect_clusters(point_cloud)
        detector.generate_loiter_rings()

        return detector

    @classmethod
    def process_obstacle_clusters_and_sample(cls,
                                            epcds_arr: np.ndarray,
                                            env_bounds: Optional[EnvironmentBounds] = None,
                                            z_range: Tuple[float, float] = (-2.5, -0.9),
                                            cluster_eps: float = 0.5,
                                            min_samples: int = 10,
                                            clearance: float = 0.2,
                                            sample_size: int = 10) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process obstacle point cloud and generate loiter rings.

        This is a convenience method that combines clustering and ring generation.
        Ports the functionality from SousVide-Semantic's process_obstacle_clusters_and_sample().

        Args:
            epcds_arr: 3xM or Mx3 array of obstacle points
            env_bounds: EnvironmentBounds instance (optional)
            z_range: Height filtering (z_min, z_max)
            cluster_eps: DBSCAN epsilon (meters)
            min_samples: Minimum points per cluster
            clearance: Safety clearance (meters)
            sample_size: Points per orbit circle

        Returns:
            (rings, centroids)
            rings: List of loiter orbit arrays
            centroids: List of cluster centers
        """
        # Create detector instance
        detector = cls(
            env_bounds=env_bounds,
            cluster_eps=cluster_eps,
            min_samples=min_samples,
            clearance=clearance
        )

        # Detect clusters
        detector.detect_clusters(epcds_arr, z_range=z_range)

        # Generate rings
        rings, centroids = detector.generate_loiter_rings(sample_size=sample_size)

        return rings, centroids

    def visualize_clusters(self, save_path: Optional[str] = None):
        """
        Visualize detected clusters using Open3D/Plotly.

        Args:
            save_path: Optional path to save HTML visualization
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for visualization")
            return

        if len(self.clusters) == 0:
            print("No clusters to visualize")
            return

        fig = go.Figure()

        # Color palette for clusters
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

        # Plot clusters
        for i, cluster in enumerate(self.clusters):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=cluster[:, 0],
                y=cluster[:, 1],
                z=cluster[:, 2],
                mode='markers',
                marker=dict(size=3, color=color),
                name=f'Cluster {i}'
            ))

        # Plot centroids
        if len(self.centroids) > 0:
            centroids = np.array(self.centroids)
            fig.add_trace(go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                marker=dict(size=8, color='black', symbol='diamond'),
                name='Centroids'
            ))

        # Plot rings
        for i, ring in enumerate(self.rings):
            if ring.shape[0] > 0:
                # Close the ring for visualization
                ring_closed = np.vstack([ring, ring[0]])
                fig.add_trace(go.Scatter3d(
                    x=ring_closed[:, 0],
                    y=ring_closed[:, 1],
                    z=ring_closed[:, 2],
                    mode='lines',
                    line=dict(width=4, color=colors[i % len(colors)]),
                    name=f'Ring {i}',
                    showlegend=False
                ))

        fig.update_layout(
            title='Obstacle Clusters and Loiter Rings',
            scene=dict(aspectmode='data'),
            width=1200,
            height=800
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            fig.show()

    def __repr__(self) -> str:
        return (f"ObstacleDetector(clusters={len(self.clusters)}, "
                f"rings={len(self.rings)}, eps={self.cluster_eps}, "
                f"min_samples={self.min_samples}, clearance={self.clearance}m)")
