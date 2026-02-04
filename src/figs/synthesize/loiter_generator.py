"""
Loiter trajectory generation for FiGS-Standalone.

Generates loiter maneuvers (spin-in-place) using FiGS Min-Snap optimization.
Ports loiter generation logic from SousVide-Semantic.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation

import figs.tsplines.min_snap as ms
import figs.utilities.trajectory_helper as th


def quat_to_yaw(q: np.ndarray) -> float:
    """
    Extract yaw angle from quaternion (xyzw format).

    Args:
        q: Quaternion in (qx, qy, qz, qw) format

    Returns:
        Yaw angle in radians
    """
    # Convert to rotation matrix and extract yaw
    R = Rotation.from_quat(q).as_matrix()
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw


class LoiterTrajectoryGenerator:
    """
    Generates loiter maneuvers using FiGS Min-Snap optimization.

    A loiter maneuver consists of three phases:
    1. Spin phase: Hover at fixed position while rotating yaw (1+ revolutions)
    2. Smooth phase: Blend from spin-end back to reference trajectory (SLERP)
    3. Tail phase: Continue following reference trajectory

    Uses the existing trajectory_helper functions for keyframe generation
    and Min-Snap optimization.
    """

    def __init__(self,
                 spin_duration: float = 4.0,
                 smooth_duration: float = 3.0,
                 tail_duration: float = 3.0,
                 spin_revs: int = 1,
                 Nco: int = 6,
                 hz: int = 20):
        """
        Initialize LoiterTrajectoryGenerator.

        Args:
            spin_duration: Duration of spin phase (seconds)
            smooth_duration: Duration of smooth transition (seconds)
            tail_duration: Duration of tail segment (seconds)
            spin_revs: Number of full 360° rotations during spin
            Nco: Number of polynomial coefficients for Min-Snap
            hz: Trajectory sampling frequency (Hz)
        """
        self.spin_duration = spin_duration
        self.smooth_duration = smooth_duration
        self.tail_duration = tail_duration
        self.spin_revs = spin_revs
        self.Nco = Nco
        self.hz = hz

    def generate_spin_trajectory(self,
                                position: np.ndarray,
                                yaw_start: float,
                                yaw_target: float,
                                base_frame_specs: Dict) -> np.ndarray:
        """
        Generate minimum-snap spin trajectory at fixed position.

        Args:
            position: (x, y, z) hover position
            yaw_start: Initial yaw angle (radians)
            yaw_target: Target yaw angle from reference trajectory
            base_frame_specs: Drone specifications dict with keys:
                - "mass": float
                - "motor_thrust_coeff": float (or "tn" for thrust normalizer)

        Returns:
            tXUd: (11+, M) trajectory array [t, pos, vel, quat, ...]
        """
        # Compute signed shortest angle difference
        dtheta = ((yaw_target - yaw_start + np.pi) % (2 * np.pi)) - np.pi
        direction = np.sign(dtheta) if dtheta != 0 else 1.0

        # Add full revolutions to minimum path
        theta_end = yaw_start + dtheta + direction * (2 * np.pi * self.spin_revs)

        # Generate keyframes for spin using trajectory_helper
        cfg = th.generate_spin_keyframes(
            name="loiter_spin",
            Nco=self.Nco,
            Nspin=self.spin_revs,
            xyz=position,
            theta0=yaw_start,
            theta1=theta_end,
            time=self.spin_duration
        )

        # Solve with Min-Snap
        result = ms.solve(cfg)
        if result is None:
            raise ValueError("Min-Snap solver failed for spin trajectory")

        Tps, CPs = result

        # Convert to full state trajectory
        tXUd_spin = th.TS_to_tXU(Tps, CPs, base_frame_specs, self.hz)

        return tXUd_spin

    def build_loiter_fragment(self,
                             tXUd_spin: np.ndarray,
                             tXUd_rrt: np.ndarray,
                             t0: float) -> np.ndarray:
        """
        Construct full loiter segment: spin → smooth → tail.

        Uses trajectory_helper.build_loiter_fragment() for the actual
        construction with SLERP interpolation.

        Args:
            tXUd_spin: Optimized spin trajectory from generate_spin_trajectory()
            tXUd_rrt: Original reference trajectory
            t0: Loiter start time in reference trajectory timeline

        Returns:
            tXUd_full: Complete loiter trajectory segment
        """
        return th.build_loiter_fragment(
            tXUd_spin,
            tXUd_rrt,
            t0,
            smooth_duration=self.smooth_duration,
            tail_duration=self.tail_duration,
            hz=self.hz
        )

    def generate_single_loiter(self,
                              tXUd_rrt: np.ndarray,
                              t0: float,
                              base_frame_specs: Dict,
                              yaw_start: Optional[float] = None) -> np.ndarray:
        """
        Generate a single loiter maneuver at time t0 along reference trajectory.

        Args:
            tXUd_rrt: (11+, N) reference trajectory array
            t0: Time at which to trigger loiter
            base_frame_specs: Drone specifications dict
            yaw_start: Optional starting yaw (random if None)

        Returns:
            tXUd_full: Complete loiter trajectory (spin + smooth + tail)
        """
        Tpd = tXUd_rrt[0]  # Time vector

        # Find RRT state at loiter time
        idx0 = np.searchsorted(Tpd, t0) - 1
        idx0 = max(0, min(idx0, len(Tpd) - 1))

        pos_rrt = tXUd_rrt[1:4, idx0]
        q_rrt = tXUd_rrt[7:11, idx0]

        # Yaw angles
        if yaw_start is None:
            yaw_start = np.random.uniform(0, 2 * np.pi)
        yaw_target = quat_to_yaw(q_rrt)

        # Generate spin trajectory
        tXUd_spin = self.generate_spin_trajectory(
            pos_rrt, yaw_start, yaw_target, base_frame_specs
        )

        # Build full fragment
        tXUd_full = self.build_loiter_fragment(tXUd_spin, tXUd_rrt, t0)

        return tXUd_full

    def generate_loiter_trajectories(self,
                                    tXUd_rrt: np.ndarray,
                                    loiter_times: np.ndarray,
                                    base_frame_specs: Dict,
                                    random_start_yaw: bool = True) -> List[np.ndarray]:
        """
        Generate multiple loiter maneuvers along reference trajectory.

        Args:
            tXUd_rrt: (11+, N) reference trajectory array
            loiter_times: (M,) array of times to trigger loiter
            base_frame_specs: Drone specifications dict
            random_start_yaw: If True, use random starting yaw for each loiter

        Returns:
            List of (11+, K) loiter trajectory arrays
        """
        loiter_trajectories = []

        for i, t0 in enumerate(loiter_times):
            try:
                yaw_start = None if random_start_yaw else 0.0

                tXUd_full = self.generate_single_loiter(
                    tXUd_rrt, t0, base_frame_specs, yaw_start
                )
                loiter_trajectories.append(tXUd_full)

                print(f"  Generated loiter {i+1}/{len(loiter_times)} at t={t0:.2f}s")

            except Exception as e:
                print(f"  Warning: Failed to generate loiter at t={t0:.2f}s: {e}")
                continue

        print(f"Successfully generated {len(loiter_trajectories)}/{len(loiter_times)} loiter trajectories")

        return loiter_trajectories

    def generate_loiter_at_position(self,
                                   position: np.ndarray,
                                   duration: float,
                                   base_frame_specs: Dict,
                                   yaw_start: float = 0.0,
                                   yaw_end: float = 2 * np.pi) -> np.ndarray:
        """
        Generate a standalone loiter trajectory at a specific position.

        Useful for creating loiter maneuvers around detected obstacles
        without requiring a reference trajectory.

        Args:
            position: (x, y, z) loiter position
            duration: Total loiter duration (spin only, no smooth/tail)
            base_frame_specs: Drone specifications dict
            yaw_start: Starting yaw angle (radians)
            yaw_end: Ending yaw angle (radians)

        Returns:
            tXUd: (11+, M) spin trajectory array
        """
        # Temporarily override spin duration
        original_duration = self.spin_duration
        self.spin_duration = duration

        try:
            tXUd_spin = self.generate_spin_trajectory(
                position, yaw_start, yaw_end, base_frame_specs
            )
        finally:
            self.spin_duration = original_duration

        return tXUd_spin

    def generate_orbit_trajectory(self,
                                 center: np.ndarray,
                                 radius: float,
                                 altitude: float,
                                 duration: float,
                                 base_frame_specs: Dict,
                                 num_waypoints: int = 12,
                                 clockwise: bool = True) -> np.ndarray:
        """
        Generate an orbit trajectory around a center point.

        Creates a circular path at fixed altitude with yaw always
        pointing toward the center (for inspection).

        Args:
            center: (x, y) center of orbit
            radius: Orbit radius (meters)
            altitude: Flight altitude (z coordinate)
            duration: Orbit duration (seconds)
            base_frame_specs: Drone specifications dict
            num_waypoints: Number of waypoints on circle
            clockwise: If True, orbit clockwise when viewed from above

        Returns:
            tXUd: (11+, M) orbit trajectory array
        """
        # Generate waypoints on circle
        direction = -1 if clockwise else 1
        angles = np.linspace(0, direction * 2 * np.pi, num_waypoints + 1)

        keyframes = {}
        for i, angle in enumerate(angles):
            t = (i / num_waypoints) * duration

            # Position on circle
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = altitude

            # Yaw pointing toward center
            yaw = angle + np.pi  # Point toward center

            is_endpoint = (i == 0 or i == num_waypoints)

            if is_endpoint:
                # Constrain position and velocity at endpoints
                # Compute tangent velocity
                speed = 2 * np.pi * radius / duration
                vx = -speed * np.sin(angle) * direction
                vy = speed * np.cos(angle) * direction
                omega = direction * 2 * np.pi / duration

                fo = [
                    [x, vx],
                    [y, vy],
                    [z, 0.0],
                    [yaw, omega]
                ]
            else:
                # Unconstrained intermediate points
                fo = [
                    [x, None, None],
                    [y, None, None],
                    [z, None, None],
                    [yaw, None, None]
                ]

            keyframes[f"fo{i}"] = {"t": t, "fo": fo}

        cfg = {"name": "orbit", "Nco": self.Nco, "keyframes": keyframes}

        # Solve with Min-Snap
        result = ms.solve(cfg)
        if result is None:
            raise ValueError("Min-Snap solver failed for orbit trajectory")

        Tps, CPs = result
        tXUd = th.TS_to_tXU(Tps, CPs, base_frame_specs, self.hz)

        return tXUd

    def get_parameters(self) -> Dict:
        """
        Get current generator parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "spin_duration": self.spin_duration,
            "smooth_duration": self.smooth_duration,
            "tail_duration": self.tail_duration,
            "spin_revs": self.spin_revs,
            "Nco": self.Nco,
            "hz": self.hz,
            "total_loiter_duration": self.spin_duration + self.smooth_duration + self.tail_duration
        }

    def __repr__(self) -> str:
        params = self.get_parameters()
        return (f"LoiterTrajectoryGenerator("
                f"spin={params['spin_duration']}s, "
                f"smooth={params['smooth_duration']}s, "
                f"tail={params['tail_duration']}s, "
                f"revs={params['spin_revs']}, "
                f"hz={params['hz']})")
