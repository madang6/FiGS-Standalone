"""
Loiter scheduling and triggering for FiGS-Standalone.

Computes when to trigger loiter maneuvers along a trajectory.
Ports scheduling logic from SousVide-Semantic.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np


class LoiterScheduler:
    """
    Computes when to trigger loiter maneuvers along a trajectory.

    Supports multiple scheduling modes:
    - regular: Fixed intervals along trajectory
    - validation: Single loiter at trajectory midpoint
    - random: Poisson-process style random intervals
    - proximity: Triggered when near detected obstacles
    """

    def __init__(self,
                 interval_duration: float = 2.0,
                 mode: str = "regular",
                 min_clearance: float = 0.5):
        """
        Initialize LoiterScheduler.

        Args:
            interval_duration: Time between loiters for regular mode (seconds)
            mode: Scheduling mode ("regular", "validation", "random", "proximity")
            min_clearance: Minimum safe clearance for filtering (meters)
        """
        self.interval_duration = interval_duration
        self.mode = mode
        self.min_clearance = min_clearance

    def compute_intervals(self,
                         tXUd: np.ndarray,
                         mode: Optional[str] = None) -> np.ndarray:
        """
        Compute loiter trigger times along trajectory.

        Args:
            tXUd: (11+, N) trajectory array with time in row 0
            mode: Override scheduling mode (uses instance mode if None)

        Returns:
            Tsps: (M,) array of loiter start times
        """
        mode = mode or self.mode
        t0 = tXUd[0, 0]
        t1 = tXUd[0, -1]
        duration = t1 - t0

        if mode == "regular":
            # Fixed intervals along trajectory
            n_intervals = max(1, int(duration // self.interval_duration))
            Tsps = t0 + np.arange(n_intervals) * self.interval_duration

        elif mode == "validation":
            # Single loiter at trajectory midpoint
            Tsps = np.array([(t0 + t1) / 2])

        elif mode == "random":
            # Poisson-process style random intervals
            n_intervals = max(1, int(duration // self.interval_duration))
            Tsps = np.sort(np.random.uniform(t0, t1 - self.interval_duration, n_intervals))

        elif mode == "end_weighted":
            # More loiters toward end of trajectory (for landing approach)
            n_intervals = max(1, int(duration // self.interval_duration))
            # Use quadratic spacing for end-weighted distribution
            t_normalized = np.linspace(0, 1, n_intervals) ** 0.5
            Tsps = t0 + t_normalized * (t1 - t0 - self.interval_duration)

        elif mode == "start_weighted":
            # More loiters toward start of trajectory
            n_intervals = max(1, int(duration // self.interval_duration))
            t_normalized = 1 - (1 - np.linspace(0, 1, n_intervals)) ** 0.5
            Tsps = t0 + t_normalized * (t1 - t0 - self.interval_duration)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'regular', 'validation', "
                           f"'random', 'end_weighted', or 'start_weighted'")

        return Tsps

    def filter_safe_intervals(self,
                             intervals: np.ndarray,
                             tXUd: np.ndarray,
                             env_bounds=None) -> np.ndarray:
        """
        Remove loiter times where position is too close to obstacles.

        Args:
            intervals: Candidate loiter times
            tXUd: Reference trajectory
            env_bounds: EnvironmentBounds instance with collision checking

        Returns:
            Filtered loiter times
        """
        if env_bounds is None:
            return intervals

        safe_intervals = []
        Tpd = tXUd[0]

        for t in intervals:
            idx = np.searchsorted(Tpd, t) - 1
            idx = max(0, min(idx, len(Tpd) - 1))
            pos = tXUd[1:4, idx]

            clearance = env_bounds.get_safe_clearance(pos)
            if clearance >= self.min_clearance:
                safe_intervals.append(t)

        if len(safe_intervals) < len(intervals):
            n_removed = len(intervals) - len(safe_intervals)
            print(f"  Filtered {n_removed} loiter times due to clearance constraints")

        return np.array(safe_intervals)

    def filter_by_velocity(self,
                          intervals: np.ndarray,
                          tXUd: np.ndarray,
                          max_velocity: float = 1.0) -> np.ndarray:
        """
        Remove loiter times where vehicle is moving too fast.

        Loitering is safer when the vehicle is already moving slowly.

        Args:
            intervals: Candidate loiter times
            tXUd: Reference trajectory
            max_velocity: Maximum velocity threshold (m/s)

        Returns:
            Filtered loiter times
        """
        safe_intervals = []
        Tpd = tXUd[0]

        for t in intervals:
            idx = np.searchsorted(Tpd, t) - 1
            idx = max(0, min(idx, len(Tpd) - 1))
            vel = tXUd[4:7, idx]
            speed = np.linalg.norm(vel)

            if speed <= max_velocity:
                safe_intervals.append(t)

        if len(safe_intervals) < len(intervals):
            n_removed = len(intervals) - len(safe_intervals)
            print(f"  Filtered {n_removed} loiter times due to velocity constraints")

        return np.array(safe_intervals)

    def compute_proximity_intervals(self,
                                   tXUd: np.ndarray,
                                   obstacle_centroids: np.ndarray,
                                   trigger_distance: float = 2.0,
                                   min_separation: float = 3.0) -> np.ndarray:
        """
        Compute loiter times based on proximity to obstacles.

        Triggers loiter when trajectory comes within trigger_distance
        of any obstacle centroid.

        Args:
            tXUd: Reference trajectory
            obstacle_centroids: (K, 3) array of obstacle centers
            trigger_distance: Distance to trigger loiter (meters)
            min_separation: Minimum time between loiters (seconds)

        Returns:
            Loiter trigger times
        """
        if len(obstacle_centroids) == 0:
            return np.array([])

        Tpd = tXUd[0]
        positions = tXUd[1:4, :].T  # (N, 3)

        trigger_times = []
        last_trigger_time = -np.inf

        for i, pos in enumerate(positions):
            t = Tpd[i]

            # Skip if too close to last trigger
            if t - last_trigger_time < min_separation:
                continue

            # Check distance to all obstacles
            distances = np.linalg.norm(obstacle_centroids - pos, axis=1)
            min_distance = np.min(distances)

            if min_distance <= trigger_distance:
                trigger_times.append(t)
                last_trigger_time = t

        return np.array(trigger_times)

    def compute_batches(self,
                       tXUd: np.ndarray,
                       batch_size: int,
                       jitter: float = 0.0) -> List[np.ndarray]:
        """
        Compute batched loiter times for parallel processing.

        Useful for generating training data where multiple rollouts
        are processed together.

        Args:
            tXUd: Reference trajectory
            batch_size: Number of loiter times per batch
            jitter: Random time jitter to add (seconds)

        Returns:
            List of (batch_size,) arrays of loiter times
        """
        # Compute base intervals
        intervals = self.compute_intervals(tXUd)

        # Add jitter
        if jitter > 0:
            intervals = intervals + np.random.uniform(-jitter, jitter, len(intervals))
            intervals = np.clip(intervals, tXUd[0, 0], tXUd[0, -1] - self.interval_duration)

        # Shuffle for randomization
        np.random.shuffle(intervals)

        # Split into batches
        batches = []
        for i in range(0, len(intervals), batch_size):
            batch = intervals[i:i + batch_size]
            if len(batch) == batch_size:  # Only full batches
                batches.append(batch)

        return batches

    def get_loiter_windows(self,
                          tXUd: np.ndarray,
                          loiter_duration: float) -> List[Tuple[float, float]]:
        """
        Get time windows for each loiter maneuver.

        Args:
            tXUd: Reference trajectory
            loiter_duration: Total loiter duration (spin + smooth + tail)

        Returns:
            List of (start_time, end_time) tuples
        """
        intervals = self.compute_intervals(tXUd)

        windows = []
        for t_start in intervals:
            t_end = t_start + loiter_duration
            if t_end <= tXUd[0, -1]:
                windows.append((t_start, t_end))

        return windows

    def validate_schedule(self,
                         intervals: np.ndarray,
                         tXUd: np.ndarray,
                         loiter_duration: float) -> Tuple[bool, str]:
        """
        Validate a loiter schedule for feasibility.

        Checks:
        - All intervals within trajectory bounds
        - No overlapping loiter windows
        - Sufficient time after last loiter

        Args:
            intervals: Proposed loiter times
            tXUd: Reference trajectory
            loiter_duration: Total loiter duration

        Returns:
            (is_valid, message)
        """
        if len(intervals) == 0:
            return True, "Empty schedule is valid"

        t0 = tXUd[0, 0]
        t1 = tXUd[0, -1]

        # Check bounds
        if np.any(intervals < t0):
            return False, f"Loiter times before trajectory start ({t0})"

        if np.any(intervals + loiter_duration > t1):
            return False, f"Loiter windows extend past trajectory end ({t1})"

        # Check overlaps
        sorted_intervals = np.sort(intervals)
        for i in range(len(sorted_intervals) - 1):
            if sorted_intervals[i] + loiter_duration > sorted_intervals[i + 1]:
                return False, f"Overlapping loiter windows at t={sorted_intervals[i]:.2f}"

        return True, f"Valid schedule with {len(intervals)} loiters"

    def get_statistics(self, intervals: np.ndarray, tXUd: np.ndarray) -> dict:
        """
        Get statistics about a loiter schedule.

        Args:
            intervals: Loiter times
            tXUd: Reference trajectory

        Returns:
            Dictionary of statistics
        """
        if len(intervals) == 0:
            return {"count": 0}

        t0, t1 = tXUd[0, 0], tXUd[0, -1]
        duration = t1 - t0

        return {
            "count": len(intervals),
            "first_time": float(intervals[0]),
            "last_time": float(intervals[-1]),
            "mean_interval": float(np.mean(np.diff(intervals))) if len(intervals) > 1 else None,
            "coverage": float((intervals[-1] - intervals[0]) / duration) if len(intervals) > 1 else 0,
            "trajectory_duration": float(duration),
            "mode": self.mode
        }

    def __repr__(self) -> str:
        return (f"LoiterScheduler(mode='{self.mode}', "
                f"interval={self.interval_duration}s, "
                f"min_clearance={self.min_clearance}m)")
