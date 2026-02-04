"""
Trajectory synthesis module for FiGS-Standalone.

Provides utilities for:
- Loiter trajectory generation
- Loiter scheduling and triggering
- Maneuver pattern library
"""

from figs.synthesize.loiter_generator import LoiterTrajectoryGenerator
from figs.synthesize.loiter_scheduler import LoiterScheduler

__all__ = [
    "LoiterTrajectoryGenerator",
    "LoiterScheduler",
]
