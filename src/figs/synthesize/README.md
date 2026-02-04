# Trajectory Synthesis Module

The `synthesize` module provides trajectory generation capabilities for FiGS-Standalone, including:

- **Loiter trajectory generation** using Min-Snap optimization
- **Loiter scheduling** with multiple triggering modes
- **Orbit trajectory generation** around obstacles

## Modules

### 1. `loiter_generator.py`

Generates loiter maneuvers (spin-in-place) using FiGS Min-Snap optimization.

#### Loiter Structure

A loiter maneuver consists of three phases:

```
┌─────────────┬──────────────┬─────────────┐
│  SPIN (4s)  │  SMOOTH (3s) │  TAIL (3s)  │
├─────────────┼──────────────┼─────────────┤
│ Fixed XYZ   │ Linear blend │ Follow RRT  │
│ Rotate yaw  │ SLERP quat   │ Original    │
│ 1+ full rev │ Transition   │ trajectory  │
└─────────────┴──────────────┴─────────────┘
```

#### Example Usage

```python
from figs.synthesize import LoiterTrajectoryGenerator, LoiterScheduler

# Create generator
loiter_gen = LoiterTrajectoryGenerator(
    spin_duration=4.0,      # Time for spin phase
    smooth_duration=3.0,    # Time for smooth transition
    tail_duration=3.0,      # Time following RRT
    spin_revs=1,            # Number of 360° rotations
    Nco=6,                  # Min-Snap polynomial coefficients
    hz=20                   # Sampling frequency
)

# Generate loiters along reference trajectory
loiter_times = np.array([2.0, 8.0, 14.0])  # Trigger times
loiter_trajectories = loiter_gen.generate_loiter_trajectories(
    tXUd_rrt,           # Reference trajectory (11+, N)
    loiter_times,       # When to trigger loiters
    frame_specs         # Drone specifications
)

# Generate standalone loiter at position
tXUd_loiter = loiter_gen.generate_loiter_at_position(
    position=np.array([0, 0, -1.5]),
    duration=6.0,
    base_frame_specs=frame_specs
)

# Generate orbit trajectory
tXUd_orbit = loiter_gen.generate_orbit_trajectory(
    center=np.array([2.0, 0.0]),
    radius=1.5,
    altitude=-1.5,
    duration=10.0,
    base_frame_specs=frame_specs
)
```

#### API Reference

**Class: `LoiterTrajectoryGenerator`**

| Method | Description |
|--------|-------------|
| `generate_single_loiter(tXUd_rrt, t0, ...)` | Generate one loiter at time t0 |
| `generate_loiter_trajectories(tXUd_rrt, times, ...)` | Generate multiple loiters |
| `generate_loiter_at_position(position, ...)` | Standalone loiter at position |
| `generate_orbit_trajectory(center, radius, ...)` | Circular orbit around point |
| `generate_spin_trajectory(position, yaw_start, ...)` | Low-level spin generation |
| `build_loiter_fragment(spin, rrt, t0)` | Combine spin + smooth + tail |
| `get_parameters()` | Get current parameters |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spin_duration` | 4.0s | Duration of spin phase |
| `smooth_duration` | 3.0s | Duration of smooth transition |
| `tail_duration` | 3.0s | Duration of tail segment |
| `spin_revs` | 1 | Number of 360° rotations |
| `Nco` | 6 | Polynomial coefficients |
| `hz` | 20 | Sampling frequency (Hz) |

---

### 2. `loiter_scheduler.py`

Computes when to trigger loiter maneuvers along a trajectory.

#### Scheduling Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `regular` | Fixed intervals | Training data generation |
| `validation` | Single midpoint | Testing |
| `random` | Poisson-process | Randomized training |
| `start_weighted` | More loiters at start | Takeoff inspection |
| `end_weighted` | More loiters at end | Landing approach |
| `proximity` | Near obstacles | Obstacle inspection |

#### Example Usage

```python
from figs.synthesize import LoiterScheduler

# Create scheduler
scheduler = LoiterScheduler(
    interval_duration=5.0,  # Time between loiters
    mode="regular",         # Scheduling mode
    min_clearance=0.5       # Safety margin (meters)
)

# Compute loiter times
loiter_times = scheduler.compute_intervals(tXUd_rrt)
print(f"Scheduled {len(loiter_times)} loiters: {loiter_times}")

# Filter by safety
from figs.scene_editing import EnvironmentBounds
env_bounds = EnvironmentBounds.from_gsplat_scene("flightroom")
loiter_times = scheduler.filter_safe_intervals(loiter_times, tXUd_rrt, env_bounds)

# Filter by velocity (only loiter when moving slowly)
loiter_times = scheduler.filter_by_velocity(loiter_times, tXUd_rrt, max_velocity=1.0)

# Proximity-based scheduling
obstacle_centroids = np.array([[2, 0, -1.5], [-2, 0, -1.5]])
loiter_times = scheduler.compute_proximity_intervals(
    tXUd_rrt,
    obstacle_centroids,
    trigger_distance=1.5,
    min_separation=4.0
)

# Validate schedule
is_valid, msg = scheduler.validate_schedule(loiter_times, tXUd_rrt, loiter_duration=10.0)
```

#### API Reference

**Class: `LoiterScheduler`**

| Method | Description |
|--------|-------------|
| `compute_intervals(tXUd, mode)` | Compute loiter trigger times |
| `filter_safe_intervals(intervals, tXUd, env_bounds)` | Filter by clearance |
| `filter_by_velocity(intervals, tXUd, max_vel)` | Filter by speed |
| `compute_proximity_intervals(tXUd, centroids, ...)` | Obstacle-based triggering |
| `compute_batches(tXUd, batch_size, jitter)` | Batched intervals |
| `get_loiter_windows(tXUd, duration)` | Get (start, end) windows |
| `validate_schedule(intervals, tXUd, duration)` | Check feasibility |
| `get_statistics(intervals, tXUd)` | Schedule statistics |

---

## Integration with Scene Editing

The synthesize module integrates with scene_editing for obstacle-aware loiter generation:

```python
from figs.scene_editing import EnvironmentBounds, ObstacleDetector, CollisionChecker
from figs.synthesize import LoiterTrajectoryGenerator, LoiterScheduler

# 1. Detect environment and obstacles
env_bounds = EnvironmentBounds.from_gsplat_scene("scene_name")
detector = ObstacleDetector(env_bounds, cluster_eps=0.5, clearance=0.3)
clusters = detector.detect_clusters(pcd_arr)
rings, centroids = detector.generate_loiter_rings()

# 2. Schedule loiters near obstacles
scheduler = LoiterScheduler(interval_duration=3.0)
loiter_times = scheduler.compute_proximity_intervals(
    tXUd_rrt,
    np.array(centroids),
    trigger_distance=2.0
)

# Filter by safety
loiter_times = scheduler.filter_safe_intervals(loiter_times, tXUd_rrt, env_bounds)

# 3. Generate loiter trajectories
loiter_gen = LoiterTrajectoryGenerator(spin_duration=4.0, spin_revs=1)
loiter_trajectories = loiter_gen.generate_loiter_trajectories(
    tXUd_rrt, loiter_times, frame_specs
)

# 4. Validate for collision
checker = CollisionChecker.from_environment_bounds(env_bounds)
for traj in loiter_trajectories:
    is_safe, info = checker.validate_trajectory(traj, min_clearance=0.3)
    if not is_safe:
        print(f"Warning: Trajectory collision at clearance {info['min_clearance_value']:.3f}m")
```

---

## Integration with Existing FiGS

### With Min-Snap Solver

The loiter generator uses `figs.tsplines.min_snap.solve()` internally:

```python
# Keyframe generation → Min-Snap solve → State trajectory
keyframes = th.generate_spin_keyframes(...)
Tps, CPs = ms.solve(keyframes)
tXUd = th.TS_to_tXU(Tps, CPs, frame_specs, hz)
```

### With MPC Controller

Loiter trajectories can be tracked with VehicleRateMPC:

```python
from figs.control.vehicle_rate_mpc import VehicleRateMPC

# Use loiter trajectory as reference
ctl = VehicleRateMPC(tXUd_loiter, policy_config, frame_config)

# Or switch between loiter and RRT
if near_obstacle:
    ctl.set_trajectory(tXUd_loiter)
else:
    ctl.set_trajectory(tXUd_rrt)
```

### With Simulator

```python
from figs.simulator import Simulator

sim = Simulator(scene, rollout, frame)

for tXUd_loiter in loiter_trajectories:
    ctl = VehicleRateMPC(tXUd_loiter, policy, frame)
    t0, tf = tXUd_loiter[0, 0], tXUd_loiter[0, -1]
    x0 = tXUd_loiter[1:11, 0]

    Tro, Xro, Uro, Imgs, _, _ = sim.simulate(ctl, t0, tf, x0)
```

---

## Configuration Examples

### Training Data Generation

```python
# Many loiters with randomization
scheduler = LoiterScheduler(interval_duration=2.0, mode="random")
loiter_gen = LoiterTrajectoryGenerator(
    spin_duration=4.0,
    smooth_duration=3.0,
    spin_revs=1,
    hz=20
)

# Generate batched intervals for parallel processing
batches = scheduler.compute_batches(tXUd_rrt, batch_size=10, jitter=0.5)
```

### Inspection Flight

```python
# Orbit around detected obstacles
loiter_gen = LoiterTrajectoryGenerator(hz=20)

for centroid in obstacle_centroids:
    tXUd_orbit = loiter_gen.generate_orbit_trajectory(
        center=centroid[:2],
        radius=1.5,
        altitude=centroid[2],
        duration=15.0,
        base_frame_specs=frame_specs,
        clockwise=True
    )
```

### Validation Testing

```python
# Single loiter at trajectory midpoint
scheduler = LoiterScheduler(mode="validation")
loiter_times = scheduler.compute_intervals(tXUd_rrt)
# Returns single time at trajectory center
```

---

## Relationship to SousVide-Semantic

This module ports loiter generation logic from SousVide-Semantic:

| SousVide-Semantic | FiGS-Standalone |
|-------------------|-----------------|
| `rollout_generator.generate_loiter_trajectories()` | `LoiterTrajectoryGenerator.generate_loiter_trajectories()` |
| `rollout_generator.compute_intervals()` | `LoiterScheduler.compute_intervals()` |
| `trajectory_helper.generate_spin_keyframes()` | Used internally (already in FiGS) |
| `trajectory_helper.build_loiter_fragment()` | Used internally (already in FiGS) |

---

## Example Workflows

See `examples/loiter_generation_example.py` for complete examples:

```bash
cd /home/admin/StanfordMSL/FiGS-Standalone
source ~/miniconda3/etc/profile.d/conda.sh && conda activate FiGS
python examples/loiter_generation_example.py
```

This demonstrates:
1. Basic loiter generation
2. Loiter with obstacle detection
3. Orbit trajectory generation
4. Standalone loiter at position
5. Scheduling mode comparison
