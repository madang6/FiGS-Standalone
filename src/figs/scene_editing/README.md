# Scene Editing Module

The `scene_editing` module provides spatial processing capabilities for FiGS-Standalone, including:

- **Dynamic environment bounds detection** from point clouds
- **Obstacle detection and clustering** using DBSCAN
- **Safe loiter orbit generation** around obstacles
- **Real-time collision checking** for trajectory validation

## Modules

### 1. `environment_bounds.py`

Dynamically detects environment bounds from point clouds, eliminating manual bound specification.

#### Key Features

- **Multiple detection methods**: percentile, AABB, convex hull, adaptive
- **Automatic spatial filtering** by altitude
- **KD-tree collision checking**
- **RRT-compatible output** format

#### Example Usage

```python
from figs.scene_editing import EnvironmentBounds

# Method 1: From GSplat scene (automatic)
env_bounds = EnvironmentBounds.from_gsplat_scene(
    scene_name="flightroom_ssv_exp",
    method="percentile",      # Options: "percentile", "aabb", "convex_hull", "adaptive"
    z_range=(-3.0, 0.0),      # Filter to flight altitude
    padding=0.5,              # Safety margin (meters)
    percentile=(5, 95)        # 5th-95th percentile
)

# Method 2: From point cloud array
pcd_arr = np.random.randn(1000, 3)  # Nx3 array
env_bounds = EnvironmentBounds.from_point_cloud(
    pcd_arr,
    method="percentile",
    padding=0.5
)

# Method 3: From pre-computed bounds
bounds_dict = {"minbound": (-5, -5, -2), "maxbound": (5, 5, 0)}
env_bounds = EnvironmentBounds.from_bounds_dict(bounds_dict)

# Check bounds
print(env_bounds)
# EnvironmentBounds(method='percentile', minbound=(-4.5, -4.5, -1.5),
#                   maxbound=(4.5, 4.5, -0.5), extent=(9.00, 9.00, 1.00)m)

# Use bounds
point = np.array([1.0, 2.0, -1.0])
within_bounds = env_bounds.is_within_bounds(point)
clearance = env_bounds.get_safe_clearance(point)
collision = env_bounds.check_collision(point, radius=0.3)

# Export for RRT
rrt_bounds = env_bounds.to_rrt_bounds()
# [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
```

#### API Reference

**Class: `EnvironmentBounds`**

| Method | Description |
|--------|-------------|
| `from_point_cloud(pcd_arr, method, ...)` | Create from Nx3 point cloud |
| `from_gsplat_scene(scene_name, ...)` | Load from GSplat scene |
| `from_bounds_dict(bounds_dict)` | Create from pre-computed bounds |
| `is_within_bounds(point)` | Check if point is within bounds |
| `check_collision(point, radius)` | KD-tree collision check |
| `get_safe_clearance(point)` | Distance to nearest obstacle |
| `to_dict()` | Export as dictionary |
| `to_rrt_bounds()` | Export for RRT planning |

---

### 2. `obstacle_detector.py`

Detects obstacle clusters using DBSCAN and generates safe loiter orbits.

#### Key Features

- **DBSCAN clustering** for obstacle grouping
- **Safe orbit generation** with clearance validation
- **Bidirectional rings** for CW/CCW loitering
- **Semantic segmentation integration**
- **Visualization** (Plotly HTML)

#### Example Usage

```python
from figs.scene_editing import ObstacleDetector, EnvironmentBounds

# Create environment bounds
env_bounds = EnvironmentBounds.from_gsplat_scene("flightroom_ssv_exp")

# Create detector
detector = ObstacleDetector(
    env_bounds=env_bounds,
    cluster_eps=0.5,        # DBSCAN neighborhood radius (meters)
    min_samples=10,         # Minimum points per cluster
    clearance=0.2           # Safety clearance (meters)
)

# Detect clusters from point cloud
clusters = detector.detect_clusters(
    point_cloud=env_bounds.pcd,
    z_range=(-2.5, -0.9)    # Height filter
)

# Generate loiter rings
rings, centroids = detector.generate_loiter_rings(
    sample_size=10,         # Waypoints per ring
    bidirectional=True,     # Create CW + CCW rings
    min_ring_points=3       # Minimum valid points
)

# Access results
print(f"Detected {len(clusters)} clusters")
print(f"Generated {len(rings)} loiter rings")

for i, (centroid, radius) in enumerate(zip(detector.centroids, detector.radii)):
    print(f"Cluster {i}: centroid={centroid}, radius={radius:.3f}m")

# Find nearest loiter target
current_pos = np.array([0, 0, -1.5])
nearest_centroid, idx, distance = detector.get_nearest_loiter_target(current_pos)

# Visualize
detector.visualize_clusters(save_path="clusters.html")
```

#### Convenience Method

```python
# All-in-one processing (ports SousVide-Semantic function)
rings, centroids = ObstacleDetector.process_obstacle_clusters_and_sample(
    epcds_arr=pcd_arr,
    env_bounds=env_bounds,
    z_range=(-2.5, -0.9),
    cluster_eps=0.5,
    min_samples=10,
    clearance=0.2,
    sample_size=10
)
```

#### API Reference

**Class: `ObstacleDetector`**

| Method | Description |
|--------|-------------|
| `detect_clusters(point_cloud, z_range)` | DBSCAN clustering |
| `generate_loiter_rings(sample_size, ...)` | Create safe orbits |
| `get_nearest_loiter_target(pos)` | Find closest cluster |
| `get_ring_by_index(index)` | Access ring by index |
| `from_semantic_segmentation(...)` | Create from semantic output |
| `process_obstacle_clusters_and_sample(...)` | All-in-one convenience method |
| `visualize_clusters(save_path)` | Plotly visualization |

**Attributes:**

- `clusters`: List of Nx3 cluster arrays
- `rings`: List of Mx3 loiter waypoint arrays
- `centroids`: List of (3,) cluster centers
- `radii`: List of orbit radii

---

### 3. `collision_checker.py`

Efficient collision detection using KD-tree spatial indexing.

#### Key Features

- **Per-timestep clearance computation**
- **Trajectory validation** with detailed statistics
- **Segment collision checking**
- **RL reward computation** (ports SousVide-Semantic)
- **Real-time updates** (rebuild KD-tree)

#### Example Usage

```python
from figs.scene_editing import CollisionChecker, EnvironmentBounds

# Create collision checker
env_bounds = EnvironmentBounds.from_gsplat_scene("flightroom_ssv_exp")
checker = CollisionChecker.from_environment_bounds(
    env_bounds,
    collision_radius=0.3  # Collision threshold (meters)
)

# Validate trajectory (tXUd format)
is_safe, info = checker.validate_trajectory(
    tXUd,                    # (11+, N) trajectory array
    min_clearance=0.5,       # Required clearance
    return_violations=True   # Include violation indices
)

print(f"Safe: {info['is_safe']}")
print(f"Collision: {info['collision_detected']}")
print(f"Min clearance: {info['min_clearance_value']:.3f}m")
print(f"Mean clearance: {info['mean_clearance']:.3f}m")

# Check line segment
start = np.array([0, 0, -1.5])
end = np.array([2, 2, -1.5])
collision = checker.check_segment_collision(start, end, num_samples=20)

# Compute clearances for positions
positions = np.array([[0, 0, -1.5], [1, 1, -1.5]])  # Nx3
clearances = checker.compute_clearances(positions)

# RL reward computation
collision_detected, collision_idx, clearances = checker.detect_collision(positions)
per_step_rewards, terminal_reward = checker.compute_collision_rewards(
    clearances,
    collision_detected,
    collision_penalty=-10.0,
    clearance_threshold=0.5
)
```

#### API Reference

**Class: `CollisionChecker`**

| Method | Description |
|--------|-------------|
| `compute_clearances(positions)` | Nearest obstacle distance |
| `detect_collision(positions)` | Detect collision + index |
| `check_segment_collision(start, end, ...)` | Line segment collision |
| `validate_trajectory(tXUd, ...)` | Full trajectory validation |
| `compute_collision_rewards(...)` | RL reward computation |
| `from_environment_bounds(env_bounds)` | Create from EnvironmentBounds |
| `update_point_cloud(pcd)` | Update obstacles |
| `get_statistics()` | Checker statistics |

---

## Integration with Existing FiGS

### With RRT Path Planning

```python
from figs.tsampling.rrt_datagen_v10 import RRTDataGen
from figs.scene_editing import EnvironmentBounds

# Detect bounds automatically
env_bounds = EnvironmentBounds.from_gsplat_scene("flightroom_ssv_exp")
rrt_bounds = env_bounds.to_rrt_bounds()

# Use in RRT
rrt = RRTDataGen(
    bounds=rrt_bounds,
    altitude=-1.5,
    obstacle_points=env_bounds.pcd,
    exclusion_radius=0.45
)

goal = np.array([3.0, 3.0])
path = rrt.generate_trajectory(goal, max_iterations=1000)
```

### With Trajectory Validation

```python
from figs.control.vehicle_rate_mpc import VehicleRateMPC
from figs.scene_editing import CollisionChecker, EnvironmentBounds

# Create checker
env_bounds = EnvironmentBounds.from_gsplat_scene("flightroom_ssv_exp")
checker = CollisionChecker.from_environment_bounds(env_bounds)

# Load and validate trajectory
ctl = VehicleRateMPC(tXUd, policy_config, frame_config)
is_safe, info = checker.validate_trajectory(ctl.tXUd)

if not is_safe:
    print(f"Trajectory unsafe! Min clearance: {info['min_clearance_value']:.3f}m")
```

### With Simulation

```python
from figs.simulator import Simulator
from figs.scene_editing import EnvironmentBounds, CollisionChecker

# Setup simulator
sim = Simulator(scene, rollout, frame)
env_bounds = EnvironmentBounds.from_gsplat_scene(scene)
checker = CollisionChecker.from_environment_bounds(env_bounds)

# Simulate
Tro, Xro, Uro, Imgs, _, _ = sim.simulate(ctl, t0, tf, x0)

# Check for collision
positions = Xro[0:3, :].T  # Extract positions
collision_detected, collision_idx, clearances = checker.detect_collision(positions)

if collision_detected:
    print(f"Collision at timestep {collision_idx}, clearance={clearances[collision_idx]:.3f}m")
```

---

## Detection Methods Comparison

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **percentile** | Robust to outliers, tunable | May be conservative | General purpose (recommended) |
| **aabb** | Tight bounds, fast | Sensitive to outliers | Clean point clouds |
| **convex_hull** | Minimal volume | Expensive, complex shapes | Precise bounds needed |
| **adaptive** | Automatic outlier removal | May over-filter | Noisy point clouds |

---

## Configuration Examples

### Conservative Safety

```python
env_bounds = EnvironmentBounds.from_gsplat_scene(
    scene_name,
    method="percentile",
    percentile=(10, 90),  # More conservative (10th-90th)
    padding=1.0           # Large safety margin
)

detector = ObstacleDetector(
    env_bounds,
    cluster_eps=0.3,      # Tighter clustering
    clearance=0.5         # Large clearance
)

checker = CollisionChecker.from_environment_bounds(
    env_bounds,
    collision_radius=0.5  # Large collision radius
)
```

### Aggressive Performance

```python
env_bounds = EnvironmentBounds.from_gsplat_scene(
    scene_name,
    method="percentile",
    percentile=(5, 95),   # Tight bounds
    padding=0.2           # Minimal margin
)

detector = ObstacleDetector(
    env_bounds,
    cluster_eps=0.7,      # Loose clustering
    clearance=0.1         # Minimal clearance
)

checker = CollisionChecker.from_environment_bounds(
    env_bounds,
    collision_radius=0.2  # Tight collision radius
)
```

---

## Troubleshooting

### No clusters detected

- **Increase `cluster_eps`**: Larger neighborhood radius
- **Decrease `min_samples`**: Lower threshold for clusters
- **Check `z_range`**: Ensure points exist in altitude range
- **Verify point cloud**: Use `detector.visualize_clusters()`

### Loiter rings have few points

- **Increase `clearance`**: More space around obstacles
- **Decrease `sample_size`**: Fewer waypoints per ring
- **Adjust `min_ring_points`**: Lower threshold
- **Check environment bounds**: Ensure navigable space exists

### High false positive collisions

- **Decrease `collision_radius`**: Tighter collision threshold
- **Increase `padding`**: Shrink navigable space
- **Check point cloud density**: May need downsampling

---

## Performance Tips

1. **KD-tree building is expensive**: Build once, query many times
2. **Point cloud size**: Downsample if >100k points (use Open3D voxel downsampling)
3. **Batch queries**: Pass Nx3 arrays instead of individual points
4. **Reuse objects**: Create EnvironmentBounds/CollisionChecker once per scene

---

## Example Workflows

See `examples/obstacle_detection_example.py` for complete examples:

```bash
cd /home/admin/StanfordMSL/FiGS-Standalone
python examples/obstacle_detection_example.py
```

This will demonstrate:
1. Dynamic bounds detection
2. Obstacle clustering
3. Collision checking
4. All-in-one processing

---

## References

- **SousVide-Semantic**: Original `process_obstacle_clusters_and_sample()` implementation
- **DBSCAN**: Ester et al. (1996) - Density-Based Spatial Clustering
- **KD-tree**: Bentley (1975) - Multidimensional Binary Search Trees
