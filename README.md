# FiGS: Flying in Gaussian Splats

FiGS is a framework for trajectory optimization and control in Gaussian Splatting environments.

## Installation

### Docker (Recommended)

The easiest way to get started is with Docker:

```bash
git clone https://github.com/madang6/FiGS-Standalone.git
cd FiGS-Standalone
docker compose build              # ~20-30 min first time
docker compose run --rm figs      # starts a shell with everything ready
```

Editable packages (figs, gemsplat, coverage_view_selection) are installed automatically on startup. Source code is bind-mounted, so edits on the host are immediately visible in the container and vice versa.

If you don't have `coverage_view_selection`, use the base config:

```bash
docker compose -f docker-compose.base.yml run --rm figs
```

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for full details, configuration options, and troubleshooting.

#### For downstream projects

FiGS serves as the base environment for other projects like [SINGER](https://github.com/madang6/SINGER). Those projects include their own `docker-compose.yml` that references the `figs:latest` image built here.

### Manual Installation (Alternative)

If you prefer not to use Docker:

```bash
git clone https://github.com/madang6/FiGS-Standalone.git
```

1) Update Submodules
```bash
cd FiGS-Standalone
git submodule update --recursive --init
```

2) Run the install.sh
```bash
bash install.sh
```

### What's Included
- Python 3.10 with numpy 1.26.4
- PyTorch 2.1.2 with CUDA support
- All core dependencies (nerfstudio, gsplat, etc.)
- FiGS package in editable mode

### Usage Examples

#### Data Storage

- Videos go in `3dgs/captures/`
- Images go in `3dgs/workspace/`
- Update the `capture_examples` variable in `notebooks/figs_generate_3dgs_example.py` with your directory names

#### Nerfstudio Commands

Process video data:
```bash
cd 3dgs/captures
ns-process-data video --data <data-directory-name> --output-dir ../workspace
```

Train a splatfacto model:
```bash
cd 3dgs/workspace
ns-train splatfacto --data <data-directory-name> \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --pipeline.model.rasterize-mode antialiased
```

Export the Gaussian splat:
```bash
ns-export gaussian-splat \
  --load-config outputs/<data-directory-name>/splatfacto/<timestamp>/config.yml \
  --output-dir outputs/<data-directory-name>/splatfacto/<timestamp>/exports
```

#### FiGS Notebooks

Process all videos in `capture_examples` (colmap, 3dgs, export):
```bash
cd notebooks
python figs_generate_3dgs_example.py
```

Additional notebooks:
```bash
python figs_capture_calibration.py
python figs_flightsim_example.py
```