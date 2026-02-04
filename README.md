# FiGS: Flying in Gaussian Splats

FiGS is a framework for trajectory optimization and control in Gaussian Splatting environments.

## Installation

### Docker (Recommended)

The easiest way to get started is with Docker:

```bash
git clone https://github.com/madang6/FiGS-Standalone.git
cd FiGS-Standalone
docker-compose build
docker-compose run figs
```

This provides a complete environment with all dependencies pre-configured.

#### Configuration

Customize paths via environment variables or a `.env` file:

```bash
# Default value
DATA_PATH=/media/admin/data/StanfordMSL/nerf_data
```

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