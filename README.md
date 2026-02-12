# FiGS: Flying in Gaussian Splats

FiGS is a framework for trajectory optimization and control in Gaussian Splatting environments.

## Installation

### Prerequisites

- NVIDIA GPU with drivers installed
- Docker Engine ([install guide](https://docs.docker.com/engine/install/ubuntu/))
- NVIDIA Container Toolkit (for GPU access inside containers):

```bash
# Add the NVIDIA container toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

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

#### If you want to remove and restart
```bash
docker rmi figs:latest
rm -rf FiGS-Standalone # remember to move FiGS-Standalone/3dgs/ out to preserve any data!
```

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