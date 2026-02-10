# Docker Setup for FiGS

## Prerequisites

- **x86_64 Linux** (ARM/Apple Silicon is not supported)
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support

## Quick Start

```bash
# 1. Build the image (first time only, ~20-30 min)
docker compose build

# 2. Start the container
docker compose run --rm figs
```

That's it. The container automatically installs FiGS, gemsplat, and coverage_view_selection in editable mode on startup. No manual setup needed.

If you don't have `coverage_view_selection` cloned as a sibling directory, use the base config instead:

```bash
docker compose -f docker-compose.base.yml run --rm figs
```

## What Happens on Startup

1. A user is created inside the container matching your host UID/GID (so files you create are owned by you)
2. `figs`, `gemsplat`, and `coverage_view_selection` are installed in editable mode (`pip install -e --no-deps`)
3. You get a bash shell in `/workspace/FiGS-Standalone`

All heavy dependencies (torch, nerfstudio, gsplat, acados) are baked into the image and don't need reinstalling.

## Editable Development

Source code is bind-mounted from the host, so:
- Edit files on the host (VS Code, etc.) and changes are immediately visible inside the container
- Edit files inside the container and changes are immediately visible on the host
- Files created inside the container are owned by your host user

## Configuration

Customize paths via environment variables or a `.env` file:

```bash
# Data directory (default shown)
DATA_PATH=/media/admin/data/StanfordMSL/nerf_data

# GPU selection (default: 0)
CUDA_VISIBLE_DEVICES=0

# X11 display for GUI apps
DISPLAY=:1
```

## Compose Files

| File | Includes | Usage |
|------|----------|-------|
| `docker-compose.yml` | FiGS + gemsplat + coverage_view_selection | `docker compose run --rm figs` |
| `docker-compose.base.yml` | FiGS + gemsplat only | `docker compose -f docker-compose.base.yml run --rm figs` |

## Mounted Volumes

| Host | Container |
|------|-----------|
| `./` (FiGS-Standalone) | `/workspace/FiGS-Standalone` |
| `../coverage_view_selection` | `/workspace/coverage_view_selection` (full config only) |
| Data directory | Same path as host (so symlinks work) |

## Downstream Projects

FiGS serves as the base environment for other projects like [SINGER](https://github.com/madang6/SINGER). Those projects include their own `docker-compose.yml` that references the `figs:latest` image built here.

## Cleanup

```bash
# Remove containers
docker compose down

# Remove unused Docker images and cache
docker system prune

# More aggressive cleanup (removes all unused images)
docker system prune -a
```

## Troubleshooting

**Container won't start with GPU errors**: Make sure the NVIDIA Container Toolkit is installed and `nvidia-smi` works on the host.

**Out of disk space**: Run `docker system prune` to clean up old images and build cache.

**Editable install fails on startup**: Check that `pyproject.toml` exists in the project root and in `gemsplat/`.
