#!/bin/bash
# FiGS Docker Build and Run Helper
# Usage:
#   ./docker-figs.sh build          # Build the image
#   ./docker-figs.sh run            # Run interactive container
#   ./docker-figs.sh run <command>  # Run a specific command

set -e

IMAGE_NAME="figs"
IMAGE_TAG="latest"
GSPLAT_VERSION="1.5.3"
NERFSTUDIO_VERSION="main"  # Branch/tag to clone from nerfstudio repo

# Detect script directory and workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

show_help() {
    cat <<EOF
FiGS Docker Helper

Usage: $0 <command> [options]

Commands:
  build [--gsplat VERSION]   Build the Docker image
  run [command]              Run interactive container (or execute command)
  shell                      Alias for 'run' with bash shell

Build Options:
  --gsplat VERSION      Set gsplat version (default: ${GSPLAT_VERSION})
                        Use 1.4.0 for legacy FiGS, 1.5.3 for nbv-splat compatibility
  --nerfstudio VERSION  Set nerfstudio branch/tag (default: ${NERFSTUDIO_VERSION})

Examples:
  $0 build                           # Build with gsplat 1.5.3
  $0 build --gsplat 1.4.0            # Build with gsplat 1.4.0
  $0 run                             # Start interactive shell
  $0 run ns-train --help             # Run nerfstudio command

Volume Mounts:
  - ${WORKSPACE_DIR}/FiGS-Standalone -> /workspace/FiGS-Standalone
  - ${WORKSPACE_DIR}/coverage_view_selection -> /workspace/coverage_view_selection
  - ~/.cache -> /home/user/.cache

First-time setup inside container:
  cd /workspace/FiGS-Standalone && pip install -e . --no-deps
  cd /workspace/FiGS-Standalone/gemsplat && pip install -e . --no-deps
  cd /workspace/coverage_view_selection && pip install -e .
  ns-install-cli

EOF
}

build_image() {
    local gsplat_ver="${GSPLAT_VERSION}"
    local nerfstudio_ver="${NERFSTUDIO_VERSION}"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gsplat)
                gsplat_ver="$2"
                shift 2
                ;;
            --nerfstudio)
                nerfstudio_ver="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo "Building FiGS Docker image..."
    echo "  gsplat version: ${gsplat_ver}"
    echo "  nerfstudio branch: ${nerfstudio_ver}"
    echo ""

    docker build \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -t "${IMAGE_NAME}:gsplat-${gsplat_ver}" \
        --build-arg GSPLAT_VERSION="${gsplat_ver}" \
        --build-arg NERFSTUDIO_VERSION="${nerfstudio_ver}" \
        -f "${SCRIPT_DIR}/Dockerfile.FiGS" \
        "${SCRIPT_DIR}"

    echo ""
    echo "Build complete!"
    echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  Also tagged: ${IMAGE_NAME}:gsplat-${gsplat_ver}"
}

run_container() {
    local cmd="${@:-/bin/bash -l}"

    echo "Starting FiGS container..."

    docker run --gpus all \
        -u "$(id -u):$(id -g)" \
        -v "${WORKSPACE_DIR}/FiGS-Standalone:/workspace/FiGS-Standalone" \
        -v "${WORKSPACE_DIR}/coverage_view_selection:/workspace/coverage_view_selection" \
        -v "${HOME}/.cache:/home/user/.cache" \
        -p 7007:7007 \
        --rm \
        -it \
        --shm-size=12gb \
        -e HOME=/workspace \
        -w /workspace \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        ${cmd}
}

# Main
case "${1:-}" in
    build)
        shift
        build_image "$@"
        ;;
    run|shell)
        shift
        run_container "$@"
        ;;
    -h|--help|help|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 --help' for usage"
        exit 1
        ;;
esac
