# FiGS: Flying in Gaussian Splats

FiGS is a framework for trajectory optimization and control in Gaussian Splatting environments.

## Installation

### Quick Start (Two Commands)
```bash
git clone https://github.com/StanfordMSL/FiGS-Standalone.git
cd FiGS-Standalone
conda env create -f environment.yml
```

That's it! The `environment.yml` handles:
- Creating the conda environment with all dependencies
- Installing FiGS and Hierarchical-Localization in editable mode
- Setting up Python path and CUDA support

### What's Included
- Python 3.10 with numpy 1.26.4
- PyTorch 2.1.2 with CUDA support
- All core dependencies (nerfstudio, gsplat, scipy, etc.)
- FiGS package in editable mode
- Hierarchical-Localization in editable mode

## Usage

See https://github.com/StanfordMSL/FiGS-Examples for how to use the package.
