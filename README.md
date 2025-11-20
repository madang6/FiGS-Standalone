# FiGS: Flying in Gaussian Splats

FiGS is a framework for trajectory optimization and control in Gaussian Splatting environments.

## Installation

### Quick Start
```bash
git clone https://github.com/StanfordMSL/FiGS-Standalone.git
```

1) Update Submodules
```bash
cd FiGS-Standalone
git submodule update --recursive --init
```

2) Install acados
```bash
# Navigate to acados folder
cd <repository-path>/FiGS-Examples/FiGS/acados/

# Compile
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install

# Add acados paths to bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```

3) Run the install.sh
```bash
bash install.sh
```

### What's Included
- Python 3.10 with numpy 1.26.4
- PyTorch 2.1.2 with CUDA support
- All core dependencies (nerfstudio, gsplat, acados, etc.)
- FiGS package in editable mode

### Usage Examples
1) nerfstudio defaults
```bash
cd 3dgs/captures
ns-process-data video --data <data-directory-name> --output-dir ../workspace 

cd 3dgs/workspace
ns-train splatfacto --data <data-directory-name> \
--pipeline.model.camera-optimizer.mode SO3xR3 \
--pipeline.model.rasterize-mode antialiased

ns-export gaussian-splat --load-config <outputs/data-directory-name/splatfacto/YYYY-YY-YY-YYYYYY/config.yml> \
--ouput-dir <outputs/data-directory-name/splatfacto/YYYY-YY-YY-YYYYYY/exports>
```

2) FiGS notebooks
```bash
cd notebooks
python figs_3dgs.py

python figs_capture_calibration.py
```