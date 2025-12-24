# FiGS: Flying in Gaussian Splats

![FiGS](figs.png)

FiGS is a framework for trajectory optimization and control in Gaussian Splatting environments.

## Installation

### Quick Start
```bash
git clone -b acados-away https://github.com/StanfordMSL/FiGS-Standalone.git
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

### Known Issues (and some fixes)
nerfstudio 1.1.5 is not up to date with COLMAP 3.13.0. This causes issues with
```
ns-process-data
```
You can apply a fix like so:
```
Update(~/miniconda3/envs/FiGS/lib/python3.10/site-packages/nerfstudio/process_dat
        a/colmap_utils.py)                                           
      126          f"--image_path {image_dir}",
      127          "--ImageReader.single_camera 1",
      128          f"--ImageReader.camera_model {camera_model.value}",
      129 -        f"--SiftExtraction.use_gpu {int(gpu)}",                  
      129 +        f"--FeatureExtraction.use_gpu {int(gpu)}",               
      130      ]
      131      if camera_mask_path is not None:
      132          feature_extractor_cmd.append(f"--ImageReader.camera_mask_
           path {camera_mask_path}")

Update(~/miniconda3/envs/FiGS/lib/python3.10/site-packages/nerfstudio/process_dat
        a/colmap_utils.py)                                           
      140      feature_matcher_cmd = [
      141          f"{colmap_cmd} {matching_method}_matcher",
      142          f"--database_path {colmap_dir / 'database.db'}",
      143 -        f"--SiftMatching.use_gpu {int(gpu)}",                    
      143 +        f"--FeatureMatching.use_gpu {int(gpu)}",                 
      144      ]
      145      if matching_method == "vocab_tree":
      146          vocab_tree_filename = get_vocab_tree()
```
this fix was found using an AI code agent.