import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

import cv2

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap

import open3d as o3d

class GSplat():
    def __init__(self, scene_config:Path) -> None:
        """
        GSplat class for rendering images from GSplat pipeline.

        Args:
            - scene_config: FiGS scene configuration dictionary.

        Variables:
            - device: Device to run the pipeline on.
            - config: Configuration for the pipeline.
            - pipeline: Pipeline for the GSplat model.
            - T_w2g: Transformation matrix from world to GSplat frame.

        """
        # Do some acrobatics to find necessary config files
        self.config_path = scene_config['path']
        self.name = scene_config['name']
        current_path = self.config_path
        print(f"Current path dict: {current_path}")
        print(f"Current path name: {current_path.name}")
        
        # Search upward for 'src' directory, then go up one level to get workspace root
        workspace_root = None
        while current_path.parent != current_path:  # Stop when reaching the filesystem root
            if current_path.name == "src":
                workspace_root = current_path.parent  # Go up one level from 'src'
                break
            current_path = current_path.parent  # Move up one level
        
        if workspace_root is None:
            raise FileNotFoundError(f"Could not find 'src' directory in parent directories of {self.config_path}")
            
        relative_target = Path("configs/perception/perception_mode.yml")
        target_path = workspace_root / relative_target
        self.perception_path = target_path

        relative_target = Path("configs/frame/carl.json")
        target_path = workspace_root / relative_target
        self.drone_path = target_path

        with open(self.drone_path, 'r') as file:
            drone_config = json.load(file)
            self.drone_config = drone_config

        # Some useful intermediate variables
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        T_w2g = np.array([
            [ 1.00, 0.00, 0.00, 0.00],
            [ 0.00,-1.00, 0.00, 0.00],
            [ 0.00, 0.00,-1.00, 0.00],
            [ 0.00, 0.00, 0.00, 1.00]
        ])

        # Class variables
        self.device = device
        self.config,self.pipeline, _, _ = eval_setup(scene_config['path'],test_mode="inference")
        self.T_w2g = T_w2g

        self.dataparser_scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
        self.dataparser_transform = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

        self.camera_out = self.generate_output_camera(self.drone_config['camera'])
        self.channels = self.drone_config['camera']['channels']
        self.width = self.camera_out.width
        self.height = self.camera_out.height

        with open(self.perception_path, 'r') as file:
            perception_mode = yaml.safe_load(file)
            self.visual_mode = perception_mode.get("visual_mode")
            self.perception_mode = perception_mode.get("perception_mode")

        if self.visual_mode not in ["rgb","dynamic","semantic_depth"]:
            raise ValueError(f"Invalid visual mode: {self.visual_mode}")
        
        if self.visual_mode=="semantic_depth":
            self.running_min = float("0")
            self.running_max = float("0")
            # Get the name of the directory 3 levels up from self.config_path
            directory_name = self.config_path.parents[2].name
            # Go up 5 levels from self.config_path
            search_path = self.config_path.parents[4]
            # Search for transforms.json in the directory sharing the string
            self.data_path = None
            for root, dirs, files in os.walk(search_path):
                if directory_name in dirs:
                    potential_path = os.path.join(root, directory_name)
                    if 'transforms.json' in os.listdir(potential_path):
                        self.data_path = potential_path
                        break

            if self.data_path is None:
                raise FileNotFoundError(f"Could not find 'transforms.json' in any directory named '{directory_name}' within {search_path}")
            with open(os.path.join(self.data_path, 'transforms.json'), 'r') as file:
                self.transforms_nerf = json.load(file)

    def generate_output_camera(self, camera_config:Dict[str,Union[int,float]]) -> Cameras:
        """
        Generate an output camera for the pipeline.

        Args:
            - camera_config: Configuration dictionary for the camera. Contains
                             width, height, channels, fx, fy, cx, cy.

        Returns:
            - camera_out: Output camera for the pipeline.
            
        """
        # Extract the camera parameters
        width,height = camera_config["width"],camera_config["height"]
        # channels = camera_config["channels"]
        fx,fy = camera_config["fx"],camera_config["fy"]
        cx,cy = camera_config["cx"],camera_config["cy"]

        # Create the camera object
        camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]
        camera_out = Cameras(
            camera_to_worlds=1.0*camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )

        camera_out = camera_out.to(self.device)

        return camera_out
    
    def render_rgb(
        self,
        camera: Cameras,
        T_c2w: np.ndarray,
        query: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Render an RGB (and optionally semantic) image from the GSplat pipeline.

        Args:
            camera:          Camera object for the pipeline.
            T_c2w:           4×4 camera→world transform.
            query:           Optional positive-language query for semantic filtering.
#FIXME
        Returns:
            - If query is None, returns:
                image_rgb: np.ndarray of shape (H,W,3), uint8
            - If query is provided, returns:
                {"rgb": image_rgb, "semantic": image_sem}, both uint8 np.ndarrays
        """
        
        if self.name.startswith("sv_") and "sfm_to_mocap_T" in self.transforms_nerf:
            dp_scale     = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
            dp_transform = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
            sfm2mocap_T  = np.asarray(self.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])
            T_w2g        = self.T_w2g

            S    = np.eye(4);      S[:3,:3] *= 1.0 / dp_scale
            T_dp = np.eye(4);      T_dp[:3,:] = dp_transform
            inv_dp = np.linalg.inv(T_dp)
            M    = sfm2mocap_T @ inv_dp
            A    = T_w2g @ M @ S
            A_inv = np.linalg.inv(A)
            T_c2g = A_inv @ T_c2w
            P_c2g = torch.from_numpy(T_c2g[:3, :]).float()
        else:
            T_c2g = self.dataparser_scale * (self.T_w2g @ T_c2w)
            P_c2g = torch.tensor(T_c2g[:3, :]).float()
            P_c2g[:3, :3] *= 1.0 / self.dataparser_scale

        camera.camera_to_worlds = P_c2g[None, :, :]
        cameras = camera.to(self.device)

        if query is not None:
            self.pipeline.model.viewer_utils.handle_language_queries(
                raw_text=query,
                is_positive=True
            )

            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(
                raw_text="",
                is_positive=False)
            
            obb_box = None

            with torch.no_grad():
                try:
                    outputs = self.pipeline.model.get_outputs_for_camera(
                        cameras,
                        obb_box=None,
                        compute_semantics=True
                    )
                except:
                    outputs = self.pipeline.model.get_outputs_for_camera(
                        cameras,
                        obb_box=None
                    )

        outputs = self.pipeline.model.get_outputs_for_camera(
            cameras,
            obb_box=None
        )

        image_d = outputs.get("depth", None)
        # Convert normalized depth to true metric depth
        depth_image = np.squeeze(image_d).cpu().numpy() * 1.0/self.dataparser_scale
        
        # Create colored visualization of depth
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colored_tensor = torch.from_numpy(depth_colored).float().cuda() / 255.0
        image_depth = depth_colored_tensor
        image_depth = image_depth.cpu().numpy()
        image_depth = (255*image_depth).astype(np.uint8)
        
        image_rgb = outputs["rgb"].cpu().numpy()
        image_rgb = (255 * image_rgb).astype(np.uint8)

        if query is None:
            return {"rgb":image_rgb, "depth": image_depth, "depth_raw": depth_image}
        elif query == "null":
            sem = outputs.get(self.perception_mode, outputs.get("similarity", outputs["rgb"]))
        else:
            sem = outputs.get(self.perception_mode, outputs.get("similarity", outputs["rgb"]))
            sem = self.render_rescale(sem)
    
        sem = apply_colormap(sem, ColormapOptions("turbo"))
        image_sem = (255 * sem.cpu().numpy()).astype(np.uint8)

        return {
            "rgb": image_rgb,
            "semantic": image_sem,
            "depth": image_depth,
            "depth_raw": depth_image,
            }    
    
    def render_rgb_old(self, camera:Cameras,T_c2w:np.ndarray) -> np.ndarray:
        """
        Render an RGB image from the GSplat pipeline.

        Args:
            - camera: Camera object for the pipeline.
            - T_c2w: Transformation matrix from camera to world frame.

        Returns:
            - image_rgb: Rendered RGB image.

        """

        if self.name.startswith("sv_") and "sfm_to_mocap_T" in self.transforms_nerf:
            # # print("Special handling for sv_ prefix")
            # dataparser_scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
            # dataparser_transform = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

            # transform = torch.eye(4)
            # transform[:3,:3] = dataparser_transform[:3,:3]

            # T_c2g = dataparser_scale * np.linalg.inv(
            #     np.asarray(self.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])
            #     ) @ T_c2w
            
            # # T_f2n = torch.tensor(self.T_w2g, dtype=torch.float32, device=T_c2w.device)
            # T_c2g = self.T_w2g @ T_c2g
            
            # # P_c2n = T_c2n[0:3,:].clone().detach().float().requires_grad_(False)
            # P_c2g = torch.tensor(T_c2g[0:3,:]).float()
            # P_c2g[:3,:3]*=1/dataparser_scale

            # 1) unpack
            dp_scale     = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
            dp_transform = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
            sfm2mocap_T  = np.asarray(self.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])
            T_w2g        = self.T_w2g                            # 4×4

            # 2) build the same A that you used on points
            S = np.eye(4);      S[:3,:3] *= 1.0 / dp_scale      # scale
            T_dp = np.eye(4);   T_dp[:3,:] = dp_transform        # dataparser
            inv_dp = np.linalg.inv(T_dp)
            M = sfm2mocap_T @ inv_dp

            A = T_w2g @ M @ S

            # 3) invert it
            A_inv = np.linalg.inv(A)

            # 4) send your camera through that inverse
            T_c2g = A_inv @ T_c2w

            # 5) grab the 3×4 for NeRF
            P_c2g = torch.from_numpy(T_c2g[:3, :]).float()
            
        else:# Extract the camera to gsplat pose
            T_c2g = self.dataparser_scale * self.T_w2g@T_c2w
            P_c2g = torch.tensor(T_c2g[0:3,:]).float()
            P_c2g[:3,:3]*=1/self.dataparser_scale

        # Render rgb image from the pose
        camera.camera_to_worlds = P_c2g[None,:3, ...]
        with torch.no_grad():
            image_rgb = self.pipeline.model.get_outputs_for_camera(camera, obb_box=None)["rgb"]

        # Convert to output image
        image_rgb = image_rgb.cpu().numpy()             # Convert to numpy
        image_rgb = (255*image_rgb).astype(np.uint8)    # Convert to uint8

        return image_rgb
#TODO
#FIXME Need to update this function to work correctly with SousVide-Semantic    
    def render(self, xcr:np.ndarray, xpr:np.ndarray=None,
               positives: Optional[str] = "",
               negatives: Optional[str] = "",
               sample_semantic_embeds: Optional[bool] = False,
               compute_semantics: Optional[bool] = True,
               debug_mode=False):
        
        if self.visual_mode == "static":
            image = self.static_render(xcr)
        elif self.visual_mode == "dynamic":
            image = self.dynamic_render(xcr,xpr)
        elif self.visual_mode == "semantic_depth":
            image = self.static_render(xcr,compute_semantics, positives, negatives, sample_semantic_embeds, debug_mode)
            image_rgb = image["rgb"]
            image_depth = image["depth"]
            image_sem = image["semantic"]

            return {"semantic": image_sem, "depth": image_depth, "rgb": image_rgb}
        else:
            raise ValueError(f"Invalid visual mode: {self.visual_mode}")
        
    def static_render(self, xcr:np.ndarray,
                      compute_semantics:Optional[bool]=False,
                      positives:Optional[str]="",negatives:Optional[str]="",
                      sample_semantic_embeds:Optional[bool]=False,
                      debug_mode:Optional[bool]=False) -> torch.Tensor:

        # Extract the pose
        T_c2n = self.T_w2g@pose2nerf_transform(np.hstack((xcr[0:3],xcr[6:10])))
        P_c2n = torch.tensor(T_c2n[0:3,:]).float()
        # P_c2n[:3,:3]*=1/self.dataparser_scale

        # rescale if using gemsplat
        if "sfm_to_mocap_T" in self.transforms_nerf:
            dataparser_scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale

            T_c2n = dataparser_scale * T_c2n
            P_c2n = T_c2n[0:3,:]
            P_c2n[:3,:3]*=1/dataparser_scale


        # render_pose_target = P_c2n.clone().detach().float().requires_grad_(False)
        render_pose_target = torch.tensor(P_c2n,device=self.device).float()
        # convert to OpenGL
        render_pose_target_gl = render_pose_target.clone()

        # render_pose_target_gl = render_pose_target_gl[None][:, :3, :]
        # render_pose_target_gl[:,:,:3]*=1/dataparser_scale

        cameras = Cameras(
            # camera_to_worlds=camera_to_world,
            fx=self.camera_out.fx,
            fy=self.camera_out.fy,
            cx=self.camera_out.cx,
            cy=self.camera_out.cy,
            width=self.camera_out.width,
            height=self.camera_out.height,
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=render_pose_target_gl[None][:, :3, :]
        )
            
        cameras = cameras.to(self.device)
        
        if self.visual_mode == "semantic" or self.visual_mode == "semantic_depth":
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=positives,
                                                                        is_positive=True)
                
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=negatives,
                                                                        is_positive=False)
            obb_box = None
            
            with torch.no_grad():
                try:
                    outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box,
                                                                        sample_semantic_embeds=sample_semantic_embeds,
                                                                        compute_semantics=compute_semantics)
                except:  
                    try:
                        outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box,
                                                                            compute_semantics=compute_semantics)
                    except:
                        outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box)

            if self.perception_mode not in outputs.keys():
                print(f"No semantic output with perception mode {self.perception_mode}")
                image_sem = outputs['rgb']
            elif self.perception_mode == "MDS":
                # image_sem = torch.mean(outputs['rgb'], dim=-1, keepdim=True).repeat(1, 1, 3)
                # image_mono = torch.tensordot(outputs['rgb'], torch.tensor([0.2989, 0.5870, 0.1140]), dims=([-1], [0]))
                # image_sem = torch.cat([image_mono, outputs['depth'], outputs['sem']], dim=1)
                image_sem = outputs[self.perception_mode]
                # image_sem[:,:,2] = self.render_rescale(image_sem[:,:,2])
            else:
                image_sem = outputs[self.perception_mode]
                image_sem = self.render_rescale(image_sem)
                image_sem = apply_colormap(image_sem, ColormapOptions("turbo"))

            image_d = outputs['depth']
            image_rgb = outputs['rgb']

            image_sem = image_sem.cpu().numpy()
            image_sem = (255*image_sem).astype(np.uint8)

            # image_d = image['depth']  #NOTE make sure to only use this when testing depth
            depth_image = np.squeeze(image_d).cpu().numpy()
            depth_normalized = cv2.normalize(depth_image,None,0,255,cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_colored_tensor = torch.from_numpy(depth_colored).float().cuda() / 255.0
            image_depth = depth_colored_tensor
            image_depth = image_depth.cpu().numpy()
            image_depth = (255*image_depth).astype(np.uint8)

            image_rgb = image_rgb.cpu().numpy()
            image_rgb = (255*image_rgb).astype(np.uint8)

            return {"semantic":image_sem, "depth":image_depth, "rgb":image_rgb}
        
        else:
            # render outputs
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=None)

            image_rgb = outputs["rgb"]
            image_rgb = image_rgb.cpu().numpy()
            image_rgb = (255*image_rgb).astype(np.uint8)

            image_depth = np.zeros_like(image_rgb)
            image_sem = np.zeros_like(image_rgb)

            return {"semantic": image_sem,"depth": image_depth,"rgb":image_rgb}
    
    def render_rescale(self, srgb_mono):
        '''This function takes a single channel semantic similarity and rescales it globally'''
        # Maintain running min/max
        if not hasattr(self, "running_min"):
            self.running_min = -1.0
        if not hasattr(self, "running_max"):
            self.running_max = 1.0

        current_min = srgb_mono.min().item()
        current_max = srgb_mono.max().item()
        self.running_min = min(self.running_min, current_min)
        self.running_max = max(self.running_max, current_max)

        similarity_clip = (srgb_mono - self.running_min) / (self.running_max - self.running_min + 1e-10)

        return similarity_clip
    
    def generate_point_cloud(self,
                            use_bounding_box: bool = False,
                            bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1),
                            bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1),
                            densify_scene: bool = False,
                            split_params: Dict = {
                                'n_split_samples': 2
                            },
                            cull_scene: bool = False,
                            cull_params: Dict = {
                                'cull_alpha_thresh': 0.1,
                                'cull_scale_thresh': 0.5
                            }
                            )->None:
        def SH2RGB(sh):
            # From INRIA
            # Base auxillary coefficient
            C0 = 0.28209479177387814
            return sh * C0 + 0.5
        
        if densify_scene:
            if cull_scene:
                # cache the previous values of all parameters
                means_prev = self.pipeline.model.means.clone()
                scales_prev = self.pipeline.model.scales.clone()
                quats_prev = self.pipeline.model.quats.clone() 
                features_dc_prev = self.pipeline.model.features_dc.clone()
                features_rest_prev = self.pipeline.model.features_rest.clone()
                opacities_prev = self.pipeline.model.opacities.clone()
                
                try:
                    clip_embeds_prev = self.pipeline.model.clip_embeds.clone()
                except AttributeError:
                    pass
        
                # cull Gaussians
                self.pipeline.model.cull_gaussians_refinement(cull_alpha_thresh=cull_params['cull_alpha_thresh'],
                                                              cull_scale_thresh=cull_params['cull_scale_thresh'],
                                                              )
                
            # split mask
            split_mask = torch.ones(len(self.pipeline.model.scales),
                                    dtype=torch.bool).to(self.device)

            # split Gaussians
            try:
                means, features_dc, features_rest, opacities, scales, quats, clip_embeds \
                    = self.pipeline.model.split_gaussians(split_mask, split_params['n_split_samples'])
                # means, features_dc, features_rest, opacities, scales, quats, \
                #     = self.pipeline.model.split_gaussians(split_mask, split_params['n_split_samples'])
            except AttributeError:
                means, features_dc, features_rest, opacities, scales, quats \
                    = self.pipeline.model.split_gaussians(split_mask, split_params['n_split_samples'])                    
                # extract the semantic embeddings
                clip_embeds = self.pipeline.model.clip_field(self.pipeline.model.means).float().detach()

            
            # 3D points
            pcd_points = means

            # colors computed from the term of order 0 in the Spherical Harmonic basis
            # coefficient of the order 0th-term in the Spherical Harmonics basis
            pcd_colors_coeff = features_dc
        else:
            # 3D points
            pcd_points = self.pipeline.model.means

            # colors computed from the term of order 0 in the Spherical Harmonic basis
            # coefficient of the order 0th-term in the Spherical Harmonics basis
            pcd_colors_coeff = self.pipeline.model.features_dc
            
            # other attributes of the Gaussian
            opacities, scales, quats \
                = self.pipeline.model.opacities, self.pipeline.model.scales, self.pipeline.model.quats
                   
            try:
                clip_embeds =  self.pipeline.model.clip_embeds
            except AttributeError:
                # extract the semantic embeddings
                clip_embeds = self.pipeline.model.clip_field(self.pipeline.model.means).float().detach()

        # color computed from the Spherical Harmonics
        pcd_colors = SH2RGB(pcd_colors_coeff).squeeze()
        
        # mask points using a bounding box
        if use_bounding_box:
            mask = ((pcd_points[:, 0] > bounding_box_min[0]) & (pcd_points[:, 0] < bounding_box_max[0])
                    & (pcd_points[:, 1] > bounding_box_min[1]) & (pcd_points[:, 1] < bounding_box_max[1])
                    & (pcd_points[:, 2] > bounding_box_min[2]) & (pcd_points[:, 2] < bounding_box_max[2])
            )
            
            pcd_points = pcd_points[mask]
            pcd_colors = pcd_colors[mask]
            
            # other attributes of the Gaussian
            opacities, scales, quats, clip_embeds \
                = opacities[mask], scales, quats[mask], clip_embeds[mask]
            # opacities, scales, quats \
            #     = opacities[mask], scales, quats[mask]#, clip_embeds[mask]
        else:
            mask = None
            
        # apply transformation to the opacities and scales
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities)

        # enviromment attributes
        env_attr = {'means': pcd_points,
                    'quats': quats,
                    'scales': scales,
                    'opacities': opacities,
                    'clip_embeds': clip_embeds
                    }

        # create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points.double().cpu().detach().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors.double().cpu().detach().numpy())
        
        # reset the values of all parameters
        if cull_scene:
            self.pipeline.model.means = torch.nn.Parameter(means_prev)
            self.pipeline.model.scales = torch.nn.Parameter(scales_prev)
            self.pipeline.model.quats = torch.nn.Parameter(quats_prev)
            self.pipeline.model.features_dc = torch.nn.Parameter(features_dc_prev)
            self.pipeline.model.features_rest = torch.nn.Parameter(features_rest_prev)
            self.pipeline.model.opacities = torch.nn.Parameter(opacities_prev)
            
            try:
                self.pipeline.model.clip_embeds = torch.nn.Parameter(clip_embeds_prev)
            except AttributeError:
                pass

        return pcd, mask, env_attr
    
    def get_semantic_point_cloud(self,
                                 positives: str = "",
                                 negatives: str = "object, things, stuff, texture",
                                 pcd_attr: Dict[str, torch.Tensor] = {}):
        # update the list of positives (objects)
        self.pipeline.model.viewer_utils.handle_language_queries(raw_text=positives,
                                                                 is_positive=True)
        
        # update the list of positives (objects)
        self.pipeline.model.viewer_utils.handle_language_queries(raw_text=negatives,
                                                                 is_positive=False)
        
        # semantic point cloud
        # CLIP features
        if 'clip_embeds' in pcd_attr.keys():
            pcd_clip = pcd_attr['clip_embeds']
        else:
            try:
                pcd_clip = self.pipeline.model.clip_embeds
            except AttributeError:
                # extract the semantic embeddings
                pcd_clip = self.pipeline.model.clip_field(self.pipeline.model.means).float().detach()

        # get the semantic outputs
        pcd_clip = {'clip': pcd_clip}
        semantic_pcd = self.pipeline.model.get_semantic_outputs(pcd_clip)
        
        return semantic_pcd
    
def get_splat(map:str) -> GSplat:
    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main_dir_path = os.getcwd()
    nerf_dir_path = os.path.join(workspace_path,"gsplats")

    map_folder = os.path.join(nerf_dir_path,'outputs',map)
    for root, _, files in os.walk(map_folder):
        if 'config.yml' in files:
            nerf_cfg_path = os.path.join(root, 'config.yml')
            print(f"Found config.yml in {root}")

    # Go into NeRF data folder and get NeRF object
    os.chdir(nerf_dir_path)
    splat = GSplat(Path(nerf_cfg_path))
    os.chdir(main_dir_path)

    return splat

def pose2nerf_transform(pose):

    # Realsense to Drone Frame
    T_r2d = np.array([
        [ 0.99250, -0.00866,  0.12186,  0.10000],
        [ 0.00446,  0.99938,  0.03463, -0.03100],
        [-0.12209, -0.03383,  0.99194, -0.01200],
        [ 0.00000,  0.00000,  0.00000,  1.00000]
    ])
    
    # Drone to Flightroom Frame
    T_d2f = np.eye(4)
    T_d2f[0:3,:] = np.hstack((R.from_quat(pose[3:]).as_matrix(),pose[0:3].reshape(-1,1)))

    # Camera convention frame to realsense frame
    T_c2r = np.array([
        [ 0.0, 0.0,-1.0, 0.0],
        [ 1.0, 0.0, 0.0, 0.0],
        [ 0.0,-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0]
    ])

    # Get image transform
    T_c2n = T_d2f@T_r2d@T_c2r
    return T_c2n