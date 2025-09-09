from __future__ import annotations

# Standard library imports
import os
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from sklearn.neighbors import BallTree
from sklearn.cluster import HDBSCAN
import roma

# Local application imports
from figs.render.gsplat_semantic import GSplat
from figs.visualize.plot_trajectories import *
# from synthesize.scene_editing_utils import get_centroid

# # # # #
# # # # # Utils
# # # # #

def rescale_point_cloud(nerf,viz=False,cull=False,verbose=False,save_html=None):
    """
    Rescale point cloud from NeRF scene data.
    
    Parameters:
        nerf: NeRF object
        viz: Whether to visualize the point cloud
        cull: Whether to apply culling based on bounds
        verbose: Whether to print debug information
        save_html: Path to save the point cloud as HTML file (if provided)
    
    Returns:
        tuple: (epcds, env_pcd_scaled.T, epcds_bounds, env_pcd, env_pcd_mask, env_attr)
    """
    # viz = False
    # verbose = True

    # cull = True
    
    # Generate the point cloud of the environment
    env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True)
    # env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(
    #     use_bounding_box=True,bounding_box_max=(0.50,1.00,0.3),bounding_box_min=(-0.50,-1.00,0.0))

    cl, ind = env_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    filtered_pcd = env_pcd.select_by_index(ind)

    dataparser_scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    dataparser_transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

    # if viz:
    #     epcds = o3d.geometry.PointCloud()
    #     env_pcd_unscaled = np.asarray(filtered_pcd.points, dtype=np.float64).reshape(-1, 3)
    #     print(f"(1) Unscaled Point Cloud:")
    #     epcds.points = o3d.utility.Vector3dVector(env_pcd_unscaled)
    #     env_pcd_colors = np.asarray(filtered_pcd.colors, dtype=np.float64).reshape(-1, 3)
    #     epcds.colors = o3d.utility.Vector3dVector(env_pcd_colors)
    #     # o3d.visualization.draw_plotly([epcds])

    # Apply the inverse of the dataparser transform to the point cloud
    env_pcd_scaled = np.asarray(filtered_pcd.points).T / dataparser_scale
    env_pcd_scaled = np.vstack((env_pcd_scaled, np.ones((1, env_pcd_scaled.shape[1]))))

    # Handle deprecated approach
    if nerf.name.startswith("sv_"):
        print("Special handling for sv_ prefix")
        transform = torch.eye(4)
        transform[:3,:] = dataparser_transform
        invtransform = np.asarray(torch.linalg.inv(transform))

        # ns-train dataparser scale&transform
        # env_pcd_scaled = invtransform @ env_pcd_scaled
        env_pcd_scaled = np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ invtransform @ np.asarray(env_pcd_scaled)
    
    env_pcd_scaled = nerf.T_w2g @ env_pcd_scaled
    env_pcd_scaled = env_pcd_scaled[:3, :].T

    epcds = o3d.geometry.PointCloud()
    epcds.points=o3d.utility.Vector3dVector(env_pcd_scaled)
    epcds.colors=o3d.utility.Vector3dVector(np.asarray(filtered_pcd.colors))

    minbound = np.percentile(env_pcd_scaled,10, axis=0).tolist()
    maxbound = np.percentile(env_pcd_scaled,90, axis=0).tolist()
    if cull:
        # minbound = np.array([-6.5, -3.75, -2.5]).tolist()
        # maxbound = np.array([6.5, 3.75, 0.0]).tolist()
        cullbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=minbound, max_bound=maxbound)
        # translation_vector = np.array([0, -6, 0])
        # cullbox.translate(translation_vector)  # By default, relative=True
        epcds = epcds.crop(cullbox)

    epcds_aabb = epcds.get_axis_aligned_bounding_box()
    bx, by, bz = epcds_aabb.get_extent()

    if verbose:
        if not cull:
            print("Theoretical Bounding Box:")
        print(f"Bounding Range: {bx}, {by}, {bz}")
        print(f"Minbound: {minbound}", f"Maxbound: {maxbound}")

    # if viz:
    #     print(f"(2) Scaled Point Cloud:")
    #     o3d.visualization.draw_plotly([epcds])

    # Save to HTML if requested
    if save_html:
        print(f"Saving point cloud to HTML: {save_html}")
        
        # Use the exact same approach as visualize_multiple_trajectories
        # Get point cloud data
        pts = np.asarray(epcds.points)
        cols = np.clip(np.asarray(epcds.colors), 0, 1)
        rgb = (cols * 255).astype(int)
        rgb_strs = [f"rgb({r},{g},{b})" for r,g,b in rgb]

        # Create the main figure
        fig = go.Figure(layout=dict(width=1200, height=800))

        # Add point cloud with inverted Z axis for right-side-up display
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=-pts[:,1], z=-pts[:,2],
            mode="markers",
            marker=dict(size=2, color=rgb_strs),
            name="Environment",
            showlegend=False
        ))
        
        # Add title to the figure
        fig.update_layout(title=f'Rescaled Point Cloud - {getattr(nerf, "name", "Unknown Scene")}')
        
        # Save the figure as HTML
        fig.write_html(save_html)
        print(f"Point cloud saved to: {save_html}")

    epcds_bounds = {"minbound": minbound, "maxbound": maxbound}

    return epcds, env_pcd_scaled.T, epcds_bounds, env_pcd, env_pcd_mask, env_attr

def get_points(path: Path, positives: str, negatives: str, threshold: float, filter_radius: float, enable_visualization_pcd=False):

    # # # # #
    # # # # # Config Path
    # # # # #

    # rootdir = Path("/home/admin/StanfordMSL/")
    # SFTIpth = rootdir / "SFTI-Program"
    # outpth  = SFTIpth / "nerf_data/outputs/"
    # scnpth  = outpth / "sv_806_3_nerfstudio/gemsplat"
    modelpth = path

    # # mode
    gaussian_splatting = True

    if gaussian_splatting:
        # Gaussian Splatting
        config_path = Path(f"modelpth / config.yml")
    else:
        # Nerfacto
        config_path = Path(f"<Enter the path to your config file.>")

    # rescale factor
    res_factor = None

    # option to enable visualization of the environment point cloud
    # enable_visualization_pcd = True

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize NeRF
    nerf = NeRF(config_path=modelpth / "config.yml",
                test_mode="inference", #"inference", "eval"
                dataset_mode="test",
                device=device)

    # camera intrinsics
    H, W, K = nerf.get_camera_intrinsics()
    K = K.to(device)

    # poses in test dataset
    poses = nerf.get_poses()

    # images for evaluation
    eval_imgs = nerf.get_images()

    # generate the point cloud of the environment
    env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True)
    # env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True,bounding_box_max=(6.5,2.75,2.5),bounding_box_min=(-6.50,-2.75,0.0))

    if enable_visualization_pcd:
        # visualize point cloud
        o3d.visualization.draw_plotly([env_pcd]) 

    # list of positives
    # e.g., kitchen: ['babynurser bottle', 'red apple', 'kettle']

    # update list of negatives ['things', 'stuff', 'object', 'texture'] -> 'object, things, stuff, texture'

    # option to render the point cloud of the entire environment or from a camera
    camera_semantic_pcd = False

    if camera_semantic_pcd:
        # camera pose
        cam_pose = poses[9]
        
        # generate semantic RGB-D point cloud
        cam_rgb, cam_pcd_points, gem_pcd, depth_mask, outputs = nerf.generate_RGBD_point_cloud(pose=cam_pose,
                                                                                            save_image=True,
                                                                                            filename='figures/eval.png',
                                                                                            compute_semantics=True,
                                                                                            positives=positives,
                                                                                            negatives=negatives)
        
        
        # apply the depth mask to the semantic outputs
        semantic_info = outputs
        
        if depth_mask is not None:
            semantic_info['similarity'] = semantic_info['similarity'][depth_mask]
    else:   
        # get the semantic outputs
        semantic_info = nerf.get_semantic_point_cloud(positives=positives,
                                                    negatives=negatives,
                                                    pcd_attr=env_attr)
        
        # initial point cloud for semantic-conditioning
        gem_pcd = env_pcd

    # # #
    # # # Generating a Semantic-Conditioned Point Cloud
    # # # 

    # threshold for masking the point cloud
    threshold_mask = threshold

    # scaled similarity
    sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
    sc_sim = sc_sim / (sc_sim.max() + 1e-6)

    # mask
    similarity_mask = (sc_sim > threshold_mask).squeeze().reshape(-1,).cpu().numpy()

    # masked point cloud
    masked_pcd_pts = np.asarray(gem_pcd.points)[similarity_mask, ...]
    masked_pcd_color = np.asarray(gem_pcd.colors)[similarity_mask, ...]

    # # #
    # # # Visualizing  a Semantic-Conditioned Point Cloud
    # # # 

    # semantic-conditioned point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(masked_pcd_pts)
    pcd.colors = o3d.utility.Vector3dVector(masked_pcd_color)

    if enable_visualization_pcd:
        # visualize point cloud
        o3d.visualization.draw_plotly([pcd])

    # # # # #
    # # # # # Scene Editing from the Gaussian Means: Objects to Target
    # # # # #  

    # option to print debug info
    print_debug_info: bool = True

    # Set of all fruits
    # fruits = {'fruit', 'apple', 'orange', 'pear', 'tomato'}

    # objects to move
    objects: List[str] = [positives]


    # # # # # #
    # # # # # # Scene Editing from the Gaussian Means: Generate a Sample Trajectory and Task
    # # # # # #  

    # # objects: List[str] = ['saucepan', 'glass lid', 'knife', 'orange']
    # # object_to_target: Dict[str, str] = {'saucepan': targets[0],
    # #                                     'glass lid': targets[0],
    # #                                     'knife': targets[1],
    # #                                     'orange': targets[1] 
    # #                                     }

    # outputs for each object
    obj_outputs = {}

    # offset for each object
    offsets = np.zeros((len(objects), 3))

    # filter-size - radius
    # filter_radius = [0.075, 0.05, 0.05, 0.05]
    filter_radius = [filter_radius]
    # filter_radius = [0.25, 0.15, 0.1, 0.05]
    # TODO
    filter_radius.extend([filter_radius[-1]] * (len(objects) - len(filter_radius)))

    # similarity threshold - multiple objects
    # threshold_obj = [0.6, 0.9, 0.9, 0.97]
    threshold_obj = [threshold]
    # TODO
    threshold_obj.extend([threshold_obj[-1]] * (len(objects) - len(threshold_obj)))

    for idx, obj in enumerate(objects):
    # if idx < 2:
        # continue

        print('*' * 50)
        print(f'Processing Object: {obj}')
        print('*' * 50)
        
    # prior information on the object masks
    # obj_priors: Dict = {
    #     'mask_prior': table_attr['raw_similarity']
    # }

        # source location
        src_centroid, src_z_bounds, scene_pcd, similarity_mask, other_attr = get_centroid(nerf=nerf,
                                                                                        env_pcd=env_pcd,
                                                                                        pcd_attr=env_attr,
                                                                                        positives=objects[idx],
                                                                                        negatives='object, things, stuff, texture',
                                                                                        threshold=threshold_obj[idx],
                                                                                        visualize_pcd=False,
                                                                                        enable_convex_hull=True,
                                                                                        enable_spherical_filter=True,
                                                                                        enable_clustering=False,
                                                                                        filter_radius=filter_radius[idx],
                                                                                        obj_priors={},#obj_priors,
                                                                                        use_Mahalanobis_distance=True)

        # object
        object_pcd_points = np.asarray(scene_pcd.points)[similarity_mask]
        object_pcd_colors = np.asarray(scene_pcd.colors)[similarity_mask]
        object_pcd_sim = other_attr['raw_similarity'][similarity_mask].cpu().numpy()

        # if any(item in obj for item in ['pot', 'pan', 'lid']):
        # plane-fitting
        pcd_clus = o3d.geometry.PointCloud()
        pcd_clus.points = o3d.utility.Vector3dVector(object_pcd_points[:, :3])
        pcd_clus.colors = o3d.utility.Vector3dVector(object_pcd_colors)

        if enable_visualization_pcd:
            fig = o3d.visualization.draw_plotly([pcd_clus])

    # remove outliers
        # pcd, inlier_ind = pcd_clus.remove_radius_outlier(nb_points=30, radius=0.03) # r=0.03 maybe nb_points=5
        pcd, inlier_ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
            
        if enable_visualization_pcd:
            print('*' * 50)
            print('Post-Outlier-Removal')
            fig = o3d.visualization.draw_plotly([pcd])
            # fig.show()
            print('*' * 50)
    
    return nerf, {"env_pcd": env_pcd, "pcd": pcd, "inlier_ind": inlier_ind, "pcd_clus": pcd_clus, 
                  "src_centroid": src_centroid, "src_z_bounds": src_z_bounds, "scene_pcd": scene_pcd, 
                  "similarity_mask": similarity_mask, "other_attr": other_attr, "object_pcd_points": object_pcd_points, 
                  "object_pcd_colors": object_pcd_colors, "object_pcd_sim": object_pcd_sim}

def in_convex_hull(points, convex_hull):
    return Delaunay(convex_hull).find_simplex(points) >= 0

def spherical_filter(source_points, target_points, radius=0.1,
                     use_Mahalanobis_distance: bool = True):
    if use_Mahalanobis_distance:
        # using a BallTree
        ball_tree = BallTree(source_points,
                        metric="mahalanobis",
                                V=np.diag(np.concatenate(
                                        (np.repeat([1], 3), 
                                        np.repeat([50], 3),
                                        [300]
                                        )
                                    ))
                                )
        
        # find the points within a sphere at each target point
        groups = ball_tree.query_radius(target_points, radius)
        
        # indices
        inds = np.unique(list(itertools.chain.from_iterable(groups)))
        
        return inds
    else:
        # using a KDTree
        kd_tree = cKDTree(source_points[:, :3])
        
        # find the points within a sphere at each target point
        groups = kd_tree.query_ball_point(target_points[:, :3], radius)
        
        # indices
        inds = np.unique(list(itertools.chain.from_iterable(groups)))
        
        return inds
    
def get_centroid(nerf: NeRF,
                 env_pcd,
                 pcd_attr: Dict,
                 positives: str,
                 negatives: str = "floor,walls,object,things,stuff,texture",
                 threshold: float = 0.85,
                 visualize_pcd: bool = True,
                 enable_convex_hull: bool = False,
                 enable_spherical_filter: bool = False,
                 enable_clustering: bool = False,
                 print_debug_info: bool = False,
                 filter_radius: float = 0.05,
                 obj_priors: Dict = {},
                 use_Mahalanobis_distance: bool = True):
    
    # get the semantic outputs
    semantic_info = nerf.get_semantic_point_cloud(positives=positives, negatives=negatives,
                                                  pcd_attr=pcd_attr)
    
    # initial point cloud for the scene
    scene_pcd = env_pcd
         
    # points in the point cloud
    scene_pcd_pts = np.asarray(scene_pcd.points)
    scene_pcd_colors = np.asarray(scene_pcd.colors)
            
    # threshold for masking the point cloud
    threshold_mask = threshold

    # scaled similarity
    sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
    sc_sim = sc_sim / (sc_sim.max() + 1e-6)

    # depth mask
    depth_mask = None
    
    if len(scene_pcd.points) != semantic_info["similarity"].shape[0]:
        raise ValueError('The Cosine similarity should be computed for all points in the point cloud!')

    similarity_mask = (sc_sim > threshold_mask).squeeze().reshape(-1,).cpu().numpy()

    # masked point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(scene_pcd.points)[similarity_mask, ...])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(scene_pcd.colors)[similarity_mask, ...])

    if visualize_pcd:
        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()
    
    # # # # #
    # # # # # Outlier Removal
    # # # # #

    # threshold based on the standard deviation of average distances
    std_ratio = 0.01

    # remove outliers
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)

    # pcd, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.05)

    if visualize_pcd:
        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()
        
    # Point Cloud
    pcd_pts = np.asarray(pcd.points)
        
    # update the similarity mask
    similarity_mask_subset = np.zeros_like(similarity_mask[similarity_mask == True], dtype=bool)
    similarity_mask_subset[ind] = True
    
    similarity_mask_out = similarity_mask
    similarity_mask_out[similarity_mask == True] = similarity_mask_subset
    
    if enable_spherical_filter:
        # apply a spherical filter
        rel_inds = spherical_filter(source_points=np.concatenate((scene_pcd_pts,
                                                                  scene_pcd_colors,
                                                                  sc_sim.cpu().numpy()),
                                                                  axis=-1),
                                    target_points=np.concatenate((scene_pcd_pts[similarity_mask_out],
                                                                  scene_pcd_colors[similarity_mask_out],
                                                                  sc_sim.cpu().numpy()[similarity_mask_out]),
                                                                  axis=-1),
                                    radius=filter_radius,
                                    use_Mahalanobis_distance=use_Mahalanobis_distance)
        rel_inds = np.array(list(rel_inds)).astype(int)
        
        # update the mask
        similarity_mask_out[rel_inds] = True
        
        if print_debug_info:
            print(f'Spherical Filter Before : {len(pcd_pts)}, After: {len(rel_inds)}')
    
    # update the point cloud
    pcd_pts = scene_pcd_pts[similarity_mask_out]
    
    if enable_convex_hull:
        # compute the convex hull
        convex_hull = ConvexHull(pcd_pts)
        
        # examine the convex hull
        convex_hull_mask = in_convex_hull(scene_pcd_pts, pcd_pts[convex_hull.vertices])
        
        if print_debug_info:
            print(f'Convex Hull Proc. Before : {len(pcd_pts)}, After: {len(convex_hull_mask.nonzero()[0])}')
    else:
        convex_hull_mask = np.zeros(len(scene_pcd_pts), dtype=bool)
        
    # update the similarity mask
    similarity_mask_out = np.logical_or(similarity_mask_out, convex_hull_mask)
    
    # TODO: Remove
    if visualize_pcd:
        # point cloud
        pcd_pts = scene_pcd_pts[similarity_mask_out]
        color_masked = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]
        
        if visualize_pcd:
            print(len(pcd_pts))
        
        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors = o3d.utility.Vector3dVector(color_masked)

        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()
        
    # apply prior information
    if obj_priors != {}:
        if 'mask_prior' in obj_priors.keys():
            # compute the softmax
            probs = torch.nn.functional.softmax(torch.cat((obj_priors['mask_prior'], 
                                                          semantic_info["similarity"]),
                                                          dim=-1),
                                                dim=-1)
            
            # maximizer
            pb_argmax = torch.argmax(probs, dim=-1, keepdim=True)
            
            # update the similarity mask
            similarity_mask_out = np.logical_and(similarity_mask_out, (pb_argmax == 1).squeeze().cpu().numpy())
        
    # TODO: Remove
    if visualize_pcd:
        # point cloud
        pcd_pts = scene_pcd_pts[similarity_mask_out]
        color_masked = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]
        
        if visualize_pcd:
            print(len(pcd_pts))
        
        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors = o3d.utility.Vector3dVector(color_masked)

        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()
    
    # Clustering
    if enable_clustering:
        # clustering
        hdb_scan = HDBSCAN(min_cluster_size=5, # default,
                            #  max_cluster_size=4,
                            metric='mahalanobis',
                            metric_params={
                                'V': np.diag(np.concatenate(
                                    (np.repeat([1], 3), 
                                        np.repeat([5], 3),
                                        [10]
                                        )
                                ))
                            }
                            )

        # fit the data
        hdb_scan.fit(np.concatenate((scene_pcd_pts[similarity_mask_out],
                                     np.asarray(scene_pcd.colors)[similarity_mask_out, ...],
                                     sc_sim[similarity_mask_out, ...].cpu().numpy()),
                                     axis=-1)
        )
        
        # all labels
        point_to_labels = hdb_scan.labels_.astype(int)
        unique_labels = set(hdb_scan.labels_)
        
        if print_debug_info:
            print(f'Unique Labels: {unique_labels}')
            
        # assigned colors
        colors = np.array([plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))])

        # visualize the clusters
        vis_pcd_pts = scene_pcd_pts[similarity_mask_out]
        vis_colors = colors[point_to_labels][:, :3]

        if visualize_pcd:
            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)

            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()
            
        # Select the cluster containing the point with the maximum similarity measure.
        
        # index of the maximizer of the similarity metric
        max_sim_idx = torch.argmax(sc_sim[similarity_mask_out])

        # Get the cluster containing the most similar point
        sel_cluster = point_to_labels[max_sim_idx]

        if visualize_pcd:
            vis_pcd_pts = scene_pcd_pts[similarity_mask_out][point_to_labels == sel_cluster]
            vis_colors = color_masked[point_to_labels == sel_cluster]
        
            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)

            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()
        
        # mask from the clustering procedure
        clustering_mask = (point_to_labels == sel_cluster)
        
        # update the similarity mask
        similarity_mask_out[similarity_mask_out == True] = np.logical_and(similarity_mask_out[similarity_mask_out == True],
                                                                          clustering_mask)

        # points in the point cloud
        pts_cond = np.asarray(pcd.points)
        
        if enable_convex_hull:
            # compute the convex hull
            convex_hull = ConvexHull(pts_cond)
            
            # examine the convex hull
            convex_hull_mask = in_convex_hull(scene_pcd_pts, pts_cond[convex_hull.vertices])
            
            if print_debug_info:
                print(f'Convex Hull Proc. Before : {len(pts_cond)}, After: {len(convex_hull_mask.nonzero()[0])}')

        # update the similarity mask
        similarity_mask_out = np.logical_or(similarity_mask_out, convex_hull_mask)

        if visualize_pcd:
            vis_pcd_pts = scene_pcd_pts[similarity_mask_out]
            vis_colors = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]

            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)
            
            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()
    
    # compute the desired centroid
    centroid = np.mean(pcd_pts, axis=0)

    if visualize_pcd:
        # point cloud
        pcd_pts = scene_pcd_pts[similarity_mask_out]
        color_masked = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]
        
        if print_debug_info:
            print(len(pcd_pts))
        
        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors = o3d.utility.Vector3dVector(color_masked)

        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()
    
    # compute the bounds in the z-direction
    z_bounds = (np.amin(pcd_pts[:, -1]), np.amax(pcd_pts[:, -1]))
    
    # other attributes
    obj_attributes = {
        'convex_hull': convex_hull_mask if enable_convex_hull else [],
        'spherical_filter': rel_inds if enable_spherical_filter else [],
        'clustering_mask': clustering_mask if enable_clustering else [],
        'raw_similarity': semantic_info["raw_similarity"]
    }
    
    return centroid, z_bounds, scene_pcd, similarity_mask_out, obj_attributes

def get_interpolated_gaussians(means_a: NDArray, means_b: NDArray,
                               quats_a: NDArray, quats_b: NDArray,
                               steps: int = 10) -> List[float]:
    """Return interpolation of poses with specified number of steps.
    Args:
        means_a: initial means
        means_b: final means
        quats_a: initial quaternions
        quats_b: final quaternions
        steps: number of steps the interpolated path should contain
    """
    
    # device 
    device = means_a.device
    
    # normalize the quaternions
    quats_a = torch.nn.functional.normalize(quats_a, dim=-1)
    quats_b = torch.nn.functional.normalize(quats_b, dim=-1)
    
    # t_steps    
    ts_a = torch.linspace(0, 1, steps // 2).to(device)[:, None, None]
    
    ts_b = torch.linspace(0, 1, steps - len(ts_a)).to(device)[:, None, None]
    
    # mean
    center = 0.5 * (means_a + means_b)
    center[:, -1] += 0.2

    # interpolated means 
    interpolated_means_a = (1 - ts_a) * means_a[None] + ts_a * center[None]

    # interpolated means 
    interpolated_means_b = (1 - ts_b) * center[None] + ts_b * means_b[None]
    
    interpolated_means = torch.cat((interpolated_means_a, interpolated_means_b),
                                    dim=0)
        
    # interpolate quaternions
    interpolated_quats = roma.utils.unitquat_slerp(quats_a, quats_b, torch.linspace(0, 1, steps).to(device))
        
    return interpolated_means, interpolated_quats

def plot_point_cloud(simulator,
                     ROs: Optional[Union[
                            Tuple[np.ndarray, np.ndarray, np.ndarray],
                            List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = None,
                     n_fr: Optional[int] = None):
    def load_config_file(base_path, subfolder, filename):
        config_path = os.path.join(base_path, subfolder)
        for root, _, files in os.walk(config_path):
            if filename in files:
                with open(os.path.join(root, filename), 'r') as file:
                    return yaml.safe_load(file), os.path.join(root, filename)
        raise FileNotFoundError(f"{filename} not found in {config_path}")
    
    def _get_frame_segments(x: np.ndarray, scale: float) -> List[Tuple[np.ndarray,np.ndarray,str]]:
        """
        Given a 14-element state vector x (pos in x[0:3], quat in x[6:10]) 
        and a scale, return (start, end, color) for each of the 3 body-axes.
        """
        dims = np.diag([0.6, 0.6, -0.2])   # same dims you used
        frame_body = scale * dims
        pos  = x[0:3]
        quat = x[6:10]
        R_mat = R.from_quat(quat).as_matrix()
        colors = ["red","green","blue"]
        segments = []
        for j, col in enumerate(colors):
            arm = R_mat @ frame_body[j,:]
            if j == 2:
                start = pos.copy()
            else:
                start = pos - arm
            end = pos + arm
            segments.append((start, end, col))
        return segments

    # Load scene configuration
    config,cfg_path = load_config_file(
        simulator.configs_path, 'course', f'{simulator.gsplat.name}.yml')

    epcds, epcds_arr, epcds_bounds, pcd, pcd_mask, pcd_attr = rescale_point_cloud(simulator.gsplat)

    pts  = np.asarray(epcds.points)
    cols = np.clip(np.asarray(epcds.colors), 0, 1)
    rgb  = (cols * 255).astype(int)
    rgb_strs = [f"rgb({r},{g},{b})" for r,g,b in rgb]

    # 2) Build the Figure with just the points
    fig = go.Figure(layout=dict(width=1500, height=1500))
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="markers",
        marker=dict(size=2, color=rgb_strs),
        showlegend=False
    ))

    # # 3) Now add each cylinder mesh
    # for cyl in cylinders:
    #     verts = np.asarray(cyl.vertices)
    #     tris  = np.asarray(cyl.triangles)
    #     # pick the uniform color you used in Open3D: [0, 0.5, 1.0] → rgb(0,128,255)
    #     cyl_color = 'rgb(0,128,255)'
    #     fig.add_trace(go.Mesh3d(
    #         x=verts[:,0], y=verts[:,1], z=verts[:,2],
    #         i=tris[:,0], j=tris[:,1], k=tris[:,2],
    #         opacity=1.0,
    #         color=cyl_color,
    #         showlegend=False
    #     ))

    # # once you have your point cloud `pts` (N×3 array):
    min_pt, max_pt = pts.min(axis=0), pts.max(axis=0)
    center = (min_pt + max_pt) * 0.5

    # 1) axis limits
    # Get axis limits from config file if available
    minbound = [float(x) if x not in [None, 'None'] else None for x in config.get('minbound', [None, None, None])]
    maxbound = [float(x) if x not in [None, 'None'] else None for x in config.get('maxbound', [None, None, None])]
    bounds = np.array([
        [minbound[i] if minbound[i] is not None else pts[:, i].min(),
            maxbound[i] if maxbound[i] is not None else pts[:, i].max()]
        for i in range(3)
    ])
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]
    # zmin,zmax = -5,0
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    # 2) camera (as before)
    cam_R    = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max() * 1.5
    az, el = np.deg2rad(45), np.deg2rad(10)
    eye = dict(
    x=cam_R*np.cos(el)*np.cos(az),
    y=cam_R*np.cos(el)*np.sin(az),
    z=cam_R*np.sin(el)
    )

    if ROs is not None:
        if isinstance(ROs, tuple):
            ROs = [ROs]
        traj_colors = ["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]
        axis_len = np.linalg.norm([dx,dy,dz]) * 0.03

        for i, RO in enumerate(ROs):
            tXU = th.RO_to_tXU(RO)
            x, y, z = tXU[1,:], tXU[2,:], tXU[3,:]
            # spline path
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=traj_colors[i%len(traj_colors)], width=4),
                name=f"traj_{i}"
            ))

            # pick frames: start, end, and every n_fr
            frame_idxs = {0, tXU.shape[1]-1}
            if n_fr:
                frame_idxs |= set(range(n_fr, tXU.shape[1], n_fr))

            for idx_fr in sorted(frame_idxs):
                xvec = tXU[1:14, idx_fr]           # same shape your quad_frame expects
                for start, end, col in _get_frame_segments(xvec, axis_len):
                    fig.add_trace(go.Scatter3d(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        mode="lines",
                        line=dict(color=col, width=2),
                        showlegend=False
                    ))

    # 3) update layout
    if simulator.gsplat.name.startswith("sv_"):
        fig.update_layout(
        scene_camera=dict(eye=eye,
                up=dict(x=0, y=0, z=1)),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=dx, y=dy, z=dz),
            xaxis=dict(title='x [m]',
                    range=[xmin, xmax], autorange=False),
            yaxis=dict(title='y [m]',
                    range=[ymax, ymin], autorange=False),
            zaxis=dict(title='z [m]',
                    range=[zmax, zmin], autorange=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=1500, height=1500,
        showlegend=False
        )
    else:
        fig.update_layout(
        scene_camera=dict(eye=eye,
                up=dict(x=0, y=0, z=1)),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=dx, y=dy, z=dz),
            xaxis=dict(title='',
                    range=[xmin, xmax], autorange=False),
            yaxis=dict(title='',
                    range=[ymax, ymin], autorange=False),
            zaxis=dict(title='',
                    range=[zmax, zmin], autorange=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=1500, height=1500,
        showlegend=False
        )
    fig.show()