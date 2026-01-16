# Importing the necessary libraries
import figs.render.capture_generation as pg
import figs.visualize.plot_trajectories as pt
import figs.visualize.generate_videos as gv
import figs.scene_editing.scene_editing_utils as scdt

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC

import os
os.environ["ACADOS_SOURCE_DIR"] = "/data/<username>/FiGS-Standalone/acados"
os.environ["LD_LIBRARY_PATH"] = os.getenv("LD_LIBRARY_PATH", "") + "/data/<username>/FiGS-Standalone/acados/lib"

import numpy as np

# print(os.getenv("ACADOS_SOURCE_DIR"))
# print(os.getenv("LD_LIBRARY_PATH"))

import ctypes
ctypes.CDLL("/data/<username>/FiGS-Standalone/acados/lib/libqpOASES_e.so")
ctypes.CDLL("/data/<username>/FiGS-Standalone/acados/lib/libblasfeo.so")
ctypes.CDLL("/data/<username>/FiGS-Standalone/acados/lib/libhpipm.so")
ctypes.CDLL("/data/<username>/FiGS-Standalone/acados/lib/libacados.so")

#%%
# FiGS Capture Examples (scene_name, capture_name)
capture_examples = [
    # 'backroom'
    # 'sv_1007_gemsplat'
    'packardpark'
]

# FiGS Simulate Examples (scene_name, rollout_name, frame_name, policy_name, course_name)
simulate_examples = [
    # ('flightroom', 'baseline', 'carl', 'vrmpc_fr', 'extended_traj_track'),
    # ('backroom',   'baseline', 'carl', 'vrmpc_fr', 'cluttered_env_track'),
    # ('mid_gate',   'baseline', 'carl', 'vrmpc_fr', 'robustness_track'),
    ('packardpark',   'baseline', 'carl', 'vrmpc_rrt', 'track_spiral')
    # ('sv_917_3_left_gemsplat', 'baseline', 'carl', 'vrmpc_rrt', 'inward_spiral'),
    # ('sv_1007_gemsplat', 'baseline', 'carl', 'vrmpc_fr', 'robustness_track'),
]

# query = 'ladder'

#%%

# Simulate within the FiGS environment
for scene, rollout, frame, policy, course in simulate_examples:
    print("=============================================================")
    print(f"Simulating {scene} scene with {course} course")
    print("-------------------------------------------------------------")

    # Load the policy and simulator
    sim = Simulator(scene,rollout,frame)
    ctl = VehicleRateMPC(course,policy,frame)

    # Use the ideal trajectory in VehicleRateMPC to get initial conditions and final time
    t0,tf,x0 = ctl.tXUd[0,0],ctl.tXUd[0,-1],ctl.tXUd[1:11,0]

    # Simulate the policy
    Tro,Xro,Uro,Imgs,_,_ = sim.simulate(ctl,t0,tf,x0)

    # Output the results
    gv.images_to_mp4(Imgs["rgb"],'test_space/'+course+'_'+scene+'.mp4', ctl.hz)       # Save the video
    # pt.plot_RO_spatial((Tro,Xro,Uro))                               # Plot the spatial trajectory

    # scdt.plot_point_cloud(sim,
    #                       (Tro,Xro,Uro),
    #                       50)

    # Clear the memory of the policy to avoid improper re-initialization of ACADOS
    del ctl