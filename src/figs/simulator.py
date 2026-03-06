import os
import shutil
import json
import yaml
import time
import statistics
import torch
import cv2
import numpy as np
import figs.utilities.trajectory_helper as th

from pathlib import Path
from typing import Type,Union,Tuple
from acados_template import AcadosSimSolver, AcadosSim
from figs.control.base_controller import BaseController
from figs.dynamics.model_equations import export_quadcopter_ode_model
from figs.dynamics.model_specifications import generate_specifications
from figs.render.gsplat_semantic import GSplat

# from gemsplat.gemsplat.clip import model
# from sousvide.flight import vision_preprocess_alternate as vp
# import sousvide.flight.vision_preprocess_groundedsam as vpg
# from sousvide.flight.vision_processor_base import VisionProcessorBase


class Simulator:
    """
    Class to simulation in FiGS
    """

    def __init__(self,
                 scene_name:str,rollout_name:str='baseline',
                 frame_name:Union[None,str]=None,
                 configs_path:Path=None,gsplats_path:Path=None) -> None:
        """
        The FiGS simulator simulates flying in a Gaussian Splat by using an ACADOS integrator
        (solver) to rollout a trajectory in a Gaussian Splat (gsplat) according to a control
        policy (policy) and simulation configuration (conFiG).

        For efficiency, the gsplat and conFiG are tied to individual Simulator objects. The
        solver and policy can be swapped out during runtime. This allows us to abstract away
        the JSON-based configuration and C backend of ACADOS. Note that every time the solver
        gets updated, the conFiG must also be uploaded with new drone specifications.

        Args:
            - scene_name:       Name of the scene to load.
            - rollout_name:     Rollout config to load.
            - frame_name:       Name of the frame to load (None if not instantiating with a frame).
            - configs_path:     Path to the directory containing the JSON files.
            - gsplats_path:     Path to the directory containing the gsplats.

        Attributes:
            - gsplat:           Gaussian Splat of the scene.
            - conFiG:           Dictionary holding simulation configurations (frequency, noise, delay and drone specs).
            - solver:           An ACADOS integrator for the drone dynamics.
            - policy:           Policy to control the drone (an ACADOS OCP based MPC by default).
            - configs_path:     Path to the configuration directory.
            - workspace_path:   Path to the gsplat directory.
        """

        # Set the configuration directory
        if configs_path is None:
            self.configs_path = Path(__file__).parent.parent.parent/'configs'
        else:
            self.configs_path = configs_path

        # Set the gsplat directory
        if gsplats_path is None:
            self.workspace_path = Path(__file__).parent.parent.parent/'3dgs'/'workspace'
        else:
            self.workspace_path = gsplats_path/'workspace'

        # Set the perception path
        self.perception_path = self.configs_path/"perception"/("perception_mode.yml")

        # Instantiate empty attributes
        self.gsplat = None
        self.conFiG = {"rollout":{},"drone":{},"perception":{}}
        self.solver = None

        # Load the attributes
        self.load_scene(scene_name)
        self.load_rollout(rollout_name)
        self.load_perception()

        if frame_name is not None:
            self.load_frame(frame_name)
    
    def load_perception(self):
        with open(self.perception_path, 'r') as file:
            perception_type = yaml.safe_load(file)
            visual_mode = perception_type.get("visual_mode")
            perception_type = perception_type.get("perception_type")

        if visual_mode not in ["rgb","semantic_depth"]:
            raise ValueError(f"Invalid visual mode: {visual_mode}")
        elif visual_mode == "semantic_depth":
            self.conFiG["perception"] = "semantic_depth"
            self.conFiG["perception_type"] = perception_type
            print(f"rendering simulation with {perception_type}.")
        else:
            self.conFiG["perception"] = "rgb"

    def load_scene(self, scene_name:str):
        """
        Loads/Updates the gsplat attribute given a scene name.

        Args:
            - scene_name:     Name of the scene to load.
        """

        # Get current and workspace directories
        curr_path,work_path = Path(os.getcwd()),self.workspace_path

        # Find the GSplat configuration
        search_path = work_path/'outputs'/scene_name
        yaml_configs = list(search_path.rglob("*.yml"))
    
        if len(yaml_configs) == 0:
            raise ValueError(f"The search path '{search_path}' did not return any configurations.")
        elif len(yaml_configs) > 1:
            raise ValueError(f"The search path '{search_path}' returned multiple configurations. Please specify a unique configuration within the directory.")
        else:
            gsplat_config = {"name":scene_name,"path":yaml_configs[0]}

        # Load GSplat (from the workspace directory to avoid path issues)
        os.chdir(work_path)
        gsplat = GSplat(gsplat_config)
        os.chdir(curr_path)

        # Update attribute(s)
        self.gsplat = gsplat

    def load_rollout(self, rollout:Union[str,dict]):
        """
        Loads/Updates the conFiG attribute given a rollout name.

        Args:
            - rollout:   Type of rollout to load.
        """

        # Check if rollout is a string or dictionary
        if isinstance(rollout,str):
            # Load the rollout config
            json_config = self.configs_path/"rollout"/(rollout+".json")

            if not json_config.exists():
                raise ValueError(f"The json file '{json_config}' does not exist.")
            else:
                # Load the json configuration
                with open(json_config) as file:
                    rollout_config = json.load(file)
        else:
            rollout_config = rollout

        # Update attribute(s)
        self.conFiG["rollout"] = rollout_config
        
    def load_frame(self, frame:Union[str,dict]):
        """
        Loads the solver attribute.

        Args:
            - frame_name:     Name of the frame to load.
        """
        
        # Check if rollout is a string or dictionary
        if isinstance(frame,str):
            # Load the frame config
            json_config = self.configs_path/"frame"/(frame+".json")

            if not json_config.exists():
                raise ValueError(f"The json file '{json_config}' does not exist.")
            else:
                # Load the json configuration
                with open(json_config) as file:
                    frame_config = json.load(file)
        else:
            frame_config = frame

        # Clear previous solver
        del self.solver
        
        # Some useful intermediate variables
        drn_spec = generate_specifications(frame_config)
        sim_json = 'figs_sim_solver.json'

        # Generate the simulator
        sim = AcadosSim()
        sim.model = export_quadcopter_ode_model(drn_spec["m"],drn_spec["tn"])  
        sim.solver_options.T = 1/self.conFiG["rollout"]["frequency"]
        sim.solver_options.integrator_type = 'IRK'

        solver = AcadosSimSolver(sim, json_file=sim_json, verbose=False)

        # Clean up the ACADOS generation files
        os.remove(os.path.join(os.getcwd(),sim_json))
        shutil.rmtree(sim.code_export_directory)
        
        # Update attribute(s)
        self.solver = solver
        self.conFiG["drone"] = drn_spec
    
    def simulate(self,policy:Type[BaseController],
                 t0:float,tf:int,x0:np.ndarray,obj:Union[None,np.ndarray]|None=None,
                 query:str|None=None,
                 vision_processor:Union[None,"VisionProcessorBase"]=None,
                 validation:bool=False,
                 verbose:bool=False
                 ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Simulates the flight.

#FIXME  Args:
            - t0:       Initial time.
            - tf:       Final time.
            - x0:       Initial state.
            - obj:      Objective to use for the simulation.
        """
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # Check if frame is loaded
        if self.solver is None:
            raise ValueError("Frame has not been loaded. Please load a frame before simulating.")

        # Unpack Variables
        hz_sim = self.conFiG["rollout"]["frequency"]
        t_dly = self.conFiG["rollout"]["delay"]
        mu_md_s = np.array(self.conFiG["rollout"]["model_noise"]["mean"])
        std_md_s = np.array(self.conFiG["rollout"]["model_noise"]["std"])
        mu_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["mean"])
        std_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["std"])
        use_fusion = self.conFiG["rollout"]["sensor_model_fusion"]["use_fusion"]
        Wf = np.diag(self.conFiG["rollout"]["sensor_model_fusion"]["weights"])
        nx,nu = self.conFiG["drone"]["nx"],self.conFiG["drone"]["nu"]
        cam_cfg = self.conFiG["drone"]["camera"]
        height,width,channels = cam_cfg["height"],cam_cfg["width"],cam_cfg["channels"]
        T_c2b = self.conFiG["drone"]["T_c2b"]

        perception = self.conFiG["perception"]
        perception_type = self.conFiG["perception_type"]

        # Derived Variables
        n_sim2ctl = int(hz_sim/policy.hz)  # Number of simulation steps per control step
        mu_md = mu_md_s*(1/n_sim2ctl)           # Scale model mean noise to control rate
        std_md = std_md_s*(1/n_sim2ctl)         # Scale model std noise to control rate
        dt = np.round(tf-t0)
        Nsim = int(dt*hz_sim)
        Nctl = int(dt*policy.hz)
        n_delay = int(t_dly*hz_sim)
        Wf_sn,Wf_md = Wf,1-Wf

        # Rollout Variables
        Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((nx,Nctl+1)),np.zeros((nu,Nctl))
        Iro_lists = {}  # channel_name -> list of frames (built dynamically)
        Xro[:,0] = x0

        # Diagnostics Variables
        Tsol = np.zeros((4,Nctl))
        Adv = np.zeros((nu,Nctl))
        
        # Transient Variables
        xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
        ucm = np.array([-self.conFiG["drone"]['m']/self.conFiG["drone"]['tn'],0.0,0.0,0.0])
        udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
        zcr = torch.zeros(policy.nzcr) if isinstance(policy.nzcr, int) else None

        # Instantiate camera object
        camera = self.gsplat.generate_output_camera(cam_cfg)

        if verbose:
            times = []

        # Rollout
        for i in range(Nsim):
            # Get current time and state
            tcr = t0+i/hz_sim

            # Control
            if i % n_sim2ctl == 0:
                # Get current image
                Tb2w = th.xv_to_T(xcr)
                T_c2w = Tb2w@T_c2b

                def _render_rgb(camera, T_c2w, query=None):
                    extra_ch = getattr(self.gsplat, "extra_channels", None)
                    if extra_ch is None and query is None:
                        return self.gsplat.render_rgb(camera, T_c2w)
                    elif extra_ch is None and query is not None:
                        return self.gsplat.render_rgb(camera, T_c2w, query)
                    elif extra_ch is not None and query is None:
                        return self.gsplat.render_rgb(camera, T_c2w, extra_channels=extra_ch)
                    elif extra_ch is not None and query is not None:
                        return self.gsplat.render_rgb(camera, T_c2w, query, extra_channels=extra_ch)

                if vision_processor is not None and perception == "semantic_depth" and perception_type == "clipseg" and query is not None:
                    image_dict = _render_rgb(camera, T_c2w)
                    icr_rgb = image_dict["rgb"]
                    icr_depth = image_dict["depth"]
                    start = time.time()
                    icr, _ = vision_processor.process(image=icr_rgb, prompt=query)
                    end = time.time()
                    if verbose:
                        times.append(end-start)
                elif perception == "semantic_depth" and perception_type != "clipseg" and query is not None:
                    image_dict = _render_rgb(camera, T_c2w, query)
                    icr = image_dict["semantic"]
                    icr_rgb = image_dict["rgb"]
                    icr_depth = image_dict["depth"]

                    if validation:
                        icr_val, _ = vision_processor.process(image=icr_rgb, prompt=query)
                else:
                    image_dict = _render_rgb(camera, T_c2w)
                    icr = image_dict["rgb"]

                # Add sensor noise and syncronize estimated state
                if use_fusion:
                    xsn += np.random.normal(loc=mu_sn,scale=std_sn)
                    xsn = Wf_sn@xsn + Wf_md@xcr
                else:
                    xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
                xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

                # Generate controller command
                ucm,zcr,adv,tsol = policy.control(tcr,xsn,ucm,obj,icr,zcr)

                # Update delay buffer
                udl[:,0] = udl[:,1]
                udl[:,1] = ucm

            # Extract delayed command
            uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

            # Simulate both estimated and actual states
            xcr = self.solver.simulate(x=xcr,u=uin)
            if use_fusion:
                xsn = self.solver.simulate(x=xsn,u=uin)

            # Add model noise
            xcr = xcr + np.random.normal(loc=mu_md,scale=std_md)
            xcr[6:10] = th.obedient_quaternion(xcr[6:10],xpr[6:10])

            # Update previous state
            xpr = xcr
            
            # Store values
            if i % n_sim2ctl == 0:
                k = i//n_sim2ctl

                # Store rendered channels dynamically
                if query is not None:
                    image_dict["semantic"] = icr  # may be from vision_processor
                    if validation and perception_type != "clipseg":
                        image_dict["validation"] = icr_val

                for ch_name, ch_img in image_dict.items():
                    if ch_name == "depth_raw":
                        continue  # skip raw float arrays
                    if ch_name not in Iro_lists:
                        Iro_lists[ch_name] = []
                    Iro_lists[ch_name].append(ch_img)

                Tro[k] = tcr
                Xro[:,k+1] = xcr
                Uro[:,k] = ucm
                Tsol[:,k] = tsol
                Adv[:,k] = adv

        if verbose:
            total_time = sum(times)
            print(f"Total inference time: {total_time:.2f} s")
            print(f"Average time/frame: {statistics.mean(times)*1000:.1f} ms")
            print(f"Median time/frame: {statistics.median(times)*1000:.1f} ms")
            print(f"Min time/frame: {min(times)*1000:.1f} ms")
            print(f"Max time/frame: {max(times)*1000:.1f} ms")

        # Stack collected frames into arrays
        Iro = {name: np.stack(frames) for name, frames in Iro_lists.items()}

        # Log final time
        Tro[Nctl] = t0+Nsim/hz_sim

        return Tro,Xro,Uro,Iro,Tsol,Adv
    

    def simulate_rollout_dino(self, model_a,
                            policy_a, policy_b,
                            videos_dir:Path,
                            t0:float,tf:int,x0:np.ndarray,obj:Union[None,np.ndarray]|None=None,
                            tXUi:np.ndarray|None=None,
                            query:str|None=None,
                            vision_processor:Union[None,"VisionProcessorBase"]=None,
                            loiter_spin:bool=False,
                            check_end:bool=True,
                            validation:bool=False,
                            verbose:bool=False
                            ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
            """
            Simulates the flight.

    #FIXME  Args:
                - t0:       Initial time.
                - tf:       Final time.
                - x0:       Initial state.
                - obj:      Objective to use for the simulation.
            """
            def log(*args, **kwargs):
                if verbose:
                    print(*args, **kwargs)

            # Check if frame is loaded
            if self.solver is None:
                raise ValueError("Frame has not been loaded. Please load a frame before simulating.")

            # Unpack Variables
            hz_sim = self.conFiG["rollout"]["frequency"]
            t_dly = self.conFiG["rollout"]["delay"]
            mu_md_s = np.array(self.conFiG["rollout"]["model_noise"]["mean"])
            std_md_s = np.array(self.conFiG["rollout"]["model_noise"]["std"])
            mu_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["mean"])
            std_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["std"])
            use_fusion = self.conFiG["rollout"]["sensor_model_fusion"]["use_fusion"]
            Wf = np.diag(self.conFiG["rollout"]["sensor_model_fusion"]["weights"])
            nx,nu = self.conFiG["drone"]["nx"],self.conFiG["drone"]["nu"]
            cam_cfg = self.conFiG["drone"]["camera"]
            height,width,channels = cam_cfg["height"],cam_cfg["width"],cam_cfg["channels"]
            T_c2b = self.conFiG["drone"]["T_c2b"]

            perception = self.conFiG["perception"]
            perception_type = self.conFiG["perception_type"]

            policy = policy_a

            # Derived Variables
            n_sim2ctl = int(hz_sim/policy.hz)       # Number of simulation steps per control step
            mu_md = mu_md_s*(1/n_sim2ctl)           # Scale model mean noise to control rate
            std_md = std_md_s*(1/n_sim2ctl)         # Scale model std noise to control rate
            # We will run the first 4s twice (i.e. add an extra 4s to the total runtime)
            extra_seconds = 4.0
            dt = np.round(tf - t0)
            dt_total = dt + extra_seconds
            Nsim = int(dt_total * hz_sim)
            Nctl = int(dt_total * policy.hz)
            n_delay = int(t_dly*hz_sim)
            Wf_sn,Wf_md = Wf,1-Wf

            # Rollout Variables
            Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((nx,Nctl+1)),np.zeros((nu,Nctl))
            Imgs_rgb = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            Imgs_sem = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            Imgs_depth = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            if validation:
                Imgs_val = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            # Iro = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            Xro[:,0] = x0

            # Diagnostics Variables
            Tsol = np.zeros((4,Nctl))
            Adv = np.zeros((nu,Nctl))
            
            # Transient Variables
            xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
            ucm = np.array([-self.conFiG["drone"]['m']/self.conFiG["drone"]['tn'],0.0,0.0,0.0])
            udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
            zcr = torch.zeros(policy.nzcr) if isinstance(policy.nzcr, int) else None

            # Instantiate camera object
            camera = self.gsplat.generate_output_camera(cam_cfg)

            if verbose:
                times = []

            try:
                # Allow callers to pass either str or Path-like
                videos_path = Path(videos_dir)
            except Exception:
                # Fall back to using as string path
                videos_path = Path(str(videos_dir))
            videos_path.mkdir(parents=True, exist_ok=True)

            frame_count = 0
            early_exit = False
            # Track whether we've applied the 4s "replay" of tXUi to the policy
            policy_switched = False
            # track the last wall-clock print time in simulation seconds
            # (use simulation time tcr - t0 for accuracy instead of frame count/hz)
            if not hasattr(self, '_last_video_time_print_sec'):
                # initialize to a negative value so first print occurs immediately
                self._last_video_time_print_sec = -1.0
            
            # Rollout
            for i in range(Nsim):
                # Get current time and state
                tcr = t0+i/hz_sim

                # Control
                if i % n_sim2ctl == 0:
                    # Get current image
                    Tb2w = th.xv_to_T(xcr)
                    T_c2w = Tb2w@T_c2b

                    if vision_processor is not None and perception == "semantic_depth" and perception_type == "clipseg" and query is not None:
                        image_dict = self.gsplat.render_rgb(camera,T_c2w)
                        # img_cr = icr["semantic"]
                        icr_rgb = image_dict["rgb"]
                        icr_depth = image_dict["depth"]
                        start = time.time()
                        icr, scaled = vision_processor.process(image=icr_rgb, prompt=query)
                        end = time.time()
                        if verbose:
                            times.append(end-start)
                    elif perception == "semantic_depth" and perception_type != "clipseg" and query is not None:
                        image_dict = self.gsplat.render_rgb(camera,T_c2w,query)
                        icr = image_dict["semantic"]
                        icr_rgb = image_dict["rgb"]
                        icr_depth = image_dict["depth"]

                        if validation:
                            icr_val, scaled = vision_processor.process(image=icr_rgb, prompt=query)
                    else:
                        image_dict = self.gsplat.render_rgb(camera,T_c2w)
                        icr = image_dict["rgb"]


                # Calculate elapsed time since start of simulation
                    video_time_elapsed = tcr - t0
                    
                    # ==== LOITER CALIBRATION LOGIC ====
                    # Handle loiter spin calibration if enabled
                    if loiter_spin and not policy_switched:
                        t10 = time.time()
                        
                        # Periodically log elapsed simulation time
                        print_interval_seconds = 1.0
                        if (video_time_elapsed - self._last_video_time_print_sec) >= print_interval_seconds:
                            if verbose:
                                print(f"video_time_elapsed={video_time_elapsed:.2f}s (frame {frame_count})")
                            self._last_video_time_print_sec = video_time_elapsed

                        # Phase 1: Calibration phase (first 'extra_seconds' seconds)
                        # Just collect data without activating control
                            overlay, scaled, present, _extras = model_a.grounded_sam_hf_inference(
                                icr_rgb,
                                query,
                                resize_output_to_input=True,
                                use_refinement=False,
                                use_smoothing=False,
                                scene_change_threshold=1.0,
                                verbose=False,
                            )
                            # log("Calibration phase:", found, sim_score, area_frac)
                            icr_rgb = overlay

                            if present:
                                policy=policy_b
                                policy_switched = True
                                cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                                camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)
                        
                        # Phase 2: Active detection phase 
                        # After calibration period but before switching to final behavior
                    #NOTE DEBUGGING STUFF HERE THIS SHOULD BE COMMENTED OUT
                        # elif video_time_elapsed < 2*extra_seconds - 2:
                        #     found, sim_score, area_frac, overlay = clipseg.loiter_calibrate(
                        #         logits=scaled,     
                        #         frame_img=icr_rgb,  
                        #         active_arm=True     # Now we can activate control
                        #     )
                        #     icr_rgb = overlay
                        #     policy = policy_b
                        #     policy_switched = True
                            
                        #     # Save the overlay image showing detection
                        #     cv2.imwrite(str(videos_path / "overlay_best.png"), overlay)

                        #     # Set up new camera configuration for the switched policy
                        #     cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                        #     camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)
                    #
                        elif video_time_elapsed < 2*extra_seconds + 2 and not policy_switched:
                            overlay, scaled, present, _extras = model_a.grounded_sam_hf_inference(
                                icr_rgb,
                                query,
                                resize_output_to_input=True,
                                use_refinement=False,
                                use_smoothing=False,
                                scene_change_threshold=1.0,
                                verbose=False,
                            )
                            # log("Calibration phase:", found, sim_score, area_frac)
                            icr_rgb = overlay

                            if present:
                                policy=policy_b
                                policy_switched = True
                                cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                                camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)
                            
                            # If target found and policy not yet switched
                            # if found and not policy_switched:
                            #     policy = policy_b
                            #     policy_switched = True
                                
                            #     # Save the overlay image showing detection
                            #     cv2.imwrite(str(videos_path / "overlay_best.png"), overlay)

                            #     # Get detailed information about the detected region
                            #     cnt, cur_area_px, cur_sol, cur_ecc = clipseg._largest_contour_from_mask(clipseg.loiter_mask)
                            #     d = clipseg._match_shape_distance(clipseg.loiter_cnt, cnt) if cnt is not None else float('inf')
                                
                            #     # Print detection stats for debugging
                            #     print(f"\nRegion Statistics:"
                            #         f"\n  largest_area={clipseg.loiter_area_frac*100:.1f}%"
                            #         f"\n  best scoring area={clipseg.loiter_max:.3f}"
                            #         f"\n  sim_score={sim_score:.3f}"
                            #         f"\n  area_frac={area_frac*100:.1f}%"
                            #         f"\n  sim_score_diff={clipseg.loiter_max - sim_score:.3f}"
                            #         f"\n  area_frac_diff={(clipseg.loiter_area_frac - area_frac)*100:.1f}%"
                            #         f"\nMatching Criteria:"
                            #         f"\n  shape_distance={d:.3f} (threshold={clipseg.shape_thresh:.3f})"
                            #         f"\n  solidity={cur_sol:.3f} (ref={clipseg.loiter_solidity:.3f} ±{clipseg.sol_tol*100:.1f}%)"
                            #         f"\n  eccentricity={cur_ecc:.3f} (ref={clipseg.loiter_eccentricity:.3f} ±{clipseg.ecc_tol*100:.1f}%)")
                                
                                # Set up new camera configuration for the switched policy
                                # cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                                # camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)
                                # log("Policy B Camera Resolution:", cam_cfg_1["width"], cam_cfg_1["height"])

                        # Track timing for performance monitoring
                        t11 = time.time()
                        times.append(t11 - t10)
                        frame_count += 1
                        
                        # Periodically report frame processing times
                        if frame_count % 50 == 0:
                            avg_ms = statistics.mean(times[-50:]) * 1e3
                            print(f"  Frame {frame_count}: avg {avg_ms:.1f} ms/frame")

                    # # Switch policy after 2x spin
                    # if loiter_spin and not policy_switched and (video_time_elapsed > (2*extra_seconds + 2)):
                    #     policy_switched = True
                    #     policy = policy_b
                    #     cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                    #     camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)

                    # ==== POLICY SWITCHING LOGIC (NON-LOITER MODE) ====
                    # Switch policy after 2x spin if not in loiter mode
                    if not loiter_spin and not policy_switched and (video_time_elapsed > (2*extra_seconds + 2)):
                        policy_switched = True
                        policy = policy_b
                        cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                        camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)

                    # ==== QUERY PROXIMITY CHECK ====
                    # Check if we're close enough to the query to exit
                    if policy_switched and check_end:
                        depth_raw = image_dict["depth_raw"]
                        exit_flag, decision_data = vp.check_depth_similarity_overlap(
                            depth_image=depth_raw,
                            similarity_image=scaled,
                        )
                        if exit_flag:
                            print("\033[95m📍 HOVERING!!!, Close to query ✨\033[0m")
                            early_exit = exit_flag
                            # early_exit_image = decision_data.get("depth_mask", None)
                            log("Query proximity exit condition met")
                    
                    # ==== COLLISION DETECTION ====
                    # Check if we're too close to obstacles
                    if policy_switched and check_end:
                        collision, collision_depth_mask = vp.close_enough_for_collision(image_dict["depth_raw"])
                        if collision:
                            print("\033[91m💥 COLLISION DETECTED!!! 💥\033[0m")
                            early_exit = True
                            decision_data = {"collision_depth_mask": collision_depth_mask}
                            log("Collision detected, triggering early exit")

                    # ==== SENSOR MODEL UPDATE ====
                    # Add sensor noise and syncronize estimated state
                    if use_fusion:
                        xsn += np.random.normal(loc=mu_sn,scale=std_sn)
                        xsn = Wf_sn@xsn + Wf_md@xcr
                    else:
                        xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
                    xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

                    # ==== EARLY EXIT HANDLING ====
                    # Handle early exit conditions (target found or collision)
                    if early_exit:
                        log("Early exit triggered, saving decision images")
                        # Save decision images for debugging
                        for key, image in decision_data.items():
                            cv2.imwrite(str(videos_path / f"{key}.png"), image)
                        # Break from main simulation loop
                        break

                    # ==== CONTROLLER COMMAND GENERATION ====
                    # Generate control command based on current policy
                    if policy_switched:
                        # If policy switched, use the new camera and process image
                        image_dict_b = self.gsplat.render_rgb(camera_1, T_c2w)
                        icr_rgb_b = image_dict_b["rgb"]
                        # Process image with vision processor for semantic information
                        icr_resized, _ = vision_processor.process(image=icr_rgb_b, prompt=query)
                        # Generate control command using the semantic image
                        ucm, zcr, adv, tsol = policy.control(tcr, xsn, ucm, obj, icr_resized, zcr)
                        # log("Using switched policy with semantic image")
                    else:
                        # Use the original policy and image
                        ucm, zcr, adv, tsol = policy.control(tcr, xsn, ucm, obj, icr, zcr)

                    # Update delay buffer for command smoothing
                    udl[:,0] = udl[:,1]
                    udl[:,1] = ucm

                # ==== SIMULATION STEP ====
                # Extract appropriate delayed command
                uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

                # Simulate both estimated and actual states
                xcr = self.solver.simulate(x=xcr, u=uin)
                if use_fusion:
                    xsn = self.solver.simulate(x=xsn, u=uin)

                # Add model noise to simulate real-world uncertainty
                xcr = xcr + np.random.normal(loc=mu_md, scale=std_md)
                xcr[6:10] = th.obedient_quaternion(xcr[6:10], xpr[6:10])

                # Update previous state
                xpr = xcr
                
                # ==== STORE SIMULATION DATA ====
                if i % n_sim2ctl == 0:
                    k = i//n_sim2ctl
                    
                    # Store images based on whether query was provided
                    if query is not None:
                        # Store semantic, RGB and depth images
                        Imgs_sem[k,:,:,:] = icr
                        Imgs_rgb[k,:,:,:] = icr_rgb
                        Imgs_depth[k,:,:,:] = icr_depth                
                        # Store validation image if needed
                        if validation and perception_type != "clipseg":
                            Imgs_val[k,:,:,:] = icr_val
                    else:
                        # Just store RGB image if no query provided
                        Imgs_rgb[k,:,:,:] = icr
                    
                    # Store timing and state information
                    Tro[k] = tcr
                    Xro[:,k+1] = xcr
                    Uro[:,k] = ucm
                    Tsol[:,k] = tsol
                    Adv[:,k] = adv

            # ==== PRINT TIMING STATISTICS ====
            if verbose and times:
                total_time = sum(times)
                print(f"Total inference time: {total_time:.2f} s")
                print(f"Average time/frame: {statistics.mean(times)*1000:.1f} ms")
                print(f"Median time/frame: {statistics.median(times)*1000:.1f} ms")
                print(f"Min time/frame: {min(times)*1000:.1f} ms")
                print(f"Max time/frame: {max(times)*1000:.1f} ms")

            # ==== PREPARE RETURN DATA ====
            # Package images based on what was collected
            if validation and perception_type != "clipseg" and query is not None:
                Iro = {"semantic": Imgs_sem, "depth": Imgs_depth, "rgb": Imgs_rgb, "validation": Imgs_val}
            elif query is not None:
                Iro = {"semantic": Imgs_sem, "depth": Imgs_depth, "rgb": Imgs_rgb}
            else:
                Iro = {"rgb": Imgs_rgb}

            # Log final time
            Tro[Nctl] = t0 + Nsim/hz_sim

            return Tro, Xro, Uro, Iro, Tsol, Adv
    

    def simulate_rollout_clipseg(self, policy_a, policy_b,
                            videos_dir:Path,
                            t0:float,tf:int,x0:np.ndarray,obj:Union[None,np.ndarray]|None=None,
                            tXUi:np.ndarray|None=None,
                            query:str|None=None,
                            vision_processor:Union[None,"VisionProcessorBase"]=None,
                            loiter_spin:bool=False,
                            check_end:bool=True,
                            validation:bool=False,
                            verbose:bool=False
                            ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
            """
            Simulates the flight.

    #FIXME  Args:
                - t0:       Initial time.
                - tf:       Final time.
                - x0:       Initial state.
                - obj:      Objective to use for the simulation.
            """
            def log(*args, **kwargs):
                if verbose:
                    print(*args, **kwargs)

            # Check if frame is loaded
            if self.solver is None:
                raise ValueError("Frame has not been loaded. Please load a frame before simulating.")

            # Unpack Variables
            hz_sim = self.conFiG["rollout"]["frequency"]
            t_dly = self.conFiG["rollout"]["delay"]
            mu_md_s = np.array(self.conFiG["rollout"]["model_noise"]["mean"])
            std_md_s = np.array(self.conFiG["rollout"]["model_noise"]["std"])
            mu_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["mean"])
            std_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["std"])
            use_fusion = self.conFiG["rollout"]["sensor_model_fusion"]["use_fusion"]
            Wf = np.diag(self.conFiG["rollout"]["sensor_model_fusion"]["weights"])
            nx,nu = self.conFiG["drone"]["nx"],self.conFiG["drone"]["nu"]
            cam_cfg = self.conFiG["drone"]["camera"]
            height,width,channels = cam_cfg["height"],cam_cfg["width"],cam_cfg["channels"]
            T_c2b = self.conFiG["drone"]["T_c2b"]

            perception = self.conFiG["perception"]
            perception_type = self.conFiG["perception_type"]

            policy = policy_a

            # Derived Variables
            n_sim2ctl = int(hz_sim/policy.hz)       # Number of simulation steps per control step
            mu_md = mu_md_s*(1/n_sim2ctl)           # Scale model mean noise to control rate
            std_md = std_md_s*(1/n_sim2ctl)         # Scale model std noise to control rate
            # We will run the first 4s twice (i.e. add an extra 4s to the total runtime)
            extra_seconds = 4.0
            dt = np.round(tf - t0)
            dt_total = dt + extra_seconds
            Nsim = int(dt_total * hz_sim)
            Nctl = int(dt_total * policy.hz)
            n_delay = int(t_dly*hz_sim)
            Wf_sn,Wf_md = Wf,1-Wf

            # Rollout Variables
            Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((nx,Nctl+1)),np.zeros((nu,Nctl))
            Imgs_rgb = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            Imgs_sem = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            Imgs_depth = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            if validation:
                Imgs_val = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            # Iro = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
            Xro[:,0] = x0

            # Diagnostics Variables
            Tsol = np.zeros((4,Nctl))
            Adv = np.zeros((nu,Nctl))
            
            # Transient Variables
            xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
            ucm = np.array([-self.conFiG["drone"]['m']/self.conFiG["drone"]['tn'],0.0,0.0,0.0])
            udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
            zcr = torch.zeros(policy.nzcr) if isinstance(policy.nzcr, int) else None

            # Instantiate camera object
            camera = self.gsplat.generate_output_camera(cam_cfg)

            if verbose:
                times = []

            try:
                # Allow callers to pass either str or Path-like
                videos_path = Path(videos_dir)
            except Exception:
                # Fall back to using as string path
                videos_path = Path(str(videos_dir))
            videos_path.mkdir(parents=True, exist_ok=True)

            frame_count = 0
            early_exit = False
            # Track whether we've applied the 4s "replay" of tXUi to the policy
            policy_switched = False
            # track the last wall-clock print time in simulation seconds
            # (use simulation time tcr - t0 for accuracy instead of frame count/hz)
            if not hasattr(self, '_last_video_time_print_sec'):
                # initialize to a negative value so first print occurs immediately
                self._last_video_time_print_sec = -1.0
            
            # Rollout
            for i in range(Nsim):
                # Get current time and state
                tcr = t0+i/hz_sim

                # Control
                if i % n_sim2ctl == 0:
                    # Get current image
                    Tb2w = th.xv_to_T(xcr)
                    T_c2w = Tb2w@T_c2b

                    if vision_processor is not None and perception == "semantic_depth" and perception_type == "clipseg" and query is not None:
                        image_dict = self.gsplat.render_rgb(camera,T_c2w)
                        # img_cr = icr["semantic"]
                        icr_rgb = image_dict["rgb"]
                        icr_depth = image_dict["depth"]
                        start = time.time()
                        icr, scaled = vision_processor.process(image=icr_rgb, prompt=query)
                        end = time.time()
                        if verbose:
                            times.append(end-start)
                    elif perception == "semantic_depth" and perception_type != "clipseg" and query is not None:
                        image_dict = self.gsplat.render_rgb(camera,T_c2w,query)
                        icr = image_dict["semantic"]
                        icr_rgb = image_dict["rgb"]
                        icr_depth = image_dict["depth"]

                        if validation:
                            icr_val, scaled = vision_processor.process(image=icr_rgb, prompt=query)
                    else:
                        image_dict = self.gsplat.render_rgb(camera,T_c2w)
                        icr = image_dict["rgb"]


                # Calculate elapsed time since start of simulation
                    video_time_elapsed = tcr - t0
                    
                    # ==== LOITER CALIBRATION LOGIC ====
                    # Handle loiter spin calibration if enabled
                    if loiter_spin:
                        t10 = time.time()
                        
                        # Periodically log elapsed simulation time
                        print_interval_seconds = 1.0
                        if (video_time_elapsed - self._last_video_time_print_sec) >= print_interval_seconds:
                            if verbose:
                                print(f"video_time_elapsed={video_time_elapsed:.2f}s (frame {frame_count})")
                            self._last_video_time_print_sec = video_time_elapsed

                        # Phase 1: Calibration phase (first 'extra_seconds' seconds)
                        # Just collect data without activating control
                        if video_time_elapsed < extra_seconds:
                            found, sim_score, area_frac, overlay = clipseg.loiter_calibrate(
                                logits=scaled,      # Similarity map from clipseg
                                frame_img=icr_rgb,  # Original RGB frame
                                active_arm=False    # Don't activate control yet
                            )
                            # log("Calibration phase:", found, sim_score, area_frac)
                            icr_rgb = overlay
                        
                        # Phase 2: Active detection phase 
                        # After calibration period but before switching to final behavior
                    #NOTE DEBUGGING STUFF HERE THIS SHOULD BE COMMENTED OUT
                        # elif video_time_elapsed < 2*extra_seconds - 2:
                        #     found, sim_score, area_frac, overlay = clipseg.loiter_calibrate(
                        #         logits=scaled,     
                        #         frame_img=icr_rgb,  
                        #         active_arm=True     # Now we can activate control
                        #     )
                        #     icr_rgb = overlay
                        #     policy = policy_b
                        #     policy_switched = True
                            
                        #     # Save the overlay image showing detection
                        #     cv2.imwrite(str(videos_path / "overlay_best.png"), overlay)

                        #     # Set up new camera configuration for the switched policy
                        #     cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                        #     camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)
                    #
                        elif video_time_elapsed < 2*extra_seconds + 2:
                            found, sim_score, area_frac, overlay = clipseg.loiter_calibrate(
                                logits=scaled,     
                                frame_img=icr_rgb,  
                                active_arm=True     # Now we can activate control
                            )
                            icr_rgb = overlay
                            
                            # If target found and policy not yet switched
                            if found and not policy_switched:
                                policy = policy_b
                                policy_switched = True
                                
                                # Save the overlay image showing detection
                                cv2.imwrite(str(videos_path / "overlay_best.png"), overlay)

                                # Get detailed information about the detected region
                                cnt, cur_area_px, cur_sol, cur_ecc = clipseg._largest_contour_from_mask(clipseg.loiter_mask)
                                d = clipseg._match_shape_distance(clipseg.loiter_cnt, cnt) if cnt is not None else float('inf')
                                
                                # Print detection stats for debugging
                                print(f"\nRegion Statistics:"
                                    f"\n  largest_area={clipseg.loiter_area_frac*100:.1f}%"
                                    f"\n  best scoring area={clipseg.loiter_max:.3f}"
                                    f"\n  sim_score={sim_score:.3f}"
                                    f"\n  area_frac={area_frac*100:.1f}%"
                                    f"\n  sim_score_diff={clipseg.loiter_max - sim_score:.3f}"
                                    f"\n  area_frac_diff={(clipseg.loiter_area_frac - area_frac)*100:.1f}%"
                                    f"\nMatching Criteria:"
                                    f"\n  shape_distance={d:.3f} (threshold={clipseg.shape_thresh:.3f})"
                                    f"\n  solidity={cur_sol:.3f} (ref={clipseg.loiter_solidity:.3f} ±{clipseg.sol_tol*100:.1f}%)"
                                    f"\n  eccentricity={cur_ecc:.3f} (ref={clipseg.loiter_eccentricity:.3f} ±{clipseg.ecc_tol*100:.1f}%)")
                                
                                # Set up new camera configuration for the switched policy
                                cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                                camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)
                                # log("Policy B Camera Resolution:", cam_cfg_1["width"], cam_cfg_1["height"])

                        # Track timing for performance monitoring
                        t11 = time.time()
                        times.append(t11 - t10)
                        frame_count += 1
                        
                        # Periodically report frame processing times
                        if frame_count % 50 == 0:
                            avg_ms = statistics.mean(times[-50:]) * 1e3
                            print(f"  Frame {frame_count}: avg {avg_ms:.1f} ms/frame")

                    # Switch policy after 2x spin
                    if loiter_spin and not policy_switched and (video_time_elapsed > (2*extra_seconds + 2)):
                        policy_switched = True
                        policy = policy_b
                        cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                        camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)

                    # ==== POLICY SWITCHING LOGIC (NON-LOITER MODE) ====
                    # Switch policy after 2x spin if not in loiter mode
                    if not loiter_spin and not policy_switched and (video_time_elapsed > (2*extra_seconds + 2)):
                        policy_switched = True
                        policy = policy_b
                        cam_cfg_1 = self.conFiG["drone"]["camera_1"]
                        camera_1 = self.gsplat.generate_output_camera(cam_cfg_1)

                    # ==== QUERY PROXIMITY CHECK ====
                    # Check if we're close enough to the query to exit
                    if policy_switched and check_end:
                        depth_raw = image_dict["depth_raw"]
                        exit_flag, decision_data = vp.check_depth_similarity_overlap(
                            depth_image=depth_raw,
                            similarity_image=scaled,
                        )
                        if exit_flag:
                            print("\033[95m📍 HOVERING!!!, Close to query ✨\033[0m")
                            early_exit = exit_flag
                            # early_exit_image = decision_data.get("depth_mask", None)
                            log("Query proximity exit condition met")
                    
                    # ==== COLLISION DETECTION ====
                    # Check if we're too close to obstacles
                    if policy_switched and check_end:
                        collision, collision_depth_mask = vp.close_enough_for_collision(image_dict["depth_raw"])
                        if collision:
                            print("\033[91m💥 COLLISION DETECTED!!! 💥\033[0m")
                            early_exit = True
                            decision_data = {"collision_depth_mask": collision_depth_mask}
                            log("Collision detected, triggering early exit")

                    # ==== SENSOR MODEL UPDATE ====
                    # Add sensor noise and syncronize estimated state
                    if use_fusion:
                        xsn += np.random.normal(loc=mu_sn,scale=std_sn)
                        xsn = Wf_sn@xsn + Wf_md@xcr
                    else:
                        xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
                    xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

                    # ==== EARLY EXIT HANDLING ====
                    # Handle early exit conditions (target found or collision)
                    if early_exit:
                        log("Early exit triggered, saving decision images")
                        # Save decision images for debugging
                        for key, image in decision_data.items():
                            cv2.imwrite(str(videos_path / f"{key}.png"), image)
                        # Break from main simulation loop
                        break

                    # ==== CONTROLLER COMMAND GENERATION ====
                    # Generate control command based on current policy
                    if policy_switched:
                        # If policy switched, use the new camera and process image
                        image_dict_b = self.gsplat.render_rgb(camera_1, T_c2w)
                        icr_rgb_b = image_dict_b["rgb"]
                        # Process image with vision processor for semantic information
                        icr_resized, _ = vision_processor.process(image=icr_rgb_b, prompt=query)
                        # Generate control command using the semantic image
                        ucm, zcr, adv, tsol = policy.control(tcr, xsn, ucm, obj, icr_resized, zcr)
                        # log("Using switched policy with semantic image")
                    else:
                        # Use the original policy and image
                        ucm, zcr, adv, tsol = policy.control(tcr, xsn, ucm, obj, icr, zcr)

                    # Update delay buffer for command smoothing
                    udl[:,0] = udl[:,1]
                    udl[:,1] = ucm

                # ==== SIMULATION STEP ====
                # Extract appropriate delayed command
                uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

                # Simulate both estimated and actual states
                xcr = self.solver.simulate(x=xcr, u=uin)
                if use_fusion:
                    xsn = self.solver.simulate(x=xsn, u=uin)

                # Add model noise to simulate real-world uncertainty
                xcr = xcr + np.random.normal(loc=mu_md, scale=std_md)
                xcr[6:10] = th.obedient_quaternion(xcr[6:10], xpr[6:10])

                # Update previous state
                xpr = xcr
                
                # ==== STORE SIMULATION DATA ====
                if i % n_sim2ctl == 0:
                    k = i//n_sim2ctl
                    
                    # Store images based on whether query was provided
                    if query is not None:
                        # Store semantic, RGB and depth images
                        Imgs_sem[k,:,:,:] = icr
                        Imgs_rgb[k,:,:,:] = icr_rgb
                        Imgs_depth[k,:,:,:] = icr_depth                
                        # Store validation image if needed
                        if validation and perception_type != "clipseg":
                            Imgs_val[k,:,:,:] = icr_val
                    else:
                        # Just store RGB image if no query provided
                        Imgs_rgb[k,:,:,:] = icr
                    
                    # Store timing and state information
                    Tro[k] = tcr
                    Xro[:,k+1] = xcr
                    Uro[:,k] = ucm
                    Tsol[:,k] = tsol
                    Adv[:,k] = adv

            # ==== PRINT TIMING STATISTICS ====
            if verbose and times:
                total_time = sum(times)
                print(f"Total inference time: {total_time:.2f} s")
                print(f"Average time/frame: {statistics.mean(times)*1000:.1f} ms")
                print(f"Median time/frame: {statistics.median(times)*1000:.1f} ms")
                print(f"Min time/frame: {min(times)*1000:.1f} ms")
                print(f"Max time/frame: {max(times)*1000:.1f} ms")

            # ==== PREPARE RETURN DATA ====
            # Package images based on what was collected
            if validation and perception_type != "clipseg" and query is not None:
                Iro = {"semantic": Imgs_sem, "depth": Imgs_depth, "rgb": Imgs_rgb, "validation": Imgs_val}
            elif query is not None:
                Iro = {"semantic": Imgs_sem, "depth": Imgs_depth, "rgb": Imgs_rgb}
            else:
                Iro = {"rgb": Imgs_rgb}

            # Log final time
            Tro[Nctl] = t0 + Nsim/hz_sim

            return Tro, Xro, Uro, Iro, Tsol, Adv
    
#     def simulate_rollout(self,policy_a:Type[BaseController],policy_b:Type[BaseController],
#                          videos_dir:Path,
#                          t0:float,tf:int,x0:np.ndarray,obj:Union[None,np.ndarray]|None=None,
#                          tXUi:np.ndarray|None=None,
#                          query:str|None=None,
#                          clipseg:bool=False,
#                          validation:bool=False,
#                          verbose:bool=False
#                          ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
#         """
#         Simulates the flight.

# #FIXME  Args:
#             - t0:       Initial time.
#             - tf:       Final time.
#             - x0:       Initial state.
#             - obj:      Objective to use for the simulation.
#         """
#         def log(*args, **kwargs):
#             if verbose:
#                 print(*args, **kwargs)

#         # Check if frame is loaded
#         if self.solver is None:
#             raise ValueError("Frame has not been loaded. Please load a frame before simulating.")

#         # Unpack Variables
#         hz_sim = self.conFiG["rollout"]["frequency"]
#         t_dly = self.conFiG["rollout"]["delay"]
#         mu_md_s = np.array(self.conFiG["rollout"]["model_noise"]["mean"])
#         std_md_s = np.array(self.conFiG["rollout"]["model_noise"]["std"])
#         mu_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["mean"])
#         std_sn = np.array(self.conFiG["rollout"]["sensor_noise"]["std"])
#         use_fusion = self.conFiG["rollout"]["sensor_model_fusion"]["use_fusion"]
#         Wf = np.diag(self.conFiG["rollout"]["sensor_model_fusion"]["weights"])
#         nx,nu = self.conFiG["drone"]["nx"],self.conFiG["drone"]["nu"]
#         cam_cfg = self.conFiG["drone"]["camera"]
#         height,width,channels = cam_cfg["height"],cam_cfg["width"],cam_cfg["channels"]
#         T_c2b = self.conFiG["drone"]["T_c2b"]

#         perception = self.conFiG["perception"]
#         perception_type = self.conFiG["perception_type"]

#         policy = policy_a

#         # Derived Variables
#         n_sim2ctl = int(hz_sim/policy.hz)       # Number of simulation steps per control step
#         mu_md = mu_md_s*(1/n_sim2ctl)           # Scale model mean noise to control rate
#         std_md = std_md_s*(1/n_sim2ctl)         # Scale model std noise to control rate
#         # We will run the first 4s twice (i.e. add an extra 4s to the total runtime)
#         extra_seconds = 4.0
#         dt = np.round(tf - t0)
#         dt_total = dt + extra_seconds
#         Nsim = int(dt_total * hz_sim)
#         Nctl = int(dt_total * policy.hz)
#         n_delay = int(t_dly*hz_sim)
#         Wf_sn,Wf_md = Wf,1-Wf

#         # Rollout Variables
#         Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((nx,Nctl+1)),np.zeros((nu,Nctl))
#         Imgs_rgb = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
#         Imgs_sem = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
#         Imgs_depth = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
#         if validation:
#             Imgs_val = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
#         # Iro = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
#         Xro[:,0] = x0

#         # Diagnostics Variables
#         Tsol = np.zeros((4,Nctl))
#         Adv = np.zeros((nu,Nctl))
        
#         # Transient Variables
#         xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
#         ucm = np.array([-self.conFiG["drone"]['m']/self.conFiG["drone"]['tn'],0.0,0.0,0.0])
#         udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
#         zcr = torch.zeros(policy.nzcr) if isinstance(policy.nzcr, int) else None

#         # Instantiate camera object
#         camera = self.gsplat.generate_output_camera(cam_cfg)

#         if verbose:
#             times = []

#         frame_count = 0
#         early_exit = False
#         # Track whether we've applied the 4s "replay" of tXUi to the policy
#         _tXUi_applied = False
#         # track the last wall-clock print time in simulation seconds
#         # (use simulation time tcr - t0 for accuracy instead of frame count/hz)
#         if not hasattr(self, '_last_video_time_print_sec'):
#             # initialize to a negative value so first print occurs immediately
#             self._last_video_time_print_sec = -1.0
        
#         # Rollout
#         for i in range(Nsim):
#             # Get current time and state
#             tcr = t0+i/hz_sim

#             # Control
#             if i % n_sim2ctl == 0:
#                 # Get current image
#                 Tb2w = th.xv_to_T(xcr)
#                 T_c2w = Tb2w@T_c2b

#                 if clipseg is not None and perception == "semantic_depth" and perception_type == "clipseg" and query is not None: 
#                     image_dict = self.gsplat.render_rgb(camera,T_c2w)
#                     # img_cr = icr["semantic"]
#                     icr_rgb = image_dict["rgb"]
#                     icr_depth = image_dict["depth"]
#                     start = time.time()
#                     icr, scaled = clipseg.clipseg_hf_inference(image=icr_rgb, prompt=query)
#                     end = time.time()
#                     if verbose:
#                         times.append(end-start)
#                 elif perception == "semantic_depth" and perception_type != "clipseg" and query is not None:
#                     image_dict = self.gsplat.render_rgb(camera,T_c2w,query)
#                     icr = image_dict["semantic"]
#                     icr_rgb = image_dict["rgb"]
#                     icr_depth = image_dict["depth"]

#                     if validation:
#                         icr_val, scaled = clipseg.clipseg_hf_inference(image=icr_rgb, prompt=query)
#                 else:
#                     image_dict = self.gsplat.render_rgb(camera,T_c2w)
#                     icr = image_dict["rgb"]
#                 # if perception == "semantic_depth" and query is not None:
#                 #     img_dict = self.gsplat.render_rgb(camera,T_c2w,query)
#                 #     icr = img_dict["semantic"]
#                 # else:
#                 #     icr = self.gsplat.render_rgb(camera,T_c2w)

#             #NOTE Here we're injecting the loiter_calibrate filter
#                 t10 = time.time()
#                 # Use precise simulation time elapsed (current sim time - start time)
#                 video_time_elapsed = tcr - t0

#                 # After the first 4s have elapsed, apply the provided tXUi to the policy
#                 # by shifting its time entries by the current sim time so the policy's
#                 # internal timing continues correctly when we "replay" the first 4s.
#                 # This is a one-time operation guarded by _tXUi_applied.
#                 # Only apply the provided tXUi after the first 4 seconds of sim time
#                 # if (not _tXUi_applied) and (tXUi is not None) and (video_time_elapsed >= extra_seconds):
#                 #     try:
#                 #         shifted = np.array(tXUi, copy=True)
#                 #         # If 1-D, assume it's a vector of times and shift all entries
#                 #         if shifted.ndim == 1:
#                 #             if verbose:
#                 #                 print(f"tXUi is 1-D, shape={shifted.shape}; shifting by tcr={tcr}")
#                 #             shifted = shifted + tcr
#                 #             policy.tXUd = shifted
#                 #         else:
#                 #             if verbose:
#                 #                 print(f"tXUi is multi-D, shape={np.array(tXUi).shape}")
#                 #             # For multi-d arrays assume first row is times (monotonic).
#                 #             times_row = np.array(tXUi[0, :])
#                 #             # Find the column within the first `extra_seconds` whose
#                 #             # state (rows 7:11) is closest to current estimated xsn[7:11].
#                 #             try:
#                 #                 # Candidate columns are those with original time <= extra_seconds
#                 #                 candidate_mask = times_row <= extra_seconds
#                 #                 candidate_idxs = np.where(candidate_mask)[0]
#                 #                 if verbose:
#                 #                     print(f"Found {candidate_idxs.size} candidate columns within first {extra_seconds}s: indices={candidate_idxs[:10]}")

#                 #                 if candidate_idxs.size > 0:
#                 #                     # extract the state rows for comparison; guard shape
#                 #                     try:
#                 #                         # Work with a full numpy copy to inspect shapes
#                 #                         traj_arr = np.array(tXUi)
#                 #                         n_rows = traj_arr.shape[0]
#                 #                         r0, r1 = 7, 11
#                 #                         # If either array is too short, fall back to time-based split
#                 #                         if n_rows <= r0 or xsn.size <= r0:
#                 #                             if verbose:
#                 #                                 print("tXUi or xsn too short for rows 7:11; falling back to time-based split")
#                 #                             closest = int(np.searchsorted(times_row, tcr))
#                 #                             closest = max(0, min(closest, times_row.size))
#                 #                         else:
#                 #                             # Clip r1 to available rows
#                 #                             r1_actual = min(r1, n_rows, xsn.size)
#                 #                             traj_states = traj_arr[r0:r1_actual, :]
#                 #                             x_slice = np.array(xsn[r0:r1_actual]).reshape(-1, 1)
#                 #                             # Only consider candidate columns
#                 #                             cand_states = traj_states[:, candidate_idxs]
#                 #                             # Ensure shapes align
#                 #                             if cand_states.shape[0] != x_slice.shape[0]:
#                 #                                 if verbose:
#                 #                                     print(f"Shape mismatch after slicing: cand_states.shape={cand_states.shape}, x_slice.shape={x_slice.shape}; falling back to time-based split")
#                 #                                 closest = int(np.searchsorted(times_row, tcr))
#                 #                                 closest = max(0, min(closest, times_row.size))
#                 #                             else:
#                 #                                 diffs = cand_states - x_slice
#                 #                                 dists = np.linalg.norm(diffs, axis=0)
#                 #                                 # choose earliest index among those with minimal distance
#                 #                                 dmin = float(np.min(dists))
#                 #                                 ties = np.where(np.isclose(dists, dmin))[0]
#                 #                                 rel_idx = int(ties[0])
#                 #                                 closest = int(candidate_idxs[rel_idx])
#                 #                                 if verbose:
#                 #                                     print(f"Chosen closest idx={closest} (distance={dists[rel_idx]:.4f}) using xsn[{r0}:{r1_actual}]={x_slice.ravel()}")
#                 #                     except Exception as e:
#                 #                         # If anything unexpected happens, fallback to safe default
#                 #                         if verbose:
#                 #                             print(f"Error selecting closest based on states: {e}; falling back to index 0")
#                 #                         closest = 0
#                 #                 else:
#                 #                     # No candidates in first 4s: fall back to first future column
#                 #                     closest = int(np.searchsorted(times_row, tcr))
#                 #                     closest = max(0, min(closest, times_row.size))
#                 #                     if verbose:
#                 #                         print(f"No candidates in first {extra_seconds}s; using time-based closest={closest}")
#                 #             except Exception as e:
#                 #                 # Fallback to safe default
#                 #                 if verbose:
#                 #                     print(f"Error selecting closest based on states: {e}; falling back to index 0")
#                 #                 closest = 0

#                 #             # Build current prefix (start -> closest) and continuing shifted trajectory
#                 #             try:
#                 #                 if verbose:
#                 #                     print(f"closest future column index = {closest}")
#                 #                 current_traj = np.array(tXUi[:, :closest], copy=True)
#                 #                 if verbose:
#                 #                     print(f"current_traj.shape = {current_traj.shape}")
#                 #                 shifted_full = np.array(tXUi, copy=True)
#                 #                 shifted_full[0, :] = shifted_full[0, :] + tcr
#                 #                 if verbose:
#                 #                     print(f"shifted_full first-row head = {shifted_full[0,:min(5, shifted_full.shape[1])]}")
#                 #                 continuing_traj = shifted_full
#                 #                 # Concatenate along columns (axis=1)
#                 #                 policy.tXUd = np.concatenate([current_traj, continuing_traj], axis=1)
#                 #                 if verbose:
#                 #                     print(f"Assigned concatenated tXUd with shape {policy.tXUd.shape}")
#                 #             except Exception as e:
#                 #                 # If shapes unexpected, fall back to assigning the shifted full trajectory
#                 #                 if verbose:
#                 #                     print(f"Failed to concat prefix + shifted_full: {e}; falling back to shifted full")
#                 #                 shifted = np.array(tXUi, copy=True)
#                 #                 shifted[0, ...] = shifted[0, ...] + tcr
#                 #                 policy.tXUd = shifted
#                 #                 if verbose:
#                 #                     print(f"Assigned fallback shifted tXUd with shape {policy.tXUd.shape}")

#                 #         _tXUi_applied = True
#                 #         if verbose:
#                 #             print("Applied shifted tXUi to policy at sim time", tcr)
#                 #     except Exception as e:
#                 #         print(f"Warning: could not apply tXUi to policy: {e}")

#                 # print video_time_elapsed at a slower rate (approx. once per second)
#                 # Use simulation-seconds based last-print timestamp so printing
#                 # cadence is correct even if frame_count/hz_sim rounding causes drift.
#                 print_interval_seconds = 1.0
#                 if (video_time_elapsed - self._last_video_time_print_sec) >= print_interval_seconds:
#                     if verbose:
#                         print(f"video_time_elapsed={video_time_elapsed:.2f}s (frame {frame_count})")
#                     self._last_video_time_print_sec = video_time_elapsed

#                 if video_time_elapsed < extra_seconds:
#                     found, sim_score, area_frac, overlay = clipseg.loiter_calibrate(
#                         logits=scaled,           # your logits/similarity map
#                         frame_img=icr_rgb,       # original frame in RGB
#                         active_arm=False
#                     )
#                     # found, sim_score, area_frac = model.loiter_calibrate(logits=scaled,
#                     #                                                     active_arm=False)
#                 else:
#                     found, sim_score, area_frac, overlay = clipseg.loiter_calibrate(
#                         logits=scaled,           # your logits/similarity map
#                         frame_img=icr_rgb,         # original frame in BGR
#                         active_arm=True
#                     )
#                     # found, sim_score, area_frac = model.loiter_calibrate(logits=scaled,
#                     #                                                     active_arm=True)
#                     if found:
#                         policy=policy_b

#                         # save best mask
#                         # Make sure videos_dir is a Path and exists
#                         try:
#                             # Allow callers to pass either str or Path-like
#                             videos_path = Path(videos_dir)
#                         except Exception:
#                             # Fall back to using as string path
#                             videos_path = Path(str(videos_dir))
#                         videos_path.mkdir(parents=True, exist_ok=True)

#                         overlay = clipseg._make_overlay(icr_rgb, clipseg.loiter_mask)
#                         cv2.imwrite(str(videos_path / "overlay_best.png"), overlay)
#                         # print stats
#                         print(f"largest_area={clipseg.loiter_area_frac*100:.1f}% "
#                                 f", best scoring area={clipseg.loiter_max:.3f} "
#                                 f", sim_score={sim_score:.3f} "
#                                 f", area_frac={area_frac*100:.1f}% "
#                                 f", sim_score_diff={clipseg.loiter_max - sim_score:.3f} "
#                                 f", area_frac_diff={(clipseg.loiter_area_frac - area_frac)*100:.1f}%")
#                     elif video_time_elapsed >= 2*extra_seconds + 2 and not found:
#                         # If calibration did not find a good loiter mask, set early_exit
#                         # so we break out of both the control and rollout loops and
#                         # let the function finish gracefully.
#                         early_exit = True
#                         # break out of the current control branch; the outer rollout
#                         # loop will check `early_exit` and break as well.
#                         # (We don't use `break` here because we're inside an if-block,
#                         #  so just continue to the normal loop flow and let the
#                         #  check after the control block handle breaking.)

#                 if video_time_elapsed < 2*extra_seconds + 2:
#                     icr = overlay
#                     icr_rgb = overlay
#                 t11 = time.time()

#                 times.append(t11 - t10)
#                 frame_count += 1
#                 if frame_count % 50 == 0:
#                     avg_ms = statistics.mean(times[-50:]) * 1e3
#                     print(f"  Frame {frame_count}: avg {avg_ms:.1f} ms/frame")

#                 # Add sensor noise and syncronize estimated state
#                 if use_fusion:
#                     xsn += np.random.normal(loc=mu_sn,scale=std_sn)
#                     xsn = Wf_sn@xsn + Wf_md@xcr
#                 else:
#                     xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
#                 xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

#                 # If early_exit was triggered during loiter calibration, break
#                 # out of the rollout loop so the function can finish gracefully.
#                 if early_exit:
#                     # set the rollout index so that storing logic uses current k
#                     # and then break from the main rollout loop
#                     # Note: we break out of the outermost `for i in range(Nsim)`
#                     break

#                 # Generate controller command
#                 ucm,zcr,adv,tsol = policy.control(tcr,xsn,ucm,obj,icr,zcr)

#                 # Update delay buffer
#                 udl[:,0] = udl[:,1]
#                 udl[:,1] = ucm

#             # Extract delayed command
#             uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

#             # Simulate both estimated and actual states
#             xcr = self.solver.simulate(x=xcr,u=uin)
#             if use_fusion:
#                 xsn = self.solver.simulate(x=xsn,u=uin)

#             # Add model noise
#             xcr = xcr + np.random.normal(loc=mu_md,scale=std_md)
#             xcr[6:10] = th.obedient_quaternion(xcr[6:10],xpr[6:10])

#             # Update previous state
#             xpr = xcr
            
#             # Store values
#             if i % n_sim2ctl == 0:
#                 k = i//n_sim2ctl

#                 if query is not None:
#                     # if isinstance(icr, tuple):
#                     #     for idx, item in enumerate(icr):
#                     #         print(f"icr[{idx}] type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
#                     Imgs_sem[k,:,:,:] = icr
#                     Imgs_rgb[k,:,:,:] = icr_rgb
#                     Imgs_depth[k,:,:,:] = icr_depth                
#                     if validation and perception_type != "clipseg":
#                         Imgs_val[k,:,:,:] = icr_val
#                 else:
#                     Imgs_rgb[k,:,:,:] = icr
#                 Tro[k] = tcr
#                 Xro[:,k+1] = xcr
#                 Uro[:,k] = ucm
#                 Tsol[:,k] = tsol
#                 Adv[:,k] = adv

#         if verbose:
#             total_time = sum(times)
#             print(f"Total inference time: {total_time:.2f} s")
#             print(f"Average time/frame: {statistics.mean(times)*1000:.1f} ms")
#             print(f"Median time/frame: {statistics.median(times)*1000:.1f} ms")
#             print(f"Min time/frame: {min(times)*1000:.1f} ms")
#             print(f"Max time/frame: {max(times)*1000:.1f} ms")

#         if validation and perception_type != "clipseg" and query is not None:
#             Iro = {"semantic":Imgs_sem,"depth":Imgs_depth,"rgb":Imgs_rgb,"validation":Imgs_val}
#         elif query is not None:
#             Iro = {"semantic":Imgs_sem,"depth":Imgs_depth,"rgb":Imgs_rgb}
#         else:
#             Iro = {"rgb":Imgs_rgb}

#         # Log final time
#         Tro[Nctl] = t0+Nsim/hz_sim

#         return Tro,Xro,Uro,Iro,Tsol,Adv
