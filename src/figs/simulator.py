import os
import shutil
import json
import yaml
import time
import statistics
import torch
import numpy as np
import figs.utilities.trajectory_helper as th

from pathlib import Path
from typing import Type,Union,Tuple
from acados_template import AcadosSimSolver, AcadosSim
from figs.control.base_controller import BaseController
from figs.dynamics.model_equations import export_quadcopter_ode_model
from figs.dynamics.model_specifications import generate_specifications
from figs.render.gsplat_semantic import GSplat

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
            self.configs_path = Path(__file__).parent.parent.parent.parent/'configs'
        else:
            self.configs_path = configs_path

        # Set the gsplat directory
        if gsplats_path is None:
            self.workspace_path = Path(__file__).parent.parent.parent.parent/'gsplats'/'workspace'
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
                 clipseg:bool=False,
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

        # Rollout
        for i in range(Nsim):
            # Get current time and state
            tcr = t0+i/hz_sim

            # Control
            if i % n_sim2ctl == 0:
                # Get current image
                Tb2w = th.xv_to_T(xcr)
                T_c2w = Tb2w@T_c2b

                if clipseg is not None and perception == "semantic_depth" and perception_type == "clipseg" and query is not None: 
                    image_dict = self.gsplat.render_rgb(camera,T_c2w)
                    # img_cr = icr["semantic"]
                    icr_rgb = image_dict["rgb"]
                    icr_depth = image_dict["depth"]
                    start = time.time()
                    icr = clipseg.clipseg_hf_inference(image=icr_rgb, prompt=query)
                    end = time.time()
                    if verbose:
                        times.append(end-start)
                elif perception == "semantic_depth" and perception_type != "clipseg" and query is not None:
                    image_dict = self.gsplat.render_rgb(camera,T_c2w,query)
                    icr = image_dict["semantic"]
                    icr_rgb = image_dict["rgb"]
                    icr_depth = image_dict["depth"]

                    if validation:
                        icr_val = clipseg.clipseg_hf_inference(image=icr_rgb, prompt=query)
                else:
                    image_dict = self.gsplat.render_rgb(camera,T_c2w)
                    icr = image_dict["rgb"]
                # if perception == "semantic_depth" and query is not None:
                #     img_dict = self.gsplat.render_rgb(camera,T_c2w,query)
                #     icr = img_dict["semantic"]
                # else:
                #     icr = self.gsplat.render_rgb(camera,T_c2w)

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

                if query is not None:
                    Imgs_sem[k,:,:,:] = icr
                    Imgs_rgb[k,:,:,:] = icr_rgb
                    Imgs_depth[k,:,:,:] = icr_depth                
                    if validation and perception_type != "clipseg":
                        Imgs_val[k,:,:,:] = icr_val
                else:
                    Imgs_rgb[k,:,:,:] = icr
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

        if validation and perception_type != "clipseg" and query is not None:
            Iro = {"semantic":Imgs_sem,"depth":Imgs_depth,"rgb":Imgs_rgb,"validation":Imgs_val}
        elif query is not None:
            Iro = {"semantic":Imgs_sem,"depth":Imgs_depth,"rgb":Imgs_rgb}
        else:
            Iro = {"rgb":Imgs_rgb}

        # Log final time
        Tro[Nctl] = t0+Nsim/hz_sim

        return Tro,Xro,Uro,Iro,Tsol,Adv