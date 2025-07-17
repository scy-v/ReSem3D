from isaacgym import gymtorch
import os
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion
import torch, hydra, zerorpc
import copy
import json
from m3p2i_aip.planners.motion_planner import m3p2i
from m3p2i_aip.planners.task_planner import task_planner
from m3p2i_aip.config.config_store import ExampleConfig
from scipy.spatial.transform import Rotation as R
import cloudpickle
import re
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.planners.motion_planner.cost_functions import Objective
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
from m3p2i_aip.utils import path_utils
curr_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(curr_path, "configs", "mppi_config/")
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
path_utils.get_config_path = lambda: config_path

class REACTIVE_TAMP:
    def __init__(self, cfg) -> None:
        self.sim = wrapper.IsaacGymWrapper(
            cfg.isaacgym,
            cfg.env_type,
            num_envs=cfg.mppi.num_samples,
            viewer=False,
            device=cfg.mppi.device,
            cube_on_shelf=cfg.cube_on_shelf,
        )
        
        self.cfg = cfg
        self.device = self.cfg.mppi.device

        self.motion_planner = m3p2i.M3P2I(
            cfg,
            dynamics=self.dynamics, 
            running_cost=self.run_cost
        )
        self.motion_planner.gripper_command = "open"
        
        self.variable_vars = {
            k: getattr(self, k)
            for k in dir(self) if callable(getattr(self, k)) and not k.startswith("_")
        } 
        
        init_pos = torch.tensor(self.sim.env_cfg[0].init_pos)
        fixed_part = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.sim._root_state[:] = torch.cat((init_pos, fixed_part)).unsqueeze(0).unsqueeze(0).to(self.device)
        self.sim.set_actor_root_state_tensor(self.sim._root_state)
        
    def run_tamp(self, dof_state, target_dict, compute_cost):
        # Set rollout state from sim state
        import time
        start = time.time()
        
        self.sim._dof_state[:] = bytes_to_torch(dof_state)
        self.target_dict = json.loads(bytes_to_torch(target_dict))
        self.compute_cost = bytes_to_torch(compute_cost)
        self.sim.set_dof_state_tensor(self.sim._dof_state)

        print("--------Compute optimal action--------")
        
        output =  torch_to_bytes(
            self.motion_planner.command(self.sim._dof_state[0])[0]
        )
        end = time.time()
        print("FPS: ", 1/ (end-start), end-start)
        
        return output
    
    def dynamics(self, _, u, t=None):
        self.sim.set_dof_velocity_target_tensor(u)
        self.sim.step()
        states = torch.stack([self.sim.robot_pos[:, 0], 
                              self.sim.robot_vel[:, 0], 
                              self.sim.robot_pos[:, 1], 
                              self.sim.robot_vel[:, 1]], dim=1) # [num_envs, 4]
        return states, u
    
    def run_cost(self, _):
        lvars = {}
        self.variable_vars.update(self.target_dict)
        exec(self.compute_cost, self.variable_vars, lvars)
        return lvars["ret_val"]()

    def compute_reach_cost(self, pos1, pos2):
        """Compute reach cost between two positions."""
        pos1, pos2 = self.to_tensor_cuda(pos1), self.to_tensor_cuda(pos2)
        pos2[2] += self.sim.env_cfg[0].init_pos[2]
        return torch.linalg.norm(pos1 - pos2, axis=1)

    def compute_height_cost(self, pos1, pos2):
        """Compute height cost between two positions."""
        pos1, pos2 = self.to_tensor_cuda(pos1), self.to_tensor_cuda(pos2)
        pos2 += self.sim.env_cfg[0].init_pos[2]  
        pos2 = pos2 + torch.zeros_like(pos1) 
        diff = pos1 - pos2
        return torch.abs(diff) 

    def compute_orientation_cost(self, wxyz1, wxyz2):
        """Compute orientation cost between two quaternions."""
        wxyz1, wxyz2 = self.to_tensor_cuda(wxyz1), self.to_tensor_cuda(wxyz2)
        return 2 * torch.acos(torch.clamp(torch.sum(wxyz1 * wxyz2, dim=-1), min=-1.0, max=1.0))
    
    def to_tensor_cuda(self, x):
        """Convert input to tensor on device."""
        return torch.tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device)

    def mppi_get_ee_pose(self):
        """
        Get the end-effector position and orientation from the hand link state.
        """
        # Get hand link states (position + quaternion)
        hand_state = self.sim.get_actor_link_by_name("panda", "panda_hand")
        num_poses = hand_state.shape[0]

        # Convert quaternions to rotation matrices
        quats = hand_state[:, 3:7]
        rot_matrices = quaternion_to_matrix(quats[:, [3, 0, 1, 2]])

        # Build hand pose transform matrix
        hand_pose_T = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_poses, 1, 1)
        hand_pose_T[:, :3, :3] = rot_matrices
        hand_pose_T[:, :3, 3] = hand_state[:, :3]

        # Define hand-to-end-effector transform (translation + 180° rotation around x-axis)
        hand_to_ee_T = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_poses, 1, 1)
        hand_to_ee_T[:, :3, 3] = torch.tensor([0, 0, 0.1034], device=self.device)
        hand_to_ee_R = R.from_euler('x', 180, degrees=True).as_matrix()
        hand_to_ee_T[:, :3, :3] = torch.tensor(hand_to_ee_R, dtype=torch.float32, device=self.device)

        # Compute end-effector pose by combining transforms
        ee_T = torch.matmul(hand_pose_T, hand_to_ee_T)
        ee_pos = ee_T[:, :3, 3]
        ee_quat = matrix_to_quaternion(ee_T[:, :3, :3])

        return ee_pos, ee_quat

    def transform_ee_to_point(self, ee_pos, ee_wxyz, point_to_ee):
        """
        Transform a point from end-effector to base_link coordinates.

        Args:
            ee_pos (torch.Tensor): End-effector positions in base_link frame, shape (N, 3).
            ee_wxyz (torch.Tensor): End-effector orientations as quaternions (w, x, y, z), shape (N, 4).
            point_to_ee (list or array): Single point coordinates in end-effector frame, length 3.

        Returns:
            torch.Tensor: Transformed point coordinates in base_link frame, shape (N, 3).
        """
        # Convert point to tensor and expand to batch size
        point_to_ee = torch.tensor(point_to_ee, dtype=torch.float32, device=ee_pos.device)
        point_to_ee = point_to_ee.unsqueeze(0).expand(ee_pos.shape[0], -1)  # (N, 3)

        # Convert quaternion (w,x,y,z) to rotation matrix
        ee_rot_mat = quaternion_to_matrix(ee_wxyz)  # (N, 3, 3)

        # Rotate and translate point
        point_base = torch.bmm(ee_rot_mat, point_to_ee.unsqueeze(-1)).squeeze(-1) + ee_pos  # (N, 3)

        return point_base

    def compute_world_vector_cost(self, ee_wxyz, world_axis, ee_axis='z', degree=0, use_positive_direction=True):
        """
        Compute the angular difference cost between end-effector orientation and a world axis.

        Args:
            ee_wxyz (torch.Tensor): Quaternions of end-effector orientations, shape (N, 4), format (w, x, y, z).
            world_axis (str): Target world axis, one of 'x', 'y', 'z'.
            ee_axis (str): End-effector axis to compare, one of 'x', 'y', 'z'. Default is 'z'.
            degree (float): Target angle in degrees. Default is 0.
            use_positive_direction (bool): Whether to use the positive angle between vectors. Default is True.

        Returns:
            torch.Tensor: Angular cost (absolute difference) in degrees, shape (N,).
        """
        axis_vectors = {
            "x": torch.tensor([1, 0, 0], dtype=torch.float32, device=ee_wxyz.device),
            "y": torch.tensor([0, 1, 0], dtype=torch.float32, device=ee_wxyz.device),
            "z": torch.tensor([0, 0, 1], dtype=torch.float32, device=ee_wxyz.device),
        }

        if world_axis not in axis_vectors:
            raise ValueError(f"Invalid world_axis '{world_axis}', choose from 'x', 'y', 'z'.")
        if ee_axis not in axis_vectors:
            raise ValueError(f"Invalid ee_axis '{ee_axis}', choose from 'x', 'y', 'z'.")

        target_vec = axis_vectors[world_axis].expand(ee_wxyz.shape[0], -1)
        ee_vec = axis_vectors[ee_axis].expand(ee_wxyz.shape[0], -1)

        rotation_matrices = quaternion_to_matrix(ee_wxyz)  # (N, 3, 3)
        ee_vec_rot = torch.bmm(rotation_matrices, ee_vec.unsqueeze(-1)).squeeze(-1)

        dot = torch.sum(ee_vec_rot * target_vec, dim=1)
        angle_rad = torch.acos(torch.clamp(dot if use_positive_direction else -dot, -1.0, 1.0))
        angle_deg = angle_rad * (180.0 / torch.pi)

        cost = torch.abs(angle_deg - degree)
        return cost

@hydra.main(version_base=None, config_path=config_path, config_name="config_panda")
def run_reactive_tamp(cfg: ExampleConfig):
    reactive_tamp = REACTIVE_TAMP(cfg)
    planner = zerorpc.Server(reactive_tamp)
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__== "__main__":
    run_reactive_tamp()