"""
Environment for basic obstacle avoidance controlling a robotic arm from UR.

In this environment the obstacle is only moving up and down in a vertical line in front of the robot.
The goal is for the robot to stay within a predefined minimum distance to the moving obstacle.
When feasible the robot should continue to the original configuration, 
otherwise wait for the obstacle to move away before proceeding
"""
import math
import gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from typing import Tuple
from robo_gym.envs.ur5.ur5_base_avoidance_env import UR5BaseAvoidanceEnv


DEBUG = True
MINIMUM_DISTANCE = 0.3 # the distance [cm] the robot should keep to the obstacle
JOINT_POSITIONS = [-0.78,-1.31,-1.31,-2.18,1.57,0.0]

class MovingBoxTargetUR5(UR5BaseAvoidanceEnv):
    
    max_episode_steps = 1000
            
    # TODO: add typing to method head
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position = None):
        # Set initial state of the Robot Server
        if fixed_object_position:
            # Object in a fixed position
            string_params = {"object_0_function": "fixed_position"}
            float_params = {"object_0_x": fixed_object_position[0], 
                            "object_0_y": fixed_object_position[1], 
                            "object_0_z": fixed_object_position[2]}
        else:
            # Object moving up and down
            z_amplitude = np.random.default_rng().uniform(low=0.09, high=0.35)
            z_frequency = 0.125
            z_offset = np.random.default_rng().uniform(low=0.2, high=0.6)
            
            string_params = {"object_0_function": "triangle_wave"}
            float_params = {"object_0_x": -0.13, 
                            "object_0_y": 0.30, 
                            "object_0_z_amplitude": z_amplitude,
                            "object_0_z_frequency": z_frequency, 
                            "object_0_z_offset": z_offset}

        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        return state_msg

    def reset(self, joint_positions = None, fixed_object_position = None) -> np.array:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            fixed_object_position (list[3]): x,y,z fixed position of object

        Returns:
            np.array: Environment state.

        """
        self.state = super().reset(joint_positions = joint_positions, fixed_object_position = fixed_object_position)   

        return self.state

    def _reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        env_state = self._robot_server_state_to_env_state(rs_state)

        reward = 0
        done = False
        info = {}
        
        # Reward weights
        close_distance_weight = -2
        delta_joint_weight = 1
        action_usage_weight = 1
        rapid_action_weight = -0.2

        # Difference in joint position current vs. starting position
        delta_joint_pos = env_state[9:15]

        # Calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        distance_to_target = np.linalg.norm(target_coord - ee_coord)   
                
        # Reward staying close to the predefined joint position
        if abs(env_state[-6:]).sum() < 0.1 * action.size:
            reward += delta_joint_weight * (1 - (abs(delta_joint_pos).sum()/(0.1 * action.size))) * (1/1000)
        
        # Reward for not acting
        if abs(action).sum() <= action.size:
            reward += action_usage_weight * (1 - (np.square(action).sum()/action.size)) * (1/1000)

        # Negative reward if actions change to rapidly between steps
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.5:
                reward += rapid_action_weight * (1/1000)
            
        # Negative reward if the obstacle is close than the predefined minimum distance
        if distance_to_target < MINIMUM_DISTANCE:
            reward += close_distance_weight * (1/self.max_episode_steps) 

        collision = True if rs_state[25] == 1 else False
        if collision:
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord
            self.last_position_on_success = []

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord
            self.last_position_on_success = []


        return reward, done, info

    # TODO: once normalization is gone this method can be merged with URBaseEnv
    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        
        self.state, reward, done, info = super().step(action)

        return self.state, reward, done, info
    
# TODO: yaw is different from the iros env -> check 
class MovingBoxTargetUR5Sim(MovingBoxTargetUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=tabletop_sphere50.world \
        yaw:=3.14\
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving2points \
        n_objects:=1.0 \
        object_0_model_name:=sphere50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBoxTargetUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class MovingBoxTargetUR5Rob(MovingBoxTargetUR5):
    real_robot = True 

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving