"""
Environment for basic obstacle avoidance controlling a robotic arm from UR.

In this environment the obstacle is only moving up and down in a vertical line in front of the robot.
The goal is for the robot to stay within a predefined minimum distance to the moving obstacle.
When feasible the robot should continue to the original configuration, 
otherwise wait for the obstacle to move away before proceeding
"""
import numpy as np
from typing import Tuple
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.envs.ur.ur_base_avoidance_env import URBaseAvoidanceEnv

# base, shoulder, elbow, wrist_1, wrist_2, wrist_3
JOINT_POSITIONS = [-1.57, -1.31, -1.31, -2.18, 1.57, 0.0]
DEBUG = True
MINIMUM_DISTANCE = 0.3 # the distance [cm] the robot should keep to the obstacle


class BasicAvoidanceUR(URBaseAvoidanceEnv):
    """Universal Robots UR basic obstacle avoidance environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        fix_base (bool): Wether or not the base joint stays fixed or is moveable. Defaults to False.
        fix_shoulder (bool): Wether or not the shoulder joint stays fixed or is moveable. Defaults to False.
        fix_elbow (bool): Wether or not the elbow joint stays fixed or is moveable. Defaults to False.
        fix_wrist_1 (bool): Wether or not the wrist 1 joint stays fixed or is moveable. Defaults to False.
        fix_wrist_2 (bool): Wether or not the wrist 2 joint stays fixed or is moveable. Defaults to False.
        fix_wrist_3 (bool): Wether or not the wrist 3 joint stays fixed or is moveable. Defaults to True.
        ur_model (str): determines which ur model will be used in the environment. Defaults to 'ur5'.
        include_polar_to_elbow (bool): determines wether or not the polar coordinates to the elbow joint are included in the state. Defaults to False. 

    Attributes:
        ur (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    
    max_episode_steps = 1000
            
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position = None) -> robot_server_pb2.State:
        if fixed_object_position:
            state_msg = super()._set_initial_robot_server_state(rs_state=rs_state, fixed_object_position=fixed_object_position)
            return state_msg
    
        z_amplitude = np.random.default_rng().uniform(low=0.09, high=0.35)
        z_frequency = 0.125
        z_offset = np.random.default_rng().uniform(low=0.2, high=0.6)
        
        string_params = {"object_0_function": "triangle_wave"}
        float_params = {"object_0_x": 0.12, 
                        "object_0_y": 0.34, 
                        "object_0_z_amplitude": z_amplitude,
                        "object_0_z_frequency": z_frequency, 
                        "object_0_z_offset": z_offset}
        state = {}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, 
                                            string_params = string_params, state_dict = rs_state)
        return state_msg


    def reset(self, joint_positions = JOINT_POSITIONS, fixed_object_position = None) -> np.array:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            fixed_object_position (list[3]): x,y,z fixed position of object
        """
        self.prev_action = np.zeros(6)

        state = super().reset(joint_positions = joint_positions, fixed_object_position = fixed_object_position)   

        return state

    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
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

        # Calculate distance to the obstacle
        obstacle_coord = np.array([rs_state['object_0_to_ref_translation_x'], rs_state['object_0_to_ref_translation_y'], rs_state['object_0_to_ref_translation_z']])
        ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'], rs_state['ee_to_ref_translation_z']])
        forearm_coord = np.array([rs_state['forearm_to_ref_translation_x'], rs_state['forearm_to_ref_translation_y'], rs_state['forearm_to_ref_translation_z']])
        distance_to_ee = np.linalg.norm(obstacle_coord - ee_coord) 
        distance_to_forearm = np.linalg.norm(obstacle_coord - forearm_coord) 
        distance_to_target = np.min([distance_to_ee, distance_to_forearm])
                
        # Reward staying close to the predefined joint position
        if abs(env_state[-6:]).sum() < 0.1 * action.size:
            reward += delta_joint_weight * (1 - (abs(delta_joint_pos).sum()/(0.1 * action.size))) * (1/1000)
        
        # Reward for not acting
        if abs(action).sum() <= action.size:
            reward += action_usage_weight * (1 - (np.square(action).sum()/action.size)) * (1/1000)

        # Negative reward if actions change to rapidly between steps
        for i in range(len(action)):
            if abs(action[i] - self.prev_action[i]) > 0.5:
                reward += rapid_action_weight * (1/1000)
            
        # Negative reward if the obstacle is close than the predefined minimum distance
        if distance_to_target < MINIMUM_DISTANCE:
            reward += close_distance_weight * (1/self.max_episode_steps) 
        
        # Check if there is a collision
        collision = True if rs_state['in_collision'] == 1 else False
        if collision:
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = obstacle_coord
            self.last_position_on_success = []

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = obstacle_coord
            self.last_position_on_success = []


        return reward, done, info

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list: action = np.array(action)
        
        state, reward, done, info = super().step(action)

        self.prev_action = self.add_fixed_joints(action)

        return state, reward, done, info
    
class BasicAvoidanceURSim(BasicAvoidanceUR, Simulation):
    cmd = "roslaunch ur_robot_server ur_robot_server.launch \
        world_name:=tabletop_sphere50.world \
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        rs_mode:=1moving2points \
        n_objects:=1.0 \
        object_0_model_name:=sphere50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur5', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        BasicAvoidanceUR.__init__(self, rs_address=self.robot_server_ip, ur_model=ur_model, **kwargs)

class BasicAvoidanceURRob(BasicAvoidanceUR):
    real_robot = True 

# roslaunch ur_robot_server ur_robot_server.launch ur_model:=ur5 real_robot:=true rviz_gui:=true gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 rs_mode:=moving