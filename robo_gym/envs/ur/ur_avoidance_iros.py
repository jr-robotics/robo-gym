"""
Environment for more complex obstacle avoidance controlling a robotic arm from UR.

In this environment the obstacle trajectory is generated with a randomized 3d spline.
The goal of the robot is to avoid dynamic obstacles while following pre-planned trajectory, 
which was programmed with the UR’s teach pendant. 
This trajectory is sampled at a frequency of 20 Hz.
"""
# TODO: add sentence that this is the environment used in the submission to iros 2021 

import os, copy, json
import numpy as np
import gym
from typing import Tuple
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.envs.ur.ur_base_avoidance_env import URBaseAvoidanceEnv

DEBUG = True
MINIMUM_DISTANCE = 0.45 # the distance [cm] the robot should keep to the obstacle

class IrosEnv03URTraining(URBaseAvoidanceEnv):
    """Universal Robots UR IROS environment. Obstacle avoidance while keeping a fixed trajectory.

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

    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, ur_model='ur5', include_polar_to_elbow=True, **kwargs) -> None:
        super().__init__(rs_address, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model, include_polar_to_elbow)

        file_name = 'trajectory_iros_2021.json'
        file_path = os.path.join(os.path.dirname(__file__), 'robot_trajectories', file_name)
        with open(file_path) as json_file:
            self.trajectory = json.load(json_file)['trajectory']

    def _set_initial_robot_server_state(self, rs_state, fixed_object_position = None) -> robot_server_pb2.State:
        if fixed_object_position:
            state_msg = super()._set_initial_robot_server_state(rs_state=rs_state, fixed_object_position=fixed_object_position)
            return state_msg

        n_sampling_points = int(np.random.default_rng().uniform(low=8000, high=12000))

        string_params = {"object_0_function": "3d_spline_ur5_workspace"}
        
        float_params = {"object_0_x_min": -1.0, "object_0_x_max": 1.0, "object_0_y_min": -1.0, "object_0_y_max": 1.0, \
                        "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                        "n_sampling_points": n_sampling_points}
        state = {}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, 
                                            string_params = string_params, state_dict = rs_state)
        return state_msg

    def reset(self, fixed_object_position = None) -> np.array:
        """Environment reset.

        Args:
            fixed_object_position (list[3]): x,y,z fixed position of object
        """
        # Initialize state machine variables
        self.state_n = 0 
        self.elapsed_steps_in_current_state = 0 
        self.target_reached = 0
        self.target_reached_counter = 0

        self.obstacle_coords = []

        joint_positions = self._get_joint_positions()

        self.state = super().reset(joint_positions = joint_positions, fixed_object_position = fixed_object_position)
            
        return self.state

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        self.elapsed_steps_in_current_state += 1
        
        self.state, reward, done, info = super().step(action)

        if self.target_reached:
            self.state_n +=1
            # Restart from state 0 if the full trajectory has been completed
            self.state_n = self.state_n % len(self.trajectory)
            self.elapsed_steps_in_current_state = 0
            self.target_reached_counter += 1
            self.target_reached = 0

        return self.state, reward, done, info

    def _reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        env_state = self._robot_server_state_to_env_state(rs_state)

        # TODO: move back to step function?
        # Check if the robot is at the target position
        # Joint positions 
        joint_positions = []
        joint_positions_keys = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                            'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position']
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        
        if self.target_point_flag:
            if np.isclose(self._get_joint_positions(), joint_positions, atol = 0.1).all():
                self.target_reached = 1

        # TODO: remove as soon as rs state is a dictonary
        # Save obstacle position
        self.obstacle_coords.append(np.array([rs_state['object_0_position_x'], rs_state['object_0_position_y'], rs_state['object_0_position_z']]))

        reward = 0
        done = False
        info = {}
        
        # Reward weights
        close_distance_weight = -2.0
        delta_joint_weight = 1.0
        action_usage_weight = 1.5
        rapid_action_weight = -0.2
        collision_weight = -0.1
        target_reached_weight = 0.05

        # Calculate distance to the target
        target_coord = np.array([rs_state['object_0_position_x'], rs_state['object_0_position_y'], rs_state['object_0_position_z']])
        ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'], rs_state['ee_to_ref_translation_z']])
        forearm_coord = np.array([rs_state['forearm_to_ref_translation_x'], rs_state['forearm_to_ref_translation_y'], rs_state['forearm_to_ref_translation_z']])
        distance_to_ee = np.linalg.norm(np.array(target_coord)-np.array(ee_coord))
        distance_to_elbow = np.linalg.norm(np.array(target_coord)-np.array(forearm_coord))
            
        # Reward staying close to the predefined joint position
        delta_joint_pos = env_state[9:15]
        for i in range(action.size):
            if abs(delta_joint_pos[i]) < 0.1:
                temp = delta_joint_weight * (1 - (abs(delta_joint_pos[i]))/0.1) * (1/1000) 
                temp = temp/action.size
                reward += temp
        
        # Reward for not acting
        if abs(action).sum() <= action.size:
            reward += action_usage_weight * (1 - (np.square(action).sum()/action.size)) * (1/1000)

        # Negative reward if actions change to rapidly between steps
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.4:
                reward += rapid_action_weight * (1/1000)

        # Negative reward if the obstacle is closer than the predefined minimum distance
        if (distance_to_ee < MINIMUM_DISTANCE) or (distance_to_elbow < MINIMUM_DISTANCE):
            reward += close_distance_weight * (1/1000)

        # Reward for reaching a predefined waypoint on the trajectory
        if self.target_reached:
            reward += target_reached_weight

        # Check if robot is in collision
        collision = True if rs_state['in_collision'] == 1 else False
        if collision:
            # Negative reward for a collision with the robot itself, the obstacle or the scene
            reward = collision_weight
            done = True
            info['final_status'] = 'collision'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
        
        if done:
            info['targets_reached'] = self.target_reached_counter
            info['obstacle_coords'] = self.obstacle_coords

        return reward, done, info

    def _robot_server_state_to_env_state(self, rs_state) -> np.array:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.
        """
        state = super()._robot_server_state_to_env_state(rs_state)

        trajectory_joint_position = self.ur.normalize_joint_values(self._get_joint_positions())
        target_point_flag = copy.deepcopy(self.target_point_flag)

        state = np.concatenate((state, trajectory_joint_position, [target_point_flag]))

        return state

    def _get_observation_space(self) -> gym.spaces.Box:
        """Get environment observation space."""

        # Joint position range tolerance
        pos_tolerance = np.full(6,0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(6, -1.0), pos_tolerance)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        
        max_delta_start_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_delta_start_positions = np.subtract(np.full(6, -1.0), pos_tolerance)

        # Target coordinates (with respect to forearm frame) range
        target_forearm_range = np.full(3, np.inf)
        
        # Definition of environment observation_space
        if self.include_polar_to_elbow:
            max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, target_forearm_range, max_joint_positions, [1]))
            min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, -target_forearm_range, min_joint_positions, [0]))
        else:
            max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, np.zeros(3), max_joint_positions, [1]))
            min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, np.zeros(3), min_joint_positions, [0]))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)


    def _get_joint_positions(self) -> np.array:
        """Get robot joint positions with standard indexing."""

        if self.elapsed_steps_in_current_state < len(self.trajectory[self.state_n]):
            joint_positions = copy.deepcopy(self.trajectory[self.state_n][self.elapsed_steps_in_current_state])
            self.target_point_flag = 0
        else:
            # Get last point of the trajectory segment
            joint_positions = copy.deepcopy(self.trajectory[self.state_n][-1])
            self.target_point_flag = 1
        
        return joint_positions

class IrosEnv03URTrainingSim(IrosEnv03URTraining, Simulation):
    cmd = "roslaunch ur_robot_server ur_robot_server.launch \
        world_name:=tabletop_sphere50.world \
        yaw:=-0.78\
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=true \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving2points \
        n_objects:=1.0 \
        object_0_model_name:=sphere50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur5', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03URTraining.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03URTrainingRob(IrosEnv03URTraining):
    real_robot = True

# TODO: check if this roslaunch is correct
# TODO: add roslaunch for the realsense skeleton tracker
# roslaunch ur_robot_server ur_robot_server.launch ur_model:=ur5 real_robot:=true rviz_gui:=true gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target



"""
Testing Environment for more complex obstacle avoidance controlling a robotic arm from UR.
In contrast to the training environment the obstacle trajectories are fixed instead of random generation.
"""
class IrosEnv03URTestFixedSplines(IrosEnv03URTraining):
    ep_n = 0 

    # TODO: add typing to method head
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position = None):
        if fixed_object_position:
            state_msg = super()._set_initial_robot_server_state(rs_state=rs_state, fixed_object_position=fixed_object_position)
            return state_msg
        
        string_params = {"object_0_function": "fixed_trajectory"}
        float_params = {"object_0_trajectory_id": self.ep_n%50}
        state = {}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, 
                                string_params = string_params, state_dict = rs_state)
        return state_msg

    def reset(self):
        self.state = super().reset()
        
        self.ep_n +=1

        return self.state

class IrosEnv03URTestFixedSplinesSim(IrosEnv03URTestFixedSplines, Simulation):
    cmd = "roslaunch ur_robot_server ur_robot_server.launch \
        world_name:=tabletop_sphere50.world \
        yaw:=-0.78\
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving2points \
        n_objects:=1.0 \
        object_trajectory_file_name:=splines_ur5 \
        object_0_model_name:=sphere50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur5', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03URTestFixedSplines.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03URTestFixedSplinesRob(IrosEnv03URTestFixedSplines):
    real_robot = True

# roslaunch ur_robot_server ur_robot_server.launch ur_model:=ur5 real_robot:=true rviz_gui:=true gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target
