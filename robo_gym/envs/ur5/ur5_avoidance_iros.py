"""
Environment for more complex obstacle avoidance controlling a robotic arm from UR.

In this environment the obstacle trajectory is generated with a randomized 3d spline.
The goal of the robot is to avoid dynamic obstacles while following pre-planned trajectory, 
which was programmed with the UR’s teach pendant. 
This trajectory is sampled at a frequency of 20 Hz.
"""
# TODO: add sentence that this is the environment used in the submission to iros 2021 

import os, random, copy, json
import numpy as np
from robo_gym.envs.ur5.ur5_base_avoidance_env import UR5BaseAvoidanceEnv
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from typing import Tuple

DEBUG = True
MINIMUM_DISTANCE = 0.45 # the distance [cm] the robot should keep to the obstacle

class IrosEnv03UR5Training(UR5BaseAvoidanceEnv):
    max_episode_steps = 1000

    def __init__(self, rs_address=None, **kwargs) -> None:
        super().__init__(rs_address, **kwargs)

        file_name = 'trajectory_iros_2021.json'
        file_path = os.path.join(os.path.dirname(__file__), 'robot_trajectories', file_name)
        with open(file_path) as json_file:
            self.trajectory = json.load(json_file)['trajectory']

    # TODO: add typing to method head
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position = None):
        if fixed_object_position:
            # Object in a fixed position
            string_params = {"object_0_function": "fixed_position"}
            float_params = {"object_0_x": fixed_object_position[0], 
                            "object_0_y": fixed_object_position[1], 
                            "object_0_z": fixed_object_position[2]}
        else:
            n_sampling_points = int(np.random.default_rng().uniform(low=8000, high=12000))
            
            string_params = {"object_0_function": "3d_spline_ur5_workspace"}
            
            float_params = {"object_0_x_min": -1.0, "object_0_x_max": 1.0, "object_0_y_min": -1.0, "object_0_y_max": 1.0, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}

        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        return state_msg

    def reset(self, fixed_object_position = None) -> np.array:
        """Environment reset.

        Args:
            fixed_object_position (list[3]): x,y,z fixed position of object

        Returns:
            np.array: Environment state.
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

    # TODO: once we use a dictonary to handle the rs state this method should be redone
    def print_state_action_info(self, rs_state, action) -> None:
        env_state = self._robot_server_state_to_env_state(rs_state)

        print('Action:', action)
        print('Last A:', self.last_action)
        print('Distance EE: {:.2f} Polar1: {:.2f} Polar2: {:.2f}'.format(env_state[0], env_state[1], env_state[2]))
        print('Distance Elbow: {:.2f} Polar1: {:.2f} Polar2: {:.2f}'.format(env_state[-6], env_state[-5], env_state[-4]))
        print('Elbow Cartesian: x={:.2f}, y={:.2f}, z={:.2f}'.format(env_state[-3], env_state[-2], env_state[-1]))
        
        print('Joint Positions: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[3:9]))
        print('Joint PosDeltas: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[9:15]))
        print('Current Desired: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[15:21]))
        print('Is current disred a target?', env_state[21])
        print('Targets reached', self.target_reached_counter)

        print('Target Reached: {}'.format(self.target_reached))
        print('State number: {}'.format(self.state_n))

        print()


    def _reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        env_state = self._robot_server_state_to_env_state(rs_state)

        # TODO: move back to step function?
        # Check if the robot is at the target position
        if self.target_point_flag:
            if np.isclose(self._get_joint_positions(), self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12]), atol = 0.1).all():
                self.target_reached = 1

        # TODO: remove as soon as rs state is a dictonary
        # Save obstacle position
        self.obstacle_coords.append(rs_state[0:3])

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
        target_coord = rs_state[0:3]
        ee_coord = rs_state[18:21]
        elbow_coord = rs_state[26:29]
        distance_to_ee = np.linalg.norm(np.array(target_coord)-np.array(ee_coord))
        distance_to_elbow = np.linalg.norm(np.array(target_coord)-np.array(elbow_coord))
            
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
        collision = True if rs_state[25] == 1 else False
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

        if DEBUG: 
            self.print_state_action_info(rs_state, action)
            print('Distance 1: {:.2f} Distance 2: {:.2f}'.format(distance_to_ee, distance_to_elbow))

        return reward, done, info

    def _robot_server_state_to_env_state(self, rs_state) -> np.array:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the end effector frame
        target_coord = rs_state[0:3]
        
        ee_to_ref_frame_translation = np.array(rs_state[18:21])
        ee_to_ref_frame_quaternion = np.array(rs_state[21:25])
        ee_to_ref_frame_rotation = R.from_quat(ee_to_ref_frame_quaternion)
        ref_frame_to_ee_rotation = ee_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_ee_quaternion = ref_frame_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_ee_translation = -ref_frame_to_ee_rotation.apply(ee_to_ref_frame_translation)

        target_coord_ee_frame = utils.change_reference_frame(target_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        #ur_j_vel = self.ur._ros_joint_list_to_ur_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # desired joint positions
        desired_joints = self.ur.normalize_joint_values(self._get_joint_positions())
        delta_joints = ur_j_pos_norm - desired_joints
        target_point_flag = copy.deepcopy(self.target_point_flag)

        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the forearm

        forearm_to_ref_frame_translation = rs_state[26:29]
        forearm_to_ref_frame_quaternion = rs_state[29:33]
        forearm_to_ref_frame_rotation = R.from_quat(forearm_to_ref_frame_quaternion)
        ref_frame_to_forearm_rotation = forearm_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_forearm_quaternion = ref_frame_to_forearm_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_forearm_translation = -ref_frame_to_forearm_rotation.apply(forearm_to_ref_frame_translation)

        target_coord_forearm_frame = utils.change_reference_frame(target_coord,ref_frame_to_forearm_translation,ref_frame_to_forearm_quaternion)
        target_polar_forearm = utils.cartesian_to_polar_3d(target_coord_forearm_frame)

        if DEBUG:
            print('Object coords in ref frame', target_coord)
            print('Object coords in ee frame', target_coord_ee_frame)
            print('Object polar coords in ee frame', target_polar)
            print('Object coords in forearm frame', target_coord_forearm_frame)
            print('Object polar coords in forearm frame', target_polar_forearm)
        # Compose environment state
        state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints, desired_joints, [target_point_flag], target_polar_forearm))

        return state

    # observation space should be fine
    def _get_observation_space(self) -> spaces.Box:
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

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
        max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, max_joint_positions, [1], target_forearm_range))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, min_joint_positions, [0], -target_forearm_range))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_robot_server_state_len(self) -> spaces.Box:

        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """
        target = [0.0]*6
        ur_j_pos = [0.0]*6
        ur_j_vel = [0.0]*6
        ee_to_ref_frame_transform = [0.0]*7
        ur_collision = [0.0]
        forearm_to_ref_frame_transform = [0.0]*7
        rs_state = target + ur_j_pos + ur_j_vel + ee_to_ref_frame_transform + ur_collision + forearm_to_ref_frame_transform

        return len(rs_state)

    def _get_joint_positions(self) -> np.array:
        """Get desired robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        if self.elapsed_steps_in_current_state < len(self.trajectory[self.state_n]):
            joint_positions = copy.deepcopy(self.trajectory[self.state_n][self.elapsed_steps_in_current_state])
            self.target_point_flag = 0
        else:
            # Get last point of the trajectory segment
            joint_positions = copy.deepcopy(self.trajectory[self.state_n][-1])
            self.target_point_flag = 1
        

        return joint_positions



class IrosEnv03UR5TrainingSim(IrosEnv03UR5Training, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
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
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03UR5Training.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03UR5TrainingRob(IrosEnv03UR5Training):
    real_robot = True

# TODO: check if this roslaunch is correct
# TODO: add roslaunch for the realsense skeleton tracker
# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target



"""
Testing Environment for more complex obstacle avoidance controlling a robotic arm from UR.
In contrast to the training environment the obstacle trajectories are fixed instead of random generation.
"""
class IrosEnv03UR5TestFixedSplines(IrosEnv03UR5Training):
    ep_n = 0 

    # TODO: add typing to method head
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position = None):
        if fixed_object_position:
            # Object in a fixed position
            string_params = {"object_0_function": "fixed_position"}
            float_params = {"object_0_x": fixed_object_position[0], 
                            "object_0_y": fixed_object_position[1], 
                            "object_0_z": fixed_object_position[2]}
        else:        
            string_params = {"object_0_function": "fixed_trajectory"}
            float_params = {"object_0_trajectory_id": self.ep_n%50}

        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        return state_msg

    def reset(self):
        self.state = super().reset()
        
        self.ep_n +=1

        return self.state

class IrosEnv03UR5TestFixedSplinesSim(IrosEnv03UR5TestFixedSplines, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
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
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03UR5TestFixedSplines.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03UR5TestFixedSplinesRob(IrosEnv03UR5TestFixedSplines):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target
