#!/usr/bin/env python3

from copy import deepcopy
import sys, math, copy, random
import numpy as np
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




JOINT_POSITIONS = [0.0, -2.5, 1.5, 0, -1.4, 0]
RANDOM_JOINT_OFFSET = [0.65, 0.25, 0.5, 3.14, 0.4, 3.14]
class UR5BaseEnv(gym.Env):
    """Universal Robots UR5 base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        ur (:obj:): Robot utilities object.
        observation_space (:obj:): Environment observation space.
        action_space (:obj:): Environment action space.
        distance_threshold (float): Minimum distance (m) from target to consider it reached.
        abs_joint_pos_range (np.array): Absolute value of joint positions range`.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False
    max_episode_steps = 300

    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, **kwargs):
        self.ur = ur_utils.UR(model="ur5")
        self.elapsed_steps = 0

        self.fix_base = fix_base
        self.fix_shoulder = fix_shoulder
        self.fix_elbow = fix_elbow
        self.fix_wrist_1 = fix_wrist_1
        self.fix_wrist_2 = fix_wrist_2
        self.fix_wrist_3 = fix_wrist_3

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.seed()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur.get_max_joint_positions()
        self.last_position_on_success = []
        self.last_action = None
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, joint_positions = None, ee_target_pose = None):
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.
        
        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        
        # Set initial robot joint positions
        if joint_positions:
            assert len(joint_positions) == 6
            self.joint_positions = joint_positions
        else:
            self._set_joint_positions(JOINT_POSITIONS)

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.joint_positions)

        # Set target End Effector pose
        if ee_target_pose:
            assert len(ee_target_pose) == 6
        else:
            ee_target_pose = self._get_target_pose()

        # Set initial state of the Robot Server

        # Object in a fixed position
        string_params = {"object_0_function": "fixed_position"}
        float_params = {"object_0_x": ee_target_pose[0], 
                        "object_0_y": ee_target_pose[1], 
                        "object_0_z": ee_target_pose[2]}
        
        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)


        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            if not np.isclose(joint_positions, self.joint_positions, atol=0.1).all():
                raise InvalidStateError('Reset joint positions are not within defined range')
            
        return self.state

    def _reward(self, rs_state, action):
        return 0, False, {}

    def add_fixed_joints(self, action) -> np.array:
        action = action.tolist()
        fixed_joints = np.array([self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_1, self.fix_wrist_2, self.fix_wrist_3])
        fixed_joint_indices = np.where(fixed_joints)[0]

        joints_position_norm = self.ur.normalize_joint_values(joints=self._get_joint_positions())

        temp = []
        for joint in range(len(fixed_joints)):
            if joint in fixed_joint_indices:
                temp.append(joints_position_norm[joint])
            else:
                temp.append(action.pop(0))
        return np.array(temp)

    def env_action_to_rs_action(self, action) -> np.array:
        """Convert environment action to Robot Server action"""
        action = self.add_fixed_joints(action)
        rs_action = copy.deepcopy(action)

        # Scale action
        rs_action = np.multiply(rs_action, self.abs_joint_pos_range)
        # Convert action indexing from ur to ros
        rs_action = self.ur._ur_joint_list_to_ros_joint_list(rs_action)

        return action, rs_action        

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list: action = np.array(action)
            
        self.elapsed_steps += 1

        # Check if the action is contained in the action space
        if not self.action_space.contains(action):
            raise InvalidActionError()

        # Convert environment action to robot server action
        action, rs_action = self.env_action_to_rs_action(action)

        # Send action to Robot Server and get state
        rs_state = self.client.send_action_get_state(rs_action.tolist()).state
        
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self._reward(rs_state=rs_state, action=action)
        self.last_action = action

        return self.state, reward, done, info

    def render():
        pass

    def _get_robot_server_state_len(self):
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
        rs_state = target + ur_j_pos + ur_j_vel + ee_to_ref_frame_transform + ur_collision

        return len(rs_state)

    def _get_env_state_len(self):
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """

        target_polar = [0.0]*3
        ur_j_pos = [0.0]*6
        ur_j_vel = [0.0]*6
        env_state = target_polar + ur_j_pos + ur_j_vel

        return len(env_state)

    # ? move to ee env
    def _set_joint_positions(self, joint_positions) -> None:
        """Set robot joint positions with standard indexing."""
        assert len(joint_positions) == 6

        if RANDOM_JOINT_OFFSET:
            joint_positions_low = np.array(joint_positions) - np.array(RANDOM_JOINT_OFFSET) 
            joint_positions_high = np.array(joint_positions) + np.array(RANDOM_JOINT_OFFSET) 

        self.joint_positions = np.random.default_rng().uniform(low=joint_positions_low, high=joint_positions_high)
    
    def _get_joint_positions(self) -> np.array:
        """Get robot joint positions with standard indexing."""
        return np.array(self.joint_positions)

    # ? move to ee env
    def _get_target_pose(self):
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        return self.ur.get_random_workspace_pose()

    def _robot_server_state_to_env_state(self, rs_state):
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
        ur_j_vel = self.ur._ros_joint_list_to_ur_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # Compose environment state
        state = np.concatenate((target_polar, ur_j_pos_norm, ur_j_vel))

        return state

    def _get_observation_space(self):
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
        # Joint velocities range 
        max_joint_velocities = np.array([np.inf] * 6)
        min_joint_velocities = - np.array([np.inf] * 6)
        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_joint_velocities))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
    
    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        fixed_joints = [self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_1, self.fix_wrist_2, self.fix_wrist_3]
        num_control_joints = len(fixed_joints) - sum(fixed_joints)

        return spaces.Box(low=np.full((num_control_joints), -1.0), high=np.full((num_control_joints), 1.0), dtype=np.float32)




