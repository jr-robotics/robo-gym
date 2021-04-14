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
from robo_gym.envs.ur5.ur5_base_env import UR5BaseEnv



JOINT_POSITIONS = [0.0, -2.5, 1.5, 0, -1.4, 0]
class UR5BaseAvoidanceEnv(UR5BaseEnv):
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
    def reset(self, joint_positions = None, fixed_object_position = None) -> np.array:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            fixed_object_position (list[3]): x,y,z fixed position of object

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None
        
        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # Initialize desired joint positions
        if joint_positions:
            assert len(joint_positions) == 6
            self.joint_positions = joint_positions
        else:
            self._set_joint_positions(JOINT_POSITIONS)

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self._get_joint_positions())

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state, fixed_object_position)
        
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
        
        # Check if current position is in the range of the desired joint positions
        joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        if not np.isclose(joint_positions, self.joint_positions, atol=0.1).all():
            raise InvalidStateError('Reset joint positions are not within defined range')        

        return self.state


    def add_fixed_joints(self, action) -> np.array:
        action = action.tolist()
        fixed_joints = np.array([self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_1, self.fix_wrist_2, self.fix_wrist_3])
        fixed_joint_indices = np.where(fixed_joints)[0]

        temp = []
        for joint in range(len(fixed_joints)):
            if joint in fixed_joint_indices:
                temp.append(0)
            else:
                temp.append(action.pop(0))
        return np.array(temp)

    def env_action_to_rs_action(self, action) -> np.array:
        """Convert environment action to Robot Server action"""
        action = self.add_fixed_joints(action)
        
        # TODO remove from here later
        if self.last_action is None:
            self.last_action = action

        rs_action = copy.deepcopy(action)

        joint_positions = self._get_joint_positions() + action

        rs_action = self.ur._ur_joint_list_to_ros_joint_list(joint_positions)

        return action, rs_action         








