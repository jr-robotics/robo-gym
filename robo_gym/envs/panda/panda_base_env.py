#!/usr/bin/env python3
from __future__ import annotations
import copy
import numpy as np
import gymnasium as gym
from typing import Tuple, Any
from robo_gym.utils import panda_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation
import time

# TODO this is the same as the neutral pose in the config, use that
# panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7
JOINT_POSITIONS = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315,
                   0.029840531355804868, 1.5411935298621688, 0.7534486589746342]


class PandaBaseEnv(gym.Env):
    """Franka Emika Panda Robot base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        fix_joint1 (bool): Whether joint1 stays fixed. Defaults to False.
        fix_joint2 (bool): Whether joint2 stays fixed. Defaults to False.
        fix_joint3 (bool): Whether joint3 stays fixed. Defaults to False.
        fix_joint4 (bool): Whether joint4 stays fixed. Defaults to False.
        fix_joint5 (bool): Whether joint5 stays fixed. Defaults to False.
        fix_joint6 (bool): Whether joint6 stays fixed. Defaults to False.
        fix_joint7 (bool): Whether joint7 stays fixed. Defaults to True.
        panda_model (str): determines which panda model will be used in the environment. Defaults to 'panda' (there exists no other version).
        rs_state_to_info (bool): Whether the state obtained from the robot server is to be included in the info dict. Defaults to false.
        action_mode (str): How the robot is controlled via the actions. One of "abs_pos" (default), "delta_pos", "abs_vel", or "delta_vel.
        action_cycle_rate (float): Frequency of actions in Hz. Defaults to 250.
        max_velocity_scale_factor (float): Ratio of the robot joints' maximum velocities to use as the maximum in velocity-based action modes (no effect in position-based action modes). Defaults to 0.1.
    Attributes:
        panda (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False
    max_episode_steps = 7000

    ACTION_MODE_ABS_POS = "abs_pos"
    ACTION_MODE_DELTA_POS = "delta_pos"
    ACTION_MODE_ABS_VEL = "abs_vel"
    ACTION_MODE_DELTA_VEL = "delta_vel"

    def __init__(self, rs_address=None, fix_joint1=False, fix_joint2=False, fix_joint3=False, fix_joint4=False, fix_joint5=False, fix_joint6=False, fix_joint7=True, panda_model='panda', rs_state_to_info=False, action_mode=ACTION_MODE_ABS_POS, action_cycle_rate = 250, max_velocity_scale_factor = 0.1, **kwargs):
        self.panda = panda_utils.PANDA(model=panda_model)
        self.action_mode = action_mode
        self.action_cycle_rate = action_cycle_rate
        self.max_velocity_scale_factor = max_velocity_scale_factor
        self.elapsed_steps = 0
        self.now = time.time()

        self.rs_state_to_info = rs_state_to_info

        self.fix_joint1 = fix_joint1
        self.fix_joint2 = fix_joint2
        self.fix_joint3 = fix_joint3
        self.fix_joint4 = fix_joint4
        self.fix_joint5 = fix_joint5
        self.fix_joint6 = fix_joint6
        self.fix_joint7 = fix_joint7

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        # for action_mode abs_pos
        self.max_joint_pos_range = self.panda.get_max_joint_positions()
        self.min_joint_pos_range = self.panda.get_min_joint_positions()
        # for action_mode abs_vel
        self.abs_joint_vel_range = self.max_velocity_scale_factor * self.panda.get_max_joint_velocities()
        # for action_mode delta_pos
        self.delta_joint_pos_range = self.abs_joint_vel_range
        # for action_mode delta_vel
        self.delta_joint_vel_range = (1.0/self.action_cycle_rate) * self.panda.get_max_joint_accelerations()

        self.rs_state = None

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def _set_initial_robot_server_state(self, rs_state) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        state_msg = robot_server_pb2.State(state=state, float_params=float_params,
                                           string_params=string_params, state_dict=rs_state)
        return state_msg

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Environment reset.

        Args:
            joint_positions (list[7] or np.array[7]): robot joint positions in radians. Order is defined by
        
        Returns:
            np.array: Environment state.

        """
        super().reset(seed=seed)
        if options is None:
            options = {}
        joint_positions = options["joint_positions"] if "joint_positions" in options else None

        if joint_positions: 
            assert len(joint_positions) == 7
        else:
            joint_positions = JOINT_POSITIONS

        self.elapsed_steps = 0

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        state = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)

        # Set initial robot joint positions
        self._set_joint_positions(joint_positions)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state)
        
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")
        
        # Check if real robot server is running with arguments that match the current environment configuration
        # (action mode, control rate - differences could be dangerous)    
        args = self.client.get_arguments()
        if not args:
            raise RobotServerError("args error")
        if args.action_cycle_rate != self.action_cycle_rate:
            raise RobotServerError("action_cycle_rate is not matching")
        if args.action_mode != self.action_mode:
            raise RobotServerError("action_mode is not matching")

        # Get Robot Server state
        rs_state = self.client.get_state_msg().state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)

        # Convert the initial state from Robot Server format to environment format
        state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            raise InvalidStateError()

        # Check if current position is in the range of the initial joint positions
        for joint in self.joint_positions.keys():
            if not np.isclose(self.joint_positions[joint], rs_state[joint], atol=0.05):
                raise InvalidStateError('Reset joint positions are not within defined range')

        self.rs_state = rs_state

        return state, {}

    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        done = False
        info = {}
        # Check if robot is in collision
        collision = True if rs_state['in_collision'] == 1 else False
        if collision:
            done = True
            info['final_status'] = 'collision'

        elif self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'

        return 0, done, info


    def add_fixed_joints(self, action) -> np.ndarray:
        action = action.tolist()
        fixed_joints = np.array([self.fix_joint1, self.fix_joint2, self.fix_joint3, self.fix_joint4, self.fix_joint5, self.fix_joint6, self.fix_joint7])
        fixed_joint_indices = np.where(fixed_joints)[0]

        joint_pos_names = ['joint1_position', 'joint2_position', 'joint3_position',
                           'joint4_position', 'joint5_position', 'joint6_position', 'joint7_position']
        joint_positions_dict = self._get_joint_positions()

        joint_positions = np.array([joint_positions_dict.get(joint_pos) for joint_pos in joint_pos_names])

        joints_position_norm = self.panda.normalize_joint_values(joints=joint_positions)


        temp = []
        for joint in range(len(fixed_joints)):
            if joint in fixed_joint_indices:
                if self.action_mode == self.ACTION_MODE_ABS_POS:
                    temp.append(joints_position_norm[joint])
                else:
                    # 0 if we are in any delta or any velocity mode
                    temp.append(0)
            else:
                temp.append(action.pop(0))
        return np.array(temp)

    def env_action_to_rs_action(self, action) -> np.ndarray:
        """Convert environment action to Robot Server action"""
        rs_action = copy.deepcopy(action)
        length = len(rs_action)

        if self.action_mode == self.ACTION_MODE_DELTA_POS:
            rs_action = np.multiply(rs_action, self.delta_joint_pos_range)
        elif self.action_mode == self.ACTION_MODE_ABS_VEL:
            rs_action = np.multiply(rs_action, self.abs_joint_vel_range)
        elif self.action_mode == self.ACTION_MODE_DELTA_VEL:
            rs_action = np.multiply(rs_action, self.delta_joint_vel_range)
        else:
            # self.action_mode == self.ACTION_MODE_ABS_POS:
            for i in range(length):
                # TODO some caching could help here
                rs_action[i] = self.min_joint_pos_range[i] + (self.max_joint_pos_range[i] - self.min_joint_pos_range[i]) * (rs_action[i] + 1) / 2

        # Convert action indexing from panda to ros
        rs_action = self.panda._panda_joint_list_to_ros_joint_list(rs_action)

        # print("Env action: " + str(action) + "\n RS action: " + str(rs_action) + "\n")

        return rs_action

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        if type(action) == list: action = np.array(action)

        action = action.astype(np.float32)
        # for checking publish Frequency -------------------------------------------------------------------------------------------------------
        self.elapsed_steps += 1
        #future = time.time()
        #elapsed_timez = future - self.now
        #print('action freq:', 1/elapsed_timez, 'Hz')
        #self.now = future
        # -----------------------------------------------------------------------------------------------------------------------------------

        # Check if the action is contained in the action space
        if not self.action_space.contains(action):
            raise InvalidActionError()

        # Add missing joints which were fixed at initialization
        action = self.add_fixed_joints(action)

        # Convert environment action to robot server action
        rs_action = self.env_action_to_rs_action(action)

        # Send action to Robot Server and get state
        rs_state = self.client.send_action_get_state(rs_action.tolist()).state_dict
        self._check_rs_state_keys(rs_state)

        # Convert the state from Robot Server format to environment format
        state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            raise InvalidStateError()

        self.rs_state = rs_state

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self.reward(rs_state=rs_state, action=action)
        if self.rs_state_to_info: info['rs_state'] = self.rs_state

        return state, reward, done, False, info

    def get_rs_state(self):
        return self.rs_state

    def render(self):
        pass

    def get_robot_server_composition(self) -> list:
        rs_state_keys = [
            'joint1_position',
            'joint2_position',
            'joint3_position',
            'joint4_position',
            'joint5_position',
            'joint6_position',
            'joint7_position',


            'joint1_velocity',
            'joint2_velocity',
            'joint3_velocity',
            'joint4_velocity',
            'joint5_velocity',
            'joint6_velocity',
            'joint7_velocity',

            'joint1_effort',
            'joint2_effort',
            'joint3_effort',
            'joint4_effort',
            'joint5_effort',
            'joint6_effort',
            'joint7_effort',

            'ee_to_ref_translation_x',
            'ee_to_ref_translation_y',
            'ee_to_ref_translation_z',
            'ee_to_ref_rotation_x',
            'ee_to_ref_rotation_y',
            'ee_to_ref_rotation_z',
            'ee_to_ref_rotation_w',

            'in_collision'
        ]
        return rs_state_keys

    def _get_robot_server_state_len(self) -> int:
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.
        """
        return len(self.get_robot_server_composition())

    def _check_rs_state_keys(self, rs_state) -> None:
        keys = self.get_robot_server_composition()
        if not len(keys) == len(rs_state.keys()):
            raise InvalidStateError("Robot Server state keys to not match. Different lengths.")

        for key in keys:
            if key not in rs_state.keys():
                raise InvalidStateError("Robot Server state keys to not match")

    def _set_joint_positions(self, joint_positions) -> None:
        """Set desired robot joint positions with standard indexing."""
        # Set initial robot joint positions
        self.joint_positions = {}
        self.joint_positions['joint1_position'] = joint_positions[0]
        self.joint_positions['joint2_position'] = joint_positions[1]
        self.joint_positions['joint3_position'] = joint_positions[2]
        self.joint_positions['joint4_position'] = joint_positions[3]
        self.joint_positions['joint5_position'] = joint_positions[4]
        self.joint_positions['joint6_position'] = joint_positions[5]
        self.joint_positions['joint7_position'] = joint_positions[6]

    def _get_joint_positions(self) -> dict:
        """Get robot joint positions with standard indexing."""
        return self.joint_positions

    def _get_joint_positions_as_array(self) -> np.ndarray:
        """Get robot joint positions with standard indexing."""
        joint_positions = []
        joint_positions.append(self.joint_positions['joint1_position'])
        joint_positions.append(self.joint_positions['joint2_position'])
        joint_positions.append(self.joint_positions['joint3_position'])
        joint_positions.append(self.joint_positions['joint4_position'])
        joint_positions.append(self.joint_positions['joint5_position'])
        joint_positions.append(self.joint_positions['joint6_position'])
        joint_positions.append(self.joint_positions['joint7_position'])
        return np.array(joint_positions)

    def get_joint_name_order(self) -> list:
        return ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']

    def _robot_server_state_to_env_state(self, rs_state) -> np.ndarray:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Joint positions 
        joint_positions = []
        joint_positions_keys = ['joint1_position', 'joint2_position', 'joint3_position',
                                'joint4_position', 'joint5_position', 'joint6_position', 'joint7_position']
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        # Normalize joint position values
        joint_positions = self.panda.normalize_joint_values(joints=joint_positions)

        # Joint Velocities
        joint_velocities = [] 
        joint_velocities_keys = ['joint1_velocity', 'joint2_velocity', 'joint3_velocity',
                                 'joint4_velocity', 'joint5_velocity', 'joint6_velocity', 'joint7_velocity']
        for velocity in joint_velocities_keys:
            joint_velocities.append(rs_state[velocity])
        joint_velocities = np.array(joint_velocities)
        joint_velocities = self.panda.normalize_velocity_values(velocites=joint_velocities)
        #print(joint_velocities)


        # Joint Efforts
        joint_efforts = [] 
        joint_efforts_keys = ['joint1_effort', 'joint2_effort', 'joint3_effort',
                              'joint4_effort', 'joint5_effort', 'joint6_effort', 'joint7_effort']
        for effort in joint_efforts_keys:
            joint_efforts.append(rs_state[effort])
        joint_efforts = np.array(joint_efforts)

        # Compose environment state
        state = np.concatenate((joint_positions, joint_velocities, joint_efforts))
        # state = np.concatenate((joint_positions, joint_velocities))
        #print('state is:', state)
        return state.astype(np.float32)

    def _get_observation_space(self) -> gym.spaces.Box:
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """
        # Joint position range tolerance
        pos_tolerance = np.full(7, 0.1)

        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(7, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(7, -1.0), pos_tolerance)
        # Joint velocities range
        max_joint_velocities = np.array([np.inf] * 7)
        min_joint_velocities = -np.array([np.inf] * 7)
        # Joint effort range
        max_joint_efforts = np.array([np.inf] * 7)
        min_joint_efforts = -np.array([np.inf] * 7)
        # Definition of environment observation_space
        max_obs = np.concatenate((max_joint_positions, max_joint_velocities, max_joint_efforts))
        min_obs = np.concatenate((min_joint_positions, min_joint_velocities, min_joint_efforts))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_action_space(self) -> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        fixed_joints = [self.fix_joint1, self.fix_joint2, self.fix_joint3, self.fix_joint4, self.fix_joint5, self.fix_joint6, self.fix_joint7]
        num_control_joints = len(fixed_joints) - sum(fixed_joints)

        return gym.spaces.Box(low=np.full(num_control_joints, -1.0), high=np.full(num_control_joints, 1.0), dtype=np.float32)


class EmptyEnvironmentPandaSim(PandaBaseEnv, Simulation):
    cmd = "roslaunch panda_robot_server panda_robot_server.launch \
        world_name:=empty_no_gravity.world \
        reference_frame:=world \
        rviz_gui:=false \
        gazebo_gui:=true \
        rs_mode:=only_robot "

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, panda_model='panda', action_mode=PandaBaseEnv.ACTION_MODE_ABS_POS, action_cycle_rate = 250, max_velocity_scale_factor = 0.1, **kwargs):
        self.cmd = self.cmd + ' action_mode:=' + action_mode + ' action_cycle_rate:=' + str(action_cycle_rate) + ' max_velocity_scale_factor:=' + str(max_velocity_scale_factor) + ' '
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        PandaBaseEnv.__init__(self, rs_address=self.robot_server_ip, panda_model=panda_model, action_mode=action_mode, action_cycle_rate=action_cycle_rate, max_velocity_scale_factor=max_velocity_scale_factor, **kwargs)


class EmptyEnvironmentPandaRob(PandaBaseEnv):
    real_robot = True

# roslaunch panda_robot_server panda_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 rs_mode:=moving
