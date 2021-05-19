#!/usr/bin/env python3
import copy
import numpy as np
import gym
from scipy.spatial.transform import Rotation as R
from robo_gym.utils import utils
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
from robo_gym.envs.ur.ur_base_env import URBaseEnv


DEBUG = True
# base, shoulder, elbow, wrist_1, wrist_2, wrist_3
JOINT_POSITIONS = [0.0, -2.5, 1.5, 0, -1.4, 0]
class URBaseAvoidanceEnv(URBaseEnv):
    """Universal Robots UR avoidance base environment.

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
    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, ur_model='ur5', include_polar_to_elbow=False, rs_state_to_info=True, **kwargs):
        self.include_polar_to_elbow = include_polar_to_elbow
        super().__init__(rs_address, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model)
        
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        # Set initial state of the Robot Server
        if fixed_object_position:
            # Object in a fixed position
            string_params = {"object_0_function": "fixed_position"}
            float_params = {"object_0_x": fixed_object_position[0], 
                            "object_0_y": fixed_object_position[1], 
                            "object_0_z": fixed_object_position[2]}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, 
                                            string_params = string_params, state_dict = rs_state)
        return state_msg

    def reset(self, joint_positions = JOINT_POSITIONS, fixed_object_position = None) -> np.array:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            fixed_object_position (list[3]): x,y,z fixed position of object

        Returns:
            np.array: Environment state.

        """        
        self.elapsed_steps = 0

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        state = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)

        # Initialize desired joint positions
        if joint_positions: 
            assert len(joint_positions) == 6
        else:
            joint_positions = JOINT_POSITIONS

        # Set initial robot joint positions
        self._set_joint_positions(joint_positions)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state, fixed_object_position)

        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

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

        return state

    def _robot_server_state_to_env_state(self, rs_state) -> np.array:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Object polar coordinates
        # Transform cartesian coordinates of object to polar coordinates 
        # with respect to the end effector frame
        object_coord = np.array([
            rs_state['object_0_to_ref_translation_x'], 
            rs_state['object_0_to_ref_translation_y'],
            rs_state['object_0_to_ref_translation_z']])

        ee_to_ref_frame_translation = np.array([
            rs_state['ee_to_ref_translation_x'], 
            rs_state['ee_to_ref_translation_y'],
            rs_state['ee_to_ref_translation_z']])

        ee_to_ref_frame_quaternion = np.array([
            rs_state['ee_to_ref_rotation_x'], 
            rs_state['ee_to_ref_rotation_y'],
            rs_state['ee_to_ref_rotation_z'],
            rs_state['ee_to_ref_rotation_w']])

        ee_to_ref_frame_rotation = R.from_quat(ee_to_ref_frame_quaternion)
        ref_frame_to_ee_rotation = ee_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_ee_quaternion = ref_frame_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_ee_translation = -ref_frame_to_ee_rotation.apply(ee_to_ref_frame_translation)

        object_coord_ee_frame = utils.change_reference_frame(object_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        object_polar = utils.cartesian_to_polar_3d(object_coord_ee_frame)


        # Joint positions 
        joint_positions = []
        joint_positions_keys = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                            'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position']
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        # Normalize joint position values
        joint_positions = self.ur.normalize_joint_values(joints=joint_positions)

        # joint positions at start
        starting_joints = self.ur.normalize_joint_values(self._get_joint_positions_as_array())
        # difference in position from start to current
        delta_joints = joint_positions - starting_joints
        
        # Transform cartesian coordinates of object to polar coordinates 
        # with respect to the forearm
        forearm_to_ref_frame_translation = np.array([
            rs_state['forearm_to_ref_translation_x'], 
            rs_state['forearm_to_ref_translation_y'],
            rs_state['forearm_to_ref_translation_z']])

        forearm_to_ref_frame_quaternion = np.array([
            rs_state['forearm_to_ref_rotation_x'], 
            rs_state['forearm_to_ref_rotation_y'],
            rs_state['forearm_to_ref_rotation_z'],
            rs_state['forearm_to_ref_rotation_w']])
        forearm_to_ref_frame_rotation = R.from_quat(forearm_to_ref_frame_quaternion)
        ref_frame_to_forearm_rotation = forearm_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_forearm_quaternion = ref_frame_to_forearm_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_forearm_translation = -ref_frame_to_forearm_rotation.apply(forearm_to_ref_frame_translation)

        object_coord_forearm_frame = utils.change_reference_frame(object_coord,ref_frame_to_forearm_translation,ref_frame_to_forearm_quaternion)
        object_polar_forearm = utils.cartesian_to_polar_3d(object_coord_forearm_frame)

        # Compose environment state
        if self.include_polar_to_elbow:
            state = np.concatenate((object_polar, joint_positions, delta_joints, object_polar_forearm))
        else:
            state = np.concatenate((object_polar, joint_positions, delta_joints, np.zeros(3)))

        return state
    

    def _get_observation_space(self) -> gym.spaces.Box:
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

        # Definition of environment observation_space
        if self.include_polar_to_elbow:
            max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, target_range))
            min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, -target_range))
        else:
            max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, np.zeros(3)))
            min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, np.zeros(3)))


        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

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
        rs_action = copy.deepcopy(action)

        joint_positions = self._get_joint_positions_as_array() + action

        rs_action = self.ur._ur_joint_list_to_ros_joint_list(joint_positions)

        return rs_action   

    def _get_robot_server_state_len(self) -> int:

        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """
        return len(self.get_robot_server_composition())  
    
    def get_robot_server_composition(self) -> list:
        rs_state_keys = [
            'object_0_to_ref_translation_x', 
            'object_0_to_ref_translation_y',
            'object_0_to_ref_translation_z',
            'object_0_to_ref_rotation_x',
            'object_0_to_ref_rotation_y',
            'object_0_to_ref_rotation_z',
            'object_0_to_ref_rotation_w',

            'base_joint_position',
            'shoulder_joint_position',
            'elbow_joint_position',
            'wrist_1_joint_position',
            'wrist_2_joint_position',
            'wrist_3_joint_position',

            'base_joint_velocity',
            'shoulder_joint_velocity',
            'elbow_joint_velocity',
            'wrist_1_joint_velocity',
            'wrist_2_joint_velocity',
            'wrist_3_joint_velocity',

            'ee_to_ref_translation_x',
            'ee_to_ref_translation_y',
            'ee_to_ref_translation_z',
            'ee_to_ref_rotation_x',
            'ee_to_ref_rotation_y',
            'ee_to_ref_rotation_z',
            'ee_to_ref_rotation_w',

            'forearm_to_ref_translation_x',
            'forearm_to_ref_translation_y',
            'forearm_to_ref_translation_z',
            'forearm_to_ref_rotation_x',
            'forearm_to_ref_rotation_y',
            'forearm_to_ref_rotation_z',
            'forearm_to_ref_rotation_w',

            'in_collision']
        return rs_state_keys









