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
    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, ur_model='ur5', include_polar_to_elbow=False, **kwargs):
        self.include_polar_to_elbow = include_polar_to_elbow
        super().__init__(rs_address, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model)
        
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}

        # Set initial state of the Robot Server
        if fixed_object_position:
            # Object in a fixed position
            string_params = {"object_0_function": "fixed_position"}
            float_params = {"object_0_x": fixed_object_position[0], 
                            "object_0_y": fixed_object_position[1], 
                            "object_0_z": fixed_object_position[2]}
            state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
            return state_msg

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

        # TODO: reorder as soon as rs state is a dict
        # Compose environment state
        if self.include_polar_to_elbow:
            state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints, target_polar_forearm))
        else:
            state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints, np.zeros(3)))

        return state

    def _get_env_state_len(self) -> int:
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """
        object_polar_coords_ee = [0.0]*3
        ur_j_pos = [0.0]*6
        ur_j_delta = [0.0]*6
        object_polar_coords_elbow = [0.0]*3
        env_state = object_polar_coords_ee + ur_j_pos + ur_j_delta + object_polar_coords_elbow

        return len(env_state)

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
        action = self.add_fixed_joints(action)
        
        # TODO remove from here later
        if self.last_action is None:
            self.last_action = action

        rs_action = copy.deepcopy(action)

        joint_positions = self._get_joint_positions() + action

        rs_action = self.ur._ur_joint_list_to_ros_joint_list(joint_positions)

        return action, rs_action   

    def _get_robot_server_state_len(self) -> int:

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










