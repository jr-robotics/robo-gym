from __future__ import annotations
import copy
import numpy as np
import gymnasium as gym
from typing import Tuple, Any
from scipy.spatial.transform import Rotation as R
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
from robo_gym.utils import utils
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.envs.panda.panda_base_env import PandaBaseEnv

import matplotlib.pyplot as plt

# joint1, joint2, joint3, joint4, joint5,joint6, joint7
JOINT_POSITIONS = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315,
                   0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
RANDOM_JOINT_OFFSET = [1.5, 0.25, 0.5, 1.0, 0.4, 3.14, 1.]  
# distance to target that need to be reached
DISTANCE_THRESHOLD = 0.1

class EndEffectorPositioningPanda(PandaBaseEnv):
    """Panda end effector positioning environment.

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
    Attributes:
        panda (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.successful_ending = False
        self.last_position = np.zeros(7)

        self.episode_counter = 0
        self.log_current_data = None
        self.current_target_pose = None
        self.log_test_data = np.empty((0, 4), int)

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
        # Target coordinates range
        target_range = np.full(3, np.inf)
        # Joint velocities range
        max_joint_velocities = np.array([np.inf] * 7)
        min_joint_velocities = - np.array([np.inf] * 7)
        # Joint efforts range
        max_joint_efforts = np.array([np.inf] * 7)
        min_joint_efforts = - np.array([np.inf] * 7)
        # Cartesian coords of the target location
        max_target_coord = np.array([np.inf] * 3)
        min_target_coord = - np.array([np.inf] * 3)
        # Cartesian coords of the end effector
        max_ee_coord = np.array([np.inf] * 3)
        min_ee_coord = - np.array([np.inf] * 3)
        # Previous action
        max_action = np.array([1.01] * 7)
        min_action = - np.array([1.01] * 7)
        # Definition of environment observation_space
        max_obs = np.concatenate(
            (target_range, max_joint_positions, max_joint_velocities, max_joint_efforts, max_target_coord, max_ee_coord, max_action))
        min_obs = np.concatenate(
            (-target_range, min_joint_positions, min_joint_velocities, min_joint_efforts, min_target_coord, min_ee_coord, min_action))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _set_initial_robot_server_state(self, rs_state, ee_target_pose) -> robot_server_pb2.State:
    
    	# set the target position
        string_params = {"object_0_function": "fixed_position"}
        float_params = {"object_0_x": ee_target_pose[0],
                        "object_0_y": ee_target_pose[1],
                        "object_0_z": ee_target_pose[2]}
        state = {}

        state_msg = robot_server_pb2.State(state=state, float_params=float_params, string_params=string_params,
                                           state_dict=rs_state)
        return state_msg

    def _robot_server_state_to_env_state(self, rs_state) -> np.ndarray:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Target polar coordinates
        # Transform cartesian coordinates of target to polar coordinates
        # with respect to the end effector frame
        target_coord = np.array([
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

        target_coord_ee_frame = utils.change_reference_frame(target_coord, ref_frame_to_ee_translation,
                                                             ref_frame_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

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

        # Joint Efforts
        joint_efforts = []
        joint_efforts_keys = ['joint1_effort', 'joint2_effort', 'joint3_effort',
                              'joint4_effort', 'joint5_effort', 'joint6_effort', 'joint7_effort']
        for effort in joint_efforts_keys:
            joint_efforts.append(rs_state[effort])
        joint_efforts = np.array(joint_efforts)

        # Compose environment state
        state = np.concatenate((target_polar, joint_positions, joint_velocities, joint_efforts, target_coord,
                                ee_to_ref_frame_translation, self.previous_action))

        return state.astype(np.float32)

    def get_robot_server_composition(self) -> list:
        rs_state_keys = [
            'object_0_to_ref_translation_x',
            'object_0_to_ref_translation_y',
            'object_0_to_ref_translation_z',
            'object_0_to_ref_rotation_x',
            'object_0_to_ref_rotation_y',
            'object_0_to_ref_rotation_z',
            'object_0_to_ref_rotation_w',

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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Environment reset.

            options:
                joint_positions (list[7] or np.array[7]): robot joint positions in radians.
                ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.
                randomize_start (bool): if True the starting position is randomized defined by the RANDOM_JOINT_OFFSET
                continue_on_success (bool): if True the next robot will continue from it current position when last episode was a success

            Returns:
                np.array: Environment state.
                dict: info

        """
        super(PandaBaseEnv, self).reset(seed=seed, options=options)

        if options is None:
            options = {}
        joint_positions = options["joint_positions"] if "joint_positions" in options else None
        ee_target_pose = options["ee_target_pose"] if "ee_target_pose" in options else None
        randomize_start = options["randomize_start"] if "randomize_start" in options else None
        continue_on_success = options["continue_on_success"] if "continue_on_success" in options else None
        
        if joint_positions:
            assert len(joint_positions) == 7
        else:
            joint_positions = JOINT_POSITIONS

        self.elapsed_steps = 0
        self.previous_action = np.zeros(7)

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        state = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)

        # Randomize initial robot joint positions
        if randomize_start:
            joint_positions_low = np.array(joint_positions) - np.array(RANDOM_JOINT_OFFSET)
            joint_positions_high = np.array(joint_positions) + np.array(RANDOM_JOINT_OFFSET)
            joint_positions = np.random.default_rng().uniform(low=joint_positions_low, high=joint_positions_high)

        # Continue from last position if last episode was a success
        if self.successful_ending and continue_on_success:
            joint_positions = self.last_position

        # Set initial robot joint positions
        self._set_joint_positions(joint_positions)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set target End Effector pose
        if ee_target_pose:
            assert len(ee_target_pose) == 7
        else:
            ee_target_pose = self._get_target_pose()

        self.current_target_pose = np.reshape(ee_target_pose[0:3], (1, 3))
        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state, ee_target_pose)

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

        return state, {}

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        if type(action) == list: action = np.array(action)

        action = action.astype(np.float32)
        #print('action is:', action)

        state, reward, done, _, info = super().step(action)
        #print('state is:', state)
        self.previous_action = self.add_fixed_joints(action)

        if done:
            self.episode_counter += 1
            # Logging the target position data and the final status for evaluation
            self.log_test_data = np.append(self.log_test_data, self.log_current_data, axis=0)
            np.save('target_coordinate_and_final_status_data.npy', self.log_test_data)

            print('episode number:', self.episode_counter)
            if info['final_status'] == 'success':
                self.successful_ending = True

                joint_positions = []
                joint_positions_keys = ['joint1_position', 'joint2_position', 'joint3_position',
                                        'joint4_position', 'joint5_position', 'joint6_position', 'joint7_position']

                for position in joint_positions_keys:
                    joint_positions.append(self.rs_state[position])
                joint_positions = np.array(joint_positions)
                self.last_position = joint_positions

        return state, reward, done, False, info

    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        reward = 0
        done = False
        info = {}

        # Reward weight for reaching the goal position
        g_w = 2
        # Reward weight for collision (ground, table or self)
        c_w = -1
        # Reward weight according to the distance to the goal
        d_w = -0.005

        # Calculate distance to the target
        target_coord = np.array([rs_state['object_0_to_ref_translation_x'], rs_state['object_0_to_ref_translation_y'],
                                 rs_state['object_0_to_ref_translation_z']])

        ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'],
                             rs_state['ee_to_ref_translation_z']])
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)

        # Reward base
        reward += d_w * euclidean_dist_3d

        if euclidean_dist_3d <= DISTANCE_THRESHOLD:
            reward = g_w * 1
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord
            self.log_current_data = np.append(1, self.current_target_pose)
            self.log_current_data = np.append(np.array([[2]]), self.current_target_pose, axis=1)

        if rs_state['in_collision']:
            reward = c_w * 1
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord
            self.log_current_data = np.append(np.array([[0]]), self.current_target_pose, axis=1)

        elif self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            info['target_coord'] = target_coord
            self.log_current_data = np.append(np.array([[1]]), self.current_target_pose, axis=1)

        return reward, done, info

    def _get_target_pose(self) -> np.ndarray:
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.
            
      	For generating samples in entire workspace:    return self.panda.get_random_workspace_pose() 
      	For generationg samples only infront of the robot :    return self.panda.get_random_pose_in_front()

        """
        #return self.panda.get_random_workspace_pose() 
        return self.panda.get_random_pose_in_front()


class EndEffectorPositioningPandaSim(EndEffectorPositioningPanda, Simulation):
    cmd = "roslaunch panda_robot_server panda_robot_server.launch \
        world_name:=tabletop_sphere50_no_collision_no_gravity.world \
        reference_frame:=world \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        rs_mode:=1object \
        n_objects:=1.0 \
        object_0_model_name:=sphere50_no_collision \
        object_0_frame:=target "

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, panda_model='panda', action_mode=PandaBaseEnv.ACTION_MODE_ABS_POS, action_cycle_rate = 250, max_velocity_scale_factor = 0.1, **kwargs):
        self.cmd = self.cmd + ' action_mode:=' + action_mode + ' action_cycle_rate:=' + str(action_cycle_rate) + ' max_velocity_scale_factor:=' + str(max_velocity_scale_factor) + ' '
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        EndEffectorPositioningPanda.__init__(self, rs_address=self.robot_server_ip, panda_model=panda_model, action_mode=action_mode, action_cycle_rate=action_cycle_rate, max_velocity_scale_factor=max_velocity_scale_factor, **kwargs)


class EndEffectorPositioningPandaRob(EndEffectorPositioningPanda):
    real_robot = True

# roslaunch ur_robot_server ur_robot_server.launch ur_model:=ur5 real_robot:=true rviz_gui:=true gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 objects_controller:=true rs_mode:=1object n_objects:=1.0 object_0_frame:=target
