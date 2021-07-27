import copy
import numpy as np
import gym
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
from robo_gym.utils import utils
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.envs.ur.ur_base_env import URBaseEnv

# base, shoulder, elbow, wrist_1, wrist_2, wrist_3
JOINT_POSITIONS = [0.0, -2.5, 1.5, -1.5, -1.4, 0.0]
RANDOM_JOINT_OFFSET = [1.5, 0.25, 0.5, 1.0, 0.4, 3.14]
# distance to target that need to be reached
DISTANCE_THRESHOLD = 0.1

class ReachAndAvoidUR(URBaseEnv):
    """Universal Robots UR end effector positioning environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        fix_base (bool): Wether or not the base joint stays fixed or is moveable. Defaults to False.
        fix_shoulder (bool): Wether or not the shoulder joint stays fixed or is moveable. Defaults to False.
        fix_elbow (bool): Wether or not the elbow joint stays fixed or is moveable. Defaults to False.
        fix_wrist_1 (bool): Wether or not the wrist 1 joint stays fixed or is moveable. Defaults to False.
        fix_wrist_2 (bool): Wether or not the wrist 2 joint stays fixed or is moveable. Defaults to False.
        fix_wrist_3 (bool): Wether or not the wrist 3 joint stays fixed or is moveable. Defaults to True.
        ur_model (str): determines which ur model will be used in the environment. Default to 'ur5'.

    Attributes:
        ur (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.
    """

    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, ur_model='ur5', rs_state_to_info=True, restrict_wrist_1=True, **kwargs):
        super().__init__(rs_address, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model, rs_state_to_info)
        
        self.restrict_wrist_1 = restrict_wrist_1

        self.successful_ending = False
        self.last_position = np.zeros(6)

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
        # Joint velocities range 
        max_joint_velocities = np.array([np.inf] * 6)
        min_joint_velocities = - np.array([np.inf] * 6)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        # Cartesian coords of the target location
        max_target_coord = np.array([np.inf] * 3)
        min_target_coord = - np.array([np.inf] * 3)
        # Cartesian coords of the end effector
        max_ee_coord = np.array([np.inf] * 3)
        min_ee_coord = - np.array([np.inf] * 3)
        # Previous action
        max_action = np.array([1.01] * 6)
        min_action = - np.array([1.01] * 6)
        # Obstacle polar coordinates to end effector
        max_obstacle_polar = np.full(3, np.inf)
        min_obstacle_polar = -np.full(3, np.inf)
        # Obstacle coords
        max_obstacle_coord = np.full(3, np.inf)
        min_obstacle_coord = -np.full(3, np.inf)

        max_obs = np.concatenate((target_range, max_joint_positions, max_joint_velocities, max_target_coord, max_ee_coord, max_action, max_obstacle_polar, max_obstacle_coord))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_joint_velocities, min_target_coord, min_ee_coord, min_action, min_obstacle_polar, min_obstacle_coord))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _set_initial_robot_server_state(self, rs_state, ee_target_pose, fixed_object_position = None) -> robot_server_pb2.State:
        if fixed_object_position:
            string_params = {"object_0_function": "fixed_position", 
                            "object_1_function": "fixed_position"}
            float_params = {"object_0_x": ee_target_pose[0], 
                            "object_0_y": ee_target_pose[1], 
                            "object_0_z": ee_target_pose[2],
                            "object_1_x": fixed_object_position[0], 
                            "object_1_y": fixed_object_position[1], 
                            "object_1_z": fixed_object_position[2]}
        else: 
            n_sampling_points = int(np.random.default_rng().uniform(low=16000, high=20000))
            
            string_params = {"object_0_function": "fixed_position", 
                            "object_1_function": "3d_spline_ur5_workspace"  }
            float_params = {"object_0_x": ee_target_pose[0], 
                            "object_0_y": ee_target_pose[1], 
                            "object_0_z": ee_target_pose[2],
                            "object_1_x_min": -1.0, 
                            "object_1_x_max": 1.0,
                            "object_1_y_min": -1.0,
                            "object_1_y_max": 1.0,
                            "object_1_z_min": 0.1,
                            "object_1_z_max": 1.0,
                            "object_1_n_points": 10,
                            "object_1_n_sampling_points": n_sampling_points}
        state = {}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, 
                                            string_params = string_params, state_dict = rs_state)
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

        target_coord_ee_frame = utils.change_reference_frame(target_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)


        # Joint positions 
        joint_positions = []
        joint_positions_keys = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                            'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position']
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        # Normalize joint position values
        joint_positions = self.ur.normalize_joint_values(joints=joint_positions)

        # Joint Velocities
        joint_velocities = [] 
        joint_velocities_keys = ['base_joint_velocity', 'shoulder_joint_velocity', 'elbow_joint_velocity',
                            'wrist_1_joint_velocity', 'wrist_2_joint_velocity', 'wrist_3_joint_velocity']
        for velocity in joint_velocities_keys:
            joint_velocities.append(rs_state[velocity])
        joint_velocities = np.array(joint_velocities)

        object_coord = np.array([
            rs_state['object_1_to_ref_translation_x'], 
            rs_state['object_1_to_ref_translation_y'],
            rs_state['object_1_to_ref_translation_z']])

        object_coord_ee_frame = utils.change_reference_frame(object_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        object_polar = utils.cartesian_to_polar_3d(object_coord_ee_frame)



        # Compose environment state
        state = np.concatenate((target_polar, joint_positions, joint_velocities, target_coord, ee_to_ref_frame_translation, self.previous_action, object_polar, object_coord))
        
        return state

    def get_robot_server_composition(self) -> list:
        rs_state_keys = [
            'object_0_to_ref_translation_x', 
            'object_0_to_ref_translation_y',
            'object_0_to_ref_translation_z',
            'object_0_to_ref_rotation_x',
            'object_0_to_ref_rotation_y',
            'object_0_to_ref_rotation_z',
            'object_0_to_ref_rotation_w',

            'object_1_to_ref_translation_x', 
            'object_1_to_ref_translation_y',
            'object_1_to_ref_translation_z',
            'object_1_to_ref_rotation_x',
            'object_1_to_ref_rotation_y',
            'object_1_to_ref_rotation_z',
            'object_1_to_ref_rotation_w',

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


    def reset(self, joint_positions = JOINT_POSITIONS, ee_target_pose = None, randomize_start=False, continue_on_success=False, fixed_object_position = None) -> np.ndarray:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.
            randomize_start (bool): if True the starting position is randomized defined by the RANDOM_JOINT_OFFSET
            continue_on_success (bool): if True the next robot will continue from it current position when last episode was a success
        """
        if joint_positions: 
            assert len(joint_positions) == 6
        else:
            joint_positions = JOINT_POSITIONS

        self.elapsed_steps = 0
        self.previous_action = np.zeros(6)

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
            assert len(ee_target_pose) == 6
        else:
            ee_target_pose = self._get_target_pose()

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
            
        self.successful_ending = False
        return state

    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        done = False
        info = {}

        # Check if robot is in collision
        collision = True if rs_state['in_collision'] == 1 else False
        if collision:
            done = True
            info['final_status'] = 'collision'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            
        
        return 0, done, info

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list: action = np.array(action)
        
        state, reward, done, info = super().step(action)
        self.previous_action = self.add_fixed_joints(action)

        if done:
            if info['final_status'] == 'success':
                self.successful_ending = True

                joint_positions = []
                joint_positions_keys = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                                        'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position']

                for position in joint_positions_keys:
                    joint_positions.append(self.rs_state[position])
                joint_positions = np.array(joint_positions)
                self.last_position = joint_positions
        
        return state, reward, done, info

    def env_action_to_rs_action(self, action) -> np.array:
        """Convert environment action to Robot Server action"""
        rs_action = copy.deepcopy(action)

        if self.restrict_wrist_1:
            min_action = -1
            max_action = 1
            max_wrist1 = 0.31
            min_wrist1 = -3.48
            wrist1 = (((rs_action[3] - min_action) * (max_wrist1 - min_wrist1)) / (max_action- min_action)) + min_wrist1
            # Scale action
            rs_action = np.multiply(rs_action, self.abs_joint_pos_range)
            rs_action[3] = wrist1
            # Convert action indexing from ur to ros
            rs_action = self.ur._ur_joint_list_to_ros_joint_list(rs_action)
        else:
            rs_action = super().env_action_to_rs_action(rs_action)

        return rs_action

    def _get_target_pose(self) -> np.array:
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        return self.ur.get_random_workspace_pose()



class ReachAndAvoidURSim(ReachAndAvoidUR, Simulation):
    cmd = "roslaunch ur_robot_server ur_robot_server.launch \
        world_name:=tabletop_sphere50_green_sphere50_no_collision.world \
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.1 \
        action_cycle_rate:=10 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        rs_mode:=2objects2points \
        n_objects:=2.0 \
        object_0_model_name:=green_sphere50_no_collision \
        object_0_frame:=target \
        object_1_model_name:=sphere50 \
        object_1_frame:=obstacle1"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur5', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ReachAndAvoidUR.__init__(self, rs_address=self.robot_server_ip, ur_model=ur_model, **kwargs)

class ReachAndAvoidURRob(ReachAndAvoidUR):
    real_robot = True 

# roslaunch ur_robot_server ur_robot_server.launch ur_model:=ur5 real_robot:=true rviz_gui:=true gui:=true reference_frame:=base_link max_velocity_scale_factor:=0.1 action_cycle_rate:=10 rs_mode:=moving
