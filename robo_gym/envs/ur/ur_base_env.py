#!/usr/bin/env python3
import copy
import numpy as np
import gym
from typing import Tuple
from robo_gym.utils import ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation


# TODO: test live share
# TODO: claudia email
# TODO: remove env state len function is not used anywhere
# TODO: fix composition in the base env. How do i control what i will receive?
# TODO: fix names for a flat dictionary representing the rs state
# TODO: remove the function "get_env_state_len"
# TODO: this grpc thing can only handle float types right?

# TODO: insert check of all necessary keys are present on submit

# TODO: common name for target and obstacle or different name for them?

rs_state_keys = dict.fromkeys([
    'target_x',
    'target_y',
    'target_z',
    !'target_y',
    !'target_y',
    !'target_y',
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
    !'ee_frame_x',
    !'ee_frame_x',
    !'ee_frame_x',
    !'ee_frame_x',
    !'ee_frame_x',
    !'ee_frame_x',
    !'ee_frame_x',
    !'ee_frame_x',
    'in_collision'
])

rs_state = {
    'target_x': 0.0,
    'target_y': 0.0,
    'target_z': 0.0,
    !'target_y': 0.0,
    !'target_y': 0.0,
    !'target_y': 0.0,
    'base_joint_position': 0.0,
    'shoulder_joint_position': 0.0,
    'elbow_joint_position': 0.0,
    'wrist_1_joint_position': 0.0,
    'wrist_2_joint_position': 0.0,
    'wrist_3_joint_position': 0.0,
    'base_joint_velocity': 0.0,
    'shoulder_joint_velocity': 0.0,
    'elbow_joint_velocity': 0.0,
    'wrist_1_joint_velocity': 0.0,
    'wrist_2_joint_velocity': 0.0,
    'wrist_3_joint_velocity': 0.0,
    !'ee_frame_x': 0.0,
    !'ee_frame_x': 0.0,
    !'ee_frame_x': 0.0,
    !'ee_frame_x': 0.0,
    !'ee_frame_x': 0.0,
    !'ee_frame_x': 0.0,
    !'ee_frame_x': 0.0,
    !'ee_frame_x': 0.0,
    'in_collision': 0.0
}
target = [0.0]*6
ur_j_pos = [0.0]*6
ur_j_vel = [0.0]*6
ee_to_ref_frame_transform = [0.0]*7
ur_collision = [0.0]
rs_state = target + ur_j_pos + ur_j_vel + ee_to_ref_frame_transform + ur_collision



JOINT_POSITIONS = [0.0, -2.5, 1.5, 0, -1.4, 0]
class URBaseEnv(gym.Env):
    """Universal Robots UR base environment.

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
    real_robot = False
    max_episode_steps = 300

    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, ur_model='ur5', **kwargs):
        self.ur = ur_utils.UR(model=ur_model)
        self.elapsed_steps = 0

        self.fix_base = fix_base
        self.fix_shoulder = fix_shoulder
        self.fix_elbow = fix_elbow
        self.fix_wrist_1 = fix_wrist_1
        self.fix_wrist_2 = fix_wrist_2
        self.fix_wrist_3 = fix_wrist_3

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur.get_max_joint_positions()
        self.last_action = None
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")


    def _set_initial_robot_server_state(self, rs_state) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}

        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        return state_msg

    def reset(self, joint_positions = None) -> np.array:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.
        
        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

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

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state)
        
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
        
        # Check if current position is in the range of the initial joint positions
        joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        if not np.isclose(joint_positions, self.joint_positions, atol=0.1).all():
            raise InvalidStateError('Reset joint positions are not within defined range')
            
        return self.state

    def _reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        done = False
        info = {}

        # Check if robot is in collision
        collision = True if rs_state[25] == 1 else False
        if collision:
            done = True
            info['final_status'] = 'collision'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            
        
        return 0, done, info

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

        return self.state, reward, done, info

    def render():
        pass

    def _get_robot_server_state_len(self) -> int:
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """
        #TODO: remove target from rs state
        target = [0.0]*6
        ur_j_pos = [0.0]*6
        ur_j_vel = [0.0]*6
        ee_to_ref_frame_transform = [0.0]*7
        ur_collision = [0.0]
        rs_state = target + ur_j_pos + ur_j_vel + ee_to_ref_frame_transform + ur_collision

        return len(rs_state)

    def _get_env_state_len(self) -> int:
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """
        ur_j_pos = [0.0]*6
        ur_j_vel = [0.0]*6
        env_state = ur_j_pos + ur_j_vel

        return len(env_state)

    def _set_joint_positions(self, joint_positions) -> None:
        """Set desired robot joint positions with standard indexing."""
        assert len(joint_positions) == 6
        self.joint_positions = copy.deepcopy(joint_positions)

    def _get_joint_positions(self) -> np.array:
        """Get robot joint positions with standard indexing."""
        return np.array(self.joint_positions)

    def _get_target_pose(self) -> np.array:
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        return self.ur.get_random_workspace_pose()

    def _robot_server_state_to_env_state(self, rs_state) -> np.array:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        ur_j_vel = self.ur._ros_joint_list_to_ur_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # Compose environment state
        state = np.concatenate((ur_j_pos_norm, ur_j_vel))

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
        # Joint velocities range 
        max_joint_velocities = np.array([np.inf] * 6)
        min_joint_velocities = -np.array([np.inf] * 6)
        # Definition of environment observation_space
        max_obs = np.concatenate((max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((min_joint_positions, min_joint_velocities))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
    
    def _get_action_space(self)-> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        fixed_joints = [self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_1, self.fix_wrist_2, self.fix_wrist_3]
        num_control_joints = len(fixed_joints) - sum(fixed_joints)

        return gym.spaces.Box(low=np.full((num_control_joints), -1.0), high=np.full((num_control_joints), 1.0), dtype=np.float32)


# TODO: remove object target
class EmptyEnvironmentURSim(URBaseEnv, Simulation):
    cmd = "roslaunch ur_robot_server ur_robot_server.launch \
        world_name:=tabletop_sphere50.world \
        yaw:=-0.78 \
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1object \
        n_objects:=1.0 \
        object_0_model_name:=sphere50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur5', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        URBaseEnv.__init__(self, rs_address=self.robot_server_ip, ur_model=ur_model, **kwargs)

class EmptyEnvironmentURRob(URBaseEnv):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving
