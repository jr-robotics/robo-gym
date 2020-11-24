#!/usr/bin/env python3

from copy import deepcopy
import sys, math, copy, random
import numpy as np
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

class UR5Env(gym.Env):
    """Universal Robots UR5 base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        ur5 (:obj:): Robot utilities object.
        observation_space (:obj:): Environment observation space.
        action_space (:obj:): Environment action space.
        distance_threshold (float): Minimum distance (m) from target to consider it reached.
        abs_joint_pos_range (np.array): Absolute value of joint positions range`.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False
    max_episode_steps = 300

    def __init__(self, rs_address=None, **kwargs):
        self.ur5 = ur_utils.UR5()
        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.seed()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur5.get_max_joint_positions()
        self.initial_joint_positions_low = np.zeros(6)
        self.initial_joint_positions_high = np.zeros(6)
        self.last_position_on_success = []
        self.prev_rs_state = None
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, initial_joint_positions = None, ee_target_pose = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None
        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        

        print(self.last_position_on_success)

        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            ur5_initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            ur5_initial_joint_positions = self.last_position_on_success
        else:
            ur5_initial_joint_positions = self._get_initial_joint_positions()

        rs_state[6:12] = self.ur5._ur_5_joint_list_to_ros_joint_list(ur5_initial_joint_positions)

        # Set target End Effector pose
        if ee_target_pose:
            assert len(ee_target_pose) == 6
        else:
            ee_target_pose = self._get_target_pose()

        rs_state[0:6] = ee_target_pose

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.tolist())
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
            joint_positions = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
            tolerance = 0.1
            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    raise InvalidStateError('Reset joint positions are not within defined range')


        # go one empty action and check if there is a collision
        if not self.real_robot:
            action = self.state[3:3+len(self.action_space.sample())]
            _, _, done, info = self.step(action)
            self.elapsed_steps = 0
            if done:
                raise InvalidStateError('Reset started in a collision state')
            
        return self.state

    def _reward(self, rs_state, action):
        return 0, False, {}

    def step(self, action):
        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action = np.multiply(rs_action, self.abs_joint_pos_range)
        # Convert action indexing from ur5 to ros
        rs_action = self.ur5._ur_5_joint_list_to_ros_joint_list(rs_action)
        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
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
        ee_to_base_transform = [0.0]*7
        ur_collision = [0.0]
        rs_state = target + ur_j_pos + ur_j_vel + ee_to_base_transform + ur_collision

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

    def _set_initial_joint_positions_range(self):
        self.initial_joint_positions_low = np.array([-0.65, -2.75, 1.0, -3.14, -1.7, -3.14])
        self.initial_joint_positions_high = np.array([0.65, -2.0, 2.5, 3.14, -1.0, 3.14])

    def _get_initial_joint_positions(self):
        """Generate random initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        self._set_initial_joint_positions_range()
        # Random initial joint positions
        joint_positions = np.random.default_rng().uniform(low=self.initial_joint_positions_low, high=self.initial_joint_positions_high)

        return joint_positions

    def _get_target_pose(self):
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """

        return self.ur5.get_random_workspace_pose()

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
        
        ee_to_base_translation = rs_state[18:21]
        ee_to_base_quaternion = rs_state[21:25]
        ee_to_base_rotation = R.from_quat(ee_to_base_quaternion)
        base_to_ee_rotation = ee_to_base_rotation.inv()
        base_to_ee_quaternion = base_to_ee_rotation.as_quat()
        base_to_ee_translation = - ee_to_base_translation

        target_coord_ee_frame = utils.change_reference_frame(target_coord,base_to_ee_translation,base_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
        ur_j_vel = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur5.normalize_joint_values(joints=ur_j_pos)

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
        # Joint positions range tolerance
        vel_tolerance = np.full(6,0.5)
        # Joint velocities range used to determine if there is an error in the sensor readings
        max_joint_velocities = np.add(self.ur5.get_max_joint_velocities(), vel_tolerance)
        min_joint_velocities = np.subtract(self.ur5.get_min_joint_velocities(), vel_tolerance)
        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_joint_velocities))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
    
    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.full((6), -1.0), high=np.full((6), 1.0), dtype=np.float32)

class EndEffectorPositioningUR5(UR5Env):

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)

        # Reward base
        reward = -1 * euclidean_dist_3d
        
        joint_positions = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
        joint_positions_normalized = self.ur5.normalize_joint_values(joint_positions)
        delta = np.abs(np.subtract(joint_positions_normalized, action))
        reward = reward - (0.05 * np.sum(delta))

        if euclidean_dist_3d <= self.distance_threshold:
            reward = 100
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord

        # Check if robot is in collision
        if rs_state[25] == 1:
            collision = True
        else:
            collision = False

        if collision:
            reward = -400
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            info['target_coord'] = target_coord

        return reward, done, info
    
    
class EndEffectorPositioningUR5DoF5(UR5Env):
        
    def reset(self, initial_joint_positions = None, ee_target_pose = None, type='continue'):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.
            type (random or continue):
                random: reset at a random position within the range defined in _set_initial_joint_positions_range
                continue: if the episode terminated with success the robot starts the next episode from this location, otherwise it starts at a random position

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None
        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        print(self.last_position_on_success)

        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            ur5_initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            ur5_initial_joint_positions = self.last_position_on_success
        else:
            ur5_initial_joint_positions = self._get_initial_joint_positions()

        print(ur5_initial_joint_positions)
        rs_state[6:12] = self.ur5._ur_5_joint_list_to_ros_joint_list(ur5_initial_joint_positions)
        # Set target End Effector pose
        if ee_target_pose:
            assert len(ee_target_pose) == 6
        else:
            ee_target_pose = self._get_target_pose()

        rs_state[0:6] = ee_target_pose

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.tolist())
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
            joint_positions = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
            tolerance = 0.1
            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    raise InvalidStateError('Reset joint positions are not within defined range')

        # go one empty action and check if there is a collision
        if not self.real_robot:
            action = self.state[3:3+len(self.action_space.sample())]
            _, _, done, _ = self.step(action)
            self.elapsed_steps = 0
            if done:
                raise InvalidStateError('Reset started in a collision state')
            
        return self.state


    def _set_initial_joint_positions_range(self):
        self.initial_joint_positions_low = np.array([-0.65, -2.75, 1.0, -3.14, -1.7, 0.0])
        self.initial_joint_positions_high = np.array([0.65, -2.0, 2.5, 3.14, -1.3, 0.0])

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)

        # Reward base
        reward = -1 * euclidean_dist_3d
        
        joint_positions = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
        joint_positions_normalized = self.ur5.normalize_joint_values(copy.deepcopy(joint_positions))
        
        delta = np.abs(np.subtract(joint_positions_normalized, action))
        reward = reward - (0.05 * np.sum(delta))

        if euclidean_dist_3d <= self.distance_threshold:
            
            reward = 100
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord
            self.last_position_on_success = joint_positions
            
        # Check if robot is in collision
        if rs_state[25] == 1:
            collision = True
        else:
            collision = False

        if collision:
            reward = -400
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord
            self.last_position_on_success = []

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            info['target_coord'] = target_coord
            self.last_position_on_success = []

        return reward, done, info

    def step(self, action):
        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action =  np.append(action, [0.0])

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action = np.multiply(rs_action, self.abs_joint_pos_range)
        # Convert action indexing from ur5 to ros
        rs_action = self.ur5._ur_5_joint_list_to_ros_joint_list(rs_action)
        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
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
    
    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)


class EndEffectorPositioningUR5Sim(EndEffectorPositioningUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        EndEffectorPositioningUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class EndEffectorPositioningUR5Rob(EndEffectorPositioningUR5):
    real_robot = True


class EndEffectorPositioningUR5DoF5Sim(EndEffectorPositioningUR5DoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        EndEffectorPositioningUR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class EndEffectorPositioningUR5DoF5Rob(EndEffectorPositioningUR5DoF5):
    real_robot = True

class MovingBoxTargetUR5(UR5Env):
    max_episode_steps = 1000
            
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
        # Joint positions range tolerance
        vel_tolerance = np.full(6,0.5)
        # Joint velocities range used to determine if there is an error in the sensor readings
        max_joint_velocities = np.add(self.ur5.get_max_joint_velocities(), vel_tolerance)
        min_joint_velocities = np.subtract(self.ur5.get_min_joint_velocities(), vel_tolerance)

        max_start_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_start_positions = np.subtract(np.full(6, -1.0), pos_tolerance)
        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_joint_velocities, max_start_positions))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_joint_velocities, min_start_positions))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def reset(self, initial_joint_positions = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None
        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            ur5_initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            ur5_initial_joint_positions = self.last_position_on_success
        else:
            ur5_initial_joint_positions = self._get_initial_joint_positions()

        rs_state[6:12] = self.ur5._ur_5_joint_list_to_ros_joint_list(ur5_initial_joint_positions)


        # Set initial state of the Robot Server
        z_amplitude = np.random.default_rng().uniform(low=0.09, high=0.35)
        z_frequency = 0.125
        z_offset = np.random.default_rng().uniform(low=0.2, high=0.6)
        
        string_params = {"function": "triangle_wave"}
        float_params = {"x": 0.13, "y": -0.30, "z_amplitude": z_amplitude, "z_frequency": z_frequency, "z_offset": z_offset}
        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        self.prev_rs_state = copy.deepcopy(rs_state)

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # save start position
        self.start_position = self.state[3:9]

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
            tolerance = 0.1
            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    raise InvalidStateError('Reset joint positions are not within defined range')

        return self.state

    # learns something but is unsatisfactory
    # def _reward(self, rs_state, action):
    #     reward = 0
    #     done = False
    #     info = {}

    #     # Calculate distance to the target
    #     target_coord = np.array(rs_state[0:3])
    #     ee_coord = np.array(rs_state[18:21])
    #     euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
        
    #     wanted_distance = 0.3 # m
    #     threshold = 0.1
    #     # threshold punishment

    #     if (euclidean_dist_3d < wanted_distance + threshold) and (euclidean_dist_3d > wanted_distance - threshold):
    #         reward += .1
    #     elif euclidean_dist_3d  < wanted_distance - threshold:
    #         reward -= .5
    #     elif euclidean_dist_3d > wanted_distance - 3*threshold:
    #         reward -= .05
    #     elif euclidean_dist_3d > wanted_distance - 2*threshold:
    #         reward -= .01

            
    #     # Check if robot is in collision
    #     if rs_state[25] == 1:
    #         collision = True
    #     else:
    #         collision = False

    #     if collision:
    #         reward = -100
    #         done = True
    #         info['final_status'] = 'collision'
    #         info['target_coord'] = target_coord
    #         self.last_position_on_success = []

    #     if self.elapsed_steps >= self.max_episode_steps:
    #         done = True
    #         info['final_status'] = 'max_steps_exceeded'
    #         info['target_coord'] = target_coord
    #         self.last_position_on_success = []

    #     return reward, done, info

    # def _reward(self, rs_state, action):
    #     reward = 0
    #     done = False
    #     info = {}

    #     minimum_distance = 0.3 # m
    #     maximum_distance = 0.6 # m

    #     # Calculate distance to the target
    #     target_coord = np.array(rs_state[0:3])
    #     ee_coord = np.array(rs_state[18:21])
    #     distance_to_target = np.linalg.norm(target_coord - ee_coord)   

    #     env_state = self._robot_server_state_to_env_state(rs_state)

    #     polar_1 = abs(env_state[1] * 180/math.pi)
    #     polar_2 = abs(env_state[2] * 180/math.pi)

    #     # reward polar coords close to zero
    #     # polar 1 weighted by max_steps
    #     p1_r = (1 - (polar_1 / 90)) * (1/1000)
    #     # if p1_r <= 0.001:
    #     #     reward += p1_r
        
    #     # polar 1 weighted by max_steps
    #     p2_r = (1 - (polar_2 / 90)) * (1/1000)
    #     # if p2_r <= 0.001:
    #     #     reward += p2_r



    #     delta_joint_pos = env_state[3:9] - env_state[-6:]

    #     dr = 0
    #     if abs(delta_joint_pos).sum() < 0.5:
    #         dr = 4 * (1 - (abs(delta_joint_pos).sum()/0.5)) * (1/1000)
    #         reward += dr
        
    #     act_r = 0
    #     if abs(action).sum() < 0.4:
    #         act_r = 4 * (1 - (abs(action).sum()/0.4)) * (1/1000)
    #         reward += act_r

    #     dist = 0
    #     if distance_to_target < minimum_distance:
    #         dist = -30 * (1/1000)
    #         reward += dist

    #     dist_max = 0
    #     if distance_to_target > maximum_distance:
    #         dist_max = -2 * (1/1000)
    #         reward += dist_max

    #     # Check if robot is in collision
    #     if rs_state[25] == 1:
    #         collision = True
    #     else:
    #         collision = False

    #     if collision:
    #         reward = -10
    #         done = True
    #         info['final_status'] = 'collision'
    #         info['target_coord'] = target_coord
    #         self.last_position_on_success = []

    #     if self.elapsed_steps >= self.max_episode_steps:
    #         done = True
    #         info['final_status'] = 'success'
    #         info['target_coord'] = target_coord
    #         self.last_position_on_success = []

    #     print('reward composition: p1 =', p1_r, 'p2 =', p2_r, 'dr =', dr, 'no_act =', act_r, 'min_dist =', dist, 'max_dist', dist_max)

    #     return reward, done, info

    ## Second Reward with squared weighted punishment for movement
    ## NOTE: works nice but diverges with longer training periods
    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        minimum_distance = 0.3 # m
        maximum_distance = 0.6 # m

        # Calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        distance_to_target = np.linalg.norm(target_coord - ee_coord)   

        env_state = self._robot_server_state_to_env_state(rs_state)

        polar_1 = abs(env_state[1] * 180/math.pi)
        polar_2 = abs(env_state[2] * 180/math.pi)

        # reward polar coords close to zero
        # polar 1 weighted by max_steps
        p1_r = (1 - (polar_1 / 90)) * (1/1000)
        # if p1_r <= 0.001:
        #     reward += p1_r
        
        # polar 1 weighted by max_steps
        p2_r = (1 - (polar_2 / 90)) * (1/1000)
        # if p2_r <= 0.001:
        #     reward += p2_r



        delta_joint_pos = env_state[3:9] - env_state[-6:]

        print('action', action)

        dr = 0
        if abs(delta_joint_pos).sum() < 0.5:
            dr = 1 * (1 - (abs(delta_joint_pos).sum()/0.5)) * (1/1000)
            reward += dr
        
        act_r = 0 
        if abs(action).sum() <= action.size:
            act_r = 1 * (1 - (np.square(action).sum()/action.size)) * (1/1000)
            reward += act_r

        dist = 0
        if distance_to_target < minimum_distance:
            dist = -2 * (1/1000) 
            reward += dist

        dist_max = 0
        if distance_to_target > maximum_distance:
            dist_max = -1 * (1/1000)
            reward += dist_max

        # Check if robot is in collision
        if rs_state[25] == 1:
            collision = True
        else:
            collision = False

        if collision:
            #reward = -5
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord
            self.last_position_on_success = []

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord
            self.last_position_on_success = []

        print('reward composition:', 'dr =', round(dr, 5), 'no_act =', round(act_r, 5), 'min_dist =', round(dist, 5), 'max_dist', round(dist_max, 5))

        return reward, done, info

    def step(self, action):
        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action = np.array(action)
        
        # action =  np.append(action, [0.0])

        # Convert environment action to Robot Server action
        # rs_action = copy.deepcopy(action)
        # Scale action
        # action = np.multiply(action, self.abs_joint_pos_range[1:4])
        # convert action
        # current_joint_positions = self.ur5._ros_joint_list_to_ur5_joint_list(self.prev_rs_state[6:12])[1:4]
        # print('current joint positions:', current_joint_positions)
        # print('action to take:', action)
        # rs_action = action + current_joint_positions
        # print('converted action:', action)

        
        # Convert environment action to Robot Server action
        initial_joint_positions = self._get_initial_joint_positions()
        # print(initial_joint_positions)
        if action.size == 3:
            initial_joint_positions[1:4] = initial_joint_positions[1:4] + action
        elif action.size == 5:
            initial_joint_positions[0:5] = initial_joint_positions[0:5] + action
        elif action.size == 6:
            initial_joint_positions = initial_joint_positions + action


        # print(initial_joint_positions)
        rs_action = initial_joint_positions


        # rs_action = np.array([initial_joint_positions[0], rs_action[0], rs_action[1], rs_action[2], initial_joint_positions[4], initial_joint_positions[5]])

        

        # Convert action indexing from ur5 to ros
        rs_action = self.ur5._ur_5_joint_list_to_ros_joint_list(rs_action)
        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        self.prev_rs_state = copy.deepcopy(rs_state)
        # print('rs_state joints', rs_state[6:12])
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
    
    def _set_initial_joint_positions_range(self):
        self.initial_joint_positions_low = np.array([-0.9, -1.5, -1.5, -3.14, 1.3, 0.0])
        self.initial_joint_positions_high = np.array([-0.7, -1.1, -1.1, 3.14, 1.7, 0.0])

    def _get_initial_joint_positions(self):
        """Get initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        self._set_initial_joint_positions_range()
        # Fixed initial joint positions
        joint_positions = np.array([-0.78,-1.31,-1.31,-2.18,1.57,0.0])

        return joint_positions

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
        
        ee_to_base_translation = rs_state[18:21]
        ee_to_base_quaternion = rs_state[21:25]
        ee_to_base_rotation = R.from_quat(ee_to_base_quaternion)
        base_to_ee_rotation = ee_to_base_rotation.inv()
        base_to_ee_quaternion = base_to_ee_rotation.as_quat()
        base_to_ee_translation = - ee_to_base_translation

        target_coord_ee_frame = utils.change_reference_frame(target_coord,base_to_ee_translation,base_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
        ur_j_vel = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur5.normalize_joint_values(joints=ur_j_pos)

        # start joint positions
        start_joints = self.ur5.normalize_joint_values(self._get_initial_joint_positions())

        # Compose environment state
        state = np.concatenate((target_polar, ur_j_pos_norm, ur_j_vel, start_joints))

        return state

class MovingBoxTargetUR5DoF3(MovingBoxTargetUR5):

    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.full((3), -1.0), high=np.full((3), 1.0), dtype=np.float32)

class MovingBoxTargetUR5DoF5(MovingBoxTargetUR5):

    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

    
class MovingBoxTargetUR5Sim(MovingBoxTargetUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=3.14\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        obstacle_controller:=true \
        target_mode:=moving \
        target_model_name:=box100"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBoxTargetUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class MovingBoxTargetUR5DoF3Sim(MovingBoxTargetUR5DoF3, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=3.14\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        obstacle_controller:=true \
        target_mode:=moving \
        target_model_name:=box100"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBoxTargetUR5DoF3.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class MovingBoxTargetUR5DoF5Sim(MovingBoxTargetUR5DoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=3.14\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        obstacle_controller:=true \
        target_mode:=moving \
        target_model_name:=box100"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBoxTargetUR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

    
class MovingBox3DSplineTargetUR5(MovingBoxTargetUR5):

    def reset(self, initial_joint_positions = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None
        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            ur5_initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            ur5_initial_joint_positions = self.last_position_on_success
        else:
            ur5_initial_joint_positions = self._get_initial_joint_positions()

        rs_state[6:12] = self.ur5._ur_5_joint_list_to_ros_joint_list(ur5_initial_joint_positions)


        # Set initial state of the Robot Server
        
        z_amplitude = np.random.default_rng().uniform(low=0.09, high=0.35)
        z_frequency = 0.125
        z_offset = np.random.default_rng().uniform(low=0.2, high=0.6)
        
        string_params = {"function": "3d_spline"}
        float_params = {"x_min": 0.2, "x_max": 0.4, "y_min": -0.2, "y_max": -0.8, "z_min": 0.1, "z_max": 1.0, \
                        "n_points": 10, "n_sampling_points": 4000}
        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        self.prev_rs_state = copy.deepcopy(rs_state)

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # save start position
        self.start_position = self.state[3:9]

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions = self.ur5._ros_joint_list_to_ur5_joint_list(rs_state[6:12])
            tolerance = 0.1
            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    raise InvalidStateError('Reset joint positions are not within defined range')
            
        return self.state

class MovingBox3DSplineTargetUR5DoF3(MovingBox3DSplineTargetUR5):

    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.full((3), -1.0), high=np.full((3), 1.0), dtype=np.float32)

class MovingBox3DSplineTargetUR5DoF5(MovingBox3DSplineTargetUR5):

    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class MovingBox3DSplineTargetUR5Sim(MovingBox3DSplineTargetUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=3.14\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        obstacle_controller:=true \
        target_mode:=moving \
        target_model_name:=box100"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBox3DSplineTargetUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class MovingBox3DSplineTargetUR5Rob(MovingBox3DSplineTargetUR5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving

class MovingBox3DSplineTargetUR5DoF3Sim(MovingBox3DSplineTargetUR5DoF3, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=3.14\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        obstacle_controller:=true \
        target_mode:=moving \
        target_model_name:=box100"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBox3DSplineTargetUR5DoF3.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class MovingBox3DSplineTargetUR5DoF3Rob(MovingBox3DSplineTargetUR5DoF3):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving

class MovingBox3DSplineTargetUR5DoF5Sim(MovingBox3DSplineTargetUR5DoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=3.14\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        obstacle_controller:=true \
        target_mode:=moving \
        target_model_name:=box100"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBox3DSplineTargetUR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class MovingBox3DSplineTargetUR5DoF5Rob(MovingBox3DSplineTargetUR5DoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving

