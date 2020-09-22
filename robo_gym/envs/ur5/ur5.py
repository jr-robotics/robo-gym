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

    def __init__(self, rs_address=None, max_episode_steps=300, **kwargs):
        self.ur5 = ur_utils.UR5()
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.full((6), -1.0), high=np.full((6), 1.0), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur5.get_max_joint_positions()
        self.initial_joint_positions_low = np.zeros(6)
        self.initial_joint_positions_high = np.zeros(6)

        self.last_position_on_success = []
        

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
        if not self.client.set_state(copy.deepcopy(rs_state.tolist())):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state())))

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
        action = self.state[3:3+len(self.action_space.sample())]
        _, _, done, info = self.step(action)
        self.elapsed_steps = 0
        if done:
            raise InvalidStateError('Reset started in a collision state')
            
        return self.state

    def _reward(self, rs_state, action):
        return 0, False

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
        rs_state = self.client.get_state()
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
    def __init__(self, rs_address=None, max_episode_steps=300, **kwargs):

        self.ur5 = ur_utils.UR5()
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur5.get_max_joint_positions()
        
        self.last_position_on_success = []

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")
    
    
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
        if not self.client.set_state(copy.deepcopy(rs_state.tolist())):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state())))

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
        rs_state = self.client.get_state()
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

