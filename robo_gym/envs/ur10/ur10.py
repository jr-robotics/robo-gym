#!/usr/bin/env python3

import sys, math, copy, random
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation

class UR10Env(gym.Env):
    """Universal Robots UR10 base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        ur10 (:obj:): Robot utilities object.
        observation_space (:obj:): Environment observation space.
        action_space (:obj:): Environment action space.
        distance_threshold (float): Minimum distance (m) from target to consider it reached.
        abs_joint_pos_range (np.array): Absolute value of joint positions range`.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False

    def __init__(self, rs_address=None, max_episode_steps=500, **kwargs):

        self.ur10 = ur_utils.UR10()
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.full((6), -1.0), high=np.full((6), 1.0), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur10.get_max_joint_positions()

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.elapsed_steps = 0

        self.last_action = None
        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set initial robot joint positions
        ur10_initial_joint_positions = self._get_initial_joint_positions()
        rs_state[6:12] = self.ur10._ur_10_joint_list_to_ros_joint_list(ur10_initial_joint_positions)

        # Generate target End Effector pose
        rs_state[0:6] = self.ur10.get_random_workspace_pose()

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
        # Convert action indexing from ur10 to ros
        rs_action = self.ur10._ur_10_joint_list_to_ros_joint_list(rs_action)
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
        ee_pose = [0.0]*6
        ur_collision = [0.0]
        rs_state = target + ur_j_pos + ur_j_vel + ee_pose + ur_collision

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

    def _get_initial_joint_positions(self):
        """Generate random initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """

        # Minimum initial joint positions
        low = np.array([-0.65, -2.75, 1.0, -3.14, -1.7, -3.14])
        # Maximum initial joint positions
        high = np.array([0.65, -2.0, 2.5, 3.14, 1.7, 3.14])
        # Random initial joint positions
        joint_positions = np.random.default_rng().uniform(low=low, high=high)

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
        target_coord = rs_state[0:3]
        ee_coord = rs_state[18:21]
        target_polar = utils.cartesian_to_polar_3d(target_coord,ee_coord)

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur10._ros_joint_list_to_ur10_joint_list(rs_state[6:12])
        ur_j_vel = self.ur10._ros_joint_list_to_ur10_joint_list(rs_state[12:18])

        # Compose environment state
        state = np.concatenate((target_polar,ur_j_pos, ur_j_vel))

        return state

    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Joint position range tolerance
        pos_tolerance = np.full(6,0.3)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(self.ur10.get_max_joint_positions(), pos_tolerance)
        min_joint_positions = np.subtract(self.ur10.get_min_joint_positions(), pos_tolerance)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        # Joint positions range tolerance
        vel_tolerance = np.full(6,0.3)
        # Joint velocities range used to determine if there is an error in the sensor readings
        max_joint_velocities = np.add(self.ur10.get_max_joint_velocities(), vel_tolerance)
        min_joint_velocities = np.subtract(self.ur10.get_min_joint_velocities(), vel_tolerance)
        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_joint_velocities))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

class EndEffectorPositioningUR10(UR10Env):

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        euclidean_dist_3d = np.linalg.norm(target_coord-ee_coord)

        # Reward base
        base_reward = -50*euclidean_dist_3d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        if euclidean_dist_3d <= self.distance_threshold:
            reward = 100
            done = True
            info['final_status'] = 'success'

        # Check if robot is in collision
        if rs_state[24] == 1:
            collision = True
        else:
            collision = False

        if collision:
            reward = -200
            done = True
            info['final_status'] = 'collision'
        
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info

class EndEffectorPositioningAntiShakeUR10(UR10Env):

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        euclidean_dist_3d = np.linalg.norm(target_coord-ee_coord)

        # Reward base
        base_reward = - 50*euclidean_dist_3d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward


        # Creating punishment for rapid movement
        penalty = 0
        if self.last_action is None:
            self.last_action = action
        else:
            for a in range(len(self.last_action)):
                if abs(self.last_action[a] - action[a]) > 1:
                    penalty -= 5
                elif abs(self.last_action[a] - action[a]) > 0.5:
                    penalty -= 1
                elif abs(self.last_action[a] - action[a]) > 0.2:
                    penalty -= 0.2
            reward += penalty
            self.last_action = action

        if euclidean_dist_3d <= self.distance_threshold:
            reward = 100
            done = True
            info['final_status'] = 'success'

        # Check if robot is in collision
        if rs_state[24] == 1:
            collision = True
        else:
            collision = False

        if collision:
            reward = -200
            done = True
            info['final_status'] = 'collision'
        
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info

class EndEffectorPositioningUR10Sim(EndEffectorPositioningUR10, Simulation):
    cmd = "roslaunch ur_robot_server ur10_sim_robot_server.launch max_torque_scale_factor:=0.5 max_velocity_scale_factor:=0.5 speed_scaling:=0.5"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        EndEffectorPositioningUR10.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class EndEffectorPositioningUR10Rob(EndEffectorPositioningUR10):
    real_robot = True

class EndEffectorPositioningAntiShakeUR10Sim(EndEffectorPositioningAntiShakeUR10, Simulation):
    cmd = "roslaunch ur_robot_server ur10_sim_robot_server.launch max_torque_scale_factor:=0.5 max_velocity_scale_factor:=0.5 speed_scaling:=0.5"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        EndEffectorPositioningAntiShakeUR10.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class EndEffectorPositioningAntiShakeUR10Rob(EndEffectorPositioningAntiShakeUR10):
    real_robot = True
