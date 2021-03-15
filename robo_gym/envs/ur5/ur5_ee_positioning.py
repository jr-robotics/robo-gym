from copy import deepcopy
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces

from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

from robo_gym.envs.ur5.ur5_base_env import UR5BaseEnv


IGNORE_WRIST_3 = True

class EndEffectorPositioningUR5(UR5BaseEnv):

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
        
        joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        joint_positions_normalized = self.ur.normalize_joint_values(joint_positions)
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


class EndEffectorPositioningUR5DoF5(UR5BaseEnv):
        
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
            self.initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            self.initial_joint_positions = self.last_position_on_success
        else:
            self.initial_joint_positions = self._get_desired_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.initial_joint_positions)

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
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            if not np.isclose(joint_positions, self.initial_joint_positions, atol=0.1).all():
                raise InvalidStateError('Reset joint positions are not within defined range')

        # go one empty action and check if there is a collision
        if not self.real_robot:
            action = self.state[3:3+len(self.action_space.sample())]
            _, _, done, _ = self.step(action)
            self.elapsed_steps = 0
            if done:
                raise InvalidStateError('Reset started in a collision state')
            
        return self.state

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
        
        joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        joint_positions_normalized = self.ur.normalize_joint_values(copy.deepcopy(joint_positions))
        
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
        # Convert action indexing from ur to ros
        rs_action = self.ur._ur_joint_list_to_ros_joint_list(rs_action)
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
    
    def _get_action_space(self):
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        if IGNORE_WRIST_3:
            return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)
        else:
            return spaces.Box(low=np.full((6), -1.0), high=np.full((6), 1.0), dtype=np.float32)




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

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving