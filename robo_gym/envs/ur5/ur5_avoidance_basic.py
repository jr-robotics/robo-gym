"""
Environment for basic obstacle avoidance controlling a robotic arm from UR.

In this environment the obstacle is only moving up and down in a vertical line in front of the robot.
The goal is for the robot to stay within a predefined minimum distance to the moving obstacle.
When feasible the robot should continue to the original configuration, 
otherwise wait for the obstacle to move away before proceeding
"""

import math, copy
from robo_gym.envs.ur5.ur5_base_env import UR5BaseEnv
import numpy as np
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from typing import Tuple

DEBUG = True
DOF = 5 # degrees of freedom the robotic arm can use [3, 5, 6]
MINIMUM_DISTANCE = 0.3 # the distance [cm] the robot should keep to the obstacle
DESIRED_JOINT_POSITIONS = [-0.78,-1.31,-1.31,-2.18,1.57,0.0]

class MovingBoxTargetUR5(UR5BaseEnv):
    
    max_episode_steps = 1000

    def _get_action_space(self) -> spaces.Box:
        return spaces.Box(low=np.full((DOF), -1.0), high=np.full((DOF), 1.0), dtype=np.float32)
            
    def _get_observation_space(self) -> spaces.Box:
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
        max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)


    def reset(self) -> np.array:
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
        self.initial_joint_positions = self._get_desired_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.initial_joint_positions)


        # Set initial state of the Robot Server
        z_amplitude = np.random.default_rng().uniform(low=0.09, high=0.35)
        z_frequency = 0.125
        z_offset = np.random.default_rng().uniform(low=0.2, high=0.6)
        
        string_params = {"object_0_function": "triangle_wave"}
        float_params = {"object_0_x": 0.13, 
                        "object_0_y": -0.30, 
                        "object_0_z_amplitude": z_amplitude,
                        "object_0_z_frequency": z_frequency, 
                        "object_0_z_offset": z_offset}
        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
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
        if not np.isclose(joint_positions, self.initial_joint_positions, atol=0.1).all():
            raise InvalidStateError('Reset joint positions are not within defined range')        

        return self.state

    def _reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        # TODO: remove print when not needed anymore
        env_state = self._robot_server_state_to_env_state(rs_state)

        reward = 0
        done = False
        info = {}
        
        # Reward weights
        close_distance_weight = -2
        delta_joint_weight = 1
        action_usage_weight = 1
        rapid_action_weight = -0.2

        # Difference in joint position current vs. starting position
        delta_joint_pos = env_state[9:15]

        # Calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        distance_to_target = np.linalg.norm(target_coord - ee_coord)   
                
        # Reward staying close to the predefined joint position
        if abs(env_state[-6:]).sum() < 0.1 * action.size:
            reward += delta_joint_weight * (1 - (abs(delta_joint_pos).sum()/(0.1 * action.size))) * (1/1000)
        
        # Reward for not acting
        if abs(action).sum() <= action.size:
            reward += action_usage_weight * (1 - (np.square(action).sum()/action.size)) * (1/1000)

        # Negative reward if actions change to rapidly between steps
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.5:
                reward += rapid_action_weight * (1/1000)
            
        # Negative reward if the obstacle is close than the predefined minimum distance
        if distance_to_target < MINIMUM_DISTANCE:
            reward += close_distance_weight * (1/self.max_episode_steps) 

        collision = True if rs_state[25] == 1 else False
        if collision:
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord
            self.last_position_on_success = []

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord
            self.last_position_on_success = []
        

        if DEBUG: self.print_state_action(rs_state, action)

        return reward, done, info

    def print_state_action(self, rs_state, action) -> None:
        env_state = self._robot_server_state_to_env_state(rs_state)

        print('Prev Action:', self.last_action)
        print('Action:', action)
        print('Distance: {:.3f}'.format(env_state[0]))
        print('Polar 1 (degree): {:.3f}'.format(env_state[1] * 180/math.pi))
        print('Polar 2 (degree): {:.3f}'.format(env_state[2] * 180/math.pi))
        print('Joint Positions: [1]:{:.3f} [2]:{:.3f} [3]:{:.3f} [4]:{:.3f} [5]:{:.3f} [6]:{:.3f}'.format(*env_state[3:9]))
        print('Joint PosDeltas: [1]:{:.3f} [2]:{:.3f} [3]:{:.3f} [4]:{:.3f} [5]:{:.3f} [6]:{:.3f}'.format(*env_state[9:15]))
        print('Sum of Deltas: {:.2e}'.format(sum(abs(env_state[9:15]))))
        print()

    def _action_to_delta_joint_pos(self, action) -> np.array:
        """Convert environment action to Robot Server action"""

        desired_joint_positions = self._get_desired_joint_positions()
        if action.size == 3:
            desired_joint_positions[1:4] = desired_joint_positions[1:4] + action
        elif action.size == 5:
            desired_joint_positions[0:5] = desired_joint_positions[0:5] + action
        elif action.size == 6:
            desired_joint_positions = desired_joint_positions + action
        else:
            raise InvalidStateError('DOF setting has unsupported value [3, 5, 6]')
        return desired_joint_positions
        

    # TODO: once normalization is gone this method can be merged with URBaseEnv
    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        self.elapsed_steps += 1

        # Check if the action is within the action space
        action = np.array(action)
        if self.last_action is None:
            self.last_action = action
        
        # Convert environment action to Robot Server action
        rs_action = self._action_to_delta_joint_pos(action)

        # ? Scaling in URBaseEnv

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
        self.last_action = action

        return self.state, reward, done, info
    
    def _get_desired_joint_positions(self) -> np.array:
        """Get desired robot joint positions with standard indexing."""
        return np.array(DESIRED_JOINT_POSITIONS)


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
        ur_j_vel = self.ur._ros_joint_list_to_ur_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # start joint positions
        start_joints = self.ur.normalize_joint_values(self._get_desired_joint_positions())
        delta_joints = ur_j_pos_norm - start_joints

        # Compose environment state
        state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints))

        return state

class MovingBoxTargetUR5Sim(MovingBoxTargetUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=3.14\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=moving \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        MovingBoxTargetUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class MovingBoxTargetUR5Rob(MovingBoxTargetUR5):
    real_robot = True 

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving