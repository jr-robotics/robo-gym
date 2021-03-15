from copy import deepcopy
import math, copy
from robo_gym.envs.ur5.ur5_base_env import UR5BaseEnv
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import random
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

DEBUG = True


class IrosEnv03UR5Training(UR5BaseEnv):
    def __init__(self, rs_address=None, **kwargs):
        self.ur = ur_utils.UR(model="ur5")
        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.seed()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur.get_max_joint_positions()
        self.last_position_on_success = []
        self.prev_rs_state = None
        self.last_action = None

        file_name = 'ur5_pickplace_trajectories.json'
        file_path = os.path.join(os.path.dirname(__file__), 'robot_trajectories', file_name)
        with open(file_path) as json_file:
            self.trajectories = json.load(json_file)
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def reset(self, initial_joint_positions = None, type='random', reward_weights=[0.0]*7):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """

        # Default Configuration 
        # [obstacle_coordinates[3], current_joint[5], delta_joint[5], desired_joint[5], flag[1], obstacle_two_coordinates[3], [0.0]*3]

        # 1.	[polar_coords]
        # 2.    [obstacle_cartesian[3], current_joint[5], delta_joint[5], desired_joint[5], flag[1], ee_cartesian[3], elbow_cartesian[3]]
        # 3.	[obstacle_coordinates[3], [0,0,0,0,0,0], delta_joint[5], [0,0,0,0,0,0], flag[1], obstacle_two_coordinates[3], [0.0]*3]
        # 4.	[obstacle_coordinates[3], current_joint[5], [0,0,0,0,0,0], desired_joint[5], flag[1], obstacle_two_coordinates[3], [0.0]*3]
        # 5. 	[obstacle_coordinates[3], current_joint[5], [0,0,0,0,0,0], desired_joint[5], [0], obstacle_two_coordinates[3], [0.0]*3]

        self.mask_setting = 1

        self.elapsed_steps = 0

        self.obstacle_coords = []

        # Initialize state machine variables
        self.state_n = 0 
        self.elapsed_steps_in_current_state = 0 
        self.target_reached = 0
        self.target_reached_counter = 0

        self.reward_composition = []
        
        # Pick robot trajectory
        self.trajectories_ids = ['trajectory_3', 'trajectory_7', 'trajectory_8', 'trajectory_9', 'trajectory_10', 'trajectory_11']
        self.trajectory_id = random.choice(self.trajectories_ids)
        if DEBUG:
            print('Robot Trajectory ID: ' + self.trajectory_id)

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # NOTE: maybe we can find a cleaner version when we have the final envs (we could prob remove it for the avoidance task altogether)
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            self.initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            self.initial_joint_positions = self.last_position_on_success
        else:
            self.initial_joint_positions = self._get_desired_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.initial_joint_positions)


        # TODO: We should create some kind of helper function depending on how dynamic these settings should be
        # Set initial state of the Robot Server
        n_sampling_points = int(np.random.default_rng().uniform(low= 8000, high=12000))
        
        string_params = {"object_0_function": "3d_spline_ur5_workspace"}
        
        float_params = {"object_0_x_min": -1.0, "object_0_x_max": 1.0, "object_0_y_min": -1.0, "object_0_y_max": 1.0, \
                        "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                        "n_sampling_points": n_sampling_points}

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
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            if DEBUG:
                print("Initial Joint Positions")
                print(self.initial_joint_positions)
                print("Joint Positions")
                print(joint_positions)
            if not np.isclose(joint_positions, self.initial_joint_positions, atol=0.1).all():
                raise InvalidStateError('Reset joint positions are not within defined range')
            
        return self.state

    def step(self, action):
        self.elapsed_steps += 1
        self.elapsed_steps_in_current_state += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action = np.array(action)
        if self.last_action is None:
            self.last_action = action
        
        # Convert environment action to Robot Server action
        desired_joint_positions = copy.deepcopy(self._get_desired_joint_positions())
        if action.size == 3:
            desired_joint_positions[1:4] = desired_joint_positions[1:4] + action
        elif action.size == 5:
            desired_joint_positions[0:5] = desired_joint_positions[0:5] + action
        elif action.size == 6:
            desired_joint_positions = desired_joint_positions + action

        rs_action = desired_joint_positions

        # Convert action indexing from ur to ros
        rs_action = self.ur._ur_joint_list_to_ros_joint_list(rs_action)
        # Send action to Robot Server and get state
        rs_state = self.client.send_action_get_state(rs_action.tolist()).state
        self.prev_rs_state = copy.deepcopy(rs_state)

        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)
        
        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        if DEBUG:
            print("Desired Joint Positions")
            print(self._get_desired_joint_positions())
            print("Joint Positions")
            print(self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12]))

        # Check if the robot is at the target position
        if self.target_point_flag:
            if np.isclose(self._get_desired_joint_positions(), self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12]), atol = 0.1).all():
                self.target_reached = 1
                self.state_n +=1
                # Restart from state 0 if the full trajectory has been completed
                self.state_n = self.state_n % len(self.trajectories[self.trajectory_id])
                self.elapsed_steps_in_current_state = 0
                self.target_reached_counter += 1
        if DEBUG:
            print("Target Reached: ")
            print(self.target_reached)
            print("State number: ")
            print(self.state_n)
    
        # Assign reward
        reward = 0
        done = False
        reward, done, info = self._reward(rs_state=rs_state, action=action)
        self.last_action = action
        self.target_reached = 0

        return self.state, reward, done, info

    def print_state_action_info(self, rs_state, action):
        env_state = self._robot_server_state_to_env_state(rs_state)

        print('Action:', action)
        print('Last A:', self.last_action)
        print('Distance EE: {:.2f} Polar1: {:.2f} Polar2: {:.2f}'.format(env_state[0], env_state[1], env_state[2]))
        print('Distance Elbow: {:.2f} Polar1: {:.2f} Polar2: {:.2f}'.format(env_state[-6], env_state[-5], env_state[-4]))
        print('Elbow Cartesian: x={:.2f}, y={:.2f}, z={:.2f}'.format(env_state[-3], env_state[-2], env_state[-1]))
        
        print('Joint Positions: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[3:9]))
        print('Joint PosDeltas: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[9:15]))
        print('Current Desired: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[15:21]))
        print('Is current disred a target?', env_state[21])
        print('Targets reached', self.target_reached_counter)

        print()
    # should not be used anyway
    def print_reward_composition(self):
        # self.reward_composition.append([dr, act_r, small_actions, act_delta, dist_1, self.target_reached, collision_reward])
        dr = [r[0] for r in self.reward_composition]
        act_r = [r[1] for r in self.reward_composition]
        # small_actions = [r[2] for r in self.reward_composition]
        act_delta = [r[2] for r in self.reward_composition]
        dist_1 = [r[3] for r in self.reward_composition]
        target_reached = [r[5] for r in self.reward_composition]
        collision_reward = [r[4] for r in self.reward_composition]
        # joint_delta_r = [r[5] for r in self.reward_composition]

        print('sanity check', dr, act_r, act_delta)

        print('Reward Composition of Episode:')
        print('Reward for keeping low delta joints: SUM={} MIN={}, MAX={}'.format(np.sum(dr), np.min(dr), np.max(dr)))
        print('Reward for as less as possible: SUM={} MIN={}, MAX={}'.format(np.sum(act_r), np.min(act_r), np.max(act_r)))
        # print('Reward minor actions: SUM={} MIN={}, MAX={}'.format(np.sum(small_actions), np.min(small_actions), np.max(small_actions)))
        print('Punishment for rapid movement: SUM={} MIN={}, MAX={}'.format(np.sum(act_delta), np.min(act_delta), np.max(act_delta)))
        print('Punishment for target distance: SUM={} MIN={}, MAX={}'.format(np.sum(dist_1), np.min(dist_1), np.max(dist_1)))
        print('Reward for target reached: SUM={} MIN={}, MAX={}'.format(np.sum(target_reached), np.min(target_reached), np.max(target_reached)))
        print('Punishment for collision: SUM={} MIN={}, MAX={}'.format(np.sum(collision_reward), np.min(collision_reward), np.max(collision_reward)))
        # print('Punishment for joint delta: SUM={} MIN={}, MAX={}'.format(np.sum(joint_delta_r), np.min(joint_delta_r), np.max(joint_delta_r)))

    def _reward(self, rs_state, action):
        # TODO: remove print when not needed anymore
        # print('action', action)
        env_state = self._robot_server_state_to_env_state(rs_state)

        reward = 0
        done = False
        info = {}

        # minimum the robot should keep to the obstacle
        minimum_distance = 0.45 # m
        
        target_coord = rs_state[0:3]
        ee_coord = rs_state[18:21]
        elbow_coord = rs_state[26:29]

        self.obstacle_coords.append(target_coord)
        
        distance_to_target = np.linalg.norm(np.array(target_coord)-np.array(ee_coord))
        distance_to_target_2 = np.linalg.norm(np.array(target_coord)-np.array(elbow_coord))

        # Normalize joint position values
        ur_j_pos = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # desired joint positions
        desired_joints = self.ur.normalize_joint_values(self._get_desired_joint_positions())
        delta_joint_pos = ur_j_pos_norm - desired_joints
        target_point_flag = self.target_point_flag


        distance_weight = -2.0
        delta_weight = 1.0
        action_usage_weight = 1.5
        act_delta_weight = -0.2
        collison_weight = -0.1
        target_reached_weight = 0.05
        joint_delta_weight = 0.0

        # reward for being in the defined interval of minimum_distance and maximum_distance
        dr = 0
        # if abs(delta_joint_pos).sum() < 0.5:
        #     dr = 1.5 * (1 - (sum(abs(delta_joint_pos))/0.5)) * (1/1000)
        #     reward += dr
        for i in range(len(delta_joint_pos)-1):
            if abs(delta_joint_pos[i]) < 0.1:
                dr = delta_weight * (1 - (abs(delta_joint_pos[i]))/0.1) * (1/1000) 
                dr = dr/5
                reward += dr
        
        
        # reward moving as less as possible
        act_r = 0 
        if abs(action).sum() <= action.size:
            act_r = action_usage_weight * (1 - (np.square(action).sum()/action.size)) * (1/1000)
            reward += act_r


        # punish big deltas in action
        act_delta = 0
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.4:
                a_r = act_delta_weight * (1/1000)
                act_delta += a_r
                reward += a_r
        
        dist_1 = 0
        if (distance_to_target < minimum_distance) or (distance_to_target_2 < minimum_distance):
            dist_1 = distance_weight * (1/1000) # -2
            reward += dist_1

        tr_reward = 0
        if self.target_reached:
            tr_reward += target_reached_weight
            reward += target_reached_weight

        
        # TODO: we could remove this if we do not need to punish failure or reward success
        # Check if robot is in collision
        collision_reward = 0
        collision = True if rs_state[25] == 1 else False
        if collision:
            collision_reward = collison_weight
            reward = collison_weight
            done = True
            info['final_status'] = 'collision'
            

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
        
        self.reward_composition.append([dr, act_r, act_delta, dist_1, collision_reward, tr_reward])
        # self.reward_composition.append([dr, act_r, small_actions, act_delta, dist_1, tr_reward, collision_reward])
        if done:
            self.print_reward_composition()
            info['reward_composition'] = self.reward_composition
            info['targets_reached'] = self.target_reached_counter
            info['obstacle_coords'] = self.obstacle_coords
            info['trajectory_id'] = self.trajectory_id

        self.print_state_action_info(rs_state, action)
        # ? DEBUG PRINT
        print('Distance 1: {:.2f} Distance 2: {:.2f}'.format(distance_to_target, distance_to_target_2))
        # if DEBUG: print('reward composition:', 'dr =', round(dr, 5), 'no_act =', round(act_r, 5), 'min_dist_1 =', round(dist_1, 5), 'min_dist_2 =', 'delta_act', round(act_delta, 5))
        
        # self.last_joint_positions = env_state[3:9]

        return reward, done, info

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

        # desired joint positions
        desired_joints = self.ur.normalize_joint_values(self._get_desired_joint_positions())
        delta_joints = ur_j_pos_norm - desired_joints
        target_point_flag = self.target_point_flag

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

        # Compose environment state
        state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints, desired_joints, [target_point_flag], target_polar_forearm, [0.0]*3))

        if self.mask_setting == 1: # polar
            pass
        elif self.mask_setting == 2: # cartesian and delete second polar
            state = np.concatenate((target_coord, ur_j_pos_norm, delta_joints, desired_joints, [target_point_flag], ee_to_ref_frame_translation, forearm_to_ref_frame_translation))
        elif self.mask_setting == 3: # no current no desired
            state = np.concatenate((target_polar, [0.0]*6, delta_joints, [0.0]*6, [target_point_flag], target_polar_forearm, [0.0]*3))
        elif self.mask_setting == 4: # no delta
            state = np.concatenate((target_polar, ur_j_pos_norm, [0.0]*6, desired_joints, [target_point_flag], target_polar_forearm, [0.0]*3))
        elif self.mask_setting == 5: # no flag for waypoint
            state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints, desired_joints, [0], target_polar_forearm, [0.0]*3))

        return state

    # observation space should be fine
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
        
        max_delta_start_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_delta_start_positions = np.subtract(np.full(6, -1.0), pos_tolerance)

        # Target coordinates (with respect to forearm frame) range
        target_forearm_range = np.full(3, np.inf)

        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, max_joint_positions, [1], target_forearm_range, target_forearm_range))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, min_joint_positions, [0], -target_forearm_range, -target_forearm_range))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

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
        ee_to_ref_frame_transform = [0.0]*7
        ur_collision = [0.0]
        forearm_to_ref_frame_transform = [0.0]*7
        rs_state = target + ur_j_pos + ur_j_vel + ee_to_ref_frame_transform + ur_collision + forearm_to_ref_frame_transform

        return len(rs_state)

    def _get_desired_joint_positions(self):
        """Get desired robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        
        
        if self.elapsed_steps_in_current_state < len(self.trajectories[self.trajectory_id][self.state_n]):
            joint_positions = copy.deepcopy(self.trajectories[self.trajectory_id][self.state_n][self.elapsed_steps_in_current_state])
            # Last Joint is set to 0
            joint_positions[5] = 0
            self.target_point_flag = 0
        else:
            # Get last point of the trajectory segment
            joint_positions = copy.deepcopy(self.trajectories[self.trajectory_id][self.state_n][-1])
            # Last Joint is set to 0
            joint_positions[5] = 0
            self.target_point_flag = 1
        

        return joint_positions


class IrosEnv03UR5TrainingDoF5(IrosEnv03UR5Training):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class IrosEnv03UR5TrainingSim(IrosEnv03UR5Training, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=tabletop_box50.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving2points \
        n_objects:=1.0 \
        object_0_model_name:=box50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03UR5Training.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03UR5TrainingDoF5Sim(IrosEnv03UR5TrainingDoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=tabletop_box50.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving2points \
        n_objects:=1.0 \
        object_0_model_name:=box50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03UR5TrainingDoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03UR5TrainingDoF5Rob(IrosEnv03UR5TrainingDoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target

# ? Test Environment with robot trajectories different from the ones on which it was trained.
class IrosEnv03UR5Test(IrosEnv03UR5Training):
    def reset(self, initial_joint_positions = None, type='random', reward_weights=[0.0]*7):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """

        # Default Configuration 
        # [obstacle_coordinates[3], current_joint[5], delta_joint[5], desired_joint[5], flag[1], obstacle_two_coordinates[3], [0.0]*3]

        # 1.	[polar_coords]
        # 2.    [obstacle_cartesian[3], current_joint[5], delta_joint[5], desired_joint[5], flag[1], ee_cartesian[3], elbow_cartesian[3]]
        # 3.	[obstacle_coordinates[3], [0,0,0,0,0,0], delta_joint[5], [0,0,0,0,0,0], flag[1], obstacle_two_coordinates[3], [0.0]*3]
        # 4.	[obstacle_coordinates[3], current_joint[5], [0,0,0,0,0,0], desired_joint[5], flag[1], obstacle_two_coordinates[3], [0.0]*3]
        # 5. 	[obstacle_coordinates[3], current_joint[5], [0,0,0,0,0,0], desired_joint[5], [0], obstacle_two_coordinates[3], [0.0]*3]

        self.mask_setting = 1

        self.elapsed_steps = 0

        self.obstacle_coords = []

        # Initialize state machine variables
        self.state_n = 0 
        self.elapsed_steps_in_current_state = 0 
        self.target_reached = 0
        self.target_reached_counter = 0

        self.reward_composition = []
        
        # Pick robot trajectory
        self.trajectories_ids = ['trajectory_12', 'trajectory_13']
        self.trajectory_id = random.choice(self.trajectories_ids)
        if DEBUG:
            print('Robot Trajectory ID: ' + self.trajectory_id)

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # NOTE: maybe we can find a cleaner version when we have the final envs (we could prob remove it for the avoidance task altogether)
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            self.initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            self.initial_joint_positions = self.last_position_on_success
        else:
            self.initial_joint_positions = self._get_desired_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.initial_joint_positions)


        # TODO: We should create some kind of helper function depending on how dynamic these settings should be
        # Set initial state of the Robot Server
        n_sampling_points = int(np.random.default_rng().uniform(low= 8000, high=12000))
        
        string_params = {"object_0_function": "3d_spline_ur5_workspace"}
        
        float_params = {"object_0_x_min": -1.0, "object_0_x_max": 1.0, "object_0_y_min": -1.0, "object_0_y_max": 1.0, \
                        "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                        "n_sampling_points": n_sampling_points}

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
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            if DEBUG:
                print("Initial Joint Positions")
                print(self.initial_joint_positions)
                print("Joint Positions")
                print(joint_positions)
            if not np.isclose(joint_positions, self.initial_joint_positions, atol=0.1).all():
                raise InvalidStateError('Reset joint positions are not within defined range')
            
        return self.state

class IrosEnv03UR5TestDoF5(IrosEnv03UR5Test):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class IrosEnv03UR5TestDoF5Sim(IrosEnv03UR5TestDoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=tabletop_box50.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving2points \
        n_objects:=1.0 \
        object_0_model_name:=box50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03UR5TestDoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03UR5TestDoF5Rob(IrosEnv03UR5TestDoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target

# ? Test Environment with robot trajectories different from the ones on which it was trained
# ? and fixed obstacle trajectories imported from file 

class IrosEnv03UR5TestFixedSplines(IrosEnv03UR5Training):
    ep_n = 0 

    def reset(self, initial_joint_positions = None, type='random', reward_weights=[0.0]*7):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """

        # Default Configuration 
        # [obstacle_coordinates[3], current_joint[5], delta_joint[5], desired_joint[5], flag[1], obstacle_two_coordinates[3], [0.0]*3]

        # 1.	[polar_coords]
        # 2.    [obstacle_cartesian[3], current_joint[5], delta_joint[5], desired_joint[5], flag[1], ee_cartesian[3], elbow_cartesian[3]]
        # 3.	[obstacle_coordinates[3], [0,0,0,0,0,0], delta_joint[5], [0,0,0,0,0,0], flag[1], obstacle_two_coordinates[3], [0.0]*3]
        # 4.	[obstacle_coordinates[3], current_joint[5], [0,0,0,0,0,0], desired_joint[5], flag[1], obstacle_two_coordinates[3], [0.0]*3]
        # 5. 	[obstacle_coordinates[3], current_joint[5], [0,0,0,0,0,0], desired_joint[5], [0], obstacle_two_coordinates[3], [0.0]*3]

        self.mask_setting = 1

        self.elapsed_steps = 0

        self.obstacle_coords = []

        # Initialize state machine variables
        self.state_n = 0 
        self.elapsed_steps_in_current_state = 0 
        self.target_reached = 0
        self.target_reached_counter = 0

        self.reward_composition = []
        
        # Pick robot trajectory
        self.trajectories_ids = ['trajectory_3', 'trajectory_7', 'trajectory_8', 'trajectory_9', 'trajectory_10', 'trajectory_11']
        self.trajectory_id = self.trajectories_ids[int(self.ep_n/50)]
        if DEBUG:
            print('Robot Trajectory ID: ' + self.trajectory_id)

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # NOTE: maybe we can find a cleaner version when we have the final envs (we could prob remove it for the avoidance task altogether)
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            self.initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            self.initial_joint_positions = self.last_position_on_success
        else:
            self.initial_joint_positions = self._get_desired_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.initial_joint_positions)


        # TODO: We should create some kind of helper function depending on how dynamic these settings should be
        # Set initial state of the Robot Server
        n_sampling_points = int(np.random.default_rng().uniform(low= 8000, high=12000))
        
        string_params = {"object_0_function": "fixed_trajectory"}
        float_params = {"object_0_trajectory_id": self.ep_n%50}
        
        print("Obstacle trajectory id: " + repr(self.ep_n%50))

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
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            if DEBUG:
                print("Initial Joint Positions")
                print(self.initial_joint_positions)
                print("Joint Positions")
                print(joint_positions)
            if not np.isclose(joint_positions, self.initial_joint_positions, atol=0.1).all():
                raise InvalidStateError('Reset joint positions are not within defined range')
        
        self.ep_n +=1

        return self.state

class IrosEnv03UR5TestFixedSplinesDoF5(IrosEnv03UR5TestFixedSplines):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class IrosEnv03UR5TestFixedSplinesDoF5Sim(IrosEnv03UR5TestFixedSplinesDoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=tabletop_box50.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving2points \
        n_objects:=1.0 \
        object_trajectory_file_name:=splines_ur5 \
        object_0_model_name:=box50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv03UR5TestFixedSplinesDoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv03UR5TestFixedSplinesDoF5Rob(IrosEnv03UR5TestFixedSplinesDoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target
