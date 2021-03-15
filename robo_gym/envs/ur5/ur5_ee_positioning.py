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
        if IGNORE_WRIST_3:
            action = np.append(action, [0.0])

        super().step(action)

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

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving