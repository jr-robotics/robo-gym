#!/usr/bin/env python3
from copy import deepcopy
import math, copy
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

from robo_gym.envs.ur5.ur5_avoidance import MovingBox3DSplineTargetUR5

# ? Variant B - Nice Environment with the position that the robot should keep that is changing over time. 

DEBUG = True

class ObstacleAvoidanceVarB1Box1PointUR5(MovingBox3DSplineTargetUR5):

    def _get_initial_joint_positions(self):
        """Get initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        # Fixed initial joint positions
        if self.elapsed_steps < 250:
            joint_positions = np.array([-0.78,-1.31,-1.31,-2.18,1.57,0.0])
        elif self.elapsed_steps < 500:
            joint_positions = np.array([-1.3,-1.0,-1.7,-2.18,1.57,0.0])
        elif self.elapsed_steps < 750:
            joint_positions = np.array([0.0,-1.8,-1.0,-1.0,2.0,0.0])
        else:
            joint_positions = np.array([-1.7,-0.8,-2.0,-2.0,2.5,0.0])

        return joint_positions

    def _reward(self, rs_state, action):
        # TODO: remove print when not needed anymore
        # print('action', action)
        env_state = self._robot_server_state_to_env_state(rs_state)

        reward = 0
        done = False
        info = {}

        # minimum and maximum distance the robot should keep to the obstacle
        minimum_distance = 0.3 # m
        maximum_distance = 0.6 # m

        # calculate distance to the target
        # target_coord = np.array(rs_state[0:3])
        # ee_coord = np.array(rs_state[18:21])
        distance_to_target = env_state[0]   


        
        # ! not used yet
        # ? we could train the robot to always "look" at the target just because it would look cool
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

        # TODO: depends on the stating pos being in the state, maybe find better solution
        # difference in joint position current vs. starting position
        # delta_joint_pos = env_state[3:9] - env_state[-6:]
        delta_joint_pos = env_state[9:15]

        # reward for being in the defined interval of minimum_distance and maximum_distance
        dr = 0
        if abs(env_state[-6:]).sum() < 0.5:
            dr = 1 * (1 - (abs(delta_joint_pos).sum()/0.5)) * (1/1000)
            reward += dr
        
        # reward moving as less as possible
        act_r = 0 
        if abs(action).sum() <= action.size:
            act_r = 1 * (1 - (np.square(action).sum()/action.size)) * (1/1000)
            reward += act_r

        # punish big deltas in action
        act_delta = 0
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.5:
                a_r = - 0.2 * (1/1000)
                act_delta += a_r
                reward += a_r
        
        # ? First try is to just split the distance reward
        # punish if the obstacle gets too close
        # dist_1 = 0
        # if distance_to_target < minimum_distance:
        #     dist_1 = -1 * (1/self.max_episode_steps) # -2
        #     reward += dist_1
        
        # dist_2 = 0
        # if distance_to_target_2 < minimum_distance:
        #     dist_2 = -1 * (1/self.max_episode_steps) # -2
        #     reward += dist_2

        dist_1 = 0
        if (distance_to_target < minimum_distance):
            dist_1 = -2 * (1/self.max_episode_steps) # -2
            reward += dist_1
        
        dist_2 = 0
       


        # punish if the robot moves too far away from the obstacle
        dist_max = 0
        if distance_to_target > maximum_distance:
            dist_max = -1 * (1/self.max_episode_steps)
            #reward += dist_max

        # TODO: we could remove this if we do not need to punish failure or reward success
        # Check if robot is in collision
        collision = True if rs_state[25] == 1 else False
        if collision:
            reward = -1
            done = True
            info['final_status'] = 'collision'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'

        

        if DEBUG: self.print_state_action_info(rs_state, action)
        # ? DEBUG PRINT
        if DEBUG: print('reward composition:', 'dr =', round(dr, 5), 'no_act =', round(act_r, 5), 'min_dist_1 =', round(dist_1, 5), 'min_dist_2 =', round(dist_2, 5), 'delta_act', round(act_delta, 5))


        return reward, done, info

class ObstacleAvoidanceVarB1Box1PointUR5DoF3(ObstacleAvoidanceVarB1Box1PointUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((3), -1.0), high=np.full((3), 1.0), dtype=np.float32)

class ObstacleAvoidanceVarB1Box1PointUR5DoF5(ObstacleAvoidanceVarB1Box1PointUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class ObstacleAvoidanceVarB1Box1PointUR5Sim(ObstacleAvoidanceVarB1Box1PointUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
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
        ObstacleAvoidanceVarB1Box1PointUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB1Box1PointUR5Rob(ObstacleAvoidanceVarB1Box1PointUR5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class ObstacleAvoidanceVarB1Box1PointUR5DoF3Sim(ObstacleAvoidanceVarB1Box1PointUR5DoF3, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
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
        ObstacleAvoidanceVarB1Box1PointUR5DoF3.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB1Box1PointUR5DoF3Rob(ObstacleAvoidanceVarB1Box1PointUR5DoF3):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class ObstacleAvoidanceVarB1Box1PointUR5DoF5Sim(ObstacleAvoidanceVarB1Box1PointUR5DoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
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
        ObstacleAvoidanceVarB1Box1PointUR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB1Box1PointUR5DoF5Rob(ObstacleAvoidanceVarB1Box1PointUR5DoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"
