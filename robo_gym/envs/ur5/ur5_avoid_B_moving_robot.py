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
