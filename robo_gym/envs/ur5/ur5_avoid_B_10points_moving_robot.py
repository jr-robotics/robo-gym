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

from robo_gym.envs.ur5.ur5_avoid_B_moving_robot import ObstacleAvoidanceVarB1Box1PointUR5

# ? Variant B with 10 points- Nice Environment with the position that the robot should keep that is changing over time. 

DEBUG = True

class ObstacleAvoidanceVarB10Points1Box1PointUR5(ObstacleAvoidanceVarB1Box1PointUR5):
    def _get_initial_joint_positions(self):
        """Get initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        trajectory_points_list = [[-1.2338908354388636, -1.8468645254718226, -1.691645924245016, -1.1664560476886194, 1.5948981046676636, -0.06978494325746709], [-1.2299569288836878, -1.7984626928912562, -1.0858233610736292, -1.8207653204547327, 1.5953295230865479, -0.07516128221620733], [-1.1691883246051233, -1.7866886297809046, -1.024494473134176, -1.893508259450094, 1.5946221351623535, -0.0150988737689417], [-1.0026686827289026, -1.7696564833270472, -1.043490235005514, -1.8909309546100062, 1.5925596952438354, 0.151437908411026], [-0.7516277472125452, -1.7439172903644007, -1.073974911366598, -1.8849604765521448, 1.5893827676773071, 0.40271154046058655], [-0.5001352469073694, -1.7182148138629358, -1.104424301777975, -1.8790748755084437, 1.58585786819458, 0.6542360782623291], [-0.24920732179750615, -1.6924284140216272, -1.1348746458636683, -1.8733561674701136, 1.5824764966964722, 0.9057040810585022], [-0.0022414366351526382, -1.667229477559225, -1.1648576895343226, -1.8674457708941858, 1.5791313648223877, 1.1524267196655273], [0.13228075206279755, -1.6547015349017542, -1.1880410353290003, -1.8560336271869105, 1.5773210525512695, 1.2871218919754028], [0.168461874127388, -1.6662958304034632, -1.2750933806048792, -1.7572453657733362, 1.5764577388763428, 1.3245306015014648], [0.16787464916706085, -1.697913948689596, -1.444775406514303, -1.556019131337301, 1.5761702060699463, 1.3265185356140137], [0.16695186495780945, -1.737269703541891, -1.6562674681292933, -1.3050482908831995, 1.5756900310516357, 1.3288296461105347], [0.1660171002149582, -1.776447598134176, -1.8668020407306116, -1.0552757422076624, 1.5751028060913086, 1.3311407566070557], [0.1654897928237915, -1.7998159567462366, -1.9922927061664026, -0.9059823195086878, 1.5747549533843994, 1.3325777053833008], [0.16541787981987, -1.8017204443561, -2.003066364918844, -0.8933466116534632, 1.5747908353805542, 1.33274507522583]]

        # Fixed initial joint positions
        if self.elapsed_steps < 100:
            joint_positions = np.array(trajectory_points_list[0])
        elif self.elapsed_steps < 200:
            joint_positions = np.array(trajectory_points_list[1])
        elif self.elapsed_steps < 300:
            joint_positions = np.array(trajectory_points_list[2])
        elif self.elapsed_steps < 400:
            joint_positions = np.array(trajectory_points_list[3])
        elif self.elapsed_steps < 500:
            joint_positions = np.array(trajectory_points_list[4])
        elif self.elapsed_steps < 600:
            joint_positions = np.array(trajectory_points_list[5])
        elif self.elapsed_steps < 700:
            joint_positions = np.array(trajectory_points_list[6])
        elif self.elapsed_steps < 800:
            joint_positions = np.array(trajectory_points_list[7])
        elif self.elapsed_steps < 900:
            joint_positions = np.array(trajectory_points_list[8])
        else:
            joint_positions = np.array(trajectory_points_list[9])

        return joint_positions


class ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3(ObstacleAvoidanceVarB10Points1Box1PointUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((3), -1.0), high=np.full((3), 1.0), dtype=np.float32)

class ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5(ObstacleAvoidanceVarB10Points1Box1PointUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class ObstacleAvoidanceVarB10Points1Box1PointUR5Sim(ObstacleAvoidanceVarB10Points1Box1PointUR5, Simulation):
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
        ObstacleAvoidanceVarB10Points1Box1PointUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB10Points1Box1PointUR5Rob(ObstacleAvoidanceVarB10Points1Box1PointUR5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3Sim(ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3, Simulation):
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
        ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3Rob(ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5Sim(ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5, Simulation):
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
        ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5Rob(ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"
