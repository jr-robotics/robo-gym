from __future__ import annotations

import numpy as np

from robo_gym.envs.base.robogym_env import *
from robo_gym.envs.manipulator.isaac_reach import *
from robo_gym.envs.panda.panda_base import PandaBaseEnv
from robo_gym.utils.panda_utils import Panda


class IsaacReachPanda(IsaacReachEnv):
    def __init__(self, **kwargs):
        # not too nice - repeated in super init
        self._config = kwargs

        IsaacReachPanda.set_robot_defaults(kwargs)

        obs_nodes: list[ObservationNode] | None = kwargs.get(
            RoboGymEnv.KW_OBSERVATION_NODES
        )
        if obs_nodes is None:
            # having this hardcoded index is suboptimal; 1 should be correct since 0 will be the manipulator observation node
            obs_nodes = [LastActionObservationNode(**self.get_obs_node_setup_kwargs(1))]
            kwargs[RoboGymEnv.KW_OBSERVATION_NODES] = obs_nodes

        super().__init__(**kwargs)

    @staticmethod
    def set_robot_defaults(kwargs: dict[str, Any]):

        # prepare robot model set it in the kwargs
        robot_model: Panda | None = kwargs.get(RoboGymEnv.KW_ROBOT_MODEL_OBJECT)
        if robot_model is None:
            robot_model = Panda(
                model_key=kwargs.get(PandaBaseEnv.KW_PANDA_MODEL_KEY, "panda")
            )
            kwargs[RoboGymEnv.KW_ROBOT_MODEL_OBJECT] = robot_model

        # default action rate
        kwargs.setdefault(RoboGymEnv.KW_ACTION_RATE, 30.0)

        # default max episode steps
        kwargs.setdefault(RewardNode.KW_MAX_EPISODE_STEPS, 600)

        kwargs.setdefault(ManipulatorEePosEnv.KW_CONTINUE_EXCEPT_COLLISION, False)
        # default joint positions
        # panda_finger_joint.*: 0.04
        kwargs.setdefault(
            ManipulatorBaseEnv.KW_JOINT_POSITIONS,
            [0, -0.569, 0, -2.81, 0, 3.037, 0.741],
        )

        kwargs.setdefault(ManipulatorEePosEnv.KW_EE_ROTATION_PITCH_RANGE, math.pi)

        # values for finger joints in observation
        kwargs.setdefault(
            IsaacReachEnv.KW_ISAAC_OBS_EXTRA_STATIC_JOINTS,
            # not sure about min (-1) and max (1)
            np.array([[0.04, -1, 1], [0.04, -1, 1]], dtype=np.float32),
        )

    def get_launch_cmd(self) -> str:
        # TODO make string composition more dynamic
        return f"roslaunch panda_robot_server panda_robot_server.launch \
            rviz_gui:={self._config.get(self.KW_RVIZ_GUI_FLAG, True)} \
            gazebo_gui:={self._config.get(self.KW_GAZEBO_GUI_FLAG, True)} \
            world_name:=isaactabletop_sphere50_no_collision.world \
            reference_frame:=world \
            max_velocity_scale_factor:={self._config.get(self.KW_MAX_VELOCITY_SCALE_FACTOR, .2)} \
            action_cycle_rate:={self.action_rate} \
            objects_controller:=true \
            rs_mode:=1object \
            action_mode:={self.action_mode}\
            n_objects:=1.0 \
            object_0_model_name:=sphere50_no_collision \
            object_0_frame:=target \
            z:=0.0 "


class IsaacReachPandaSim(IsaacReachPanda):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = False
        super().__init__(**kwargs)


class IsaacReachPandaRob(IsaacReachPanda):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = True
        super().__init__(**kwargs)
