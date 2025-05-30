from __future__ import annotations

from robo_gym.envs.base.robogym_env import *
from robo_gym.envs.manipulator.manipulator_base import (
    ManipulatorBaseEnv,
    ManipulatorActionNode,
)
from robo_gym.utils.panda_utils import Panda


class PandaBaseEnv(ManipulatorBaseEnv):

    KW_PANDA_MODEL_KEY = "panda_model"

    def __init__(self, **kwargs):

        PandaBaseEnv.set_robot_defaults(kwargs)

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
        kwargs.setdefault(RewardNode.KW_MAX_EPISODE_STEPS, 300)

        # default fixed joints: last joint
        prefix_fix = ManipulatorActionNode.KW_PREFIX_FIX_JOINT
        last_joint_name = robot_model.joint_names[-1]
        kw_fix_last_joint = prefix_fix + last_joint_name
        kwargs.setdefault(kw_fix_last_joint, True)

        # default joint positions
        value = [-0.018, -0.76, 0.02, -2.342, 0.03, 1.541, 0.753]

        kwargs.setdefault(ManipulatorBaseEnv.KW_JOINT_POSITIONS, value)

    def get_launch_cmd(self) -> str:
        # TODO make string composition more dynamic
        return f"roslaunch panda_robot_server panda_robot_server.launch \
        rviz_gui:={self._config.get(self.KW_RVIZ_GUI_FLAG, True)} \
        gazebo_gui:={self._config.get(self.KW_GAZEBO_GUI_FLAG, True)} \
        world_name:=empty_no_gravity.world \
        reference_frame:=world \
        max_velocity_scale_factor:={self._config.get(self.KW_MAX_VELOCITY_SCALE_FACTOR, .2)} \
        action_cycle_rate:={self.action_rate} \
        rs_mode:=only_robot \
        action_mode:={self.action_mode}"


class EmptyEnvironmentPandaSim(PandaBaseEnv):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = False
        super().__init__(**kwargs)


class EmptyEnvironmentPandaRob(PandaBaseEnv):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = True
        super().__init__(**kwargs)
