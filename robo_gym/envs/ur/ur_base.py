from typing import Any

from robo_gym.envs.base.robogym_env import RoboGymEnv, RewardNode
from robo_gym.envs.manipulator.manipulator_base import (
    ManipulatorBaseEnv,
    ManipulatorActionNode,
)
from robo_gym.utils.ur_utils import UR


class URBaseEnv2(ManipulatorBaseEnv):

    KW_UR_MODEL_KEY = "ur_model"

    def __init__(self, **kwargs):

        URBaseEnv2.set_robot_defaults(kwargs)

        super().__init__(**kwargs)

    @staticmethod
    def set_robot_defaults(kwargs: dict[str, Any]):

        # prepare UR model depending on ur_model and set it in the kwargs
        ur_model: UR | None = kwargs.get(RoboGymEnv.KW_ROBOT_MODEL_OBJECT)
        if ur_model is None:
            ur_model = UR(model_key=kwargs.get(URBaseEnv2.KW_UR_MODEL_KEY))
            kwargs[RoboGymEnv.KW_ROBOT_MODEL_OBJECT] = ur_model

        # default action rate
        RoboGymEnv.set_default(kwargs, RoboGymEnv.KW_ACTION_RATE, 20.0)

        # default max episode steps
        RoboGymEnv.set_default(kwargs, RewardNode.KW_MAX_EPISODE_STEPS, 300)

        # default fixed joints: last joint
        prefix_fix = ManipulatorActionNode.KW_PREFIX_FIX_JOINT
        last_joint_name = ur_model.joint_names[-1]
        kw_fix_last_joint = prefix_fix + last_joint_name
        RoboGymEnv.set_default(kwargs, kw_fix_last_joint, True)

        # default joint positions
        RoboGymEnv.set_default(
            kwargs,
            ManipulatorBaseEnv.KW_JOINT_POSITIONS,
            [
                0.0,
                -2.5,
                1.5,
                0.0,
                -1.4,
                0.0,
            ],
        )

    def get_launch_cmd(self) -> str:
        # TODO make string composition more dynamic
        return f"roslaunch ur_robot_server ur_robot_server.launch \
        rviz_gui:={self._config.get(self.KW_RVIZ_GUI_FLAG, True)} \
        gazebo_gui:={self._config.get(self.KW_GAZEBO_GUI_FLAG, True)} \
        world_name:=empty.world \
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:={self.action_rate} \
        rs_mode:=only_robot \
        ur_model:={self.ur_model_key}"

    @property
    def ur_model_key(self) -> str:
        return self._config.get(URBaseEnv2.KW_UR_MODEL_KEY)


class EmptyEnvironment2URSim(URBaseEnv2):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = False
        super().__init__(**kwargs)


class EmptyEnvironment2URRob(URBaseEnv2):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = True
        super().__init__(**kwargs)
