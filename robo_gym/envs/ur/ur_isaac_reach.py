from __future__ import annotations

from robo_gym.envs.base.robogym_env import *
from robo_gym.envs.manipulator.isaac_reach import *
from robo_gym.envs.ur.ur_base import URBaseEnv2
from robo_gym.utils.ur_utils import UR


class IsaacReachUR(IsaacReachEnv):
    def __init__(self, **kwargs):
        # not too nice - repeated in super init
        self._config = kwargs

        kwargs.setdefault(URBaseEnv2.KW_UR_MODEL_KEY, "ur10")

        IsaacReachUR.set_robot_defaults(kwargs)

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

        # prepare UR model depending on ur_model and set it in the kwargs
        ur_model: UR | None = kwargs.get(RoboGymEnv.KW_ROBOT_MODEL_OBJECT)
        if ur_model is None:
            ur_model = UR(model_key=kwargs.get(URBaseEnv2.KW_UR_MODEL_KEY))
            kwargs[RoboGymEnv.KW_ROBOT_MODEL_OBJECT] = ur_model

        # default action rate
        kwargs.setdefault(RoboGymEnv.KW_ACTION_RATE, 30.0)

        # default max episode steps
        kwargs.setdefault(RewardNode.KW_MAX_EPISODE_STEPS, 600)

        kwargs.setdefault(ManipulatorEePosEnv.KW_CONTINUE_EXCEPT_COLLISION, True)
        # default joint positions
        kwargs.setdefault(
            ManipulatorBaseEnv.KW_JOINT_POSITIONS, [0, -1.7120, 1.7120, 0, 0, 0]
        )

        kwargs.setdefault(ManipulatorEePosEnv.KW_EE_ROTATION_PITCH_RANGE, math.pi / 2)

    def get_launch_cmd(self) -> str:
        # TODO make string composition more dynamic
        # TODO duplicated from EndEffectorPositioning2UR
        return f"roslaunch ur_robot_server ur_robot_server.launch \
            rviz_gui:={self._config.get(self.KW_RVIZ_GUI_FLAG, True)} \
            gazebo_gui:={self._config.get(self.KW_GAZEBO_GUI_FLAG, True)} \
            world_name:=isaactabletop_sphere50_no_collision.world \
            reference_frame:=base_link \
            ee_frame:=tool1 \
            max_velocity_scale_factor:={self._config.get(self.KW_MAX_VELOCITY_SCALE_FACTOR, .2)} \
            action_cycle_rate:={self.action_rate} \
            objects_controller:=true \
            rs_mode:=1object \
            n_objects:=1.0 \
            object_0_model_name:=sphere50_no_collision \
            object_0_frame:=target \
            z:=0.0 \
            ur_model:={self.ur_model_key}"

    @property
    def ur_model_key(self) -> str:
        return self._config.get(URBaseEnv2.KW_UR_MODEL_KEY)


class IsaacReachURSim(IsaacReachUR):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = False
        super().__init__(**kwargs)


class IsaacReachURRob(IsaacReachUR):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = True
        super().__init__(**kwargs)
