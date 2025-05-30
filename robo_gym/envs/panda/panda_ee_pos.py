from __future__ import annotations

from robo_gym.envs.base.robogym_env import *
from robo_gym.envs.manipulator.ee_pos_base import ManipulatorEePosEnv
from robo_gym.envs.panda.panda_base import PandaBaseEnv


class EndEffectorPositioningPanda(ManipulatorEePosEnv):
    def __init__(self, **kwargs):
        # not too nice - repeated in super init
        self._config = kwargs

        obs_nodes: list[ObservationNode] | None = kwargs.get(
            RoboGymEnv.KW_OBSERVATION_NODES
        )
        if obs_nodes is None:
            # having this hardcoded index is suboptimal; 1 should be correct since 0 will be the manipulator observation node
            obs_nodes = [LastActionObservationNode(**self.get_obs_node_setup_kwargs(1))]
            kwargs[RoboGymEnv.KW_OBSERVATION_NODES] = obs_nodes

        PandaBaseEnv.set_robot_defaults(kwargs)

        # these are probably not too meaningful since first 6 joints are equal to UR
        value = [1.5, 0.25, 0.5, 1.0, 0.4, 3.14, 1.0]
        kwargs.setdefault(ManipulatorEePosEnv.KW_RANDOM_JOINT_OFFSET, value)

        kwargs.setdefault(ManipulatorEePosEnv.KW_EE_TARGET_VOLUME_BOUNDING_BOX, True)
        kwargs.setdefault(ManipulatorEePosEnv.KW_EE_POSITION_X_RANGE, [0.35, 0.65])
        kwargs.setdefault(ManipulatorEePosEnv.KW_EE_POSITION_Y_RANGE, [-0.2, 0.2])
        kwargs.setdefault(ManipulatorEePosEnv.KW_EE_POSITION_Z_RANGE, [0.15, 0.5])

        super().__init__(**kwargs)

    def get_launch_cmd(self) -> str:
        # TODO make string composition more dynamic
        return f"roslaunch panda_robot_server panda_robot_server.launch \
        rviz_gui:={self._config.get(self.KW_RVIZ_GUI_FLAG, True)} \
        gazebo_gui:={self._config.get(self.KW_GAZEBO_GUI_FLAG, True)} \
        world_name:=tabletop_sphere50_no_collision_no_gravity.world \
        reference_frame:=world \
        max_velocity_scale_factor:={self._config.get(self.KW_MAX_VELOCITY_SCALE_FACTOR, .2)} \
        action_cycle_rate:={self.action_rate} \
        objects_controller:=true \
        rs_mode:=1object \
        action_mode:={self.action_mode}\
        n_objects:=1.0 \
        object_0_model_name:=sphere50_no_collision \
        object_0_frame:=target "


class EndEffectorPositioningPandaSim(EndEffectorPositioningPanda):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = False
        super().__init__(**kwargs)


class EndEffectorPositioningPandaRob(EndEffectorPositioningPanda):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = True
        super().__init__(**kwargs)
