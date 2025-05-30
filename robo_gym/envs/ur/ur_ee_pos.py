from __future__ import annotations

from robo_gym.envs.base.robogym_env import *
from robo_gym.envs.manipulator.ee_pos_base import ManipulatorEePosEnv
from robo_gym.envs.ur.ur_base import URBaseEnv2


class EndEffectorPositioning2UR(ManipulatorEePosEnv):
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

        URBaseEnv2.set_robot_defaults(kwargs)
        kwargs.setdefault(RoboGymEnv.KW_ACTION_RATE, 10.0)
        kwargs.setdefault(
            ManipulatorEePosEnv.KW_RANDOM_JOINT_OFFSET, [1.5, 0.25, 0.5, 1.0, 0.4, 3.14]
        )

        super().__init__(**kwargs)

    def get_launch_cmd(self) -> str:
        # TODO make string composition more dynamic
        return f"roslaunch ur_robot_server ur_robot_server.launch \
            rviz_gui:={self._config.get(self.KW_RVIZ_GUI_FLAG, True)} \
            gazebo_gui:={self._config.get(self.KW_GAZEBO_GUI_FLAG, True)} \
            world_name:=tabletop_sphere50_no_collision.world \
            reference_frame:=base_link \
            max_velocity_scale_factor:={self._config.get(self.KW_MAX_VELOCITY_SCALE_FACTOR, .1)} \
            action_cycle_rate:={self.action_rate} \
            objects_controller:=true \
            rs_mode:=1object \
            n_objects:=1.0 \
            object_0_model_name:=sphere50_no_collision \
            object_0_frame:=target \
            ur_model:={self.ur_model_key}"

    @property
    def ur_model_key(self) -> str:
        return self._config.get(URBaseEnv2.KW_UR_MODEL_KEY)


class EndEffectorPositioning2URSim(EndEffectorPositioning2UR):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = False
        super().__init__(**kwargs)


class EndEffectorPositioning2URRob(EndEffectorPositioning2UR):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = True
        super().__init__(**kwargs)
