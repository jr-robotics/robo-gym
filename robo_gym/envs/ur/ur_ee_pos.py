from robo_gym.envs.manipulator.ee_pos_base import ManipulatorEePosEnv
from robo_gym.envs.ur.ur_base import URBaseEnv2


class EndEffectorPositioningUR2(ManipulatorEePosEnv):
    def __init__(self, **kwargs):

        URBaseEnv2.set_robot_defaults(kwargs)

        super().__init__(**kwargs)

    def get_launch_cmd(self) -> str:
        # TODO merge legacy hardcoded cmd with superclass cmd; introduce more fields to control it
        legacy_cmd = "roslaunch ur_robot_server ur_robot_server.launch \
            world_name:=tabletop_sphere50_no_collision.world \
            reference_frame:=base_link \
            max_velocity_scale_factor:=0.1 \
            action_cycle_rate:=10 \
            rviz_gui:=true \
            gazebo_gui:=true \
            objects_controller:=true \
            rs_mode:=1object \
            n_objects:=1.0 \
            object_0_model_name:=sphere50_no_collision \
            object_0_frame:=target"
        return super().get_launch_cmd()


class EndEffectorPositioning2URSim(EndEffectorPositioningUR2):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = False
        super().__init__(**kwargs)


class EndEffectorPositioning2URRob(EndEffectorPositioningUR2):

    def __init__(self, **kwargs):
        kwargs[self.KW_REAL_ROBOT] = True
        super().__init__(**kwargs)
