from __future__ import annotations

from robo_gym.envs.manipulator.ee_pos_base import *
from robo_gym.envs.manipulator.manipulator_base import *


class IsaacReachEnv(ManipulatorEePosEnv):
    """
    Base class for compatibility environments for agents trained with IsaacLab reach tasks.
    Environment actions and observations use the spaces from the corresponding IsaacLab environments.
    On the back-end, a UR robot server as for our End Effector Positioning environments is used.
    The environment nodes are responsible for achieving compatibility between that robot server and the spaces as used in IsaacLab.
    """

    KW_ISAAC_SCALE = "isaac_scale"

    def __init__(self, **kwargs):
        # not too nice - repeated in super init
        self._config = kwargs

        RoboGymEnv.set_default(kwargs, ManipulatorEePosEnv.KW_EE_ROTATION_MATTERS, True)
        RoboGymEnv.set_default(kwargs, IsaacReachEnv.KW_ISAAC_SCALE, 0.5)

        # env nodes
        action_node: ActionNode | None = kwargs.get(RoboGymEnv.KW_ACTION_NODE)
        if not action_node:
            action_node = IsaacReachActionNode(**self.get_action_node_setup_kwargs())
        kwargs[RoboGymEnv.KW_ACTION_NODE] = action_node

        super().__init__(**kwargs)

    def create_main_observation_node(self, node_index: int = 0, **kwargs):
        return IsaacReachObservationNode(**self.get_obs_node_setup_kwargs(node_index))


class IsaacReachObservationNode(ManipulatorEePosObservationNode):

    # def __init__(self, **kwargs):
    #    super().__init__(**kwargs)

    def get_observation_space_part(self) -> gym.spaces.Box:
        num_joints = len(self._robot_model.joint_names)

        # joint positions
        tolerance_abs = self._robot_model.denormalize_joint_values(
            np.array([self.joint_position_tolerance_normalized] * num_joints)
        )
        max_joint_positions = (
            self._robot_model.get_max_joint_positions()
            + tolerance_abs
            - self._robot_model.joint_positions
        )
        min_joint_positions = (
            self._robot_model.get_min_joint_positions()
            - tolerance_abs
            - self._robot_model.joint_positions
        )

        # velocities
        max_joint_velocities = np.array([np.inf] * num_joints)
        min_joint_velocities = -np.array([np.inf] * num_joints)

        # ee target
        ee_target_max = np.concatenate((np.array([np.inf] * 3), np.array([1.0] * 4)))
        ee_target_min = -1 * ee_target_max

        obs_space_max = np.concatenate(
            (max_joint_positions, max_joint_velocities, ee_target_max)
        )
        obs_space_min = np.concatenate(
            (min_joint_positions, min_joint_velocities, ee_target_min)
        )
        obs_space = gym.spaces.Box(high=obs_space_max, low=obs_space_min)
        return obs_space

    def rs_state_to_observation_part(
        self, rs_state_array: NDArray, rs_state_dict: dict[str, float], **kwargs
    ) -> NDArray:
        # previous action is added by a separate LastActionObservationNode
        joint_positions = (
            self.extract_joint_positions_from_rs_state_dict(rs_state_dict)
            - self._robot_model.joint_positions
        )
        joint_velocities = self.extract_joint_velocities_from_rs_state_dict(
            rs_state_dict
        )
        command = np.array(self.get_target_pose())
        obs = np.concatenate((joint_positions, joint_velocities, command))
        return obs


class IsaacReachActionNode(ManipulatorActionNode):

    def __init__(self, **kwargs):
        self.scale = kwargs.get(IsaacReachEnv.KW_ISAAC_SCALE, 0.5)
        super().__init__(**kwargs)

    def get_default_joint_positions(self) -> NDArray:
        return self._robot_model.joint_positions

    def env_action_to_rs_action(self, env_action: NDArray, **kwargs) -> NDArray:
        rs_action = self.robot_model.joint_positions + env_action * self.scale
        rs_action = self.robot_model.reorder_joints_for_rs(rs_action)
        return rs_action

    def get_action_space(self) -> gym.spaces.Box:
        max_joint_positions = self._robot_model.get_max_joint_positions()
        min_joint_positions = self._robot_model.get_min_joint_positions()

        action_max = (
            max_joint_positions - self._robot_model.joint_positions
        ) / self.scale
        action_min = (
            min_joint_positions - self._robot_model.joint_positions
        ) / self.scale

        action_space = gym.spaces.Box(high=action_max, low=action_min)
        return action_space
