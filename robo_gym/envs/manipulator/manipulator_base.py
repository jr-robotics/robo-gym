from __future__ import annotations

from envs.robogym_env import *
from utils.manipulator_model import ManipulatorModel


class ManipulatorBaseEnv(RoboGymEnv):

    def __init__(self, **kwargs):
        # not too nice - repeated in super init
        self._config = kwargs

        self._robot_model: ManipulatorModel | None = kwargs.get(
            RoboGymEnv.KW_ROBOT_MODEL_OBJECT
        )

        # env nodes
        action_node: ActionNode | None = kwargs.get(RoboGymEnv.KW_ACTION_NODE)
        if not action_node:
            action_node = ManipulatorActionNode(**self.get_action_node_setup_kwargs())
        kwargs[RoboGymEnv.KW_ACTION_NODE] = action_node

        obs_node: ObservationNode | None = None
        obs_nodes: list[ObservationNode] | None = kwargs.get(
            RoboGymEnv.KW_OBSERVATION_NODES
        )
        if obs_nodes is None:
            obs_nodes = []
            kwargs[RoboGymEnv.KW_OBSERVATION_NODES] = obs_nodes
        for current_node in obs_nodes:
            if isinstance(current_node, ManipulatorObservationNode):
                obs_node = current_node
                break
        if not obs_node:
            obs_node = ManipulatorObservationNode(**self.get_obs_node_setup_kwargs(0))
            obs_nodes.insert(0, obs_node)

        reward_node: RewardNode | None = kwargs.get(RoboGymEnv.KW_REWARD_NODE)
        if not reward_node:
            reward_node = ManipulatorRewardNode(**self.get_reward_node_setup_kwargs())
        kwargs[RoboGymEnv.KW_REWARD_NODE] = reward_node

        super().__init__(**kwargs)


class ManipulatorActionNode(ActionNode):
    # TODO
    # consider other action modes - current impl corresponds to ABS_POS only

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._robot_model: ManipulatorModel | None = kwargs.get(
            RoboGymEnv.KW_ROBOT_MODEL_OBJECT
        )
        self._joint_names = kwargs.get("joint_names")
        self._fixed_joint_names: list[str] = []
        self._controlled_joint_names: list[str] = []
        for joint_name in self._joint_names:
            fix_arg_name = "fix_" + joint_name
            if kwargs.get(fix_arg_name, False):
                self._fixed_joint_names.append(joint_name)
            else:
                self._controlled_joint_names.append(joint_name)

    def setup(self, **kwargs):
        super().setup(**kwargs)

    def get_action_space(self) -> gym.spaces.Box:
        length = len(self._controlled_joint_names)
        return gym.spaces.Box(
            low=np.full(length, -1.0),
            high=np.full(length, 1.0),
            dtype=np.float32,
        )

    def env_action_to_rs_action(self, env_action: NDArray, **kwargs) -> NDArray:
        # TODO: allow for custom normalization

        # optimization potential, but more concise than it was:
        # start with default positions and overwrite non-fixed joints with values from env action
        normalized_full_action = self._robot_model.normalize_joint_values(
            self._robot_model.joint_positions
        )
        source_index = 0
        for joint_index in range(len(self._joint_names)):
            joint_name = self._joint_names[joint_index]
            if joint_name in self._controlled_joint_names:
                normalized_full_action[joint_index] = env_action[source_index]
                source_index += 1

        result = self._robot_model.denormalize_joint_values(normalized_full_action)
        return result


class ManipulatorObservationNode(ObservationNode):
    # TODO implement methods

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, **kwargs):
        super().setup(**kwargs)

    def get_observation_space_part(self) -> gym.spaces.Box:
        pass

    def rs_state_to_observation_part(
        self, rs_state_array: NDArray, rs_state_dict: dict[str, float], **kwargs
    ) -> NDArray:
        pass

    def get_reset_state_part_float(self) -> dict[str, float]:
        return super().get_reset_state_part_float()

    def get_reset_state_part_str(self) -> dict[str, str]:
        return super().get_reset_state_part_str()

    def get_reset_state_part_state_dict(self) -> dict[str, float]:
        return super().get_reset_state_part_state_dict()

    def get_reset_state_part_state_array_values(self) -> NDArray:
        return super().get_reset_state_part_state_array_values()

    def get_reset_state_part_order(self) -> int:
        return super().get_reset_state_part_order()


class ManipulatorRewardNode(RewardNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, **kwargs):
        super().setup(**kwargs)

    def get_reward(
        self,
        rs_state_array: NDArray,
        rs_state_dict: dict[str, float],
        env_action: NDArray,
        **kwargs,
    ) -> Tuple[float, bool, dict]:

        done = False
        info = {}

        # Check if robot is in collision
        collision = rs_state_dict["in_collision"] == 1
        if collision:
            done = True
            info["final_status"] = "collision"

        elif self.env.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "success"

        return 0, done, info
