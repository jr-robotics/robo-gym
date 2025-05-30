from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from robo_gym.envs.base.robogym_env import (
    RoboGymEnv,
    ActionNode,
    ObservationNode,
    RewardNode,
)
from robo_gym.utils.manipulator_model import ManipulatorModel


class ManipulatorBaseEnv(RoboGymEnv):

    KW_RS_JOINT_NAMES = "rs_joint_names"
    KW_JOINT_POSITIONS = "joint_position"
    RS_STATE_KEY_SUFFIX_JOINT_POSITION = "_position"
    RS_STATE_KEY_SUFFIX_JOINT_VELOCITY = "_velocity"

    KW_ACTION_MODE = "action_mode"
    ACTION_MODE_ABS_POS = "abs_pos"
    ACTION_MODE_DELTA_POS = "delta_pos"
    ACTION_MODE_ABS_VEL = "abs_vel"
    ACTION_MODE_DELTA_VEL = "delta_vel"

    KW_MAX_VELOCITY_SCALE_FACTOR = "max_velocity_scale_factor"

    def __init__(self, **kwargs):
        # not too nice - repeated in super init
        self._config = kwargs

        self._robot_model: ManipulatorModel | None = kwargs.get(
            RoboGymEnv.KW_ROBOT_MODEL_OBJECT
        )
        if self.KW_JOINT_POSITIONS in kwargs:
            # TODO check values
            self._robot_model.joint_positions = kwargs[self.KW_JOINT_POSITIONS]

        self.action_mode = self._config.setdefault(
            self.KW_ACTION_MODE, self.ACTION_MODE_ABS_POS
        )
        # we only support this one action mode for now
        assert self.action_mode == self.ACTION_MODE_ABS_POS

        # env nodes
        action_node: ActionNode | None = kwargs.get(RoboGymEnv.KW_ACTION_NODE)
        if not action_node:
            action_node = ManipulatorActionNode(**self.get_action_node_setup_kwargs())
        kwargs[RoboGymEnv.KW_ACTION_NODE] = action_node

        RoboGymEnv.assure_instance_of_type_in_list(
            kwargs,
            RoboGymEnv.KW_OBSERVATION_NODES,
            ManipulatorObservationNode,
            True,
            self.create_main_observation_node,
            {},
        )

        reward_node: RewardNode | None = kwargs.get(RoboGymEnv.KW_REWARD_NODE)
        if not reward_node:
            reward_node = ManipulatorRewardNode(**self.get_reward_node_setup_kwargs())
        kwargs[RoboGymEnv.KW_REWARD_NODE] = reward_node

        super().__init__(**kwargs)

    def create_main_observation_node(self, node_index: int = 0, **kwargs):
        return ManipulatorObservationNode(**self.get_obs_node_setup_kwargs(node_index))


class ManipulatorActionNode(ActionNode):
    # TODO
    # consider other action modes - current impl corresponds to ABS_POS only

    KW_PREFIX_FIX_JOINT = "fix_"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._robot_model: ManipulatorModel | None = kwargs.get(
            RoboGymEnv.KW_ROBOT_MODEL_OBJECT
        )
        assert self._robot_model is not None
        self._joint_names: list[str] = self._robot_model.joint_names
        self._fixed_joint_names: list[str] = []
        self._controlled_joint_names: list[str] = []
        for joint_name in self._joint_names:
            fix_arg_name = self.KW_PREFIX_FIX_JOINT + joint_name
            if kwargs.get(fix_arg_name, False):
                self._fixed_joint_names.append(joint_name)
            else:
                self._controlled_joint_names.append(joint_name)

    def get_action_space(self) -> gym.spaces.Box:
        length = len(self._controlled_joint_names)
        return gym.spaces.Box(
            low=np.full(length, -1.0, dtype=np.float32),
            high=np.full(length, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def env_action_to_rs_action(self, env_action: NDArray, **kwargs) -> NDArray:
        # optimization potential, but more concise than it was:
        # start with default positions and overwrite non-fixed joints with values from env action
        normalized_full_action = self._robot_model.normalize_joint_values(
            np.array(self._robot_model.joint_positions, dtype=np.float32)
        )
        source_index = 0
        for joint_index in range(len(self._joint_names)):
            joint_name = self._joint_names[joint_index]
            if joint_name in self._controlled_joint_names:
                normalized_full_action[joint_index] = env_action[source_index]
                source_index += 1

        denormalized_full_action = self._robot_model.denormalize_joint_values(
            normalized_full_action
        )
        result = self._robot_model.reorder_joints_for_rs(
            denormalized_full_action
        ).astype(np.float32)
        return result

    def get_reset_state_part_state_dict(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for joint_index in range(len(self._joint_names)):
            joint_name = self._robot_model.remote_joint_names[joint_index]
            state_key = (
                joint_name + ManipulatorBaseEnv.RS_STATE_KEY_SUFFIX_JOINT_POSITION
            )
            result[state_key] = self._robot_model.joint_positions[joint_index]
        return result

    @property
    def robot_model(self):
        return self._robot_model


class ManipulatorObservationNode(ObservationNode):
    KW_JOINT_POSITION_TOLERANCE_NORMALIZED = "joint_position_tolerance_normalized"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._robot_model: ManipulatorModel | None = kwargs.get(
            RoboGymEnv.KW_ROBOT_MODEL_OBJECT
        )
        assert self._robot_model is not None
        self._joint_names: list[str] = (
            self._robot_model.joint_names
        )  # kwargs.get("joint_names")

    def get_observation_space_part(self) -> gym.spaces.Box:
        # Joint position range tolerance
        num_joints = len(self._robot_model.joint_names)
        pos_tolerance = np.full(num_joints, self.joint_position_tolerance_normalized)

        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(num_joints, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(num_joints, -1.0), pos_tolerance)
        # Joint velocities range
        max_joint_velocities = np.array([np.inf] * num_joints)
        min_joint_velocities = -np.array([np.inf] * num_joints)
        # Definition of environment observation_space
        max_obs = np.concatenate(
            (max_joint_positions, max_joint_velocities), dtype=np.float32
        )
        min_obs = np.concatenate(
            (min_joint_positions, min_joint_velocities), dtype=np.float32
        )

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def rs_state_to_observation_part(
        self, rs_state_array: NDArray, rs_state_dict: dict[str, float], **kwargs
    ) -> NDArray:
        # from old impl, but why are only the joint positions normalized?
        joint_positions = self.extract_normalized_joint_positions_from_rs_state_dict(
            rs_state_dict
        )

        # Joint Velocities
        joint_velocities = self.extract_joint_velocities_from_rs_state_dict(
            rs_state_dict
        )

        # Compose environment state
        state = np.concatenate((joint_positions, joint_velocities))

        return state.astype(np.float32)

    def extract_joint_velocities_from_rs_state_dict(self, rs_state_dict):
        joint_velocities = []
        joint_velocities_keys = [
            joint_name + ManipulatorBaseEnv.RS_STATE_KEY_SUFFIX_JOINT_VELOCITY
            for joint_name in self._robot_model.remote_joint_names
        ]
        for velocity in joint_velocities_keys:
            joint_velocities.append(rs_state_dict[velocity])
        joint_velocities = np.array(joint_velocities)
        return joint_velocities

    def extract_normalized_joint_positions_from_rs_state_dict(self, rs_state_dict):
        joint_positions = self.extract_joint_positions_from_rs_state_dict(rs_state_dict)
        # Normalize joint position values
        joint_positions = self._robot_model.normalize_joint_values(
            joints=joint_positions
        )
        return joint_positions

    def extract_joint_positions_from_rs_state_dict(self, rs_state_dict):
        joint_positions = []
        joint_positions_keys = [
            joint_name + ManipulatorBaseEnv.RS_STATE_KEY_SUFFIX_JOINT_POSITION
            for joint_name in self._robot_model.remote_joint_names
        ]
        for position in joint_positions_keys:
            joint_positions.append(rs_state_dict[position])
        joint_positions = np.array(joint_positions)
        return joint_positions

    @property
    def joint_position_tolerance_normalized(self) -> float:
        return self._config.get(
            ManipulatorObservationNode.KW_JOINT_POSITION_TOLERANCE_NORMALIZED, 0.1
        )


class ManipulatorRewardNode(RewardNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_reward(
        self,
        rs_state_array: NDArray,
        rs_state_dict: dict[str, float],
        env_action: NDArray,
        **kwargs,
    ) -> Tuple[float, bool, dict]:

        info = self.env.get_default_info()
        done = False

        # Check if robot is in collision
        collision = rs_state_dict["in_collision"] == 1
        if collision:
            done = True
            info[RoboGymEnv.INFO_KW_FINAL_STATUS] = RoboGymEnv.FINAL_STATUS_COLLISION

        elif (
            self.max_episode_steps is not None
            and self.env.elapsed_steps >= self.max_episode_steps
        ):
            done = True
            info[RoboGymEnv.INFO_KW_FINAL_STATUS] = RoboGymEnv.FINAL_STATUS_SUCCESS

        return 0.0, done, info
