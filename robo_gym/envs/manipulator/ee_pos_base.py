from typing import Tuple

import gymnasium as gym
import numpy as np
from numpy._typing import NDArray
from scipy.spatial.transform import Rotation as R

from robo_gym.envs.manipulator.manipulator_base import (
    ManipulatorBaseEnv,
    ManipulatorActionNode,
    ManipulatorObservationNode,
    ManipulatorRewardNode,
)


class ManipulatorEePosEnv(ManipulatorBaseEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO


class ManipulatorEePosActionNode(ManipulatorActionNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def env_action_to_rs_action(self, env_action: NDArray, **kwargs) -> NDArray:
        return super().env_action_to_rs_action(env_action, **kwargs)

    def get_action_space(self) -> gym.spaces.Box:
        return super().get_action_space()

    def get_reset_state_part_state_dict(self) -> dict[str, float]:
        return super().get_reset_state_part_state_dict()


class ManipulatorEePosObservationNode(ManipulatorObservationNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_observation_space_part(self) -> gym.spaces.Box:
        return super().get_observation_space_part()

    def rs_state_to_observation_part(
        self, rs_state_array: NDArray, rs_state_dict: dict[str, float], **kwargs
    ) -> NDArray:

        super_result = super().rs_state_to_observation_part(
            rs_state_array, rs_state_dict, **kwargs
        )
        rs_state = rs_state_dict
        # Target polar coordinates
        # Transform cartesian coordinates of target to polar coordinates
        # with respect to the end effector frame
        target_coord = np.array(
            [
                rs_state["object_0_to_ref_translation_x"],
                rs_state["object_0_to_ref_translation_y"],
                rs_state["object_0_to_ref_translation_z"],
            ]
        )

        ee_to_ref_frame_translation = np.array(
            [
                rs_state["ee_to_ref_translation_x"],
                rs_state["ee_to_ref_translation_y"],
                rs_state["ee_to_ref_translation_z"],
            ]
        )

        ee_to_ref_frame_quaternion = np.array(
            [
                rs_state["ee_to_ref_rotation_x"],
                rs_state["ee_to_ref_rotation_y"],
                rs_state["ee_to_ref_rotation_z"],
                rs_state["ee_to_ref_rotation_w"],
            ]
        )

        ee_to_ref_frame_rotation = R.from_quat(ee_to_ref_frame_quaternion)
        ref_frame_to_ee_rotation = ee_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_ee_quaternion = ref_frame_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_ee_translation = -ref_frame_to_ee_rotation.apply(
            ee_to_ref_frame_translation
        )

        target_coord_ee_frame = utils.change_reference_frame(
            target_coord, ref_frame_to_ee_translation, ref_frame_to_ee_quaternion
        )
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

        # Compose environment state
        state = np.concatenate(
            (
                target_polar,
                super_result,
                target_coord,
                ee_to_ref_frame_translation,
                self.previous_action,
            )
        )

        return state.astype(np.float32)


class ManipulatorEePosRewardNode(ManipulatorRewardNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_reward(
        self,
        rs_state_array: NDArray,
        rs_state_dict: dict[str, float],
        env_action: NDArray,
        **kwargs
    ) -> Tuple[float, bool, dict]:
        result: Tuple[float, bool, dict] = super().get_reward(
            rs_state_array, rs_state_dict, env_action, **kwargs
        )
        # TODO
        return result
