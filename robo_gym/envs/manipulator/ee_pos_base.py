from __future__ import annotations

import math
from typing import Tuple, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType
from numpy._typing import NDArray
from scipy.spatial.transform import Rotation as R

from robo_gym.envs.base.robogym_env import RoboGymEnv, RewardNode
from robo_gym.envs.manipulator.manipulator_base import (
    ManipulatorBaseEnv,
    ManipulatorActionNode,
    ManipulatorObservationNode,
    ManipulatorRewardNode,
)
from robo_gym.utils import utils
from robo_gym.utils.manipulator_model import ManipulatorModel


class ManipulatorEePosEnv(ManipulatorBaseEnv):

    KW_EE_TARGET_POSE = "ee_target_pose"
    KW_EE_TARGET_VOLUME_BOUNDING_BOX = "ee_target_volume_bounding_box"
    KW_EE_DISTANCE_THRESHOLD = "ee_distance_threshold"
    KW_EE_ROTATION_THRESHOLD = "ee_rotation_threshold"
    KW_EE_ROTATION_MATTERS = "ee_rotation_matters"
    KW_EE_ROTATION_ROLL_RANGE = "ee_rotation_roll_range"
    KW_EE_ROTATION_PITCH_RANGE = "ee_rotation_pitch_range"
    KW_EE_ROTATION_YAW_RANGE = "ee_rotation_yaw_range"
    KW_EE_ROTATION_QUAT_UNIQUE = "ee_rotation_quat_unique"
    KW_EE_ROTATION_RPY_SEQ = "ee_rotation_rpy_seq"
    KW_EE_POSITION_X_RANGE = "ee_rotation_pos_x_range"
    KW_EE_POSITION_Y_RANGE = "ee_rotation_pos_y_range"
    KW_EE_POSITION_Z_RANGE = "ee_rotation_pos_z_range"
    KW_CONTINUE_ON_SUCCESS = "continue_on_success"
    KW_CONTINUE_EXCEPT_COLLISION = "continue_except_collision"
    KW_RANDOMIZE_START = "randomize_start"
    KW_RANDOM_JOINT_OFFSET = "random_joint_offset"

    def __init__(self, **kwargs):
        # not too nice - repeated in super init
        self._config = kwargs

        self._robot_model: ManipulatorModel | None = kwargs.get(
            RoboGymEnv.KW_ROBOT_MODEL_OBJECT
        )
        if self.KW_JOINT_POSITIONS in kwargs:
            # TODO check values
            self._robot_model.joint_positions = kwargs[self.KW_JOINT_POSITIONS]

        RoboGymEnv.assure_instance_of_type_in_list(
            kwargs,
            RoboGymEnv.KW_OBSERVATION_NODES,
            ManipulatorEePosObservationNode,
            True,
            self.create_main_observation_node,
            {},
        )

        # Note: to rebuild legacy observation, subclass may add a LastActionObservationNode to the observation nodes
        reward_node: RewardNode | None = kwargs.get(RoboGymEnv.KW_REWARD_NODE)
        if not reward_node:
            reward_node = ManipulatorEePosRewardNode(
                **self.get_reward_node_setup_kwargs()
            )
        kwargs[RoboGymEnv.KW_REWARD_NODE] = reward_node

        super().__init__(**kwargs)

        self.last_position = np.zeros(self.get_robot_model().joint_count)
        self.successful_ending = False

    def create_main_observation_node(self, node_index: int = 0, **kwargs):
        return ManipulatorEePosObservationNode(
            **self.get_obs_node_setup_kwargs(node_index)
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _prepare_state(self):
        super()._prepare_state()

        robot_model = self.get_robot_model()

        # initial joint positions, default: from config
        joint_positions = np.array(self._config.get(self.KW_JOINT_POSITIONS))
        if (
            self._config.get(self.KW_CONTINUE_ON_SUCCESS)
            or self._config.get(self.KW_CONTINUE_EXCEPT_COLLISION)
        ) and self.successful_ending:
            # if success and continue on success: to last positions
            joint_positions = self.last_position
        elif self._config.get(self.KW_RANDOMIZE_START):
            # if randomize start: configured joint positions + random offset
            random_offset = self._config.get(self.KW_RANDOM_JOINT_OFFSET)
            if len(random_offset) == robot_model.joint_count:
                np_random_offset = np.array(random_offset)
                joint_positions = robot_model.get_random_offset_joint_positions(
                    joint_positions, np_random_offset, np_random=self.np_random
                )
        # joint positions from robot model will put it into state for new rs state to set by the Action Node
        self._robot_model.joint_positions = joint_positions
        self.successful_ending = False

        assert isinstance(self._reward_node, ManipulatorEePosRewardNode)
        # initialize ee target:
        # from configured ee target pose if any
        new_ee_target = self._config.get(self.KW_EE_TARGET_POSE)
        # else random
        if new_ee_target is None:

            roll_range = utils.get_config_range(
                self._config, self.KW_EE_ROTATION_ROLL_RANGE
            )
            pitch_range = utils.get_config_range(
                self._config, self.KW_EE_ROTATION_PITCH_RANGE
            )
            yaw_range = utils.get_config_range(
                self._config, self.KW_EE_ROTATION_YAW_RANGE
            )

            quat_unique = self._config.get(self.KW_EE_ROTATION_QUAT_UNIQUE, False)
            rpy_seq = self._config.get(self.KW_EE_ROTATION_RPY_SEQ, "xyz")

            if self._config.get(self.KW_EE_TARGET_VOLUME_BOUNDING_BOX, False):
                pos_x_range = utils.get_config_range(
                    self._config, self.KW_EE_POSITION_X_RANGE
                )
                pos_y_range = utils.get_config_range(
                    self._config, self.KW_EE_POSITION_Y_RANGE
                )
                pos_z_range = utils.get_config_range(
                    self._config, self.KW_EE_POSITION_Z_RANGE
                )

                new_ee_target = utils.create_random_bounding_box_pose_quat(
                    pos_x_range,
                    pos_y_range,
                    pos_z_range,
                    np_random=self.np_random,
                    roll_range=roll_range,
                    pitch_range=pitch_range,
                    yaw_range=yaw_range,
                    quat_unique=quat_unique,
                    seq=rpy_seq,
                )

            else:
                new_ee_target = robot_model.get_random_workspace_pose_quat(
                    np_random=self.np_random,
                    roll_range=roll_range,
                    pitch_range=pitch_range,
                    yaw_range=yaw_range,
                    quat_unique=quat_unique,
                    seq=rpy_seq,
                )
        try:
            new_ee_target = new_ee_target.tolist()
        except TypeError:
            pass
        except AttributeError:
            pass
        self._reward_node.set_ee_target(new_ee_target)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        final_status = info.get(self.INFO_KW_FINAL_STATUS)
        if terminated and (
            final_status == self.FINAL_STATUS_SUCCESS
            or self._config.get(self.KW_CONTINUE_EXCEPT_COLLISION)
            and final_status != self.FINAL_STATUS_COLLISION
        ):
            # store last position for later
            assert isinstance(self._action_node, ManipulatorActionNode)
            robot_model = self.get_robot_model()

            joint_positions = []
            joint_positions_keys = [
                joint_name + ManipulatorBaseEnv.RS_STATE_KEY_SUFFIX_JOINT_POSITION
                for joint_name in robot_model.remote_joint_names
            ]

            for position in joint_positions_keys:
                joint_positions.append(self._last_rs_state_dict[position])
            joint_positions = np.array(joint_positions)
            self.last_position = joint_positions
            # TODO extend this mechanism to also work if truncated from outside
            # may need to change the logic for going to stored joint positions
            self.successful_ending = True
        return obs, reward, terminated, truncated, info

    def get_robot_model(self) -> ManipulatorModel:
        assert isinstance(self._action_node, ManipulatorActionNode)
        return self._action_node.robot_model


class ManipulatorEePosObservationNode(ManipulatorObservationNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_observation_space_part(self) -> gym.spaces.Box:
        super_result = super().get_observation_space_part()

        # Target coordinates range
        target_range = np.full(3, np.inf)

        # Cartesian coords of the target location
        max_target_pose = np.array([np.inf] * 3)
        min_target_pose = -np.array([np.inf] * 3)

        # Cartesian coords of the end effector
        max_ee_pose = np.array([np.inf] * 3)
        min_ee_pose = -np.array([np.inf] * 3)

        if self._config.get(ManipulatorEePosEnv.KW_EE_ROTATION_MATTERS):
            quaternion_max = np.array([1.0] * 4)
            quaternion_min = -1 * quaternion_max

            max_ee_pose = np.concatenate((max_ee_pose, quaternion_max))
            min_ee_pose = np.concatenate((min_ee_pose, quaternion_min))

            max_target_pose = np.concatenate((max_target_pose, quaternion_max))
            min_target_pose = np.concatenate((min_target_pose, quaternion_min))

        # Definition of environment observation_space part
        max_obs = np.concatenate(
            (target_range, super_result.high, max_target_pose, max_ee_pose)
        ).astype(np.float32)
        min_obs = np.concatenate(
            (-target_range, super_result.low, min_target_pose, min_ee_pose)
        ).astype(np.float32)

        # vs legacy: action is added by separate LastActionObservationNode
        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def get_target_pose(self) -> list[float]:
        target_pose = []  # this default return value isn't great
        reward_node = self.env._reward_node
        if isinstance(reward_node, ManipulatorEePosRewardNode):
            target_pose = reward_node.get_ee_target()
        return target_pose

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

        target_pose = self.get_target_pose()
        target_coord = target_pose[0:3]

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

        # ordering is not great - adds before and after superclass result

        ee_rotation_for_state = []
        target_rotation_for_state = []
        if self._config.get(ManipulatorEePosEnv.KW_EE_ROTATION_MATTERS):
            ee_rotation_for_state = ee_to_ref_frame_quaternion
            target_rotation_for_state = target_pose[4:7]
        ee_rotation_for_state = np.array(ee_rotation_for_state)

        # Compose environment state
        # previous action is added by a separate LastActionObservationNode
        # BUT legacy also included fixed joints here, although they are not part of the action
        # (thus each fixed joint leads to 1 less place in the observation here, which is more consistent)
        state = np.concatenate(
            (
                target_polar,
                super_result,
                target_coord,
                target_rotation_for_state,
                ee_to_ref_frame_translation,
                ee_rotation_for_state,
                # self.previous_action,
            )
        )

        return state.astype(np.float32)


class ManipulatorEePosRewardNode(ManipulatorRewardNode):

    DEFAULT_DISTANCE_THRESHOLD = 0.1
    DEFAULT_ROTATION_THRESHOLD = 0.1

    INFO_KW_TRANSLATION_DISTANCE = "dist_coord"
    INFO_KW_ROTATION_DISTANCE = "dist_rot"
    INFO_KW_TARGET_COORD = "target_coord"
    INFO_KW_TARGET_ROT = "target_rot"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ee_target: list[float] | None = None

    def set_ee_target(self, new_target: list[float]):
        self._ee_target = new_target
        while len(self._ee_target) < 6:
            self._ee_target.append(0.0)
        # Store as 3 floats for translation and 4 floats for rotation quaternion.
        # We might already have received 7 floats; else interpret as translation and rpy here
        if len(self._ee_target) != 7:
            self._ee_target = utils.pose_quat_from_pose_rpy(np.array(self._ee_target))

    def get_ee_target(self) -> list[float]:
        return self._ee_target

    def get_reset_state_part_str(self) -> dict[str, str]:
        return {"object_0_function": "fixed_position"}

    def get_reset_state_part_float(self) -> dict[str, float]:
        return {
            "object_0_x": self._ee_target[0],
            "object_0_y": self._ee_target[1],
            "object_0_z": self._ee_target[2],
        }

    def calculate_quaternion_distance(self, quat1: NDArray, quat2: NDArray) -> float:
        """
        Get the distance between two rotation quaternions.
        Subclasses may want to use different implementations.

        Parameters:
            quat1: a quaternion
            quat2: another quaternion
        Returns:
            float: the distance between the two quaternions
        """
        return utils.rotation_error_magnitude(quat1, quat2)

    def get_reward(
        self,
        rs_state_array: NDArray,
        rs_state_dict: dict[str, float],
        env_action: NDArray,
        **kwargs,
    ) -> Tuple[float, bool, dict]:

        info = self.env.get_default_info()

        # TODO adapt and improve code taken from legacy more; replace hardcoded numbers
        rs_state = rs_state_dict
        reward = 0.0
        done = False

        # Reward weight for reaching the goal position
        g_w = 2
        # Reward weight for collision (ground, table or self)
        c_w = -1
        # Reward weight according to the distance to the goal
        d_w = -0.005

        # Reward weight according to the rotation difference to the goal
        # TODO configurable
        rotation_weight = -0.05

        # Calculate distance to the target
        target_coord = np.array(
            [
                rs_state["object_0_to_ref_translation_x"],
                rs_state["object_0_to_ref_translation_y"],
                rs_state["object_0_to_ref_translation_z"],
            ]
        )
        ee_coord = np.array(
            [
                rs_state["ee_to_ref_translation_x"],
                rs_state["ee_to_ref_translation_y"],
                rs_state["ee_to_ref_translation_z"],
            ]
        )
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)

        # Reward base
        reward += d_w * euclidean_dist_3d

        quat_target = []
        rotation_success = True
        rotation_matters = self._config.get(
            ManipulatorEePosEnv.KW_EE_ROTATION_MATTERS, False
        )

        distance_threshold = self._config.get(
            ManipulatorEePosEnv.KW_EE_DISTANCE_THRESHOLD,
            self.DEFAULT_DISTANCE_THRESHOLD,
        )
        rotation_threshold = self._config.get(
            ManipulatorEePosEnv.KW_EE_ROTATION_THRESHOLD,
            self.DEFAULT_ROTATION_THRESHOLD,
        )

        info[self.INFO_KW_TRANSLATION_DISTANCE] = euclidean_dist_3d

        if rotation_matters:
            quat_ee = np.array(
                [
                    rs_state["ee_to_ref_rotation_x"],
                    rs_state["ee_to_ref_rotation_y"],
                    rs_state["ee_to_ref_rotation_z"],
                    rs_state["ee_to_ref_rotation_w"],
                ]
            )
            quat_target = self._ee_target[3:7]

            rot_diff = self.calculate_quaternion_distance(quat_ee, quat_target)
            rot_reward = rotation_weight * rot_diff
            reward += rot_reward
            rotation_success = rot_diff < rotation_threshold
            info[self.INFO_KW_ROTATION_DISTANCE] = rot_diff

        if rotation_success and euclidean_dist_3d <= distance_threshold:
            reward = g_w * 1
            done = True
            info[RoboGymEnv.INFO_KW_FINAL_STATUS] = RoboGymEnv.FINAL_STATUS_SUCCESS

        if rs_state.get("in_collision", False):
            reward = c_w * 1
            done = True
            info[RoboGymEnv.INFO_KW_FINAL_STATUS] = RoboGymEnv.FINAL_STATUS_COLLISION

        elif self.env.elapsed_steps >= self.max_episode_steps:
            done = True
            info[RoboGymEnv.INFO_KW_FINAL_STATUS] = (
                RoboGymEnv.FINAL_STATUS_MAX_STEPS_EXCEEDED
            )

        info[self.INFO_KW_TARGET_COORD] = target_coord
        if rotation_matters:
            info[self.INFO_KW_TARGET_ROT] = quat_target

        return reward, done, info
