#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import yaml
import copy
from robo_gym.utils import utils


class ManipulatorModel:
    """Manipulator robots model class.

    Attributes:
        max_joint_positions (np.array): Maximum joint position values (rad)`.
        min_joint_positions (np.array): Minimum joint position values (rad)`.
        max_joint_velocities (np.array): Maximum joint velocity values (rad/s)`.
        min_joint_velocities (np.array): Minimum joint velocity values (rad/s)`.
        joint_names (list): Joint names (Standard Indexing)`.

    Joint Names (ROS Indexing):
    [elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint,
     wrist_3_joint]

    NOTE: Where not specified, Standard Indexing is used.
    """

    def __init__(
        self,
        model_key: str,
        file_path: str,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        override_joint_names: list[str] | None = None,
    ):

        self.model_key = model_key
        self._norm_min = norm_min
        self._norm_max = norm_max

        # Load robot parameters
        with open(file_path, "r") as stream:
            try:
                p = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        if override_joint_names:
            self.joint_names = override_joint_names
        else:
            self.joint_names = p.get("joint_names")
        if "remote_joint_names" in p:
            self.remote_joint_names = p["remote_joint_names"]
        else:
            self.remote_joint_names = self.joint_namess

        # Initialize joint limits attributes
        self.max_joint_positions = np.zeros(len(self.joint_names))
        self.min_joint_positions = np.zeros(len(self.joint_names))
        self.max_joint_velocities = np.zeros(len(self.joint_names))
        self.min_joint_velocities = np.zeros(len(self.joint_names))

        self.joint_positions = np.zeros(len(self.joint_names))

        for idx, joint in enumerate(self.joint_names):
            self.max_joint_positions[idx] = p["joint_limits"][joint]["max_position"]
            self.min_joint_positions[idx] = p["joint_limits"][joint]["min_position"]
            self.max_joint_velocities[idx] = p["joint_limits"][joint]["max_velocity"]
            self.min_joint_velocities[idx] = -p["joint_limits"][joint]["max_velocity"]

        # Workspace parameters
        self.ws_r = p["workspace_area"]["r"]
        self.ws_min_r = p["workspace_area"]["min_r"]

    def get_max_joint_positions(self) -> NDArray:

        return self.max_joint_positions

    def get_min_joint_positions(self) -> NDArray:

        return self.min_joint_positions

    def get_max_joint_velocities(self) -> NDArray:

        return self.max_joint_velocities

    def get_min_joint_velocities(self) -> NDArray:

        return self.min_joint_velocities

    def normalize_joint_value(self, source_value: float, joint_index: int) -> float:
        dest_value = float(
            np.interp(
                source_value,
                [
                    self.min_joint_positions[joint_index],
                    self.max_joint_positions[joint_index],
                ],
                [self._norm_min, self._norm_max],
            )
        )
        return dest_value

    def normalize_joint_values(self, joints: NDArray) -> NDArray:
        """Normalize joint position values

        Args:
            joints (np.array): Joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """

        result = copy.deepcopy(joints)
        for joint_index in range(len(joints)):
            source_value = float(joints[joint_index])
            dest_value = self.normalize_joint_value(source_value, joint_index)
            result[joint_index] = dest_value
        return result

    def denormalize_joint_value(self, source_value: float, joint_index: int) -> float:
        dest_value = float(
            np.interp(
                source_value,
                [self._norm_min, self._norm_max],
                [
                    self.min_joint_positions[joint_index],
                    self.max_joint_positions[joint_index],
                ],
            )
        )
        return dest_value

    def denormalize_joint_values(self, joints: NDArray) -> NDArray:
        """Map normalized joint values to joint position in min/max range

        Args:
            joints (np.array): Normalized joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """

        result = copy.deepcopy(joints)
        for joint_index in range(len(joints)):
            source_value = float(joints[joint_index])
            dest_value = self.denormalize_joint_value(source_value, joint_index)
            result[joint_index] = dest_value
        return result

    def denormalize_joint_range(self, source_range: float, joint_index: int) -> float:
        dest_value = float(
            np.interp(
                source_range,
                [0, 1],
                [
                    0,
                    abs(
                        self.max_joint_positions[joint_index]
                        - self.min_joint_positions[joint_index]
                    ),
                ],
            )
        )
        return dest_value

    def denormalize_joint_ranges(self, joints: NDArray) -> NDArray:
        result = np.zeros_like(joints)
        for joint_index in range(len(joints)):
            source_value = float(joints[joint_index])
            dest_value = self.denormalize_joint_range(source_value, joint_index)
            result[joint_index] = dest_value
        return result

    def get_random_workspace_pose_rpy(
        self,
        np_random: np.random.Generator | None = None,
        roll_range: NDArray | None = None,
        pitch_range: NDArray | None = None,
        yaw_range: NDArray | None = None,
    ) -> NDArray:
        """Get pose of a random point in the robot workspace.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """

        if np_random is None:
            np_random = np.random.default_rng()

        pose = np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            phi = np_random.uniform(low=0.0, high=2 * np.pi)
            costheta = np_random.uniform(low=0.0, high=1.0)  # [-1.0,1.0] for a sphere
            u = np_random.uniform(low=0.0, high=1.0)

            theta = np.arccos(costheta)
            r = self.ws_r * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # TODO this is UR specific
            if (x**2 + y**2) > self.ws_min_r**2:
                singularity_area = False

        roll = utils.get_uniform_from_range(np_random, roll_range, 0.0)
        pitch = utils.get_uniform_from_range(np_random, pitch_range, 0.0)
        yaw = utils.get_uniform_from_range(np_random, yaw_range, 0.0)

        pose[0:6] = [x, y, z, roll, pitch, yaw]

        return pose

    def get_random_workspace_pose_quat(
        self,
        np_random: np.random.Generator | None = None,
        roll_range: NDArray | None = None,
        pitch_range: NDArray | None = None,
        yaw_range: NDArray | None = None,
        seq="xyz",
        quat_unique: bool = False,
    ) -> NDArray:
        pose_rpy = self.get_random_workspace_pose_rpy(
            np_random, roll_range, pitch_range, yaw_range
        )
        pose_quat = utils.quat_from_euler(
            pose_rpy[3], pose_rpy[4], pose_rpy[5], seq, quat_unique
        )
        return pose_quat

    def get_random_offset_joint_positions(
        self,
        joint_positions: NDArray,
        random_offset: NDArray,
        np_random: np.random.Generator | None = None,
    ):

        if np_random is None:
            np_random = np.random.default_rng()

        joint_positions_low = joint_positions - random_offset
        joint_positions_high = joint_positions + random_offset

        joint_positions = np_random.uniform(
            low=joint_positions_low, high=joint_positions_high
        )
        return joint_positions

    def reorder_joints_from_rs(self, ros_thetas: NDArray) -> NDArray:
        return ros_thetas

    def reorder_joints_for_rs(self, thetas: NDArray) -> NDArray:
        return thetas

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)
