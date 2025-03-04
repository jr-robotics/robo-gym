#!/usr/bin/env python3

import numpy as np
from numpy.typing import NDArray
import yaml
import os
import copy

from robo_gym.utils.manipulator_model import *


class UR(ManipulatorModel):
    """Universal Robots model class.

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

    def __init__(self, model_key: str):

        assert model_key in ["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"]

        self.model = model_key

        file_name = model_key + ".yaml"
        file_path = os.path.join(os.path.dirname(__file__), "ur_parameters", file_name)

        super().__init__(model_key, file_path)

    def _swap_base_and_elbow(self, thetas: NDArray) -> NDArray:
        return np.array(
            [thetas[2], thetas[1], thetas[0], thetas[3], thetas[4], thetas[5]]
        )

    def _ros_joint_list_to_ur_joint_list(self, ros_thetas):
        """Transform joint angles list from ROS indexing to standard indexing.

        Rearrange a list containing the joints values from the joint indexes used
        in the ROS join_states messages to the standard joint indexing going from
        base to end effector.

        Args:
            ros_thetas (list): Joint angles with ROS indexing.

        Returns:
            np.array: Joint angles with standard indexing.

        """

        return self._swap_base_and_elbow(ros_thetas)

    def _ur_joint_list_to_ros_joint_list(self, thetas):
        """Transform joint angles list from standard indexing to ROS indexing.

        Rearrange a list containing the joints values from the standard joint indexing
        going from base to end effector to the indexing used in the ROS
        join_states messages.

        Args:
            thetas (list): Joint angles with standard indexing.

        Returns:
            np.array: Joint angles with ROS indexing.

        """

        return self._swap_base_and_elbow(thetas)
