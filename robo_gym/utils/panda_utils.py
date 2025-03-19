#!/usr/bin/env python3

import numpy as np
from numpy.typing import NDArray
import yaml
import os
import copy

from robo_gym.utils.manipulator_model import *


class Panda(ManipulatorModel):
    """Franka Emika Panda model class.

    Attributes:
        max_joint_positions (np.array): Maximum joint position values (rad)`.
        min_joint_positions (np.array): Minimum joint position values (rad)`.
        max_joint_velocities (np.array): Maximum joint velocity values (rad/s)`.
        min_joint_velocities (np.array): Minimum joint velocity values (rad/s)`.
        joint_names (list): Joint names (Standard Indexing)`.

    Joint Names (ROS Indexing):
    [panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5,
     panda_joint6, panda_joint7]

    NOTE: Where not specified, Standard Indexing is used.
    """

    def __init__(self, model_key: str):

        assert model_key == "panda"

        self.model = model_key

        file_name = model_key + ".yaml"
        file_path = os.path.join(
            os.path.dirname(__file__), "panda_parameters", file_name
        )

        super().__init__(model_key, file_path)
