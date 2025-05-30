#!/usr/bin/env python3
from robo_gym.utils.ur_utils import UR
import numpy as np


class TestUrUtils:
    # TODO more assertions on test results

    def test_random_workspace_pose(self):
        ur = UR(model_key="ur10")

        pose = ur.get_random_workspace_pose_rpy()
        assert 6 == len(pose)

        pose = ur.get_random_workspace_pose_rpy(np_random=np.random.default_rng())
        assert 6 == len(pose)

    def test_random_offset_joint_positions(self):
        ur = UR(model_key="ur10")
        joint_positions = np.array(
            [
                0.0,
                -2.5,
                1.5,
                0.0,
                -1.4,
                0.0,
            ],
            dtype=np.float32,
        )
        random_offset = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        positions = ur.get_random_offset_joint_positions(joint_positions, random_offset)
        assert 6 == len(positions)

        positions = ur.get_random_offset_joint_positions(
            joint_positions, random_offset, np_random=np.random.default_rng()
        )
        assert 6 == len(positions)
