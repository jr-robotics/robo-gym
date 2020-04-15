#!/usr/bin/env python3

import numpy as np
from robo_gym.utils import utils

class Mir100():
    """Mobile Industrial Robots MiR100 utilities.

    Attributes:
        max_lin_vel (float): Maximum robot's linear velocity (m/s).
        min_lin_vel (float): Minimum robot's linear velocity (m/s).
        max_ang_vel (float): Maximum robot's angular velocity (rad/s).
        min_ang_vel (float): Minimum robot's linear velocity (rad/s).

    """
    def __init__(self):

        self.max_lin_vel = 1.0
        self.min_lin_vel = -1.0

        self.max_ang_vel = 1.5
        self.min_ang_vel = -1.5

    def get_max_lin_vel(self):

        return self.max_lin_vel

    def get_min_lin_vel(self):

        return self.min_lin_vel

    def get_max_ang_vel(self):

        return self.max_ang_vel

    def get_min_ang_vel(self):

        return self.min_ang_vel

    def get_corners_positions(self,x,y,yaw):
        """Get robot's corners coordinates given the coordinates of its center.

        Args:
            x (float): x coordinate of robot's geometric center.
            y (float): y coordinate of robot's geometric center.
            yaw (float): yaw angle of robot's geometric center.

        The coordinates are given with respect to the map origin and cartesian system.

        Returns:
            list[list]: x and y coordinates of the 4 robot's corners.

        """

        robot_x_dimension = 0.9
        robot_y_dimension = 0.58
        dx = robot_x_dimension/2
        dy = robot_y_dimension/2

        delta_corners = [[dx,-dy],[dx,dy],[-dx,dy],[-dx,-dy]]
        corners = []

        for corner_xy in delta_corners:
            # Rotate point around origin
            r_xy = utils.rotate_point(corner_xy[0],corner_xy[1],yaw)
            # Translate back from origin to corner
            corners.append([sum(x) for x in zip(r_xy,[x,y])])

        return corners
