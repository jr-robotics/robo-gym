#!/usr/bin/env python3

import math
import numpy as np

def normalize_angle_rad(a):
    """Normalize angle (in radians) to +-pi

    Args:
        a (float): Angle (rad).

    Returns:
        float: Normalized angle (rad).

    """

    return (a + math.pi) % (2 * math.pi) - math.pi

def point_inside_circle(x,y,center_x,center_y,radius):
    """Check if a point is inside a circle.

    Args:
        x (float): x coordinate of the point.
        y (float): y coordinate of the point.
        center_x (float): x coordinate of the center of the circle.
        center_y (float): y coordinate of the center of the circle.
        radius (float): radius of the circle (m).

    Returns:
        bool: True if the point is inside the circle.

    """

    dx = abs(x - center_x)
    dy = abs(y - center_y)
    if dx>radius:
        return False
    if dy>radius:
        return False
    if dx + dy <= radius:
        return True
    if dx**2 + dy**2 <= radius**2:
        return True
    else:
        return False

def rotate_point(x,y,theta):
    """Rotate a point around the origin by an angle theta.

    Args:
        x (float): x coordinate of point.
        y (float): y coordinate of point.
        theta (float): rotation angle (rad).

    Returns:
        list: [x,y] coordinates of rotated point.

    """
    """
    Rotation of a point by an angle theta(rads)
    """
    x_r = x * math.cos(theta) - y * math.sin(theta)
    y_r = x * math.sin(theta) + y * math.cos(theta)
    return [x_r, y_r]

def cartesian_to_polar_2d(x_target, y_target, x_origin = 0, y_origin = 0):
    """Transform 2D cartesian coordinates to 2D polar coordinates.

    Args:
        x_target (type): x coordinate of target point.
        y_target (type): y coordinate of target point.
        x_origin (type): x coordinate of origin of polar system. Defaults to 0.
        y_origin (type): y coordinate of origin of polar system. Defaults to 0.

    Returns:
        float, float: r,theta polard coordinates.

    """

    delta_x = x_target - x_origin
    delta_y = y_target - y_origin
    polar_r = np.sqrt(delta_x**2+delta_y**2)
    polar_theta = np.arctan2(delta_y,delta_x)

    return polar_r, polar_theta

def cartesian_to_polar_3d(target, origin = [0,0,0]):
    """Transform 3D cartesian coordinates to 3D polar coordinates.

    Args:
        target (list): [x,y,z] coordinates of target point.
        origin (list): [x,y,z] coordinates of origin of polar system. Defaults to [0,0,0].

    Returns:
        list: [r,phi,theta] polar coordinates of target point.

    """

    delta_x = target[0] - origin[0]
    delta_y = target[1] - origin[1]
    delta_z = target[2] - origin[2]
    r =  np.sqrt(delta_x**2+delta_y**2+delta_z**2)
    phi = np.arctan2(delta_y,delta_x)
    theta = np.arccos(delta_z/r)

    return [r,phi,theta]

def downsample_list_to_len(data, output_len):
    """Downsample a list of values to a specific length.

    Args:
        data (list): Data to downsample.
        output_len (int): Length of the downsampled list.

    Returns:
        list: Downsampled list.

    """

    assert output_len>0

    downsample_rate = math.ceil(len(data)/output_len)
    d_data = data[::downsample_rate]

    return d_data
