#!/usr/bin/env python3

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def cartesian_to_polar_3d(cartesian_coordinates):
    """Transform 3D cartesian coordinates to 3D polar coordinates.

    Args:
        cartesian_coordinates (list): [x,y,z] coordinates of target point.

    Returns:
        list: [r,phi,theta] polar coordinates of point.

    """

    x = cartesian_coordinates[0]
    y = cartesian_coordinates[1]
    z = cartesian_coordinates[2]
    r =  np.sqrt(x**2+y**2+z**2)
    #? phi is defined in [-pi, +pi]
    phi = np.arctan2(y,x)
    #? theta is defined in [0, +pi]
    theta = np.arccos(z/r)

    return [r,theta,phi]

def downsample_list_to_len(data, output_len):
    """Downsample a list of values to a specific length.

    Args:
        data (list): Data to downsample.
        output_len (int): Length of the downsampled list.

    Returns:
        list: Downsampled list.

    """

    assert output_len > 0
    assert output_len <= len(data)

    temp = np.linspace(0, len(data)-1, num=output_len)
    temp = [int(round(x)) for x in temp]

    assert len(temp) == len(set(temp))

    ds_data = []
    for index in temp:
        ds_data.append(data[index])

    return ds_data

def change_reference_frame(point, translation, quaternion):
    """Transform a point from one reference frame to another, given
        the translation vector between the two frames and the quaternion
        between  the two frames.

    Args:
        point (array_like,shape(3,) or shape(N,3)): x,y,z coordinates of the point in the original frame
        translation (array_like,shape(3,)): translation vector from the original frame to the new frame 
        quaternion (array_like,shape(4,)): quaternion from the original frame to the new frame

    Returns:
        ndarray,shape(3,): x,y,z coordinates of the point in the new frame.
        
    """

    #point = [1,2,3]
    #point = np.array([1,2,3])
    #point = np.array([[11,12,13],[21,22,23]]) # point.shape = (2,3) # point (11,12,13)  and point (21,22,23)

    # Apply rotation
    r = R.from_quat(quaternion)
    rotated_point = r.apply(np.array(point))
    # Apply translation
    translated_point = np.add(rotated_point, np.array(translation))

    return translated_point