'''This is just a prototypical setup for getting used to pytest'''

'''List of best practices:
        - test function need to start with the keyword 'test'
        - test function names should give and indication what is being tested -> preferably long test function names

'''


# from robo_gym.utils import utils
import numpy as np
from unittest import TestCase

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

def test_polar_coords_default_origin():
    target_x = 3
    target_y = 4

    polar_r, polar_theta = cartesian_to_polar_2d(x_target=target_x, y_target=target_y)
    
    assert polar_r == 5
    TestCase().assertAlmostEqual(polar_theta, 0.927, places=3)

def test_cartesian_to_polar_2d_default_origin():
    # NOTE: Should we have multiple 'test' per function or not?
    # NOTE: Is testing one sample point enough?
    # Example 1
    target_x = 3
    target_y = 4

    polar_r, polar_theta = cartesian_to_polar_2d(x_target=target_x, y_target=target_y)
    
    assert polar_r == 5
    TestCase().assertAlmostEqual(polar_theta, 0.927, places=3)

def test_cartesian_to_polar_2d_set_origin():
    # NOTE: Should we have multiple 'test' per function or not?
    # NOTE: Is testing one sample point enough?
    # Example 1
    target_x = 3
    target_y = 4
    origin_x = 1
    origin_y = 1

    polar_r, polar_theta = cartesian_to_polar_2d(x_target=target_x, y_target=target_y, x_origin=origin_x, y_origin=origin_y)
    
    ## NOTE: not sure how to properly do this
    assert abs(polar_r - 3.61) < 0.01
    assert abs(polar_theta - 0.9827949) < 0.01


def test_cartesian_to_polar_2d_default_edge():
    # NOTE: Should we have multiple 'test' per function or not?
    # NOTE: Can I pass a list of values that should default to zero?
    # Example 1
    target_x = 0
    target_y = 0

    polar_r, polar_theta = cartesian_to_polar_2d(x_target=target_x, y_target=target_y)
    
    assert polar_r == 0
    TestCase().assertAlmostEqual(polar_theta, 0.0, places=3)

    polar_r, polar_theta = cartesian_to_polar_2d(x_target=1.5, y_target=1.5, x_origin=1.5, y_origin=1.5)
    
    assert polar_r == 0
    TestCase().assertAlmostEqual(polar_theta, 0.0, places=3)



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

    print(delta_x)
    print(delta_y)
    print(delta_z)
    
    r =  np.sqrt(delta_x**2+delta_y**2+delta_z**2)
    phi = np.arctan2(delta_y,delta_x)
    theta = np.arccos(delta_z/r)

    return [r,phi,theta]



def test_cartesian_to_polar_3d_default_origin():
    # NOTE: Should we have multiple 'test' per function or not?
    # NOTE: Is testing one sample point enough?
    # Example 1
    target = [9, 4, 5]

    r, phi, theta = cartesian_to_polar_3d(target=target)
    
    assert abs(r - 11.045361017187) < 0.01
    assert abs(phi - 0.41822432957) < 0.01
    assert abs(theta - 1.1010291108682) < 0.01


def test_cartesian_to_polar_3d_set_origin():
    target = [9.12, 4.32, 5.72]
    origin = [3.12, 12.7, 4.2]

    r, phi, theta = cartesian_to_polar_3d(target=target, origin=origin)
    
    assert abs(r - 10.418) < 0.01
    assert abs(phi - (-0.949419)) < 0.01
    assert abs(theta - 1.424401) < 0.01

import math
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


    print(d_data)
    return d_data


def test_downsample_list_to_len():
    input_length = 1000
    sample_list = [i for i in range(input_length)]
    output_len = 501 # 999
    output_len = 349

    downsampled_list = downsample_list_to_len(data=sample_list, output_len=output_len)

    assert len(downsampled_list) == output_len