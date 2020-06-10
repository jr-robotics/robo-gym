'''This is just a prototypical setup for getting used to pytest'''

'''List of best practices:
        - test function need to start with the keyword 'test'
        - test function names should give and indication what is being tested -> preferably long test function names

'''
'''
    Open questions:
        How do we want to compare floating point numbers?
            1. assert abs(X-Y) < 0.001
            2. assertAlmostEqual from unittest (works similar), but i dont think we are supposed to mix the two
        
        How do we want to test function like cartesian_to_polar in general?
            - Testing one precomputed value set?
            - Multiple precomputed value sets? How many?
            - Other ideas for functions such as the ones in utils?
'''

import pytest
from robo_gym.utils import utils


## cartesian_to_polar_2d ##

def test_cartesian_to_polar_2d_default_origin():
    target_x = 3
    target_y = 4

    polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=target_x, y_target=target_y)
    
    assert abs(polar_r - 5) < 0.01
    assert abs(polar_theta - 0.927) < 0.01


def test_cartesian_to_polar_2d_set_origin():
    target_x = 3
    target_y = 4
    origin_x = 1
    origin_y = 1

    polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=target_x, y_target=target_y, x_origin=origin_x, y_origin=origin_y)
    
    assert abs(polar_r - 3.61) < 0.01
    assert abs(polar_theta - 0.9827949) < 0.01


test_equal_points = [
    ([0, 0],[0, 0], 0, 0),
    ([1, 1], [1, 1], 0, 0),
    ([3.141, 3.141], [3.141, 3.141], 0, 0)
]
@pytest.mark.parametrize('p1, p2, expected_r, expected_theta', test_equal_points)
def test_cartesian_to_polar_2d_equal_points(p1, p2, expected_r, expected_theta):
    target_x = 0
    target_y = 0

    polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=p1[0], y_target=p1[1], x_origin=p2[0], y_origin=p2[1])
    
    assert abs(polar_r - expected_r) < 0.001
    assert abs(polar_theta - expected_theta) < 0.001


## cartesian_to_polar_3d ##
def test_cartesian_to_polar_3d_default_origin():
    target = [9, 4, 5]

    r, phi, theta = utils.cartesian_to_polar_3d(target=target)
    
    assert abs(r - 11.045361017187) < 0.01
    assert abs(phi - 0.41822432957) < 0.01
    assert abs(theta - 1.1010291108682) < 0.01


def test_cartesian_to_polar_3d_set_origin():
    target = [9.12, 4.32, 5.72]
    origin = [3.12, 12.7, 4.2]

    r, phi, theta = utils.cartesian_to_polar_3d(target=target, origin=origin)
    
    assert abs(r - 10.418) < 0.01
    assert abs(phi - (-0.949419)) < 0.01
    assert abs(theta - 1.424401) < 0.01


test_downsample = [2, 4, 5, 6, 7, 8, 9, 10, 13, 15, 24, 51, 100, 101, 200, 201, 349, 501]

@pytest.mark.parametrize('target_length', test_downsample)
def test_downsample_list_to_len(target_length):
    input_length = 1000
    sample_list = [i for i in range(input_length)]

    downsampled_list = utils.downsample_list_to_len(data=sample_list, output_len=target_length)

    assert len(downsampled_list) == target_length