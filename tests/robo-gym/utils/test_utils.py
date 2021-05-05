#!/usr/bin/env python3
import pytest
from robo_gym.utils import utils
import numpy as np


### normalize_angle_rad ###
test_normalize_angle_rad = [
    (7.123, 0.8398),
    (0.8398, 0.8398),
    (-2.47, -2.47),
    (-47.47, 2.795)

]
@pytest.mark.parametrize('a, expected_a', test_normalize_angle_rad)
def test_normalize_angle_rad(a, expected_a):
    normalized_a = utils.normalize_angle_rad(a)
    assert abs(expected_a - normalized_a) < 0.01


### point_inside_circle ###
test_inside_circle = [
    (5.47, 3.51, 6, 8, 5.23, True),
    (5.11, 2.23, 6, 8, 5.23, False),
    (5.69, 2.78, 6, 8, 5.23, True)
]
@pytest.mark.parametrize('x, y, center_x, center_y, radius, expected_result', test_inside_circle)
def test_point_inside_circle(x, y, center_x, center_y, radius, expected_result):
    is_inside = utils.point_inside_circle(x, y, center_x, center_y, radius)
    assert is_inside == expected_result

### rotate_point ###
test_rotate_points = [
    (3.13, 5.83, 0.785398, -1.91, 6.34),
    (4, 3, -2.14675498, 0.34, -4.99),
    (1, 3, -0.0, 1, 3),

]
@pytest.mark.parametrize('x, y, theta, expected_x, expected_y', test_rotate_points)
def test_rotate_point(x, y, theta, expected_x, expected_y):
    new_x, new_y = utils.rotate_point(x, y, theta)

    print(new_x, new_y)

    assert abs(new_x - expected_x) < 0.01
    assert abs(new_y - expected_y) < 0.01


### cartesian_to_polar_2d ###
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

    polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=p1[0], y_target=p1[1], x_origin=p2[0], y_origin=p2[1])
    
    assert abs(polar_r - expected_r) < 0.001
    assert abs(polar_theta - expected_theta) < 0.001


### cartesian_to_polar_3d ###
def test_cartesian_to_polar_3d():
    cartesian_coordinates = [9, 4, 5]

    r, theta, phi = utils.cartesian_to_polar_3d(cartesian_coordinates=cartesian_coordinates)
    
    assert abs(r - 11.045361017187) < 0.01
    assert abs(phi - 0.41822432957) < 0.01
    assert abs(theta - 1.1010291108682) < 0.01

### downsample_list_to_len ###
test_downsample = [
    2, 4, 5, 6, 7, 8, 9, 10, 13, 15, 24, 51, 100, 101, 200, 201, 349, 501
]
@pytest.mark.parametrize('target_length', test_downsample)
def test_downsample_list_to_len(target_length):
    input_length = 1000
    sample_list = [i for i in range(input_length)]

    downsampled_list = utils.downsample_list_to_len(data=sample_list, output_len=target_length)

    assert len(downsampled_list) == target_length

### change_reference_frame ###
def test_translation_change_reference_frame():

    point = [5,3,2]
    translation = [-11,6,-1]
    quaternion = [0,0,0,1]
    
    assert (utils.change_reference_frame(point,translation,quaternion) == [-6,9,1]).all()

def test_rotation_change_reference_frame():

    point = [-0.250,0.256,1.118]
    translation = [0.0,0.0,-0.227]
    quaternion = [0.0,0.0,1.0,0.0]
    
    assert np.allclose(a = utils.change_reference_frame(point,translation,quaternion),b =[0.250,-0.256,0.890], atol = 0.001)