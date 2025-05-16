#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def normalize_angle_rad(a):
    """Normalize angle (in radians) to +-pi

    Args:
        a (float): Angle (rad).

    Returns:
        float: Normalized angle (rad).

    """

    return (a + math.pi) % (2 * math.pi) - math.pi


def point_inside_circle(x, y, center_x, center_y, radius):
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
    if dx > radius:
        return False
    if dy > radius:
        return False
    if dx + dy <= radius:
        return True
    if dx**2 + dy**2 <= radius**2:
        return True
    else:
        return False


def rotate_point(x, y, theta):
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


def cartesian_to_polar_2d(x_target, y_target, x_origin=0, y_origin=0):
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
    polar_r = np.sqrt(delta_x**2 + delta_y**2)
    polar_theta = np.arctan2(delta_y, delta_x)

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
    r = np.sqrt(x**2 + y**2 + z**2)
    # ? phi is defined in [-pi, +pi]
    phi = np.arctan2(y, x)
    # ? theta is defined in [0, +pi]
    theta = np.arccos(z / r)

    return [r, theta, phi]


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

    temp = np.linspace(0, len(data) - 1, num=output_len)
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

    # point = [1,2,3]
    # point = np.array([1,2,3])
    # point = np.array([[11,12,13],[21,22,23]]) # point.shape = (2,3) # point (11,12,13)  and point (21,22,23)

    # Apply rotation
    r = R.from_quat(quaternion)
    rotated_point = r.apply(np.array(point))
    # Apply translation
    translated_point = np.add(rotated_point, np.array(translation))

    return translated_point


def quat_from_euler(
    roll: float, pitch: float, yaw: float, seq="zyz", quat_unique: bool = False
) -> NDArray:
    rot = R.from_euler(seq=seq, angles=[roll, pitch, yaw])
    return rot.as_quat(canonical=quat_unique)


def quat_from_euler_xyz(
    roll: float, pitch: float, yaw: float, quat_unique: bool = False
) -> NDArray:
    return quat_from_euler(roll, pitch, yaw, "xyz", quat_unique)


def quat_from_euler_zyx(
    roll: float, pitch: float, yaw: float, quat_unique: bool = False
) -> NDArray:
    return quat_from_euler(roll, pitch, yaw, "zyx", quat_unique)


def euler_from_quat(q: NDArray, seq="xyz") -> NDArray:
    rot = R.from_quat(q)
    rpy = rot.as_euler(seq=seq, degrees=False)
    return rpy


def quat_mul(q1: NDArray, q2: NDArray, quat_unique: bool = False) -> NDArray:
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    # TODO verify that this does what we want
    r_result = r1 * r2
    return r_result.as_quat(canonical=quat_unique)


def quat_inv(q: NDArray, quat_unique: bool = False) -> NDArray:
    r = R.from_quat(q)
    r_inv = r.inv()
    return r_inv.as_quat(canonical=quat_unique)


def quat_from_euler_xyz_isaac(roll: float, pitch: float, yaw: float) -> NDArray:
    # simplified, based on from Isaac Lab implementation (for comparison)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qx, qy, qz, qw])


def euler_xyz_from_quat_isaac(
    quat: NDArray,
) -> NDArray:
    """Convert rotation given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        quat: The quaternion orientation in (x, y, z, w).

    Returns:
        A tuple containing roll-pitch-yaw.

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_x, q_y, q_z, q_w = quat[0], quat[1], quat[2], quat[3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = math.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = (
        math.copysign(math.pi / 2.0, sin_pitch)
        if abs(sin_pitch) >= 1
        else math.asin(sin_pitch)
    )

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(sin_yaw, cos_yaw)

    return np.array(
        [
            roll % (2 * math.pi),
            pitch % (2 * math.pi),
            yaw % (2 * math.pi),
        ]
    )


def get_config_range(data: dict, key: str) -> NDArray:
    result = data.get(key)
    if result is not None:
        try:
            float_val = float(result)
            return np.array([float_val, float_val])
        except ValueError:
            pass
        except TypeError:
            pass
        result = np.array(result).reshape([2])
        if result[1] < result[0]:
            result[1] = result[0]
    return result


def get_uniform_from_range(
    np_random: np.random.Generator,
    range: NDArray | None,
    default_value: float = 0.0,
) -> float:
    if range is None:
        return default_value
    return np_random.uniform(low=range[0], high=range[1])


# Isaac random pose generation rewritten for our use
def create_random_bounding_box_pose_quat(
    pos_x_range: NDArray,
    pos_y_range: NDArray,
    pos_z_range: NDArray,
    np_random: np.random.Generator | None = None,
    roll_range: NDArray | None = None,
    pitch_range: NDArray | None = None,
    yaw_range: NDArray | None = None,
    seq="xyz",
    quat_unique: bool = False,
) -> NDArray:

    if np_random is None:
        np_random = np.random.default_rng()
    # sample new pose targets
    # -- position
    x = get_uniform_from_range(np_random, pos_x_range, 0.0)
    y = get_uniform_from_range(np_random, pos_y_range, 0.0)
    z = get_uniform_from_range(np_random, pos_z_range, 0.0)

    # -- orientation
    roll = get_uniform_from_range(np_random, roll_range, 0.0)
    pitch = get_uniform_from_range(np_random, pitch_range, 0.0)
    yaw = get_uniform_from_range(np_random, yaw_range, 0.0)

    # quat = quat_from_euler(roll, pitch, yaw, seq)
    quat = quat_from_euler_xyz_isaac(roll, pitch, yaw)
    if quat_unique and quat[3] < 0:
        quat = -quat

    result = np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]])
    return result


def pose_quat_from_pose_rpy(
    pose_rpy: NDArray, seq="xyz", quat_unique: bool = False
) -> NDArray:
    quat = quat_from_euler(pose_rpy[3], pose_rpy[4], pose_rpy[5], seq, quat_unique)
    result = np.array(
        [pose_rpy[0], pose_rpy[1], pose_rpy[2], quat[0], quat[1], quat[2], quat[3]]
    )
    return result


def pose_rpy_from_pose_quat(pose_quat: NDArray, seq="xyz") -> NDArray:
    quat = pose_quat[3:6]
    rpy = euler_from_quat(quat, seq)
    result = np.array(
        [pose_quat[0], pose_quat[1], pose_quat[2], rpy[0], rpy[1], rpy[2]]
    )
    return result


def rotation_error_magnitude(q1: NDArray, q2: NDArray) -> float:
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_diff = r1 * (r2.inv())
    return r_diff.magnitude()


# our representation: xyzw
# Isaac representation: wxyz (not in the Isaac methods here, they are already converted)
# but important for providing Isaac-like observations
def quat_xyzw_from_wxyz(q: NDArray) -> NDArray:
    # w = q[0]
    return np.array([q[1], q[2], q[3], q[0]])


def quat_wxyz_from_xyzw(q: NDArray) -> NDArray:
    # w = q[3]
    return np.array([q[3], q[0], q[1], q[2]])


def pose_quat_xyzw_from_wxyz(pose: NDArray) -> NDArray:
    pos = pose[0:3]
    q = pose[3:7]
    result = np.concatenate((pos, quat_xyzw_from_wxyz(q)))
    return result


def pose_quat_wxyz_from_xyzw(pose: NDArray) -> NDArray:
    pos = pose[0:3]
    q = pose[3:7]
    result = np.concatenate((pos, quat_wxyz_from_xyzw(q)))
    return result


def flatten_to_dict(obj, prefix=""):
    result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            result.update(flatten_to_dict(value, new_key))

    elif isinstance(obj, list):
        # index all elements of this list object with the same number of digits
        # to allow consistent sorting of the columns
        max_index_digits = len(str(len(obj) - 1))
        for index, value in enumerate(obj):
            index_str = str(index).zfill(max_index_digits)
            new_key = f"{prefix}[{index_str}]"
            result.update(flatten_to_dict(value, new_key))

    elif hasattr(obj, "__dict__"):
        return flatten_to_dict(obj.__dict__, prefix)

    elif hasattr(obj, "__slots__"):
        for attr in obj.__slots__:
            value = getattr(obj, attr)
            new_key = f"{prefix}.{attr}" if prefix else attr
            result.update(flatten_to_dict(value, new_key))

    else:
        result[prefix] = obj

    return result
