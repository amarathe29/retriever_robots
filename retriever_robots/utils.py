import math
import numpy as np


def euler_from_quaternion(x, y, z, w):
    """
    Converts a quaternion into standard Euler angles (Roll, Pitch, Yaw)
    in radians. Sequence: XYZ (Roll, Pitch, Yaw).
    """
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts Euler angles (Roll, Pitch, Yaw) in radians into a quaternion (x, y, z, w).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = cy * cp * cr + sy * sp * sr  # w
    q[1] = cy * cp * sr - sy * sp * cr  # x
    q[2] = cy * sp * cr + sy * cp * sr  # y
    q[3] = sy * cp * cr - cy * sp * sr  # z

    return q

def create_rotation_matrix(roll=0, pitch=0, yaw=0, units='radians'):
    """
    Creates a rotation matrix from roll, pitch, and yaw angles.
    """

    if units == 'degrees':
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x


    return R

def angle_wrap(angle: float) -> float:
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped