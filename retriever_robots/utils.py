import math
import numpy as np

from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion




def euler_from_quaternion(x, y, z, w):
    """
    Converts a quaternion into standard Euler angles (Roll, Pitch, Yaw)
    in radians. Sequence: XYZ (Roll, Pitch, Yaw).
    """
    return Rotation.from_quat([x, y, z, w]).as_euler()


def mult_quat_msgs(q1,q2):
    """
    Multiplies two quaternions represented as geometry_msgs.msg.Quaternion.
    Returns the resulting quaternion as a geometry_msgs.msg.Quaternion.
    """
    r1 = Rotation.from_quat([q1.x, q1.y, q1.z, q1.w])
    r2 = Rotation.from_quat([q2.x, q2.y, q2.z, q2.w])
    r_result = r1 * r2
    q_result = r_result.as_quat()
    
    return Quaternion(x=q_result[0], y=q_result[1], z=q_result[2], w=q_result[3])

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