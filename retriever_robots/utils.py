import math
import numpy as np

from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion, Twist
from cvxopt import matrix, sparse
from cvxopt.solvers import qp, options

options["show_progress"] = False
options["reltol"] = 1e-2
options["feastol"] = 1e-2
options["maxiters"] = 50


def euler_from_quaternion(x: float, y: float, z: float, w: float, use_extrinsics: bool = False) -> Rotation:
    """
    Converts a quaternion into standard Euler angles (Roll, Pitch, Yaw)
    in radians. Sequence: XYZ (Roll, Pitch, Yaw).
    """
    return Rotation.from_quat([x, y, z, w]).as_euler("XYZ" if not use_extrinsics else "xyz")


def mult_quat_msgs(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """
    Multiplies two quaternions represented as geometry_msgs.msg.Quaternion.
    Returns the resulting quaternion as a geometry_msgs.msg.Quaternion.
    """
    r1 = Rotation.from_quat([q1.x, q1.y, q1.z, q1.w])
    r2 = Rotation.from_quat([q2.x, q2.y, q2.z, q2.w])
    r_result = r1 * r2
    q_result = r_result.as_quat()

    return Quaternion(x=q_result[0], y=q_result[1], z=q_result[2], w=q_result[3])


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> list:
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


def create_rotation_matrix(
    roll: float = 0, pitch: float = 0, yaw: float = 0, units: str = "radians"
) -> np.array:
    """
    Creates a rotation matrix from roll, pitch, and yaw angles.
    """
    if units == "degrees":
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def angle_wrap(angle: float) -> float:
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped


def si_to_uni_vel(dxi, pose):
    linear_gain = 1
    angular_limit = np.pi / 2

    dxu = np.zeros((2, 1))
    h = pose[2]
    e_fwd = np.array([np.cos(h), np.sin(h)])  # unit vector along heading
    e_perp = np.array([-np.sin(h), np.cos(h)])  # unit vector perpendicular (left)

    # Linear velocity: projection of dxi onto the heading direction.
    dxu[0] = linear_gain * np.dot(e_fwd, dxi)

    # Angular velocity: proportional to the angle between dxi and
    # the heading, normalised so that a 90-degree error maps to angular_limit.
    dxu[1] = (
        angular_limit
        * np.arctan2(np.dot(e_perp, dxi), np.dot(e_fwd, dxi))
        / (np.pi / 2)
    )

    return dxu


def uni_to_si_vel(dxu, pose, projection_distance=0.05):
    d = projection_distance

    T = np.array([[1, 0], [0, d]])

    dxi = np.zeros((2, 1))
    h = pose[2]
    # Rotation matrix from body frame to world frame.
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    dxi = R @ T @ dxu

    return dxi


def get_robot_barrier_func(
    safety_radius: float = 0.7,
    barrier_gain: float = 100.0,
    magnitude_limit: float = 0.2,
    boundary_points: np.array = None,
    build_area_points: np.array = None,
    block_safety_radius: float = 0.15,
) -> callable:
    """
    Returns a barrier function that you feed the current cmd and robot positions to, and it will return a safe velocity. Additionally keeps the robots within the bounds of the area, and outside the build zone.

    Args:
        safety_radius (float, optional): Radius of safety for the robots to maintain (in meters?). Defaults to 0.15.
        barrier_gain (float, optional): _description_. Defaults to 100.0.
        magnitude_limit (float, optional): _description_. Defaults to 0.2.

    Returns:
        function: _description_
    """

    def barrier_func(
        unsafe_cmd: Twist,
        pose: np.array,
        neighbor_positions: np.array,
        block_positions: np.array = None,
    ) -> Twist:
        """_summary_

        Args:
            unsafe_cmd (Twist): The unsafe velocity
            pose (np.array): 3 x 1 array denoting the current pose of the robot [[x], [y], [theta]]
            neighbor_positions (np.array): 2 x M array denoting the positions of the M neighbors of the robot [[x_1, y_1],[x_2, y_2]]

        Returns:
            Twist: A safe velocity
        """
        dxu = np.array([unsafe_cmd.linear.x], [unsafe_cmd.angular.z])
        # TODO: Convert Unicycle to SI, and then back
        dxi = uni_to_si_vel(dxu, pose)
        M = (
            neighbor_positions.shape[1]
            if (neighbor_positions is not None and neighbor_positions.shape[1] > 0)
            else 0
        )
        B = (
            block_positions.shape[1]
            if (block_positions is not None and block_positions.shape[1] > 0)
            else 0
        )
        # TODO: Figure out the actual boundary points of our space
        bp = boundary_points if boundary_points is not None else np.array([0, 6, 0, 3])
        bap = (
            build_area_points
            if build_area_points is not None
            else np.array([1.5, 3, 0, 1.5])
        )

        num_constraints = M + B + 4 + 4 + 8
        A = np.zeros((num_constraints, 2))
        b = np.zeros(num_constraints)
        position = pose[:2]
        # Avoid neighbor constraings, we don't know neighbor vels, so we don't have our solver worry about them
        for neighbor_ndx in range(M):
            neighbor_position = neighbor_positions[:, neighbor_ndx]
            diff = position - neighbor_position
            h = np.dot(diff, diff) - safety_radius**2
            A[neighbor_ndx] = -2 * diff
            b[neighbor_ndx] = barrier_gain * (h**3)

        for block_ndx in range(B):
            block_position = block_positions[:, block_ndx]
            diff = position - block_position
            h = np.dot(diff, diff) - block_safety_radius**2
            A[block_ndx] = -2 * diff
            b[block_ndx] = barrier_gain * (h**3)

        # Boundary constraints
        row = M
        A[row] = [0, 1]
        b[row] = 0.4 * barrier_gain * (bp[3] - safety_radius / 2 - position[1]) ** 3
        row += 1
        A[row] = [0, -1]
        b[row] = 0.4 * barrier_gain * (position[1] - bp[2] - safety_radius / 2) ** 3
        row += 1
        A[row] = [1, 0]
        b[row] = 0.4 * barrier_gain * (bp[1] - safety_radius / 2 - position[0]) ** 3
        row += 1
        A[row] = [-1, 0]
        b[row] = 0.4 * barrier_gain * (position[0] - bp[0] - safety_radius / 2) ** 3
        row += 1

        # Keep away constraints
        A[row] = [0, -1]
        b[row] = 0.4 * barrier_gain * (bap[3] - safety_radius / 2 - position[1]) ** 3
        row += 1
        A[row] = [0, 1]
        b[row] = 0.4 * barrier_gain * (position[1] - bap[2] - safety_radius / 2) ** 3
        row += 1
        A[row] = [-1, 0]
        b[row] = 0.4 * barrier_gain * (bap[1] - safety_radius / 2 - position[0]) ** 3
        row += 1
        A[row] = [1, 0]
        b[row] = 0.4 * barrier_gain * (position[0] - bap[0] - safety_radius / 2) ** 3
        row += 1

        constraint_bounds = magnitude_limit * np.cos(np.pi / 8)
        direction_constraint = [
            [1, 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0, 1],
            [-1 / np.sqrt(2), 1 / np.sqrt(2)],
            [-1, 0],
            [-1 / np.sqrt(2), -1 / np.sqrt(2)],
            [0, -1],
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
        ]
        for d in direction_constraint:
            A[row] = d
            b[row] = constraint_bounds
            row += 1

        P = matrix(np.eye(2))
        q = matrix(-dxi.astype(float))
        G = matrix(A)
        h_vec = matrix(b)

        sol = qp(P, q, G, h_vec)
        dxi_out = np.array([sol["x"][0], sol["x"][1]])
        dxu_out = si_to_uni_vel(dxi_out, pose)
        out_cmd = Twist()
        out_cmd.linear.x = dxu_out[0]
        out_cmd.angular.z = dxu_out[1]

        return out_cmd

    return barrier_func
