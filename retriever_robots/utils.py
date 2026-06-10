import math
import numpy as np

from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion, Twist, TransformStamped
from cvxopt import matrix, sparse # type: ignore
from cvxopt.solvers import qp, options # type: ignore
from rclpy.logging import get_logger

options["show_progress"] = False
options["reltol"] = 1e-2
options["feastol"] = 1e-2
options["maxiters"] = 50


def do_transform_transform(t1: TransformStamped, t2: TransformStamped) -> TransformStamped:
    # given two transforms, produce t1 * t2
    T1 = convert_transform_to_matrix(t1)
    T2 = convert_transform_to_matrix(t2)

    T_result = T1 @ T2
    result = TransformStamped()
    result.header.stamp = t1.header.stamp
    result.header.frame_id = t1.header.frame_id
    result.child_frame_id = t2.child_frame_id
    result.transform.translation.x = T_result[0, 3]
    result.transform.translation.y = T_result[1, 3]
    result.transform.translation.z = T_result[2, 3]
    r = Rotation.from_matrix(T_result[0:3, 0:3]).as_quat()
    result.transform.rotation.x = r[0]
    result.transform.rotation.y = r[1]
    result.transform.rotation.z = r[2]
    result.transform.rotation.w = r[3]
    return result


def convert_transform_to_matrix(transform: TransformStamped) -> np.array:
    T = np.eye(4)
    T[0:3, 0:3] = Rotation.from_quat(
        [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
    ).as_matrix()
    T[0:3, 3] = np.array(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ]
    )
    return T


def euler_from_quaternion(
    x: float, y: float, z: float, w: float, use_extrinsics: bool = False
) -> Rotation:
    """
    Converts a quaternion into standard Euler angles (Roll, Pitch, Yaw)
    in radians. Sequence: XYZ (Roll, Pitch, Yaw).
    """
    return Rotation.from_quat([x, y, z, w]).as_euler(
        "XYZ" if not use_extrinsics else "xyz"
    )

def yaw_from_quaternion(q: Quaternion, use_extrinsics: bool = False) -> float:
    _, _, yaw = euler_from_quaternion(x=q.x, y=q.y, z=q.z, w=q.w,  use_extrinsics=use_extrinsics)
    return yaw


def reverse_yaw_quaternion(quat: Quaternion) -> Quaternion:
    """
    Multiplies a ROS2 quaternion by a 180 degree rotation about the z axis.
    """
    R = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    R2 = Rotation.from_quat([0, 0, 1, 0])
    q_rot = (R2 * R).as_quat()
    return Quaternion(x=q_rot[0], y=q_rot[1], z=q_rot[2], w=q_rot[3])


def mult_quat_msgs(q1: Quaternion, q2: Quaternion, flip_yaw=False) -> Quaternion:
    """
    Multiplies two quaternions represented as geometry_msgs.msg.Quaternion.
    Returns the resulting quaternion as a geometry_msgs.msg.Quaternion.
    """
    r1 = Rotation.from_quat([q1.x, q1.y, q1.z, q1.w])
    r2 = Rotation.from_quat([q2.x, q2.y, q2.z, q2.w])
    r_result = r1 * r2

    if flip_yaw:
        r3 = Rotation.from_quat([0, 0, 1, 0])
        r_result = r3 * r_result
    q_result = r_result.as_quat()

    return Quaternion(x=q_result[0], y=q_result[1], z=q_result[2], w=q_result[3])


def quaternion_from_euler(
    roll: float = 0, pitch: float = 0, yaw: float = 0, units: str = "radians"
) -> list:
    """
    Converts Euler angles (Roll, Pitch, Yaw) in radians into a Ros2 Quaternion
    """

    if units.lower() == "degrees":
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    x, y, z, w = Rotation.from_euler("XYZ", [roll, pitch, yaw]).as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)


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

    return Rotation.from_euler("XYZ", [roll, pitch, yaw]).as_matrix()


def angle_wrap(angle: float) -> float:
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped


def si_to_uni_vel(dxi: np.array, pose: np.array, projection_distance:float=0.03) -> np.array:
    d = projection_distance
    h = float(pose[2].item()) if hasattr(pose[2], "item") else float(pose[2])
    T_inv = np.array([[np.cos(h), np.sin(h)], [-np.sin(h) / d, np.cos(h) / d]])

    # Decoupled matrix multiplication
    dxu = T_inv @ dxi.reshape(2, 1)
    return dxu


def uni_to_si_vel(dxu: np.array, pose: np.array, projection_distance: float=0.05) -> np.array:
    d = projection_distance

    T = np.array([[1, 0], [0, d]])

    dxi = np.zeros((2, 1))
    h = float(pose[2].item()) if hasattr(pose[2], "item") else float(pose[2])
    # Rotation matrix from body frame to world frame.
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    dxi = R @ T @ dxu

    return dxi


def get_robot_barrier_func(
    safety_radius: float = 0.35,
    barrier_gain: float = 20.0,
    magnitude_limit: float = 0.2,
    boundary_points: np.array | None = None,
    build_area_points: np.array | None = None,
    block_safety_radius: float = 0.07,
    projection_dist: float = 0.15,
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
        neighbor_positions: np.array | None = None,
        block_positions: np.array | None = None,
    ) -> Twist:
        """_summary_

        Args:
            unsafe_cmd (Twist): The unsafe velocity
            pose (np.array): 3 x 1 array denoting the current pose of the robot [[x], [y], [theta]]
            neighbor_positions (np.array): 2 x M array denoting the positions of the M neighbors of the robot [[x_1, y_1],[x_2, y_2]]
            block_positions (np.array): 2 x B array denoting the positions of the B blocks in the space [[x_1, y_1],[x_2, y_2]]

        Returns:
            Twist: A safe velocity
        """
        if neighbor_positions is not None:
            assert neighbor_positions.shape[0] == 2, "Neighbor positions is incorrect shape"
        if block_positions is not None:
            assert block_positions.shape[0] == 2, "Block positions is incorrect shape"

        barrier_power = 3
        dxu = np.array([[unsafe_cmd.linear.x], [unsafe_cmd.angular.z]])
        projection_distance = projection_dist

        # TODO: Convert Unicycle to SI, and then back
        dxi = uni_to_si_vel(dxu, pose, projection_distance)
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

        num_constraints = M + B + 4 + 8 + 1
        A = np.zeros((num_constraints, 2))
        b = np.zeros(num_constraints)
        position = np.array(
            [
                pose[0] + projection_distance * np.cos(pose[2]),
                pose[1] + projection_distance * np.sin(pose[2]),
            ]
        ).flatten()
        constraint_idx = 0

        # Avoid neighbor constraings, we don't know neighbor vels, so we don't have our solver worry about them
        for neighbor_ndx in range(M):
            neighbor_position = neighbor_positions[:, neighbor_ndx]
            diff = position - neighbor_position
            h = np.dot(diff, diff) - safety_radius**2
            A[constraint_idx] = -2 * diff
            b[constraint_idx] = barrier_gain * (h**barrier_power)
            constraint_idx += 1

        for block_ndx in range(B):
            block_position = block_positions[:, block_ndx]
            diff = position - block_position
            h = np.dot(diff, diff) - block_safety_radius**2
            A[constraint_idx] = -2 * diff
            b[constraint_idx] = barrier_gain * (h**barrier_power)
            constraint_idx += 1

        # Boundary constraints
        A[constraint_idx] = [0, 1]
        b[constraint_idx] = (
            0.4
            * barrier_gain
            * (bp[3] - safety_radius / 2 - position[1]) ** barrier_power
        )
        constraint_idx += 1
        A[constraint_idx] = [0, -1]
        b[constraint_idx] = (
            0.4
            * barrier_gain
            * (position[1] - bp[2] - safety_radius / 2) ** barrier_power
        )
        constraint_idx += 1
        A[constraint_idx] = [1, 0]
        b[constraint_idx] = (
            0.4
            * barrier_gain
            * (bp[1] - safety_radius / 2 - position[0]) ** barrier_power
        )
        constraint_idx += 1
        A[constraint_idx] = [-1, 0]
        b[constraint_idx] = (
            0.4
            * barrier_gain
            * (position[0] - bp[0] - safety_radius / 2) ** barrier_power
        )
        constraint_idx += 1

        # Keep away constraints
        # We're approximating it as a circle because the box is too damn annoying
        center_x = (bap[0] + bap[1]) / 2.0
        center_y = (bap[2] + bap[3]) / 2.0
        circle_center = np.array([center_x, center_y])

        half_width = (bap[1] - bap[0]) / 2.0
        half_height = (bap[3] - bap[2]) / 2.0
        circle_radius = np.sqrt(half_width**2 + half_height**2)
        diff = position - circle_center
        buffer_zone = 0.03

        h = np.linalg.norm(diff) - (circle_radius + buffer_zone)
        A[constraint_idx] = -2 * diff
        b[constraint_idx] = barrier_gain * (h**barrier_power)
        constraint_idx += 1

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
            A[constraint_idx] = d
            b[constraint_idx] = constraint_bounds
            constraint_idx += 1

        P = matrix(np.eye(2))
        q = matrix(-dxi.astype(float))
        G = matrix(A)
        h_vec = matrix(b)

        sol = qp(P, q, G, h_vec)
        dxi_out = np.array([sol["x"][0], sol["x"][1]])
        dxu_out = si_to_uni_vel(dxi_out, pose, projection_distance)
        out_cmd = Twist()
        out_cmd.linear.x = dxu_out[0][0]
        out_cmd.angular.z = dxu_out[1][0]

        return out_cmd

    return barrier_func
