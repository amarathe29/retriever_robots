import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, Pose, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from retriever_msgs.action import GoToBlock  # type: ignore
import message_filters

from cv_bridge import CvBridge
import cv2
import numpy as np
from enum import Enum, auto
import math


class StateMachine(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    FIND_BLOCK_POSE = auto()
    POSITIONING = auto()
    GRABBING = auto()
    STOCKPILING = auto()
    RETURNING = auto()
    RECOVERY = auto()


def euler_from_quaternion(x, y, z, w):
    """
    Converts a quaternion into standard Euler angles (Roll, Pitch, Yaw)
    in radians. Sequence: ZYX (Yaw, Pitch, Roll).
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




class RetrieveNode(Node):
    """docstring for RetrieveNode."""

    def __init__(self, node_name, *args):
        super(RetrieveNode, self).__init__(node_name)
        # Set up subscribers
        self.bev_pose_sub = self.create_subscription(
            Pose, f"{self.get_namespace()}/pose", self.pose_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, f"{self.get_namespace()}/odom", self.odom_callback, 10
        )

        # Set up synchronizer for color and depth images
        self.color_sub = message_filters.Subscriber(
            self, Image, f"{self.get_namespace()}/camera/color/image_raw"
        )
        self.color_info = message_filters.Subscriber(
            self,
            CameraInfo,
            f"{self.get_namespace()}/camera/color/camera_info",
        )

        queue_size = 1
        slop = 0.2

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.color_info], queue_size, slop
        )

        self.ts.registerCallback(self.cam_callback)

        # Set up publisher
        self.vel_pub = self.create_publisher(
            Twist, f"{self.get_namespace()}/cmd_vel", 10
        )

        # Set up action server
        self.retrieve_action = ActionServer(
            self,
            GoToBlock,
            f"{self.get_namespace()}/gotoblock",
            self.retrieve_callback,
        )

        # AprilTag recognition stuff
        ARUCO_TAG = cv2.aruco.DICT_APRILTAG_25h9
        self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_TAG)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Save the logger in ros1 style syntax, because wtf is with the self.get_logger() BS
        self.logger = self.get_logger()
        self.logger.info(f"Launched Retrieve Node for {self.get_namespace()}")

        self.marker_size = 0.0544

        self.state = StateMachine.IDLE
        self.pose = None
        self.odom = None
        self.start_pose_set = False
        self._start_pose = None
        self.request_pose = None
        self.block_pose = None

    def pose_callback(self, msg: Pose) -> None:
        self.logger.debug(f"Received Pose: {msg}")
        self.pose = msg
        if not self.start_pose_set:
            self._start_pose = self.pose
            self.start_pose_set = True

    def odom_callback(self, msg: Odometry) -> None:
        self.logger.debug(f"Received Odometry: {msg}")
        self.odom = msg
        if self._start_pose is None:
            self._start_pose = self.odom.pose.pose

    def cam_callback(
        self, color_msg: Image, color_info: CameraInfo
    ) -> None:

        if (self.camera_matrix is None) or (self.distortion_coeffs is None):
            self.logger.warning("No camera info received yet, cannot process image")
            self.logger.info("Received camera info, saving camera matrix")
            self.camera_matrix = np.array(color_info.k, dtype=np.float64).reshape(
                (3, 3)
            )
            self.distortion_coeffs = np.array(color_info.d, dtype=np.float64)
            self.logger.debug(f"Camera matrix: {self.camera_matrix}")
            return

        if self.state == StateMachine.IDLE:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception as e:
            self.logger.error(f"Error in decoding images from camera: {e}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_parameters
        )

        self.logger.info(f"Detected {len(ids) if ids is not None else 0} ArUco marker(s)")


        if ids is not None:

            # self.logger.info(f"Detected {len(ids)} ArUco marker(s)")
            marker_corners = corners[0][0]
            self.logger.info(
                f"Marker ID: {ids[0]} | Corners: {marker_corners[0]}| All Corners: {marker_corners}"
            )
            self.logger.info(f"State Machine State: {self.state.name.lower()}")

            if (self.state == StateMachine.FIND_BLOCK_POSE):
                if (self.block_pose is None):

                    self.logger.info(f"Searching for block")

                    if len(marker_corners) != 4:
                        self.logger.error(
                            f"Detected marker {marker_corners} does not have 4 corners, cannot estimate pose"
                        )
                        return

                    self.logger.error(
                        f"All the intrinsics:\n {self.camera_matrix}, {self.distortion_coeffs}, {marker_corners}"
                    )

                    # DEBUG: write the img to file with aruco tags detected
                    im = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
                    cv2.imwrite("detected_markers.png", im)

                    obj_pts = np.array(
                        [
                            [-self.marker_size / 2, self.marker_size / 2, 0.0],
                            [self.marker_size / 2, self.marker_size / 2, 0.0],
                            [self.marker_size / 2, -self.marker_size / 2, 0.0],
                            [-self.marker_size / 2, -self.marker_size / 2, 0.0],
                        ],
                        dtype=np.float64,
                    )
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts,
                        marker_corners,
                        self.camera_matrix,
                        self.distortion_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )

                    if ok:
                        self.logger.info(f"tvec:\n{tvec}\nrvec:\n{rvec}")
                    else:
                        self.logger.error(f"PnP Solver Failed: {ok}")
                        return


                    R_marker_to_cam, _ = cv2.Rodrigues(rvec)
                    # Double checking: Image X is robot -Y, Image Y is Robot -Z, and Image Z is robot X
                    R_image_to_robot_axes = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                    R_cam_angle_to_robot = create_rotation_matrix(pitch=-30, units='degrees')
                    R_cam_to_robot = R_cam_angle_to_robot @ R_image_to_robot_axes

                    # The camera is like 10 cm in front of the robot wheel base, honestly this is probably unnecessary, I added it just in case
                    T_cam_to_robot = np.array([[-0.1], [0.0], [0.0]])

                    R_marker_to_robot = R_cam_to_robot @ R_marker_to_cam

                    # cam_x, cam_y, cam_z = tvec

                    marker_x_robot = R_marker_to_robot[:, 0].reshape(3, 1)
                    marker_x_robot[2] = 0.0
                    T_marker_to_cam = tvec.reshape(3, 1)
                    T_marker_to_robot = T_marker_to_cam @ R_cam_to_robot + T_cam_to_robot

                    self.logger.warn(
                        f"Marker center is {T_marker_to_robot[0]} m away,  {T_marker_to_robot[1]} m to the left, and {T_marker_to_robot[2]} m down)"
                    )

                    desired_dist = 0.1
                    # Approach direction is negative if tag x-axis is pointed towards the robot and positive if tag x-axis is pointed away from the robot
                    approach_direction = -np.sign(np.dot(marker_x_robot.flatten(), T_marker_to_robot.flatten())) * marker_x_robot.flatten()
                    approach_direction /= np.linalg.norm(approach_direction)

                    # TODO: If we end up putting markers on the side of the block, we need to always approach from positive Z:
                    # marker_z_robot = R_marker_to_robot[:, 2].reshape(3,1)
                    # marker_z_robot[2] = 0.0
                    # approach_direction = marker_z_robot / np.linalg.norm(marker_z_robot)

                    # TODO: Verify this is a unit vector and a meaningful one, and not just all in one direction
                    self.logger.info(f"Approach direction: {approach_direction}")

                    # Just in case self.curr_pose changes for some reason between reads to create this array, we'll store it as a var and use that
                    curr_pose = self.curr_pose
                    curr_loc = np.array(
                        [curr_pose.position.x, curr_pose.position.y, curr_pose.position.z]
                    )
                    desired_location = (
                        curr_loc + T_marker_to_robot + desired_dist * approach_direction
                    )

                    # The new location Z better be god damn 0
                    self.logger.info(
                        "=" * 20
                        + f"\nCurrent location:\n{self.odom.pose.pose.position}\nProposed new location:\n{desired_location}\n"
                        + "=" * 20
                    )

                    self.block_pose = Pose()
                    self.block_pose.position.x = desired_location[0]
                    self.block_pose.position.y = desired_location[1]
                    self.block_pose.position.z = 0.0

                    # Yaw probably needs a negative or something, I'm not sure
                    yaw = np.arctan2(approach_direction[0], approach_direction[1])
                    yaw = 0.0
                    quaternion = quaternion_from_euler(roll=0.0, pitch=0.0, yaw=yaw)

                    self.block_pose.orientation.w = quaternion[0]
                    self.block_pose.orientation.x = quaternion[1]
                    self.block_pose.orientation.y = quaternion[2]
                    self.block_pose.orientation.z = quaternion[3]
                    self.logger.info(f"Estimated grab pose: {self.block_pose}")

    def retrieve_callback(self, goal_handle) -> GoToBlock.Result:
        self.logger.info(f"Received retrieve action goal: {goal_handle.request}")
        if self.state != StateMachine.IDLE:
            self.logger.error(f"Already running an action")
            goal_handle.fail()
            result = GoToBlock.Result()
            result.success = True
            result.end_pose = self.curr_pose
            return

        feedback_msg = GoToBlock.Feedback()
        feedback_msg.block_captured = False

        goal_reached = False

        while not goal_reached:

            if self.state == StateMachine.IDLE:
                self.request_pose = goal_handle.request.goal_pose
                self.block_pose = None
                self.marker_location = None
                self.logger.info(
                    f"Received retrieve action goal: {self.request_pose}, entering NAVIGATING state"
                )
                self.state = StateMachine.NAVIGATING

            elif self.state == StateMachine.NAVIGATING:
                reached = self.go_to_pose(self.request_pose) # TODO, we can't collide with the block, so this go_to_pose must be interruptable as soon as we see the block.
                if reached:
                    self.logger.info(
                        f"Reached block pose, entering FIND_BLOCK_POSE state to locate block"
                    )
                    self.state = StateMachine.FIND_BLOCK_POSE
            elif self.state == StateMachine.FIND_BLOCK_POSE:
                if self.block_pose is not None:
                    self.logger.info(
                        f"Found block pose, entering POSITIONING state to orient around block"
                    )
                    self.state = StateMachine.POSITIONING

            elif self.state == StateMachine.POSITIONING:
                reached = self.go_to_pose(self.block_pose)
                if reached:
                    self.logger.info(
                        f"Reached grab pose, entering GRABBING state to grab block"
                    )
                    self.state = StateMachine.GRABBING

            elif self.state == StateMachine.GRABBING:
                # TODO: This should approach and capture the block
                self.logger.info(
                    f"Grabbed block, entering STOCKPILING state to stockpile block"
                )
                self.state = StateMachine.STOCKPILING
                pass

            elif self.state == StateMachine.STOCKPILING:
                # TODO: This should be changed to be the output of some function from the camera also add functionality
                feedback_msg.block_captured = True
                reached = True
                if feedback_msg.block_captured and reached:
                    self.logger.info(
                        f"Stockpiled block, entering RETURNING state to return to start"
                    )
                    self.state = StateMachine.RETURNING
                elif not feedback_msg.block_captured:
                    self.logger.info(
                        f"Failed to capture block, entering RECOVERY state to attempt recovery"
                    )
                    self.state = StateMachine.RECOVERY
                pass

            # This state is just if we want the robot to return to the starting position
            elif self.state == StateMachine.RETURNING:
                if self._start_pose is not None:
                    goal_reached = self.go_to_pose(self._start_pose)
                    if goal_reached:
                        goal_handle.succeed()
                        result = GoToBlock.Result()
                        result.success = True
                        result.end_pose = self.curr_pose
                        self.state = StateMachine.IDLE
                        return result

            elif self.state == StateMachine.RECOVERY:
                # TODO: Add some kind of recovery behavior
                pass

            feedback_msg.curr_pose = self.curr_pose
            goal_handle.publish_feedback(feedback_msg)

        result = GoToBlock.Result()
        result.success = False
        result.end_pose = self.curr_pose
        # add gracefull returning to idle and the idle position here
        return result

    def angle_wrap(self, angle: float) -> float:
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        return wrapped

    def pose_controller(self, target_pose: Pose) -> Twist:

        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        dx = target_pose.position.x - self.curr_pose.position.x
        dy = target_pose.position.y - self.curr_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q_curr = self.curr_pose.orientation
        _, _, current_yaw = euler_from_quaternion(
            q_curr.x, q_curr.y, q_curr.z, q_curr.w
        )
        angle_diff = self.angle_wrap(angle_to_target - current_yaw)
        q_desired = target_pose.orientation
        _, _, desired_yaw = euler_from_quaternion(
            q_desired.x, q_desired.y, q_desired.z, q_desired.w
        )
        desired_yaw_diff = self.angle_wrap(desired_yaw - current_yaw)

        cmd = Twist()
        linear_gain = 0.3  # Arbitrary gain cause why not
        angular_gain = 0.7

        # TODO: Double-check this works with the new changes
        if abs(angle_diff) > 0.1 and distance > 0.15:
            cmd.angular.z = angular_gain * angle_diff
        else:
            if distance > 0.1:
                cmd.linear.x = linear_gain * distance
            else:
                if abs(desired_yaw_diff) > 0.1:
                    cmd.angular.z = angular_gain * desired_yaw_diff
                else:
                    return Twist()

        return cmd

    # TODO: Implement the robot moving backwards
    def pose_controller_reverse(self, target_pose: Pose) -> Twist:
        pass

    # TODO: Implement a controller to drive the robot in smooth arcs instead of lines using control lyapunov functions or splines
    # gamma is approach angle gain, k is desired angle gain, and h is rotation error gain
    def pose_controller_clf(self, target_pose: Pose, gamma=1.0, k=3.0, h=-0.5) -> Twist:
        assert gamma > 0, f"gamma = {gamma} must be greater than 0"
        assert k > gamma, f"k = {k} must be greater than gamma = {gamma}"
        assert h > 0, f"h = {h} must be greater than 0"

        q_curr = self.curr_pose.orientation
        _, _, theta = euler_from_quaternion(q_curr.x, q_curr.y, q_curr.z, q_curr.w)
        q_desired = target_pose.orientation
        _, _, desired_theta = euler_from_quaternion(
            q_desired.x, q_desired.y, q_desired.z, q_desired.w
        )

        dx = target_pose.position.x - self.curr_pose.position.x
        dy = target_pose.position.y - self.curr_pose.position.y

        # We're going to do things in the frame of reference of the goal position, because apparently that makes behavior more consistent (shoutout to my old advisor for this controller)
        R = np.array(
            [
                [np.cos(-desired_theta), -np.sin(-desired_theta)],
                [np.sin(-desired_theta), np.cos(-desired_theta)],
            ]
        )

        # positional error from the goal's frame of reference
        position_error = R @ np.array([dx, dy])
        e = np.linalg.norm(position_error)

        # Angle to goal from goal frame
        theta_error_vec = np.arctan2(position_error[1], position_error[0])

        alpha = theta_error_vec - (theta - desired_theta)
        alpha = self.angle_wrap(alpha)

        ca = np.cos(alpha)
        sa = np.sin(alpha)

        v = gamma * e * ca

        # Prevent divide by zero errors
        sinc_alpha = 1.0 if np.abs(alpha) < 1e-6 else (sa / alpha)
        w = k * alpha * gamma * ca * sinc_alpha * (alpha + h + theta_error_vec)
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        return cmd

    def check_reached_target(self, target_pose: Pose) -> bool:
        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        dx = target_pose.position.x - self.curr_pose.position.x
        dy = target_pose.position.y - self.curr_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q_curr = self.curr_pose.orientation
        _, _, current_yaw = euler_from_quaternion(
            q_curr.x, q_curr.y, q_curr.z, q_curr.w
        )
        angle_diff = self.angle_wrap(angle_to_target - current_yaw)

        # If we haven't reached the target, return false, else we return true
        if abs(angle_diff) > 0.1 or distance > 0.1:
            return False

        return True

    def go_to_pose(self, target_pose: Pose, controller=pose_controller) -> bool:
        # for now, just pretend we went there
        self.logger.error(f"Gone to pose: {target_pose}")
        return True

        cmd = self.pose_controller(self.request_pose)
        self.vel_pub.publish(cmd)
        reached = self.check_reached_target(self.request_pose)
        return reached

    @property
    def curr_pose(self):
        if hasattr(self, "pose") and self.pose is not None:
            self.logger.debug("Using pose topic")
            return self.pose
        elif hasattr(self, "odom") and self.odom is not None:
            self.logger.debug("Using odometry topic")
            return self.odom.pose.pose
        else:
            self.logger.warning("No pose information available")
            return None


def main():
    rclpy.init()
    node = RetrieveNode("retriever_node")
    # I'll try spinning, that's a good trick!
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
