import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from geometry_msgs.msg import Twist, Pose, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from retriever_msgs.action import GoToBlock  # type: ignore
import message_filters

from cv_bridge import CvBridge 
import cv2
import numpy as np
from enum import Enum
import math

class StateMachine(Enum):
    IDLE = 0
    NAVIGATING = 1
    GRABBING = 2
    RETURNING = 3
    FIND_POSE = 4
    RECOVERY = 5

def euler_from_quaternion(w, x, y, z):
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
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            f"{self.get_namespace()}/camera/depth/image_raw",
        )
        self.depth_info = message_filters.Subscriber(
            self,
            CameraInfo,
            f"{self.get_namespace()}/camera/depth/camera_info",
        )

        queue_size = 10
        slop = 0.5

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.depth_info], queue_size, slop
        )

        self.ts.registerCallback(self.cam_callback)


        # Set up publisher
        self.vel_pub = self.create_publisher(Twist, f"{self.get_namespace()}/cmd_vel", 10)

        # Set up action server
        self.retrieve_action = ActionServer(
            self,
            GoToBlock,
            f"{self.get_namespace()}/get_block",
            self.retrieve_callback,
        )

        # AprilTag recognition stuff
        ARUCO_DICT = cv2.aruco.DICT_APRILTAG_25h9
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Save the logger in ros1 style syntax, because wtf is with the self.get_logger() BS
        self.logger = self.get_logger()
        self.logger.info(f"Launched Retrieve Node for {self.get_namespace()}")

        self.marker_size = 0.05

        self.state = StateMachine.IDLE
        self.pose = None
        self.odom = None
        self.start_pose_set = False
        self._start_pose = None
        self.request_pose = None
        self.grab_pose = None

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
        self, color_msg: Image, depth_msg: Image, depth_info: CameraInfo
    ) -> Pose:
        if self.state == StateMachine.IDLE:
            self.grab_pose = Pose()
            self.marker_location = None
            return

        if (self.camera_matrix is None) or (self.distortion_coeffs is None):
            self.logger.warning("No camera info received yet, cannot process image")
            self.logger.info("Received camera info, saving camera matrix")
            self.camera_matrix = np.array(depth_info.k, dtype=np.float64).reshape(
                (3, 3)
            )
            self.distortion_coeffs = np.array(depth_info.d, dtype=np.float64)
            self.logger.debug(f"Camera matrix: {self.camera_matrix}")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="gray")
        except Exception as e:
            self.logger.error(f"Error: {e}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width, _ = gray.shape
        corners, ids, rejected = self.detector.detectMarkers(gray)

        frame_center_y = height // 2
        frame_center_x = width // 2

        if ids is not None:

            self.logger.info(f"Detected {len(ids)} ArUco marker(s)")
            marker_corners = corners[0][0]
            self.logger.info(f"Marker ID: {ids[0]} | Corners: {marker_corners[0]}")

            bot_l = marker_corners[3]
            bot_r = marker_corners[2]
            top_r = marker_corners[1]
            top_l = marker_corners[0]

            marker_center_top = ((top_l[0] + top_r[0]) // 2, (top_l[1] + top_r[1]) // 2)
            marker_center_bot = ((bot_l[0] + bot_r[0]) // 2, (bot_l[1] + bot_r[1]) // 2)

            marker_center_x = int(
                (
                    marker_corners[0][0]
                    + marker_corners[1][0]
                    + marker_corners[2][0]
                    + marker_corners[3][0]
                )
                // 4
            )
            marker_center_y = int(
                (
                    marker_corners[0][1]
                    + marker_corners[1][1]
                    + marker_corners[2][1]
                    + marker_corners[3][1]
                )
                // 4
            )

            # Find distance to block center
            depth_val = depth[marker_center_y, marker_center_x]
            if (
                self.state == StateMachine.GRABBING
                or self.state == StateMachine.RETURNING
            ):
                self.marker_location = (marker_center_x, marker_center_y, depth_val)

            if self.state == StateMachine.FIND_POSE:

                if depth_msg.encoding == "32FC1":
                    distance = float(depth_val)
                elif depth_msg.encoding == "16UC1":
                    distance = float(depth_val) / 1000.0  # mm to meters
                else:
                    self.get_logger().warn(
                        f"Unsupported encoding: {depth_msg.encoding}"
                    )
                    return
                closer = (
                    marker_center_top
                    if marker_center_top[1] > marker_center_bot[1]
                    else marker_center_bot
                )
                further = (
                    marker_center_bot
                    if closer == marker_center_top
                    else marker_center_top
                )

                dy = further[1] - closer[1]
                dx = further[0] - closer[0]

                angle = np.arctan2(dy, dx)
                roll = 0.0  # Robot can't roll
                pitch = 0.0  # Robot can't pitch, it's only a batter

                quaternion = quaternion_from_euler(
                    roll, pitch, angle
                )

                self.grab_pose.position.x = self.odom.pose.position.x
                self.grab_pose.position.y = distance * np.tan(angle)
                self.grab_pose.position.z = 0.0  # Robot is ground vehicle

                self.grab_pose.orientation.x = quaternion[0]
                self.grab_pose.orientation.y = quaternion[1]
                self.grab_pose.orientation.z = quaternion[2]
                self.grab_pose.orientation.w = quaternion[3]

    def retrieve_callback(self, goal_handle):
        self.logger.info(f"Received retrieve action goal: {goal_handle.request}")

        feedback_msg = GoToBlock.Feedback()
        feedback_msg.block_captured = False
        
        if self.state == StateMachine.IDLE:
            self.request_pose = goal_handle.request.goal_pose
            self.state = StateMachine.NAVIGATING
            self.logger.info(
                f"Received retrieve action goal: {self.request_pose}, entering Navigation state"
            )

        if self.state == StateMachine.NAVIGATING:
            reached = self.go_to_pose(self.request_pose)
            if reached:
                self.state = StateMachine.FIND_POSE
                self.logger.info(
                    f"Reached block pose, entering Find Pose state to orient around block"
                )
        if self.state == StateMachine.FIND_POSE:
            reached = self.go_to_pose(self.grab_pose)
            if reached:
                self.state = StateMachine.GRABBING
                self.logger.info(
                    f"Reached grab pose, entering Grabbing state to attempt to grab block"
                )
        if self.state == StateMachine.GRABBING:
            # TODO: This should be changed to be the output of a function from the camera
            feedback_msg.block_captured = True


        # This state is just if we want the robot to return to the starting position
        if self.state == StateMachine.RETURNING:
            if self._start_pose is not None:
                reached = self.go_to_pose(self._start_pose)
                if reached:
                    goal_handle.succeed()
                    result = GoToBlock.Result()
                    result.success = True
                    result.end_pose = self.curr_pose
                    self.state == StateMachine.IDLE
                    return result

        feedback_msg.curr_pose = self.curr_pose
        goal_handle.publish_feedback(feedback_msg)

    def angle_wrap(self, angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def go_to_pose(self, target_pose: Pose) -> bool:
        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        dx = target_pose.position.x - self.curr_pose.position.x
        dy = target_pose.position.y - self.curr_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q = self.curr_pose.orientation
        _, _, current_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        angle_diff = self.angle_wrap(angle_to_target - current_yaw)
        desired_yaw_diff = self.angle_wrap(target_pose.orientation.z - current_yaw)

        cmd = Twist()
        p = 0.5 # Arbitrary gain cause why not

        if abs(angle_diff) > 0.1 and distance > 0.15:
            cmd.angular.z = p * angle_diff
        else:
            if distance > 0.1:
                cmd.linear.x = p * distance
            else:
                if abs(desired_yaw_diff) > 0.1:
                    cmd.angular.z = p * desired_yaw_diff
                else:
                    return True

        self.vel_pub.publish(cmd)
        return False

    @property
    def curr_pose(self):
        if hasattr(self, "pose") and self.pose is not None:
            return self.pose
        elif hasattr(self, "odom") and self.odom is not None:
            return self.odom.pose.pose
        else:
            self.logger.warning("No pose information available")
            return None

def main():
    rclpy.init()
    node = RetrieveNode("retriever_node")
    # I'll try spinning, that's a good trick!
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
