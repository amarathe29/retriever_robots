import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from retriever_msgs.action import GoToBlock  # type: ignore
import message_filters

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf_transformations
from enum import Enum


class StateMachine(Enum):
    IDLE = 0
    GRABBING = 1
    RETURNING = 2
    NAVIGATING = 3


class RetrieveNode(Node):
    """docstring for RetrieveNode."""

    def __init__(self, arg):
        super(RetrieveNode, self).__init__()
        # Set up subscribers
        self.bev_pose_sub = self.subscriber(
            f"{self.get_namespace()}/pose", Pose, self.pose_callback
        )
        self.odom_sub = self.subscriber(
            f"{self.get_namespace()}/odom", Odometry, self.odom_callback
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

        # self.cam_info_sub = self.subscriber(
        #     f"{self.get_namespace()}/camera/color/camera_info",
        #     CameraInfo,
        #     self.cam_info_callback,
        # )

        # Set up publisher
        self.vel_pub = self.publisher(Twist, f"{self.get_namespace()}/cmd_vel", 10)

        # Set up action server
        self.retrieve_action = ActionServer(
            self,
            GoToBlock,
            f"{self.get_namespace()}/get_block",
            self.retrieve_callback,
        )

        # Aruco tag recognition stuff
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

    def pose_callback(self, msg: Pose) -> None:
        self.logger.debug(f"Received Pose: {msg}")
        self.pose = msg

    def odom_callback(self, msg: Odometry) -> None:
        self.logger.debug(f"Received Odometry: {msg}")
        self.odom = msg

    # def cam_info_callback(self, msg: CameraInfo) -> None:
    # if (self.camera_matrix is None) or (self.distortion_coeffs is None):
    #     self.logger.info("Received camera info, saving camera matrix")
    #     self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
    #     self.distortion_coeffs = np.array(msg.d, dtype=np.float64)
    #     self.logger.debug(f"Camera matrix: {self.camera_matrix}")

    def cam_callback(
        self, color_msg: Image, depth_msg: Image, depth_info: CameraInfo
    ) -> Pose:

        self.new_pose = Pose()

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
        except CvBridgeError as e:
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

            # find distance to block
            depth_val = depth[marker_center_y, marker_center_x]
            if depth_msg.encoding == "32FC1":
                distance = float(depth_val)
            elif depth_msg.encoding == "16UC1":
                distance = float(depth_val) / 1000.0  # mm to meters
            else:
                self.get_logger().warn(f"Unsupported encoding: {depth_msg.encoding}")
                return
            closer = (
                marker_center_top
                if marker_center_top[1] > marker_center_bot[1]
                else marker_center_bot
            )
            further = (
                marker_center_bot if closer == marker_center_top else marker_center_top
            )

            dy = further[1] - closer[1]
            dx = further[0] - closer[0]

            angle = np.arctan2(dy, dx)
            roll = 0.0  # Robot can't roll
            pitch = 0.0  # Robot can't pitch, it's only a batter

            quaternion = tf_transformations.quaternion_from_euler(roll, pitch, angle)

            self.new_pose.position.x = self.odom.pose.position.x
            self.new_pose.position.y = distance * np.tan(angle)
            self.new_pose.position.z = width

            self.new_pose.orientation.x = quaternion[0]
            self.new_pose.orientation.y = quaternion[1]
            self.new_pose.orientation.z = quaternion[2]
            self.new_pose.orientation.w = quaternion[3]

    def retrieve_callback(self, goal_handle):
        self.logger.info(f"Received retrieve action goal: {goal_handle.request}")


def main():
    node = RetrieveNode()
    # I'll try spinning, that's a good trick!
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
