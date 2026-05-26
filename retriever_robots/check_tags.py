import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Pose
from cv_bridge import CvBridge
import cv2


MARKER_SIZE = 0.0544
OBJ_PTS = np.array(
                        [
                            [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0.0],
                            [MARKER_SIZE / 2, MARKER_SIZE / 2, 0.0],
                            [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0.0],
                            [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0.0],
                        ],
                        dtype=np.float64,
)

class CheckTags(Node):
    def __init__(self):
        super().__init__("check_tags")
        self.bridge = CvBridge()
        self.get_logger().info("CheckTags node has been started.")


        self.pub = self.create_publisher(Twist, "/asher/cmd_vel", 10)
        self.camera_matrix = None
        self.distortion_coeffs = None

        self.cam_info = self.create_subscription(
            CameraInfo,
            "/asher/camera/color/camera_info",
            self.camera_info_callback,
            1,
        )
        self.cam = self.create_subscription(
            Image,
            "/asher/camera/color/image_raw",
            self.image_callback,
            1
        )

    def camera_info_callback(self, msg):
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)

    def image_callback(self, msg):
        try:
            if self.camera_matrix is None or self.distortion_coeffs is None:
                self.get_logger().warn("Camera info not received yet.")
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.get_logger().info("Received an image.")

            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_25h9)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=parameters)
            if ids is not None:

                self.get_logger().warn(f"Found {len(ids)} tags: {ids.flatten()}")
                ok, rvec, tvec = cv2.solvePnP(
                        OBJ_PTS,
                        corners,
                        self.camera_matrix,
                        self.distortion_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                if ok:
                    self.get_logger().info(f"Tag pose: rvec={rvec.flatten()}, tvec={tvec.flatten()}")
                    # now rotate the robot to center the tag in the image
                    p = 0.1
                    T_marker_to_robot = tvec.flatten()
                    twist = Twist()
                    twist.angular.z = (min(max(-p * T_marker_to_robot[0], -0.4), 0.4))
                    self.get_logger().info(f"Publishing twist: angular.z={twist.angular.z}")

                    self.pub.publish(twist)

                else:
                    self.get_logger().error("Could not solve PnP for detected tag.")

        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CheckTags()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()