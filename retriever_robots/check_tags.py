import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Pose, PoseArray
from cv_bridge import CvBridge
import cv2

import message_filters


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

        self.vis_pub = self.create_publisher(PoseArray, "/asher/visible_blocks", 10)

        self.camera_matrix = None
        self.distortion_coeffs = None
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_25h9)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # self.cam_info = self.create_subscription(
        #     CameraInfo,
        #     "/asher/camera/color/camera_info",
        #     self.camera_info_callback,
        #     1,
        # )
        # self.cam = self.create_subscription(
        #     Image,
        #     "/asher/camera/color/image_raw",
        #     self.image_callback,
        #     1
        # )

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



    def cam_callback(self, img_msg, cam_info_msg):
        self.camera_info_callback(cam_info_msg)
        self.image_callback(img_msg)

    def camera_info_callback(self, msg):
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        try:
            if self.camera_matrix is None or self.distortion_coeffs is None:
                self.get_logger().warn("Camera info not received yet.")
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.get_logger().info("Received an image.")


            corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.parameters)
            if ids is not None:

                self.get_logger().warn(f"Found {len(ids)} tags: {ids.flatten()}")
                ok, rvec, tvec = cv2.solvePnP(
                        OBJ_PTS,
                        corners[0][0],
                        self.camera_matrix,
                        self.distortion_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                if ok:
                    self.get_logger().info(f"Tag pose: rvec={rvec.flatten()}, tvec={tvec.flatten()}")
                    # now rotate the robot to center the tag in the image
                    p = 1
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