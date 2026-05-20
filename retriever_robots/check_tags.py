import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CheckTags(Node):
    def __init__(self):
        super().__init__("check_tags")
        self.bridge = CvBridge()
        self.get_logger().info("CheckTags node has been started.")

        self.create_subscription(
            Image,
            "/asher/camera/color/image_raw",
            self.image_callback,)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.get_logger().info("Received an image.")
            # use cv2 2.4.6 to find an aruco tag in the image
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_25h9)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=parameters)
            if ids is not None:
                self.get_logger().warn(f"Found {len(ids)} tags: {ids.flatten()}")
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