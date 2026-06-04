import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped

from retriever_robots.utils import create_rotation_matrix

from retriever_msgs.msg import PoseStatus

import message_filters

from scipy.spatial.transform import Rotation




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


class DetectBlock(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.logger = self.get_logger()

        self.bridge = CvBridge()

        self.pub = self.create_publisher(Twist, f"{self.get_namespace()}/cmd_vel", 10)

        # communicates the location of the identified block back to the retriever node. This is a custom topic, not a standard ROS topic, so we can change it as needed.
        self.vis_pub = self.create_publisher(
            PoseStatus, f"{self.get_namespace()}/visible_block", 10
        )

        self.debug_pub = self.create_publisher(PoseStamped, f"{self.get_namespace()}/debug_block_pose", 10
        )

        self.camera_matrix = None
        self.distortion_coeffs = None
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_25h9)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.base_frame = self._namespaced_frame("base_link")
        self.camera_frame = self._namespaced_frame("camera_link")
        self.camera_offset = np.array([[0.15], [0.0], [0.1]])

        self.R_image_frame_to_robot = create_rotation_matrix(pitch = -90, units="degrees") @ create_rotation_matrix(roll = 90, units="degrees")
        self.R_cam_angle_to_robot = create_rotation_matrix(roll = 30, units="degrees") @ self.R_image_frame_to_robot

        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._broadcast_static_camera_transform()

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

        self.logger.info(f"Launched Block Detection Node for {self.get_namespace()}")

    def _namespaced_frame(self, frame_name):
        ns = self.get_namespace().strip("/")
        return f"{ns}/{frame_name}" if ns else frame_name

    def _broadcast_static_camera_transform(self):
        self.static_transform = TransformStamped()
        self.static_transform.header.stamp = self.get_clock().now().to_msg()
        self.static_transform.header.frame_id = self.base_frame
        self.static_transform.child_frame_id = self.camera_frame
        self.static_transform.transform.translation.x = float(self.camera_offset[0])
        self.static_transform.transform.translation.y = float(self.camera_offset[1])
        self.static_transform.transform.translation.z = float(self.camera_offset[2])

        # The stored R_cam_to_robot maps camera coordinates into robot coordinates.
        # For a TF from robot->camera, use the inverse rotation.
        quat = Rotation.from_matrix(self.R_cam_angle_to_robot.T).as_quat()
        self.static_transform.transform.rotation.x = float(quat[0])
        self.static_transform.transform.rotation.y = float(quat[1])
        self.static_transform.transform.rotation.z = float(quat[2])
        self.static_transform.transform.rotation.w = float(quat[3])

        self.static_broadcaster.sendTransform([self.static_transform])
        self.logger.info(
            f"Published static transform {self.base_frame} -> {self.camera_frame}"
        )

    def cam_callback(self, img_msg, cam_info_msg):
        self.camera_info_callback(cam_info_msg)
        self.image_callback(img_msg)

    def camera_info_callback(self, msg):
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        pose_status = PoseStatus()
        pose_status.tag_in_frame = False
        try:
            if self.camera_matrix is None or self.distortion_coeffs is None:
                self.logger.warn("Camera info not received yet.")
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            corners, ids, _ = cv2.aruco.detectMarkers(
                cv_image, self.aruco_dict, parameters=self.parameters
            )
            if ids is not None:

                self.logger.debug(f"Found {len(ids)} tags: {ids.flatten()}")
                ok, rvec, tvec = cv2.solvePnP(
                    OBJ_PTS,
                    corners[0][0],
                    self.camera_matrix,
                    self.distortion_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok:                    

                    x, y, z, w = Rotation.from_rotvec(rvec.flatten()).as_quat()


                    camera_pose = PoseStamped()
                    camera_pose.header.stamp = self.get_clock().now().to_msg()
                    camera_pose.header.frame_id = self.camera_frame
                    camera_pose.pose.position.x = float(tvec[0])
                    camera_pose.pose.position.y = float(tvec[1])
                    camera_pose.pose.position.z = float(tvec[2])
                    camera_pose.pose.orientation.x = x
                    camera_pose.pose.orientation.y = y
                    camera_pose.pose.orientation.z = z
                    camera_pose.pose.orientation.w = w

                    robot_pose = do_transform_pose_stamped(camera_pose, self.static_transform)

                    self.debug_pub.publish(robot_pose)

                    pose_status.tag_in_frame = True
                    pose_status.pose = robot_pose.pose
                    
                    
                    self.logger.info(
                        f"Tag Detected: Marker center is {robot_pose.pose.position.x} m away,  {robot_pose.pose.position.y} m to the left, and {robot_pose.pose.position.z} m down)",
                        throttle_duration_sec=1.0,
                    )

                else:
                    self.logger.debug(
                        "Could not solve PnP for detected tag.",
                        throttle_duration_sec=1.0,
                    )

            else:
                self.logger.debug(
                    "No tags detected in the image.", throttle_duration_sec=1.0
                )

            if not pose_status.tag_in_frame:
                pose_status.block_in_frame, x, y = self.segment_color(
                    cv_image
                )  # if no tags are detected, try to segment based on color as a fallback
                if pose_status.block_in_frame:
                    # create a fake pose with a y position scaled based on the negative x value of the image. Make a rough x pose based on y in frame
                    pose_status.pose.position.x = 0.5 + max(
                        min(-0.001 * (y - cv_image.shape[0] / 2), 0.5), -0.5
                    )
                    pose_status.pose.position.y = max(
                        min(-0.001 * (x - cv_image.shape[1] / 2), 0.5), -0.5
                    )
                    pose_status.pose.position.z = 0.0
                    pose_status.pose.orientation.w = 1.0
                    self.logger.debug(
                        f"Tag not detected, using color segmentation. Estimated pose: ({pose_status.pose.position.x}, {pose_status.pose.position.y}, {pose_status.pose.position.z})",
                        throttle_duration_sec=1.0,
                    )

            self.vis_pub.publish(pose_status)

        except Exception as e:
            self.logger.error(f"Error converting image: {e}")

    def segment_color(self, cv_image):
        # Convert the image to HSV color space for better color segmentation
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the block's color in HSV space
        # do orange instead
        lower_color = np.array([2, 100, 100])  # lower bound (orange color)
        upper_color = np.array([10, 255, 255])  # Example upper

        # Create a mask using the defined color bounds
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 50:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.logger.debug(f"Segmented block at pixel coordinates: ({cX}, {cY})")
                return True, cX, cY
        return False, None, None


def main(args=None):
    rclpy.init(args=args)
    node = DetectBlock("detect_block")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
