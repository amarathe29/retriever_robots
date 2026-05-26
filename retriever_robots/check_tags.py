import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Pose
from cv_bridge import CvBridge
import cv2

from retriever_robots.utils import quaternion_from_euler, create_rotation_matrix

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


        self.pub = self.create_publisher(Twist, f"{self.get_namespace()}/cmd_vel", 10)

        # communicates the location of the identified block back to the retriever node. This is a custom topic, not a standard ROS topic, so we can change it as needed.
        self.vis_pub = self.create_publisher(Pose, f"{self.get_namespace()}/visible_block", 10)

        self.camera_matrix = None
        self.distortion_coeffs = None
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_25h9)
        self.parameters = cv2.aruco.DetectorParameters_create()


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
                    # TODO: check this math and ordering

                    # now, convert rvec and tvec into a Pose in the world frame
                    R_marker_to_cam, _ = cv2.Rodrigues(rvec)
                    # Image X is robot -Y, Image Y is robot -Z, Image Z is robot X
                    R_image_to_robot_axes = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                    R_cam_angle_to_robot = create_rotation_matrix(pitch=-30, units='degrees')

                    R_cam_to_robot = R_cam_angle_to_robot @ R_image_to_robot_axes

                    R_marker_to_robot = R_cam_to_robot @ R_marker_to_cam
                    rot_vec, _ = cv2.Rodrigues(R_marker_to_robot)

                    T_cam_to_robot = np.array([[-0.1], [0], [0]]) # camera is 10cm in front of the robot axis

                    T_marker_to_cam = tvec.reshape(3, 1)
                    T_marker_to_robot = R_cam_to_robot @ T_marker_to_cam + T_cam_to_robot


                    self.logger.warn(
                        f"Marker center is {T_marker_to_robot[0]} m away,  {T_marker_to_robot[1]} m to the left, and {T_marker_to_robot[2]} m down)"
                    )

                    pose = Pose()

                    pose.position.x = float(T_marker_to_robot[0])
                    pose.position.y = float(T_marker_to_robot[1])
                    pose.position.z = float(T_marker_to_robot[2])

                    q = quaternion_from_euler(float(rot_vec[0]), float(rot_vec[1]), float(rot_vec[2]))
                    pose.orientation.x = q[1]
                    pose.orientation.y = q[2]
                    pose.orientation.z = q[3]
                    pose.orientation.w = q[0]

                    self.vis_pub.publish(pose)


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