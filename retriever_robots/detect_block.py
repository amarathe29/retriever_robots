import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Pose
from cv_bridge import CvBridge
import cv2

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
        self.bridge = CvBridge()

        self.pub = self.create_publisher(Twist, f"{self.get_namespace()}/cmd_vel", 10)

        # communicates the location of the identified block back to the retriever node. This is a custom topic, not a standard ROS topic, so we can change it as needed.
        self.vis_pub = self.create_publisher(PoseStatus, f"{self.get_namespace()}/visible_block", 10)

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

        self.logger = self.get_logger()
        self.logger.info(f"Launched Block Detection Node for {self.get_namespace()}")


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
        try:
            if self.camera_matrix is None or self.distortion_coeffs is None:
                self.logger.warn("Camera info not received yet.")
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
           
            corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.parameters)
            if ids is not None:

                self.logger.warn(f"Found {len(ids)} tags: {ids.flatten()}")
                ok, rvec, tvec = cv2.solvePnP(
                        OBJ_PTS,
                        corners[0][0],
                        self.camera_matrix,
                        self.distortion_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                if ok:

                    # now, convert rvec and tvec into a Pose in the world frame
                    R_marker_to_cam, _ = cv2.Rodrigues(rvec)
                    # Image X is robot -Y, Image Y is robot -Z, Image Z is robot X
                    R_image_to_robot_axes = np.array(
                        [[0, 0, 1], 
                         [-1, 0, 0], 
                         [0, -1, 0],]
                    )
                    R_cam_angle_to_robot = create_rotation_matrix(pitch=30, units='degrees')

                    R_cam_to_robot = R_cam_angle_to_robot @ R_image_to_robot_axes

                    R_marker_to_robot = R_cam_to_robot @ R_marker_to_cam

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


                    w,x,y,z = Rotation.from_matrix(R_marker_to_robot).as_quat()
                    pose.orientation.x = x
                    pose.orientation.y = y
                    pose.orientation.z = z
                    pose.orientation.w = w

                    pose_status.in_frame = True
                    pose_status.position = pose


                else:
                    self.logger.error("Could not solve PnP for detected tag.")
                    pose_status.in_frame = False # partial tag detected?
            else:
                pose_status.in_frame = False # no tags detected


            self.vis_pub.publish(pose_status)

        except Exception as e:
            self.logger.error(f"Error converting image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DetectBlock("detect_block")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()