import rclpy
from rclpy.node import Node

import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from std_msgs.msg import Bool

from retriever_robots.utils import do_transform_transform


RETRIEVER_DICT = {
    "/asher": "aruco_20",
    "asher": "aruco_20",
    "/mika": "aruco_21",
    "mika": "aruco_21",
}


class WorldConversionNode(Node):
    def __init__(self):
        super().__init__("world_conversion_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.valid = False


        self.pub = self.create_publisher(Bool, f"{self.get_namespace()}/world_conversion_active", 10)

    def _namespaced_frame(self, frame_name):
        ns = self.get_namespace().strip("/")
        return f"{ns}/{frame_name}" if ns else frame_name

    def timer_callback(self):
        try:
            # Listen to the transform from world to aruco tag
            world_to_aruco = self.tf_buffer.lookup_transform("world", RETRIEVER_DICT.get(self.get_namespace(), "aruco_20"), rclpy.time.Time())
            # Listen to the transform from odom to base_link
            base_link_to_odom = self.tf_buffer.lookup_transform(self._namespaced_frame("base_link"), self._namespaced_frame("odom"), rclpy.time.Time())

            aruco_to_base_link = TransformStamped()
            aruco_to_base_link.header.stamp = self.get_clock().now().to_msg()
            aruco_to_base_link.header.frame_id = RETRIEVER_DICT.get(self.get_namespace(), "aruco_20")
            aruco_to_base_link.child_frame_id = f"{self.get_namespace()}/base_link"
            aruco_to_base_link.transform.translation.x = 0.14
            aruco_to_base_link.transform.translation.y = 0.0
            aruco_to_base_link.transform.translation.z = -0.24
            aruco_to_base_link.transform.rotation.x = 0.0
            aruco_to_base_link.transform.rotation.y = 0.0
            aruco_to_base_link.transform.rotation.z = 0.0
            aruco_to_base_link.transform.rotation.w = 1.0

            world_to_base_link = do_transform_transform(world_to_aruco, aruco_to_base_link)
            world_to_odom = do_transform_transform(world_to_base_link, base_link_to_odom)

            self.valid = True


            # Broadcast the new transform from world to odom
            self.tf_broadcaster.sendTransform(world_to_odom)

        except Exception as e:
            self.get_logger().debug(f"Could not find transform: {e}")
            self.valid = False

        self.pub.publish(Bool(data=self.valid))


def main(args=None):
    rclpy.init(args=args)
    node = WorldConversionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()