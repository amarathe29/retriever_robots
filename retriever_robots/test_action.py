import rclpy
from rclpy.node import Node

from cc_interfaces.action import RetrievalTask
from cc_interfaces.msg import Block
from geometry_msgs.msg import PoseStamped, PolygonStamped, Point32, Quaternion
from rclpy.action import ActionClient

from retriever_robots.utils import quaternion_from_euler

import numpy as np


class RetrieverActionTestNode(Node):
    def __init__(self):
        super().__init__("retriever_action_test_node")
        self.retrieve_action_client = ActionClient(
            self, RetrievalTask, "asher/retrieve_block"
        )

    def send_retrieval_goal(self, position, stockpile, type=0, quat=None):
        x, y, _ = position
        if quat is None:
            quat = Quaternion()

        goal_msg = RetrievalTask.Goal()
        goal_msg.block = Block()
        goal_msg.block.type = int(type)
        goal_msg.block.pose.pose.position.x = x
        goal_msg.block.pose.pose.position.y = y
        goal_msg.block.pose.pose.position.z = 0.0
        goal_msg.block.pose.pose.orientation = quat
        goal_msg.block.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.block.pose.header.frame_id = "world"

        stockpile_msg = PolygonStamped()
        stockpile_msg.polygon.points = [
            Point32(x=stockpile[0][0], y=stockpile[0][1], z=0.0),
            Point32(x=stockpile[1][0], y=stockpile[1][1], z=0.0),
            Point32(x=stockpile[2][0], y=stockpile[2][1], z=0.0),
            Point32(x=stockpile[3][0], y=stockpile[3][1], z=0.0),
        ]

        stockpile_msg.header = goal_msg.block.pose.header

        goal_msg.stockpile = stockpile_msg

        self.retrieve_action_client.wait_for_server()
        goal = self.retrieve_action_client.send_goal_async(goal_msg)
        goal.add_done_callback(self.handle_result)

    def handle_result(self, future):

        self.goal_handle = future.result()

        if not self.goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.result_handle = self.goal_handle.get_result_async()
        self.result_handle.add_done_callback(self.process_result)

    def process_result(self, future):
        result = future.result().result

        if result.success:
            self.get_logger().info(f"Action returned success")
        else:
            self.get_logger().info(f"Action Failed")


def main():
    rclpy.init()
    point = [1.5, 3.0, 0.0]
    stockpile = np.array([[0.9, 1.25]]*4)
    node = RetrieverActionTestNode()
    quat = quaternion_from_euler(yaw=0, units="degrees")
    print(f"Going to point {point}")
    node.send_retrieval_goal(point, stockpile, 1, quat)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
