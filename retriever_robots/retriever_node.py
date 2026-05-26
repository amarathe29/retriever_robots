import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from retriever_msgs.action import GoToBlock  # type: ignore
from retriever_msgs.msg import PoseStatus  # type: ignore

from retriever_robots.utils import euler_from_quaternion, angle_wrap

import numpy as np
from enum import Enum, auto

class StateMachine(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    FIND_GRAB_POSE = auto()
    POSITIONING = auto()
    GRABBING = auto()
    STOCKPILING = auto()
    RETURNING = auto()
    RECOVERY = auto()


class RetrieveNode(Node):
    """RetrieveNode runs the state machine for each of the retriever robots."""

    def __init__(self, node_name, *args):
        super(RetrieveNode, self).__init__(node_name)
        # Set up subscribers
        self.bev_pose_sub = self.create_subscription(
            Pose, f"{self.get_namespace()}/pose", self.pose_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, f"{self.get_namespace()}/odom", self.odom_callback, 10
        )

        # our own ad hoc topic for the locations of the blocks we can see, in the robot's frame
        self.visible_block_sub = self.create_subscription(
            PoseStatus, f"{self.get_namespace()}/visible_block", self.visible_block_callback, 10
        )

        # Set up publisher
        self.vel_pub = self.create_publisher(
            Twist, f"{self.get_namespace()}/cmd_vel", 10
        )

        # Set up action server
        self.retrieve_action = ActionServer(
            self,
            GoToBlock,
            f"{self.get_namespace()}/gotoblock",
            self.retrieve_callback,
        )

        # Save the logger in ros1 style syntax, because wtf is with the self.get_logger() BS
        self.logger = self.get_logger()
        self.logger.info(f"Launched Retrieve Node for {self.get_namespace()}")

        self.marker_size = 0.0544

        self.state = StateMachine.IDLE
        self.pose = None
        self.odom = None
        self.start_pose_set = False
        self._start_pose = None
        self.request_pose = None
        self.grab_pose = None
        self.visible_count = 0

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
            self.start_pose_set = True

    def visible_block_callback(self, msg: Pose) -> None:

        if not msg.in_frame:
            self.visible_count += 1
            if self.visible_count > 10:
                # acts as some hysteresis for losing the block at 30 fps
                if self.state in [StateMachine.GRABBING, StateMachine.STOCKPILING]:
                    self.logger.warning("No block visible")
            return
        self.visible_count = 0
        self.block_pose = msg.pose

        if self.state in [StateMachine.NAVIGATING, StateMachine.FIND_GRAB_POSE, StateMachine.RECOVERY]:
            
            # TODO: cool math here to go from position of block in robot frame to robots position for optimal grasp
            # should be colinear the orientation of the block. 
            _,_,yaw = euler_from_quaternion(msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)

            positioning_distance = 0.3
            self.grab_pose = Pose()
            self.grab_pose.position.x = self.block_pose.position.x - positioning_distance * np.cos(yaw)
            self.grab_pose.position.y = self.block_pose.position.y - positioning_distance * np.sin(yaw)
            self.grab_pose.position.z = self.block_pose.position.z
            self.grab_pose.orientation = self.block_pose.orientation
            self.logger.info(f"found block at {self.block_pose}, setting grab pose to {self.grab_pose}")

        # TODO: We need to use the location of the block to update our state machine:

        # if we're grabbing, we can just check the position of the block within an envelope

        # if we're stockpiling, we want to catch if we lose sight of the block...we may need a custom message here with a bool...


    def retrieve_callback(self, goal_handle) -> GoToBlock.Result:
        """action handler for the retrieve action server"""

        self.logger.info(f"Received retrieve action goal: {goal_handle.request}")
        if self.state != StateMachine.IDLE:
            self.logger.error(f"Already running an action")
            goal_handle.fail()
            result = GoToBlock.Result()
            result.success = True
            result.end_pose = self.curr_pose
            return

        feedback_msg = GoToBlock.Feedback()
        feedback_msg.block_captured = False

        goal_reached = False

        while not goal_reached:

            if self.state == StateMachine.IDLE:
                self.request_pose = goal_handle.request.goal_pose
                self.grab_pose = None
                self.marker_location = None
                self.logger.info(
                    f"Received retrieve action goal: {self.request_pose}, entering NAVIGATING state"
                )
                self.state = StateMachine.NAVIGATING

            elif self.state == StateMachine.NAVIGATING:
                reached = self.go_to_pose(self.request_pose)
                if reached or self.grab_pose is not None:
                    self.logger.info(
                        f"Reached block pose, entering FIND_GRAB_POSE state to locate block"
                    )
                    self.state = StateMachine.FIND_GRAB_POSE
            elif self.state == StateMachine.FIND_GRAB_POSE:
                if self.grab_pose is not None:
                    self.logger.info(
                        f"Found block pose, entering POSITIONING state to orient around block"
                    )
                    self.state = StateMachine.POSITIONING
                else:
                    # TODO: do a cool spiral search outward
                    self.state = StateMachine.RECOVERY
                    # self.logger.info(
                    #     f"Block not found, searching nearby"
                    # )
            elif self.state == StateMachine.POSITIONING:
                reached = self.go_to_pose(self.grab_pose)
                if reached:
                    self.logger.info(
                        f"Reached grab pose, entering GRABBING state to grab block"
                    )
                    self.state = StateMachine.GRABBING

            elif self.state == StateMachine.GRABBING:
                # super naive, I'd rather put an bound on block position here
                reached = self.go_to_pose(self.block_pose)
                if reached:
                    self.logger.info(
                        f"Grabbed block, entering STOCKPILING state to stockpile block"
                    )
                    self.state = StateMachine.STOCKPILING

            elif self.state == StateMachine.STOCKPILING:
                # TODO: This should be changed to be the output of some function from the camera also add functionality
                feedback_msg.block_captured = True
                reached = True
                if feedback_msg.block_captured and reached:
                    self.logger.info(
                        f"Stockpiled block, entering RETURNING state to return to start"
                    )
                    self.state = StateMachine.RETURNING
                elif not feedback_msg.block_captured:
                    self.logger.info(
                        f"Failed to capture block, entering RECOVERY state to attempt recovery"
                    )
                    self.state = StateMachine.RECOVERY
                pass

            # This state is just if we want the robot to return to the starting position
            elif self.state == StateMachine.RETURNING:
                if self._start_pose is not None:
                    goal_reached = self.go_to_pose(self._start_pose)
                    if goal_reached:
                        goal_handle.succeed()
                        result = GoToBlock.Result()
                        result.success = True
                        result.end_pose = self.curr_pose
                        self.state = StateMachine.IDLE
                        return result

            elif self.state == StateMachine.RECOVERY:
                # TODO: Add some kind of recovery behavior
                if self.grab_pose is not None:
                    self.logger.info(
                        f"Attempting to recover by going to grab pose {self.grab_pose}"
                    )
                    reached = self.go_to_pose(self.grab_pose)
                    if reached:
                        self.logger.info(
                            f"Reached grab pose, entering POSITIONING state to orient around block"
                        )
                        self.state = StateMachine.POSITIONING
                pass

            feedback_msg.curr_pose = self.curr_pose
            goal_handle.publish_feedback(feedback_msg)

        result = GoToBlock.Result()
        result.success = False
        result.end_pose = self.curr_pose
        # add gracefull returning to idle and the idle position here
        return result


    def pose_controller(self, target_pose: Pose) -> Twist:

        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        self.logger.info(f"Using simple pose controller to go to {target_pose}")

        dx = target_pose.position.x - self.curr_pose.position.x
        dy = target_pose.position.y - self.curr_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q_curr = self.curr_pose.orientation
        _, _, current_yaw = euler_from_quaternion(
            q_curr.x, q_curr.y, q_curr.z, q_curr.w
        )
        angle_diff = angle_wrap(angle_to_target - current_yaw)
        q_desired = target_pose.orientation
        _, _, desired_yaw = euler_from_quaternion(
            q_desired.x, q_desired.y, q_desired.z, q_desired.w
        )
        desired_yaw_diff = angle_wrap(desired_yaw - current_yaw)

        cmd = Twist()
        linear_gain = 0.3  # Arbitrary gain cause why not
        angular_gain = 0.7

        # TODO: Double-check this works with the new changes
        if abs(angle_diff) > 0.1 and distance > 0.15:
            cmd.angular.z = angular_gain * angle_diff
        else:
            if distance > 0.1:
                cmd.linear.x = linear_gain * distance
            else:
                if abs(desired_yaw_diff) > 0.1:
                    cmd.angular.z = angular_gain * desired_yaw_diff
                else:
                    return Twist()

        return cmd

    # TODO: Implement the robot moving backwards
    def pose_controller_reverse(self, target_pose: Pose) -> Twist:
        pass

    # TODO: Implement a controller to drive the robot in smooth arcs instead of lines using control lyapunov functions or splines
    # gamma is approach angle gain, k is desired angle gain, and h is rotation error gain
    def pose_controller_clf(self, target_pose: Pose, gamma=1.0, k=3.0, h=0.5) -> Twist:
        assert gamma > 0, f"gamma = {gamma} must be greater than 0"
        assert k > gamma, f"k = {k} must be greater than gamma = {gamma}"
        assert h > 0, f"h = {h} must be greater than 0"

        self.logger.info(f"Using CLF controller to go to {target_pose}")
        q_curr = self.curr_pose.orientation
        _, _, theta = euler_from_quaternion(q_curr.x, q_curr.y, q_curr.z, q_curr.w)
        q_desired = target_pose.orientation
        _, _, desired_theta = euler_from_quaternion(
            q_desired.x, q_desired.y, q_desired.z, q_desired.w
        )

        dx = target_pose.position.x - self.curr_pose.position.x
        dy = target_pose.position.y - self.curr_pose.position.y

        # We're going to do things in the frame of reference of the goal position, because apparently that makes behavior more consistent (shoutout to my old advisor for this controller)
        R = np.array(
            [
                [np.cos(-desired_theta), -np.sin(-desired_theta)],
                [np.sin(-desired_theta), np.cos(-desired_theta)],
            ]
        )

        # positional error from the goal's frame of reference
        position_error = R @ np.array([dx, dy])
        e = np.linalg.norm(position_error)

        # Angle to goal from goal frame
        theta_error_vec = np.arctan2(position_error[1], position_error[0])

        alpha = theta_error_vec - (theta - desired_theta)
        alpha = angle_wrap(alpha)

        ca = np.cos(alpha)
        sa = np.sin(alpha)

        v = gamma * e * ca

        # Prevent divide by zero errors
        sinc_alpha = 1.0 if np.abs(alpha) < 1e-6 else (sa / alpha)
        w = k * alpha * gamma * ca * sinc_alpha * (alpha + h + theta_error_vec)
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        return cmd

    def check_reached_target(self, target_pose: Pose) -> bool:
        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        dx = target_pose.position.x - self.curr_pose.position.x
        dy = target_pose.position.y - self.curr_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q_curr = self.curr_pose.orientation
        _, _, current_yaw = euler_from_quaternion(
            q_curr.x, q_curr.y, q_curr.z, q_curr.w
        )
        angle_diff = angle_wrap(angle_to_target - current_yaw)

        # If we haven't reached the target, return false, else we return true
        if abs(angle_diff) > 0.04 or distance > 0.1:
            return False

        return True

    def go_to_pose(self, target_pose: Pose, controller=None) -> bool:
        # for now, just pretend we went there
        # self.logger.error(f"Gone to pose: {target_pose}")
        # return True

        if controller is None:
            controller = self.pose_controller

        cmd = controller(self.request_pose)
        self.vel_pub.publish(cmd)
        reached = self.check_reached_target(self.request_pose)
        return reached

    @property
    def curr_pose(self):
        if hasattr(self, "pose") and self.pose is not None:
            self.logger.debug("Using pose topic")
            return self.pose
        elif hasattr(self, "odom") and self.odom is not None:
            self.logger.debug("Using odometry topic")
            return self.odom.pose.pose
        else:
            self.logger.warning("No pose information available")
            return None


def main():
    rclpy.init()
    node = RetrieveNode("retriever_node")
    # I'll try spinning, that's a good trick!
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
