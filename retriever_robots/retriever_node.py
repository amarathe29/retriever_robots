import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, Pose, PoseStamped, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from cc_interfaces.action import RetrievalTask  # type: ignore
from cc_interfaces.msg import Block  # type: ignore
from retriever_msgs.msg import PoseStatus  # type: ignore

from retriever_robots.utils import angle_wrap, euler_from_quaternion, reverse_yaw_quaternion, quaternion_from_euler, get_robot_barrier_func

import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped

import numpy as np
from enum import Enum, auto
from copy import deepcopy


BLOCK_OFFSET = 0.4 #m
REVERSE_DISTANCE = 0.4 #m
STOCK_CLEARANCE = 1 # m clearance
class State(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    GRABBING = auto()
    STOCKPILE_PREP = auto()
    STOCKPILE_DEPOSIT = auto()
    STOCKPILE_EXIT = auto()
    STOCKPILE_DEPART = auto()
    RETURNING = auto()
    RECOVERY = auto()


class RetrieveNode(Node):
    """RetrieveNode runs the state machine for each of the retriever robots."""

    def __init__(self, node_name, *args):
        super(RetrieveNode, self).__init__(node_name)
        # Set up subscribers
        self.odom_sub = self.create_subscription(
            Odometry, f"{self.get_namespace()}/odom", self.odom_callback, 10
        )

        # our own ad hoc topic for the locations of the blocks we can see, in the robot's frame

        self.visible_block_sub = self.create_subscription(
            PoseStatus,
            f"{self.get_namespace()}/visible_block",
            self.visible_block_callback,
            10,
        )

        self.frame_ref_sub = self.create_subscription(
            Bool,
            f"{self.get_namespace()}/world_conversion_active",
            self.update_odom_state,
            10,
        )

        # Set up publisher
        self.vel_pub = self.create_publisher(
            Twist, f"{self.get_namespace()}/cmd_vel", 10
        )


        self.vis_pub = self.create_publisher(
            MarkerArray, f"{self.get_namespace()}/markers", 10
        )

        # Set up action server
        self.retrieve_action = ActionServer(
            self,
            RetrievalTask,
            f"{self.get_namespace()}/retrieve_block",
            self.retrieve_callback,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Save the logger in ros1 style syntax, because wtf is with the self.get_logger() BS
        self.logger = self.get_logger()
        self.logger.info(f"Launched Retrieve Node for {self.get_namespace()}")

        self.marker_size = 0.0544

        self.state = State.IDLE
        self.return_state = State.NAVIGATING
        self.pose = None
        self.odom = None
        self.start_pose_set = False
        self._start_pose = None

        self.block_type = None
        self.nav_pose = None
        self.observed_block_pose = None
        self.stockpile = None
        self.stockpile_safe = None
        self.recovery_pose = None

        self.missing_tag_count = 0
        self.tag_visible = False
        self.valid_tf_tree = False
        self.barrier_func = get_robot_barrier_func(boundary_points=[-3.0,3.0,-5.0,5.0])

    def odom_callback(self, msg: Odometry) -> None:
        self.logger.debug(f"Received Odometry: {msg}")
        self.odom = msg
        if self._start_pose is None:
            self._start_pose = PoseStamped()
            self._start_pose.header.frame_id = self._namespaced_frame("odom")
            self._start_pose.pose = self.odom.pose.pose
            self.start_pose_set = True

    def update_odom_state(self, valid_msg: Bool):
        self.valid_tf_tree = valid_msg.data

    def visible_block_callback(self, msg: Pose) -> None:

        self.update_visualization()

        if msg.block_in_frame and not msg.tag_in_frame:
            if self.state in [State.RECOVERY]:
                self.logger.info(
                    f"[VisibleCallback] Block visible but tag not visible, setting recovery pose to ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})",
                    throttle_duration_sec=1.0,
                )
                self.recovery_pose = msg.pose # currently unused
            return

        if not msg.tag_in_frame:
            self.missing_tag_count += 1
            if self.missing_tag_count > 10:
                # acts as some hysteresis for losing the block at 30 fps
                if self.state in [
                    State.NAVIGATING,
                    State.GRABBING,
                    State.STOCKPILE_PREP,
                ]:
                    self.tag_visible = False
                    self.logger.warning("No tag visible", throttle_duration_sec=5.0)
            return

        if self.block_type is not None and msg.type != self.block_type:
            # TODO: If we want a mislabelled block, we'd have to remove this
            self.logger.warning("Incorrect Block Size", throttle_duration_sec=5.0)
            return

        self.tag_visible = True
        self.missing_tag_count = 0
        self.observed_block_pose = msg.pose


    def print_pose_euler(self, pose: Pose) -> str:
        roll, pitch, yaw = euler_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        return f"POS: ({pose.position.x:.2f}, {pose.position.y:.2f}), EULER: (Roll: {np.degrees(roll):.2f}, Pitch: {np.degrees(pitch):.2f}, Yaw: {np.degrees(yaw):.2f})"


    def calculate_nav_pose(self, message, stock_pt_safe):
        # given a message that contains both the block pose and the stockpile pose
        block_pose = message.block.pose.pose
        
        block_pt = (block_pose.position.x, block_pose.position.y)

        travel_angle = np.math.atan2(block_pt[1] - stock_pt_safe[1], block_pt[0]-stock_pt_safe[0])

        _,_, block_angle = euler_from_quaternion(
            block_pose.orientation.x,
            block_pose.orientation.y,
            block_pose.orientation.z,
            block_pose.orientation.w,
            use_extrinsics=True,
        )

        # this is how much angle the robot needs to rotate through in order to bring the block to the stockpile (directly)
        turn_magnitude = np.abs(travel_angle - block_angle)

        pose = PoseStamped()
        pose.header = message.block.pose.header
        pose.pose.position = block_pose.position

        if turn_magnitude < np.pi/2:
            # position ourselves along the pos y-axis, facing the start
            pose.pose.position.x += BLOCK_OFFSET*np.cos(block_angle)
            pose.pose.position.y += BLOCK_OFFSET*np.sin(block_angle)
            pose.pose.orientation = reverse_yaw_quaternion(block_pose.orientation)

        else:
            pose.pose.position.x -= BLOCK_OFFSET*np.cos(block_angle)
            pose.pose.position.y -= BLOCK_OFFSET*np.sin(block_angle)
            pose.pose.orientation = block_pose.orientation


        return pose

    def save_block_pose(self):
        block_loc = deepcopy(self.observed_block_pose)
        transform = self.tf_buffer.lookup_transform(self._namespaced_frame("odom"), block_loc.header.frame_id, rclpy.time.Time())
        res = do_transform_pose_stamped(block_loc, transform)
        res.pose.orientation = self.curr_pose.pose.orientation
        return res

    def calculate_exit_pose(self):
        transform = self.tf_buffer.lookup_transform(self._namespaced_frame("odom"), self._namespaced_frame("base_link"), rclpy.time.Time())
        pos = PoseStamped()
        pos.header.frame_id = self._namespaced_frame("base_link")
        pos.header.stamp = self.get_clock().now().to_msg()
        pos.pose.position.x = -REVERSE_DISTANCE
        return do_transform_pose_stamped(pos, transform)

    def retrieve_callback(self, goal_handle) -> RetrievalTask.Result:
        """action handler for the retrieve action server"""

        self.logger.info(f"Received retrieve action goal: {goal_handle.request}")
        if self.state != State.IDLE:
            self.logger.error(f"Already running an action")
            goal_handle.abort()
            result = RetrievalTask.Result()
            result.success = False
            return


        goal_reached = False

        while not goal_reached:

            self.logger.info(
                f"Current state: {self.state.name}", throttle_duration_sec=1.0
            )

            if self.state == State.IDLE:
                stockpile_points = goal_handle.request.stockpile.polygon.points
                stock_pt = np.mean([(pt.x, pt.y) for pt in stockpile_points], axis=0)
                stock_pt_safe = (stock_pt[0] + STOCK_CLEARANCE, stock_pt[1])
                self.stockpile = PoseStamped()
                self.stockpile.header = goal_handle.request.stockpile.header
                self.stockpile.pose.position.x = stock_pt[0]
                self.stockpile.pose.position.y = stock_pt[1]
                self.stockpile.pose.orientation = quaternion_from_euler(yaw = 180, units="degrees")

                self.stockpile_safe = deepcopy(self.stockpile)
                self.stockpile_safe.pose.position.x = stock_pt_safe[0]
                self.stockpile_safe.pose.position.y = stock_pt_safe[1]

                self.nav_pose = self.calculate_nav_pose(goal_handle.request, stock_pt_safe)
                self.observed_block_pose = None
                self.block_type = goal_handle.request.block.type

                self.logger.info(
                    f"Received retrieve action goal: {self.nav_pose}, entering NAVIGATING state"
                )
                self.state = State.NAVIGATING

            elif self.state == State.NAVIGATING:
                reached = self.go_to_pose(self.nav_pose)
                if self.observed_block_pose is not None:
                    self.logger.info(
                        f"[{self.state.name}]Found block, entering GRABBING state to acquire block"
                    )
                    self.grab_pose = self.save_block_pose()

                    self.state = State.GRABBING
                elif reached and not self.tag_visible:
                    self.logger.info(
                        f"[{self.state.name}]Reached target location but no block found, entering RECOVERY state to attempt recovery"
                    )
                    self.return_state = State.GRABBING
                    self.state = State.RECOVERY
            elif self.state == State.GRABBING:
                # super naive, we should check for block in position here
                # I think this is where we take down our barriers on the specific block
                reached = self.go_to_pose(self.grab_pose, controller=self.pose_controller)
                self.logger.info(
                    f"Going to block located at {self.grab_pose.pose}",
                    throttle_duration_sec=1.0,
                )
                if reached:
                    self.grab_pose = None
                    self.logger.info(
                        f"[{self.state.name}]Grabbed block, bringing block to Safe Stockpile Point"
                    )
                    self.state = State.STOCKPILE_PREP

            elif self.state == State.STOCKPILE_PREP:
                # move to the safe stockpile location
                reached = self.go_to_pose(self.stockpile_safe)
                if self.tag_visible and reached:
                    self.logger.info(
                        f"[{self.state.name}]Reached Safe Stockpile Point, attempting DEPOSIT"
                    )
                    self.state = State.STOCKPILE_DEPOSIT
                elif not self.tag_visible:
                    self.logger.info(
                        f"[{self.state.name}]LOST the BLOCK, entering RECOVERY state to attempt recovery"
                    )
                    self.return_state = State.GRABBING
                    self.state = State.RECOVERY

            elif self.state == State.STOCKPILE_DEPOSIT:
                # move to the specific stockpile location
                reached = self.go_to_pose(self.stockpile)
                if self.tag_visible and reached:
                    self.logger.info(
                        f"[{self.state.name}]Stockpiled block, attempting to EXIT the STOCKPILE"
                    )
                    self.exit_pose = self.calculate_exit_pose()
                    self.state = State.STOCKPILE_EXIT
                elif not self.tag_visible:
                    self.logger.info(
                        f"[{self.state.name}]LOST the BLOCK, entering RECOVERY state to attempt recovery"
                    )
                    self.return_state = State.GRABBING
                    self.state = State.RECOVERY

            elif self.state == State.STOCKPILE_EXIT:
                back_up = self.go_to_pose(self.exit_pose, self.pose_controller_reverse)
                if back_up:
                    self.state = State.STOCKPILE_DEPART
                    
            elif self.state == State.STOCKPILE_DEPART:
                self.stockpile_safe.pose.orientation = reverse_yaw_quaternion(self.stockpile_safe.pose.orientation)
                reached = self.go_to_pose(self.stockpile_safe)
                if reached:
                        self.logger.info(
                            f"[{self.state.name}]Exited the stockpile successfully, returning to base"
                        )
                        self.state = State.RETURNING

            # This state is just if we want the robot to return to the starting position
            elif self.state == State.RETURNING:
                if self._start_pose is not None:
                    goal_reached = self.go_to_pose(self._start_pose, controller=self.pose_controller_clf)
                    if goal_reached:
                        self.logger.info(
                            f"[{self.state.name}]Returned to start, finishing action and returning to IDLE state"
                        )
                        goal_handle.succeed()
                        result = RetrievalTask.Result()
                        result.success = True
                        result.delivered = Block()
                        result.delivered.pose = self.observed_block_pose
                        self.state = State.IDLE
                        return result

            elif self.state == State.RECOVERY:
                cmd = Twist()

                if not self.tag_visible:
                    self.logger.warning(
                        "No recovery pose available, spinning robot",
                        throttle_duration_sec=5.0,
                    )
                    cmd.angular.z = 0.2
                    self.vel_pub.publish(cmd)
                    continue
                self.grab_pose = self.save_block_pose()
                self.state = self.return_state


        result = RetrievalTask.Result()
        result.success = False
        result.delivered = Block()
        result.delivered.pose = self.observed_block_pose if self.observed_block_pose is not None else PoseStamped()
        return result

    def pose_controller(self, target_pose: Pose) -> Twist:

        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        self.logger.info(
            f"Using simple pose controller to go to {target_pose}, from {self.curr_pose}",
            throttle_duration_sec=2.0,
        )

        dx = target_pose.position.x - self.curr_pose.pose.position.x
        dy = target_pose.position.y - self.curr_pose.pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q_curr = self.curr_pose.pose.orientation
        _, _, current_yaw = euler_from_quaternion(
            q_curr.x, q_curr.y, q_curr.z, q_curr.w, use_extrinsics=True
        )
        angle_diff = angle_wrap(angle_to_target - current_yaw)
        q_desired = target_pose.orientation
        _, _, desired_yaw = euler_from_quaternion(
            q_desired.x, q_desired.y, q_desired.z, q_desired.w, use_extrinsics=True
        )
        desired_yaw_diff = angle_wrap(desired_yaw - current_yaw)

        cmd = Twist()
        linear_gain = 0.3  # Arbitrary gain cause why not
        angular_gain = 0.7

        self.logger.info(
            f"Controller distance remaining: {distance}, angle difference: {angle_diff}, diff to desired_yaw: {desired_yaw_diff}",
            throttle_duration_sec=1.0,
        )
        if abs(angle_diff) > 0.04 and distance > 0.15:
            sgn = np.sign(angle_diff)
            cmd.angular.z = sgn * max(min(angular_gain * abs(angle_diff), 0.2), 0.05)
        else:
            if distance > 0.07:
                sgn = np.sign(distance)
                cmd.linear.x = sgn * max(min(linear_gain * distance, 0.2), 0.05)
            else:
                if abs(desired_yaw_diff) > 0.04:
                    sgn = np.sign(desired_yaw_diff)
                    cmd.angular.z = sgn * max(
                        min(angular_gain * abs(desired_yaw_diff), 0.2), 0.05
                    )
                else:
                    return Twist()

        return cmd

    # TODO: Implement the robot moving backwards better than this
    def pose_controller_reverse(self, target_pose: Pose) -> Twist:
        
        dx = target_pose.position.x - self.curr_pose.pose.position.x
        dy = target_pose.position.y - self.curr_pose.pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q_curr = self.curr_pose.pose.orientation
        _, _, current_yaw = euler_from_quaternion(
            q_curr.x, q_curr.y, q_curr.z, q_curr.w, use_extrinsics=True
        )
        angle_diff = angle_wrap(angle_to_target - (current_yaw + np.pi))
        q_desired = target_pose.orientation
        _, _, desired_yaw = euler_from_quaternion(
            q_desired.x, q_desired.y, q_desired.z, q_desired.w, use_extrinsics=True
        )
        desired_yaw_diff = angle_wrap(desired_yaw - current_yaw)

        cmd = Twist()
        linear_gain = 0.3  # Arbitrary gain cause why not
        angular_gain = 0.7

        self.logger.info(
            f"Controller distance remaining: {distance}, angle difference: {angle_diff}, diff to desired_yaw: {desired_yaw_diff}",
            throttle_duration_sec=1.0,
        )
        if abs(angle_diff) > 0.04 and distance > 0.15:
            sgn = np.sign(angle_diff)
            cmd.angular.z = sgn * max(min(angular_gain * abs(angle_diff), 0.2), 0.05)
        else:
            if distance > 0.07:
                sgn = -np.sign(distance)
                cmd.linear.x = sgn * max(min(linear_gain * abs(distance), 0.2), 0.05)
            else:
                if abs(desired_yaw_diff) > 0.04:
                    sgn = np.sign(desired_yaw_diff)
                    cmd.angular.z = sgn * max(
                        min(angular_gain * abs(desired_yaw_diff), 0.2), 0.05
                    )
                else:
                    return Twist()

        return cmd

    def pose_controller_clf_constrained(self, target_pose: Pose) -> Twist:
        return self.pose_controller_clf(target_pose, forward_constraint=True)

    # TODO: Implement a controller to drive the robot in smooth arcs instead of lines using control lyapunov functions or splines
    # gamma is approach angle gain, k is desired angle gain, and h is rotation error gain
    def pose_controller_clf(
        self, target_pose: Pose, gamma=0.7, k=1.0, h=0.7, forward_constraint=False
    ) -> Twist:
        assert gamma > 0, f"gamma = {gamma} must be greater than 0"
        assert k > gamma, f"k = {k} must be greater than gamma = {gamma}"
        assert h > 0, f"h = {h} must be greater than 0"

        self.logger.info(
            f"Using CLF controller to go to {target_pose}, from {self.curr_pose.pose}",
            throttle_duration_sec=1.0,
        )
        q_curr = self.curr_pose.pose.orientation
        _, _, theta = euler_from_quaternion(q_curr.x, q_curr.y, q_curr.z, q_curr.w)
        q_desired = target_pose.orientation
        _, _, desired_theta = euler_from_quaternion(
            q_desired.x, q_desired.y, q_desired.z, q_desired.w
        )

        dx = target_pose.position.x - self.curr_pose.pose.position.x
        dy = target_pose.position.y - self.curr_pose.pose.position.y

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
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        self.logger.info(
            f"Current angle: {theta}, desired: {desired_theta}",
            throttle_duration_sec=1.0,
        )
        self.logger.info(
            f"Controller distance remaining: {e}, angle difference: {alpha}",
            throttle_duration_sec=1.0,
        )
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        v = gamma * e * ca
        if forward_constraint:
            v = max(0.0, v)
        # Prevent divide by zero errors
        sinc_alpha = 1.0 if np.abs(alpha) < 1e-6 else (sa / alpha)
        w = k * alpha + gamma * (ca * sinc_alpha) * (alpha + h * theta_error_vec)

        if e < 0.025:
            self.logger.info("Position error low", throttle_duration_sec=1.0)
        if alpha < 0.05:
            self.logger.info("Angular error low", throttle_duration_sec=1.0)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        robo_pose = np.array([[self.curr_pose.pose.position.x],[self.curr_pose.pose.position.y],[self.curr_pose.pose.position.z]])
        safe_cmd = self.barrier_func(cmd, robo_pose, neighbor_positions=np.array([[0.9], [0.0]]), block_positions=None)
        return safe_cmd

    def check_reached_target(self, target_pose: Pose) -> bool:
        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        dx = target_pose.position.x - self.curr_pose.pose.position.x
        dy = target_pose.position.y - self.curr_pose.pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        q_curr = self.curr_pose.pose.orientation
        _, _, current_yaw = euler_from_quaternion(
            q_curr.x, q_curr.y, q_curr.z, q_curr.w
        )
        q_target = target_pose.orientation
        _, _, desired_yaw = euler_from_quaternion(
            q_target.x, q_target.y, q_target.z, q_target.w
        )
        angle_diff = angle_wrap(desired_yaw - current_yaw)

        self.logger.info(
            f"Distance remaining: {distance}, Angle difference: {angle_diff}",
            throttle_duration_sec=1.0,
        )

        # If we haven't reached the target, return false, else we return true
        if abs(angle_diff) >= 0.08 or distance >= 0.1:
            return False

        return True

    def go_to_pose(self, target_pose: PoseStamped, controller=None) -> bool:

        assert target_pose.header.frame_id == self._namespaced_frame("odom")

        if controller is None:
            controller = self.pose_controller_clf

        cmd = controller(target_pose.pose)
        self.vel_pub.publish(cmd)
        reached = self.check_reached_target(target_pose.pose)
        return reached

    def brake(self):
        self.vel_pub.publish(Twist())

    # TODO: use the world frame and published aruco tags to update the robots position
    @property
    def curr_pose(self):
        if hasattr(self, "pose") and self.pose is not None:
            self.logger.debug("Using pose topic")
            return self.pose
        elif hasattr(self, "odom") and self.odom is not None:
            self.logger.debug("Using odometry topic")
            pose = PoseStamped()
            pose.header.frame_id = self._namespaced_frame("odom")
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose = self.odom.pose.pose
            return pose
        else:
            self.logger.warning("No pose information available")
            return None

    def _namespaced_frame(self, frame_name):
        ns = self.get_namespace().strip("/")
        return f"{ns}/{frame_name}" if ns else frame_name

    def update_visualization(self):
        msg = MarkerArray()
        marker_data = {
            "curr_pose" : (self.curr_pose, (1.0,1.0,0.0)),
            "observed_block" : (self.observed_block_pose, (1.0, 0.0, 0.0)),
            "nav_pose" : (self.nav_pose, (0.0,1.0,1.0)),
            "stockpile_pose" : (self.stockpile, (0.0,1.0,0.0)),
            "stockpile_safe_pose" : (self.stockpile_safe, (0.0,0.0,1.0)),
        }

        for i, nary in enumerate(marker_data.items()):
            label, data = nary
            if data[0] is None:
                continue
            m = Marker()
            m.header = data[0].header
            m.ns = self.get_namespace()
            m.id = i
            m.text = label
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.5
            m.scale.y = 0.05
            m.scale.z = 0.05
            m.color.r = data[1][0]
            m.color.g = data[1][1]
            m.color.b = data[1][2]
            m.color.a = 1.0
            m.pose = data[0].pose
            msg.markers.append(m)


        self.vis_pub.publish(msg)


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
