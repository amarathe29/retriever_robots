from collections.abc import Callable
from copy import deepcopy
from enum import Enum, auto

import numpy as np
import rclpy
import tf2_ros
from cc_interfaces.action import RetrievalTask  # type: ignore
from cc_interfaces.msg import Block  # type: ignore
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Quaternion,
    Twist,
    PointStamped,
    PolygonStamped,
)
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from retriever_msgs.msg import PoseStatus  # type: ignore
from std_msgs.msg import Bool, Header
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_point
from visualization_msgs.msg import Marker, MarkerArray

from retriever_robots.utils import (
    angle_wrap,
    euler_from_quaternion,
    get_robot_barrier_func,
    quaternion_from_euler,
    reverse_yaw_quaternion,
    yaw_from_quaternion,
)

BLOCK_OFFSET = 0.4  # m
REVERSE_DISTANCE = 0.4  # m
STOCK_CLEARANCE = 1  # m clearance


class State(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    GRABBING = auto()
    STOCKPILE_PREP_PREP = auto()
    STOCKPILE_PREP = auto()
    STOCKPILE_DEPOSIT = auto()
    STOCKPILE_EXIT = auto()
    STOCKPILE_DEPART = auto()
    STOCKPILE_FLEE = auto()
    RETURNING = auto()
    RECOVERY = auto()


class RetrieveNode(Node):
    """RetrieveNode runs the state machine for each of the retriever robots."""

    def __init__(self, node_name: str, *args) -> None:
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

        self.build_area_sub = self.create_subscription(
            PolygonStamped, "build_polygon", self.build_area_callback, 10
        )

        self.block_sub = self.create_subscription(
            OccupancyGrid, "/block_mask", self.block_callback, 10
        )

        # Set up publisher
        self.vel_pub = self.create_publisher(
            Twist, f"{self.get_namespace()}/cmd_vel", 10
        )

        self.vis_pub = self.create_publisher(
            MarkerArray, f"{self.get_namespace()}/markers", 10
        )
        self.pc_pub = self.create_publisher(
            PointCloud2, f"{self.get_namespace()}/detected_obs", 10
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

        self.state = State.IDLE
        self.return_state = State.NAVIGATING
        self.odom = None
        self.start_pose_set = False
        self._start_pose = None

        self.block_type = None
        self.block_positions = None
        self.nav_pose = None
        self.observed_block_pose = None
        self.stockpile = None
        self.stockpile_safe = None
        self.stockpile_safe_safe = None
        self.recovery_pose = None
        self.terminus = None
        self.origin = None
        self.build_area_points = None

        self.test_pose = None # debug

        self.missing_tag_count = 0
        self.tag_visible = False
        self.valid_tf_tree = False

        # TODO: Check the name of the Jackal, also potentially use the tf tree to figure out the boundaries???
        self.neighbor_list = ["aruco_31", "aruco_15", "aruco_21"]

    def odom_callback(self, msg: Odometry) -> None:
        self.logger.debug(f"Received Odometry: {msg}")
        self.odom = msg
        if self._start_pose is None:
            self._start_pose = PoseStamped()
            self._start_pose.header.frame_id = self._namespaced_frame("odom")
            self._start_pose.pose = self.odom.pose.pose
            self.start_pose_set = True

    def update_odom_state(self, valid_msg: Bool) -> None:
        self.valid_tf_tree = valid_msg.data

    def visible_block_callback(self, msg: PoseStatus) -> None:

        self.update_visualization()

        if msg.block_in_frame and not msg.tag_in_frame:
            if self.state in [State.RECOVERY]:
                self.logger.info(
                    f"[VisibleCallback] Block visible but tag not visible, setting recovery pose to ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})",
                    throttle_duration_sec=1.0,
                )
                self.recovery_pose = msg.pose  # currently unused
            return

        if not msg.tag_in_frame:
            self.missing_tag_count += 1
            if self.missing_tag_count > 10:
                # acts as some hysteresis for losing the block at 30 fps
                if self.state in [
                    State.NAVIGATING,
                    State.GRABBING,
                    State.STOCKPILE_PREP,
                    State.STOCKPILE_PREP_PREP,
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

    def get_odom_transform(self, message):
        transform = self.tf_buffer.lookup_transform(
            self._namespaced_frame("odom"), message.header.frame_id, rclpy.time.Time()
        )
        return transform

    def calculate_nav_pose(
        self, message: RetrievalTask.Goal, stock_pt_safe: tuple[float, float]
    ) -> PoseStamped:
        # given a message that contains both the block pose and the stockpile pose
        block_pose_stamped = message.block.pose

        block_pt = (
            block_pose_stamped.pose.position.x,
            block_pose_stamped.pose.position.y,
        )

        travel_angle = np.math.atan2(
            block_pt[1] - stock_pt_safe[1], block_pt[0] - stock_pt_safe[0]
        )

        block_angle = yaw_from_quaternion(
            q=block_pose_stamped.pose.orientation, use_extrinsics=True
        )

        # this is how much angle the robot needs to rotate through in order to bring the block to the stockpile (directly)
        turn_magnitude = np.abs(travel_angle - block_angle)

        pose = block_pose_stamped

        if turn_magnitude < np.pi / 2:
            # position ourselves along the pos y-axis, facing the start
            pose.pose.position.x += BLOCK_OFFSET * np.cos(block_angle)
            pose.pose.position.y += BLOCK_OFFSET * np.sin(block_angle)
            pose.pose.orientation = reverse_yaw_quaternion(pose.pose.orientation)

        else:
            pose.pose.position.x -= BLOCK_OFFSET * np.cos(block_angle)
            pose.pose.position.y -= BLOCK_OFFSET * np.sin(block_angle)

        self.test_pose = pose

        transform = self.get_odom_transform(pose)
        pose = do_transform_pose_stamped(pose, transform)

        return pose

    def block_callback(self, msg: OccupancyGrid) -> None:
        block_arr = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width
        )
        origin = msg.info.origin
        resolution = msg.info.resolution

        self.terminus = resolution * np.array([msg.info.width, msg.info.height])
        self.origin = np.array([origin.position.x, origin.position.y])

        row_mask, col_mask = np.where(block_arr > 0)
        x_positions = (col_mask * resolution) + origin.position.x
        y_positions = (row_mask * resolution) + origin.position.y

        try:
            transform = self.tf_buffer.lookup_transform(
                self._namespaced_frame("odom"), "world", rclpy.time.Time()
            )
            transformed_x = []
            transformed_y = []
            for x, y in zip(x_positions, y_positions):
                pt = PointStamped()
                pt.header.stamp = self.get_clock().now().to_msg()
                pt.header.frame_id = "world"
                pt.point.x = x
                pt.point.y = y

                transformed_pt = do_transform_point(pt, transform)

                transformed_x.append(transformed_pt.point.x)
                transformed_y.append(transformed_pt.point.y)

            self.block_positions = np.vstack(
                (transformed_x, transformed_y)
            )
        except Exception as e:
            self.logger.warn(f"Failed to find the block positions:\n {e}", throttle_duration_sec=5.0)
            
            return


    def build_area_callback(self, msg: PolygonStamped):
        transform = self.tf_buffer.lookup_transform(
            self._namespaced_frame("odom"), "world", rclpy.time.Time()
        )
        pts = msg.polygon.points
        pt_arr = []
        for pt in [pts[0], pts[2]]:
            pt_0 = PointStamped()
            pt_0.header.stamp = self.get_clock().now().to_msg()
            pt_0.header.frame_id = "world"
            pt_0.point.x = pt.x
            pt_0.point.y = pt.y
            pt_transformed = do_transform_point(pt_0, transform)
            pt_arr.append(pt_transformed)

        self.build_area_points = [pt_arr[0].x, pt_arr[1].x, pt_arr[0].y, pt_arr[1].y]

    def save_block_pose(self) -> PoseStamped:
        block_loc = deepcopy(self.observed_block_pose)
        transform = self.get_odom_transform(block_loc)

        res = do_transform_pose_stamped(block_loc, transform)
        res.pose.orientation = self.curr_pose.pose.orientation
        return res

    def calculate_exit_pose(self) -> PoseStamped:
        transform = self.tf_buffer.lookup_transform(
            self._namespaced_frame("odom"),
            self._namespaced_frame("base_link"),
            rclpy.time.Time(),
        )

        pos = PoseStamped()
        pos.header.frame_id = self._namespaced_frame("base_link")
        pos.header.stamp = self.get_clock().now().to_msg()
        pos.pose.position.x = -REVERSE_DISTANCE
        return do_transform_pose_stamped(pos, transform)

    def stockpile_behavior(
        self, pose_location: PoseStamped, msg: str, next_state: State, controller: Callable = None
    ) -> None:
        reached = self.go_to_pose(pose_location, controller=controller)
        if self.tag_visible and reached:
            self.logger.info(f"[{self.state.name}]{msg}")
            self.exit_pose = self.calculate_exit_pose()
            self.state = next_state
        elif not self.tag_visible:
            self.send_to_recovery()

    def send_to_recovery(self, msg: str = None) -> None:
        message = (
            msg
            if msg is not None
            else f"[{self.state.name}]LOST the BLOCK, entering RECOVERY state to attempt recovery"
        )
        self.logger.info(message)
        self.return_state = State.GRABBING
        self.state = State.RECOVERY

    def create_result(
        self, success: bool, pose: PoseStamped = None
    ) -> RetrievalTask.Result:
        result = RetrievalTask.Result()
        result.success = success
        result.delivered = Block()
        result.delivered.pose = pose or PoseStamped()
        return result

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
                stock_pt_safe = (stock_pt[0] - STOCK_CLEARANCE, stock_pt[1])

                stock_pile_quat = quaternion_from_euler(
                    yaw=0, units="degrees"
                )

                self.stockpile = PoseStamped()
                self.stockpile.header = goal_handle.request.stockpile.header
                self.stockpile.pose.position.x = stock_pt[0] - 0.35
                self.stockpile.pose.position.y = stock_pt[1]
                self.stockpile.pose.orientation = stock_pile_quat

                self.stockpile_safe = PoseStamped()
                self.stockpile_safe.header = goal_handle.request.stockpile.header
                self.stockpile_safe.pose.position.x = stock_pt_safe[0]
                self.stockpile_safe.pose.position.y = stock_pt_safe[1]
                self.stockpile_safe.orientation = stock_pile_quat

                self.stockpile_safe_safe = PoseStamped()
                self.stockpile_safe_safe.header = goal_handle.request.stockpile.header
                self.stockpile_safe_safe.pose.position.x = stock_pt_safe[0]
                self.stockpile_safe_safe.pose.position.y = stock_pt_safe[1] + 1.5
                self.stockpile.pose.orientation = quaternion_from_euler(
                    yaw=270, units="degrees"
                )

                transform_stockpile = self.get_odom_transform(self.stockpile)

                self.stockpile = do_transform_pose_stamped(
                    self.stockpile, transform_stockpile
                )
                self.stockpile_safe = do_transform_pose_stamped(
                    self.stockpile_safe, transform_stockpile
                )
                self.stockpile_safe_safe = do_transform_pose_stamped(
                    self.stockpile_safe_safe, transform_stockpile
                )
        

                self.nav_pose = self.calculate_nav_pose(
                    goal_handle.request, stock_pt_safe
                )
                self.observed_block_pose = None
                self.block_type = goal_handle.request.block.type

                self.logger.info(
                    f"Received retrieve action goal: {self.nav_pose}, entering NAVIGATING state"
                )
                self.init_barriers()
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
                    self.send_to_recovery(
                        msg=f"[{self.state.name}]Reached target location but no block found, entering RECOVERY state to attempt recovery"
                    )

            elif self.state == State.GRABBING:
                # super naive, we should check for block in position here
                # I think this is where we take down our barriers on the specific block
                reached = self.go_to_pose(
                    self.grab_pose, controller=self.pose_controller
                )
                self.logger.info(
                    f"Going to block located at {self.grab_pose.pose}",
                    throttle_duration_sec=1.0,
                )
                if reached:
                    self.grab_pose = None
                    self.logger.info(
                        f"[{self.state.name}]Grabbed block, bringing block to Safe Stockpile Point"
                    )
                    self.state = State.STOCKPILE_PREP_PREP

            elif self.state == State.STOCKPILE_PREP_PREP:
                # move to a safe stockpiling location
                self.stockpile_behavior(
                    pose_location=self.stockpile_safe_safe,
                    msg="Reached Safe Safe Stockpile Point, attempting Safe Stockpile",
                    next_state=State.STOCKPILE_PREP,
                )


            elif self.state == State.STOCKPILE_PREP:
                # move to a safe stockpiling location
                self.stockpile_behavior(
                    pose_location=self.stockpile_safe,
                    msg="Reached Safe Stockpile Point, attempting DEPOSIT",
                    next_state=State.STOCKPILE_DEPOSIT,
                )

            elif self.state == State.STOCKPILE_DEPOSIT:
                # move to the specific stockpile location
                self.stockpile_behavior(
                    pose_location=self.stockpile,
                    msg="Stockpiled block, attempting to EXIT the STOCKPILE",
                    next_state=State.STOCKPILE_EXIT,
                    controller=self.pose_controller,
                )

            elif self.state == State.STOCKPILE_EXIT:
                back_up = self.go_to_pose(self.exit_pose, self.pose_controller_reverse)
                if back_up:
                    self.state = State.STOCKPILE_DEPART

            elif self.state == State.STOCKPILE_DEPART:
                exit_pose = deepcopy(self.stockpile_safe)
                exit_pose.orientation = reverse_yaw_quaternion(
                    self.stockpile_safe.pose.orientation
                )
                reached = self.go_to_pose(self.stockpile_safe)
                if reached:
                    self.logger.info(
                        f"[{self.state.name}]Exited the stockpile successfully, getting safer"
                    )
                    self.state = State.STOCKPILE_FLEE

            elif self.state == State.STOCKPILE_FLEE:
                exit_pose = deepcopy(self.stockpile_safe_safe)
                exit_pose.orientation = reverse_yaw_quaternion(
                    self.stockpile_safe_safe.pose.orientation
                )
                reached = self.go_to_pose(self.stockpile_safe_safe)
                if reached:
                    self.logger.info(
                        f"[{self.state.name}]Exited the stockpile successfully, getting safer"
                    )
                    self.state = State.RETURNING

            # This state is just if we want the robot to return to the starting position
            elif self.state == State.RETURNING:
                if self._start_pose is not None:
                    goal_reached = self.go_to_pose(
                        self._start_pose, controller=self.pose_controller_clf
                    )
                    if goal_reached:
                        self.logger.info(
                            f"[{self.state.name}]Returned to start, finishing action and returning to IDLE state"
                        )
                        goal_handle.succeed()
                        result = self.create_result(
                            success=True, pose=self.observed_block_pose
                        )
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

        result = self.create_result(success=False, pose=self.observed_block_pose)
        return result

    def pose_controller(
        self, target_pose: Pose, reversed: bool = False
    ) -> tuple[Twist, bool]:

        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return

        self.logger.info(
            f"Using simple pose controller to go to {target_pose}, from {self.curr_pose}",
            throttle_duration_sec=2.0,
        )
        reached = False
        dx = target_pose.position.x - self.curr_pose.pose.position.x
        dy = target_pose.position.y - self.curr_pose.pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        q_curr = self.curr_pose.pose.orientation
        current_yaw = yaw_from_quaternion(q=q_curr, use_extrinsics=True)

        angle_diff = (
            angle_wrap(angle_to_target - current_yaw)
            if not reversed
            else angle_wrap(angle_to_target - (current_yaw + np.pi))
        )
        q_desired = target_pose.orientation
        desired_yaw = yaw_from_quaternion(q=q_desired, use_extrinsics=True)

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
                sgn = np.sign(distance) if not reversed else -np.sign(distance)
                cmd.linear.x = sgn * max(min(linear_gain * distance, 0.2), 0.05)
            else:
                if abs(desired_yaw_diff) > 0.04:
                    sgn = np.sign(desired_yaw_diff)
                    cmd.angular.z = sgn * max(
                        min(angular_gain * abs(desired_yaw_diff), 0.2), 0.05
                    )
                else:
                    reached = True

        return cmd, reached

    # gamma is approach angle gain, k is desired angle gain, and h is rotation error gain. Low gamma will slow down linear velocity as well.
    def pose_controller_clf(
        self,
        target_pose: Pose,
        gamma: float = 0.5,
        k: float = 1.0,
        h: float = 0.6,
        forward_constraint: bool = False,
    ) -> tuple[Twist, bool]:
        assert gamma > 0, f"gamma = {gamma} must be greater than 0"
        assert k > gamma, f"k = {k} must be greater than gamma = {gamma}"
        assert h > 0, f"h = {h} must be greater than 0"

        if self.curr_pose is None:
            self.logger.warning("Current pose is unknown, cannot navigate")
            return Twist, False

        self.logger.info(
            f"Using CLF controller to go to {target_pose}, from {self.curr_pose.pose}",
            throttle_duration_sec=1.0,
        )
        reached = False
        q_curr = self.curr_pose.pose.orientation
        theta = yaw_from_quaternion(q=q_curr, use_extrinsics=True)
        q_desired = target_pose.orientation
        desired_theta = yaw_from_quaternion(q=q_desired, use_extrinsics=True)

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
                reached = True

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        curr_pose = self.curr_pose
        curr_ori = yaw_from_quaternion(
            q=curr_pose.pose.orientation, use_extrinsics=True
        )

        curr_position = [
            curr_pose.pose.position.x,
            curr_pose.pose.position.y,
        ]
        robo_pose = np.array(
            [
                [curr_position[0]],
                [curr_position[1]],
                [curr_ori],
            ]
        )
        # I don't want to risk the barriers affecting the block positions in case we use them elsewhere
        # TODO: Also doublecheck the shapes on these things
        block_positions = deepcopy(self.block_positions)
        neighbor_positions = deepcopy(self.neighbor_positions)
        try:
            safe_cmd = self.barrier_func(
                cmd,
                robo_pose,
                neighbor_positions=neighbor_positions,
                block_positions=block_positions,
            )
        except Exception as e:
            self.logger.error(
                f"Barriers are unable to produce a safe velocity: {e}\nStopping robot"
            )
            return Twist(), False
        return safe_cmd, reached

    # TODO: Implement the robot moving backwards better than this. Made it somewhat better
    def pose_controller_reverse(self, target_pose: Pose) -> tuple[Twist, bool]:
        return self.pose_controller(target_pose=target_pose, reversed=True)

    def pose_controller_clf_constrained(self, target_pose: Pose) -> tuple[Twist, bool]:
        return self.pose_controller_clf(target_pose, forward_constraint=True)

    def go_to_pose(
        self,
        target_pose: PoseStamped,
        controller: Callable[[Pose], tuple[Twist, bool]] = None,
    ) -> bool:
        return False
        assert target_pose.header.frame_id == self._namespaced_frame("odom")
        if controller is None:
            controller = self.pose_controller_clf

        cmd, reached = controller(target_pose.pose)
        self.vel_pub.publish(cmd)
        return reached

    def brake(self) -> None:
        self.vel_pub.publish(Twist())

    def _namespaced_frame(self, frame_name: str) -> str:
        ns = self.get_namespace().strip("/")
        return f"{ns}/{frame_name}" if ns else frame_name

    def update_visualization(self) -> None:
        if hasattr(self, "block_positions") and self.block_positions is not None:
            transposed_block_positions = self.block_positions.T
            zeros_col = np.zeros((transposed_block_positions.shape[0], 1))
            transposed_block_positions = np.hstack((transposed_block_positions, zeros_col))
            header = Header()
            header.frame_id = self._namespaced_frame('odom')
            header.stamp = rclpy.clock.Clock().now().to_msg() # Adjust to your node's clock if inside a class

            cloud_msg = point_cloud2.create_cloud_xyz32(header, transposed_block_positions.astype(np.float32))
            self.pc_pub.publish(cloud_msg)


        msg = MarkerArray()
        marker_data = {
            "curr_pose": (self.curr_pose, (1.0, 1.0, 0.0)),
            "observed_block": (self.observed_block_pose, (1.0, 0.0, 0.0)),
            "test_pose": (self.test_pose, (1.0,0.0,1.0)),
            "nav_pose": (self.nav_pose, (0.0, 1.0, 1.0)),
            "stockpile_pose": (self.stockpile, (0.0, 1.0, 0.0)),
            "stockpile_safe_pose": (self.stockpile_safe, (0.0, 0.0, 1.0)),
            "stockpile_safe_safe_pose": (self.stockpile_safe_safe, (1.0, 0.5, 0.0))
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

    def init_barriers(self):
        if self.terminus is not None and self.origin is not None:
            transform = self.tf_buffer.lookup_transform(
                self._namespaced_frame("odom"), "world", rclpy.time.Time()
            )
            origin_pt = PointStamped()
            origin_pt.header.stamp = self.get_clock().now().to_msg()
            origin_pt.header.frame_id = "world"
            origin_pt.point.x = self.origin[0]
            origin_pt.point.y = self.origin[1]

            terminus_pt = PointStamped()
            terminus_pt.header.stamp = origin_pt.header.stamp
            terminus_pt.header.frame_id = "world"
            terminus_pt.point.x = self.terminus[0]
            terminus_pt.point.y = self.terminus[1]

            origin_pt_fixed = do_transform_point(origin_pt, transform)
            terminus_pt_fixed = do_transform_point(terminus_pt, transform)
            boundary_points=[
                    origin_pt_fixed.point.x,
                    terminus_pt_fixed.point.x,
                    origin_pt_fixed.point.y,
                    terminus_pt_fixed.point.y,
                ]
            self.logger.info(f"Initializing barriers with boundaries: {boundary_points}\n Build area points are {self.build_area_points}")
            self.logger.info(f"Currently located at {self.curr_pose}")
            boundary_points = None
            self.barrier_func = get_robot_barrier_func(boundary_points=boundary_points,
                build_area_points=self.build_area_points,
            )
        else:
            self.logger.warning("Barriers initialized with fixed values")
            self.barrier_func = get_robot_barrier_func(
                boundary_points=[-5.0, 5.0, -5.0, 5.0],
                build_area_points=[-6, -5.5, -6, -5.5],
            )

    # TODO: use the world frame and published aruco tags to update the robots position
    @property
    def curr_pose(self) -> PoseStamped | None:
        if hasattr(self, "odom") and self.odom is not None:
            pose = PoseStamped()
            pose.header.frame_id = self._namespaced_frame("odom")
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose = self.odom.pose.pose
            return pose
        else:
            self.logger.warning("No pose information available")
            return None

    # TODO: Does it make sense to do this as a property? Yes, it does, don't question it
    @property
    def neighbor_positions(self) -> np.ndarray:
        positions = np.zeros((2, len(self.neighbor_list)))
        for ndx, neighbor_frame in enumerate(self.neighbor_list):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self._namespaced_frame("odom"), neighbor_frame, rclpy.time.Time()
                )
                positions[0, ndx] = transform.transform.translation.x
                positions[1, ndx] = transform.transform.translation.y
            except Exception as e:
                self.logger.error(f"Could not compute neighbor transform: {e}")
                positions[0, ndx] = -10
                positions[1, ndx] = -10
        return positions


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
