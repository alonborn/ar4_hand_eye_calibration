#!/usr/bin/env python3
"""
A script to follow an aruco marker with a robot arm using PyMoveit2.
"""
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_geometry_msgs import do_transform_pose
from pymoveit2 import MoveIt2
import logging
import debugpy

#view end effector position
#ros2 run tf2_ros tf2_echo base_link ee_link

class ArucoMarkerFollower(Node):

    def __init__(self):
        super().__init__("aruco_marker_follower")
        self.logger = self.get_logger()

        # Create a secondary logger for file output
        self.file_logger = logging.getLogger('aruco_pose_logger')
        self.file_logger.setLevel(logging.INFO)

        # File handler setup
        file_handler = logging.FileHandler('/home/alon/Documents/aruco_pose_log.txt')
        file_handler.setLevel(logging.INFO)

        # Add a formatter for clear log structure
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Attach the handler to the new logger
        self.file_logger.addHandler(file_handler)


        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=ReentrantCallbackGroup(),
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.moveit2.max_velocity = 1.0
        self.moveit2.max_acceleration = 1.0
        self.moveit2.planning_time = 5.0  # Timeout in seconds

        # ID of the aruco marker mounted on the robot
        self.marker_id = self.declare_parameter(
            "marker_id", 1).get_parameter_value().integer_value

        self.subscription = self.create_subscription(ArucoMarkers,
                                                     "/aruco_markers",
                                                     self.handle_aruco_markers,
                                                     1)
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose",
                                              1)

        self.target_pose_pub = self.create_publisher(
            PoseStamped, "/follow_aruco_target_pose", 1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._prev_marker_pose = None


    def write_to_logger(self, transformed_pose):
        # Extract position values
        position = transformed_pose.position
        x, y, z = round(position.x, 3), round(position.y, 3), round(position.z, 3)

           # Extract rotation (quaternion) values
        orientation = transformed_pose.orientation
        qx, qy, qz, qw = (
            round(orientation.x, 3),
            round(orientation.y, 3),
            round(orientation.z, 3),
            round(orientation.w, 3)
        )

        # Log the formatted position
        self.file_logger.info(f"Position: {x},{y},{z} | Rotation: {qx},{qy},{qz},{qw}")
        

    def handle_aruco_markers(self, msg: ArucoMarkers):
        cal_marker_pose = None
        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == self.marker_id:
                cal_marker_pose = msg.poses[i]
                break
            else:
                self.logger.info(f"Detected unexpected marker with ID: {marker_id}")

        if cal_marker_pose is None:
            self.logger.error(f"Could not find marker with ID: {self.marker_id}")
            return
        self.logger.info(f"handle aruco markers")
        # only start following if the marker pose has changed by at least 2cm
        if self._prev_marker_pose is not None:
            if ((cal_marker_pose.position.x -
                 self._prev_marker_pose.position.x)**2 +
                (cal_marker_pose.position.y -
                 self._prev_marker_pose.position.y)**2 +
                (cal_marker_pose.position.z -
                 self._prev_marker_pose.position.z)**2 > 0.02**2):
                self._prev_marker_pose = cal_marker_pose
                return

        self._prev_marker_pose = cal_marker_pose

        # get pose in robot base frame
        try:
            transformed_pose = self._transform_pose(cal_marker_pose,
                                                    "camera_color_optical_frame",
                                                    "base_link")
            self.write_to_logger(transformed_pose)
            
        except tf2_ros.LookupException as e:
            self.file_logger.info(f"Error with: cal_marker_pose:" + str(cal_marker_pose))
            self.logger.error(f"Error transforming pose: {e}")
            return

        transformed_pose.position.x = 0.04
        transformed_pose.position.y = -0.31
        transformed_pose.position.z = 0.375
        transformed_pose.orientation.x = 0.044
        transformed_pose.orientation.y = -0.702
        transformed_pose.orientation.z = 0.71
        transformed_pose.orientation.w = -0.033


        self.logger.info(f"Following marker at pose2: {transformed_pose}")
        self.move_to(transformed_pose)
        self.logger.info(f"Done moving: {transformed_pose}")

    def _transform_pose(self, pose: Pose, source_frame,
                        target_frame: str) -> Pose:
        # Get the transform from source frame to target frame
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame,
                                                    Time())
        # Transform the pose
        transformed_pose = do_transform_pose(pose, transform)
        # publish pose
        stamped_pose = PoseStamped()
        stamped_pose.header.frame_id = target_frame
        stamped_pose.pose = transformed_pose
        self.pose_pub.publish(stamped_pose)

        pose.position.z += 0.05
        transformed_pose = do_transform_pose(pose, transform)

        stamped_pose = PoseStamped()
        stamped_pose.header.frame_id = target_frame
        stamped_pose.pose = transformed_pose
        self.target_pose_pub.publish(stamped_pose)
        return transformed_pose

    def move_to(self, msg: Pose):
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose = msg

        self.moveit2.move_to_pose(pose=pose_goal)
        ret = self.moveit2.wait_until_executed()
        if ret:
            self.file_logger.info("Move successful")


def main():

    # Allow attaching the debugger remotely on port 5678
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # Uncomment this if you want to pause execution until the debugger attaches

    rclpy.init()
    node = ArucoMarkerFollower()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
