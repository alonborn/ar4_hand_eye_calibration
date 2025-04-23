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
from my_robot_interfaces.srv import MoveToPose  # Import the custom service type
import time
import debugpy
import transforms3d


class ArucoMarkerFollower(Node):

    def __init__(self):
        super().__init__("aruco_marker_follower")
        self.logger = self.get_logger()

        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]
        
        # self.moveit2 = MoveIt2(
        #     node=self,
        #     joint_names=self.arm_joint_names,
        #     base_link_name="base_link",
        #     end_effector_name="link_6",
        #     group_name="ar_manipulator",
        #     callback_group=ReentrantCallbackGroup(),
        # )
        # self.moveit2.planner_id = "RRTConnectkConfigDefault"
        # self.moveit2.max_velocity = 1.0
        # self.moveit2.max_acceleration = 1.0
        self.move_completed_successfully = False
        self.cb_group = ReentrantCallbackGroup()
        self.cb_group_aruco_marker = ReentrantCallbackGroup()

        self.move_client = self.create_client(MoveToPose,'ar_move_to_pose',callback_group=self.cb_group)
        self.arm_is_available = True
        # ID of the aruco marker mounted on the robot
        self.marker_id = self.declare_parameter(
            "marker_id", 1).get_parameter_value().integer_value

        self.subscription = self.create_subscription(ArucoMarkers,"/aruco_markers",self.handle_aruco_markers,1,callback_group=self.cb_group_aruco_marker)
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose",1)

        self.target_pose_pub = self.create_publisher(
            PoseStamped, "/follow_aruco_target_pose", 1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._prev_marker_pose = None

        # Timer: callback every 2.0 seconds
        # self.timer_callback_group = ReentrantCallbackGroup()
        # self.timer = self.create_timer(2.0, self.timer_callback,callback_group=self.timer_callback_group)

    # def timer_callback(self):
    #     self.get_logger().info('ROS loop is alive!')


    def wait_for_arm (self):
        """Wait for the arm to be available before sending a move request."""
        self.get_logger().info("waiting for arm to be available")
        while not self.arm_is_available:
            #rclpy.spin_once(self, timeout_sec=0.1) 
            time.sleep(0.01)  # Sleep for a short duration to avoid busy-waiting
            #print ("spinning")

    def response_callback(self, future):
        try:
            self.get_logger().info("Response callback")
            response = future.result()
            self.move_completed_successfully = response.success
            # Logging the status and message from the response
            #self.get_logger().info(f'Response Status: {response.success}')
            #self.get_logger().info(f'Response Message: {response.message}')
            self.arm_is_available = True
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def send_move_request(self, pose):
        # Create a request
        request = MoveToPose.Request()
        request.pose = pose
        
        # Send the request
        self.arm_is_available = False
        self.get_logger().info("Sending move request")
        future = self.move_client.call_async(request)
        self.get_logger().info("Waiting for move request to complete")
        future.add_done_callback(self.response_callback)
        #self.move_client.call_async(request).add_done_callback(self.response_callback)
              
        # Add a callback to be executed when the future is complete
        
        self.wait_for_arm()
        self.get_logger().info("Move request done")

    def handle_aruco_markers(self, msg: ArucoMarkers):
        self.get_logger().info("Received aruco markers")
        cal_marker_pose = None
        if (not self.arm_is_available):
            return
        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == self.marker_id:
                cal_marker_pose = msg.poses[i]
                break
            else:
                self.logger.info(f"Detected unexpected marker with ID: {marker_id}")

        if cal_marker_pose is None:
            return

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
        except tf2_ros.LookupException as e:
            self.logger.error(f"Error transforming pose: {e}")
            return

        # first flip the pose up side down

        quat = [
            transformed_pose.orientation.w,
            transformed_pose.orientation.x,
            transformed_pose.orientation.y,
            transformed_pose.orientation.z,
        ]
        x_180_deg_quat = [0, 1, 0, 0]
        flipped_quat = transforms3d.quaternions.qmult(quat, x_180_deg_quat)
        transformed_pose.orientation.w = flipped_quat[0]
        transformed_pose.orientation.x = flipped_quat[1]
        transformed_pose.orientation.y = flipped_quat[2]
        transformed_pose.orientation.z = flipped_quat[3]

        transformed_pose.position.z += 0.09        

        self.logger.info(f"Following marker at pose: {transformed_pose}")
        # self.move_to(transformed_pose)
        self.send_move_request(transformed_pose)

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
        
        self.send_move_request(pose_goal)

        # self.moveit2.move_to_pose(pose=pose_goal)
        # self.moveit2.wait_until_executed()


def main():

    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # Uncomment this if you want to pause execution until the debugger attaches
    # print("debugger attached")
        
    rclpy.init()
    node = ArucoMarkerFollower()
    executor = MultiThreadedExecutor(7)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
