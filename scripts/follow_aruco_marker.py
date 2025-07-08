#!/usr/bin/env python3
"""
A script to follow an aruco marker with a robot arm using PyMoveit2.
"""
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
import threading
import tf2_ros
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_geometry_msgs import do_transform_pose
from pymoveit2 import MoveIt2
import debugpy
import transforms3d
from my_robot_interfaces.srv import MoveToPose  # Import the custom service type
import time
from geometry_msgs.msg import Pose, Point, Quaternion

class ArucoMarkerFollower(Node):

    def __init__(self):
        super().__init__("aruco_marker_follower")
        self.logger = self.get_logger()
        
        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]
        self.follower_enabled = True 
        
        self.processing_lock = threading.Lock()
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

        # ID of the aruco marker mounted on the robot
        self.marker_id = self.declare_parameter(
            "marker_id", 1).get_parameter_value().integer_value

        self.subscription = self.create_subscription(ArucoMarkers,
                                                     "/aruco_markers",
                                                     self.handle_aruco_markers,
                                                     1)
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose",
                                              1)
        self.move_arm_client = self.create_client(MoveToPose, '/ar_move_to_pose')
        self.target_pose_pub = self.create_publisher(PoseStamped, "/follow_aruco_target_pose", 1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._prev_marker_pose = None
        
        # Timer: callback every 2.0 seconds
        self.timer_callback_group = ReentrantCallbackGroup()
        #self.timer = self.create_timer(1.0, self.timer_callback,callback_group=self.timer_callback_group)
        self.filtered_position = None  # holds the smoothed pose position
        self.ema_alpha = 0.3           # tune 0.1–0.5 depending on smoothness vs responsiveness


    def get_request(self, pose, is_cartesian=True):
        # Create a request
        request = MoveToPose.Request()
        request.pose = pose
        request.cartesian = is_cartesian
        return request

    def timer_callback(self):
        self.get_logger().info('ROS loop is alive!')

    def handle_aruco_markers(self, msg: ArucoMarkers):

        acquired = self.processing_lock.acquire(blocking=False)
        if not acquired:
            self.logger.info("Already processing a marker, skipping this callback.")
            return

        if not self.follower_enabled:
            self.logger.info("Follower disabled; skipping marker processing.")
            self.release_processing()
            return

        
        self.logger.info(f"Processing marker with ID: {self.marker_id}")

        cal_marker_pose = None
        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == self.marker_id:
                cal_marker_pose = msg.poses[i]
                break
            else:
                self.logger.info(f"Detected unexpected marker with ID: {marker_id}")
        try:
            if cal_marker_pose is None:
                return


            # get pose in robot base frame
            try:
                self.logger.info(f"-----------------------(1)cal_marker_pose: {cal_marker_pose}")
                transformed_pose = self._transform_pose(cal_marker_pose,
                                                        "camera_color_optical_frame",
                                                        "base_link")
                self.logger.info(f"-----------------------(2)Transformed pose: {transformed_pose}")
                self.get_logger().info(f"Transformed pose (after _transform_pose): {transformed_pose}")
                pos = transformed_pose.position

                if self.filtered_position is None:
                    # first run: initialize filter
                    self.filtered_position = [pos.x, pos.y, pos.z]
                else:
                    # update filter with EMA
                    self.filtered_position[0] = self.ema_alpha * pos.x + (1 - self.ema_alpha) * self.filtered_position[0]
                    self.filtered_position[1] = self.ema_alpha * pos.y + (1 - self.ema_alpha) * self.filtered_position[1]
                    self.filtered_position[2] = self.ema_alpha * pos.z + (1 - self.ema_alpha) * self.filtered_position[2]

                # update transformed_pose with filtered position
                transformed_pose.position.x = self.filtered_position[0]
                transformed_pose.position.y = self.filtered_position[1]
                transformed_pose.position.z = self.filtered_position[2]


                #only start following if the marker pose has changed by at least 2cm
                if self._prev_marker_pose is not None:
                    dist = (transformed_pose.position.x - self._prev_marker_pose.position.x)**2 + (transformed_pose.position.y - self._prev_marker_pose.position.y)**2 + (transformed_pose.position.z - self._prev_marker_pose.position.z)**2 
                    self.logger.info(f"Distance to previous marker pose: {dist}")
                    if (dist > 0.05**2):
                        self._prev_marker_pose = transformed_pose
                        return
                    # if (dist < 0.005**2):
                    #     self._prev_marker_pose = transformed_pose
                    #     self.logger.info(f"Marker pose has not changed enough to follow: {dist}")
                    #     return
                
                self._prev_marker_pose = transformed_pose

            except tf2_ros.LookupException as e:
                self.logger.error(f"Error transforming pose: {e}")
                return

            # first flip the pose up side down

            self.logger.info(f"Transformed pose: {transformed_pose}")
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

            transformed_pose.position.z += 0.10

            self.logger.info(f"------------------------(3)Following marker at pose: {transformed_pose}")
            # self.move_to(transformed_pose)
        except Exception as e:
            self.logger.error(f"Unexpected exception during marker processing: {e}")
        finally:
            self.logger.info(("Finished processing marker with ID: " f"{self.marker_id}"))
            self.release_processing()  # always release the lock here

            
    def release_processing(self):
        if self.processing_lock.locked():
            try:
                self.processing_lock.release()
            except RuntimeError as e:
                self.logger.warning(f"Tried to release processing lock but failed: {e}")
        else:
            self.logger.debug("Processing lock was not locked at release attempt.")

        self.joint_states_enabled = True
        
    def _transform_pose(self, pose: Pose, source_frame: str, target_frame: str) -> Pose:
        """Transforms a pose from the source_frame to the target_frame,
        publishes both the direct transformed pose and one offset in z by +5cm,
        and returns the direct transformed pose."""
        
        # Look up the transform from source_frame -> target_frame
        self.logger.info(f"------------------------(6)Transforming pose from {source_frame} to {target_frame}")
        self.get_logger().info(f"------------------------(7)tf_buffer=" + str(self.tf_buffer))
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        
        self.logger.info(f"------------------------(4)Transform: {transform}")
        # Transform the original pose
        transformed_pose = do_transform_pose(pose, transform)
        self.get_logger().info(f"------------------------(5)Transformed pose: {transformed_pose}")  
        # Publish the direct transformed pose
        stamped_pose = PoseStamped()
        stamped_pose.header.stamp = self.get_clock().now().to_msg()
        stamped_pose.header.frame_id = target_frame
        stamped_pose.pose = transformed_pose
        self.pose_pub.publish(stamped_pose)
        
        # Create a copy of the original pose with z offset +5cm before transforming
        pose_with_offset = Pose()
        pose_with_offset.position = Point(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z + 0.05  # add 5 cm in z
        )
        pose_with_offset.orientation = pose.orientation
        
        transformed_pose_offset = do_transform_pose(pose_with_offset, transform)
        
        # Publish the offset pose (useful for visualizations)
        stamped_offset_pose = PoseStamped()
        stamped_offset_pose.header.stamp = self.get_clock().now().to_msg()
        stamped_offset_pose.header.frame_id = target_frame
        stamped_offset_pose.pose = transformed_pose_offset
        self.target_pose_pub.publish(stamped_offset_pose)
        
        return transformed_pose


    def move_to(self, msg: Pose):
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose = msg

        # if self.tmp == 1:
        #     self.tmp = 0
        #     msg = Pose(position = Point (x=0.03,y=-0.33,z=0.39),orientation = Quaternion(x=0.4263,y=0.0428,z=0.0902,w=0.899))
        # else:
        #     self.tmp = 1
        #     msg = Pose(position = Point(x=0.03,y=-0.33,z=0.32),orientation = Quaternion(x=0.4263,y=0.0428,z=0.0902,w=0.899))

        self.send_move_request(pose=msg,is_cartesian=False)
        # self.moveit2.move_to_pose(pose=pose_goal)
        # self.moveit2.wait_until_executed()

    def send_move_request(self, pose, is_cartesian=True):
        self.joint_states_enabled = False

        # Construct the request
        pose = Pose(position=pose.position, orientation=pose.orientation)
        self.logger.info("Starting move request")

        request = self.get_request(pose, is_cartesian=is_cartesian)

        # Wait for the service to be available
        if not self.move_arm_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("MoveToPose service not available!")
            self.release_processing()  # release lock/flag if we can't even send the request
            return

        # Send async request and handle the response in the callback
        future = self.move_arm_client.call_async(request)
        future.add_done_callback(self.handle_response)

    # Note: do NOT release lock or reset flags here — it should only happen in handle_response


    def handle_response(self, future):
        try:
            result = future.result()
            self.logger.info(f"Move service completed successfully: {result}")
        except Exception as e:
            self.logger.error(f"Move service call failed: {e}")
        finally:
            self.logger.info(f"Finished processing marker with ID: {self.marker_id}")
            self.release_processing()
            
    # def call_service_blocking(self, client, request, timeout_sec=10.0):
    #     if not client.wait_for_service(timeout_sec=2.0):
    #         self.logger.error("Service not available")
    #         return None

    #     future = client.call_async(request)
    #     future.add_done_callback(self.handle_response)

def main():
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()  # Uncomment this if you want to pause execution until the debugger attaches
    print("debugger attached")

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
