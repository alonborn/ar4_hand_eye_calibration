#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from collections import defaultdict, deque
from geometry_msgs.msg import Point
import debugpy
import copy

class ArucoPoseEstimator(Node):
    """Node to estimate the pose of ArUco markers in the camera image.
    
    Run the following command to visualize the output:
    
    ros2 run image_view image_view image:=/aruco_image 
    """

    def __init__(self):
        super().__init__('aruco_pose_estimator')

        self.bridge = CvBridge()

        # Image subscriber
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Correct topic
            self.image_callback2,
            10)

        # CameraInfo subscriber for color camera
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',  # Correct topic
            self.camera_info_callback,
            10)
        self.camera_info_received = False  # Flag to track if we have camera info


        # Service to provide latest Aruco marker pose
        self.pose_publisher = self.create_publisher(Point, '/aruco_pose', 10)

        # Initialize camera parameters (will be filled from CameraInfo)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Aruco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Image publisher
        self.image_publisher = self.create_publisher(Image, '/aruco_image', 10)
        self.get_logger().info(f"initialization complete")


        self.pose_buffer_size = 30  # Or whatever you want
        self.tvecs_buffer = defaultdict(lambda: deque(maxlen=self.pose_buffer_size))
        self.rvecs_buffer = defaultdict(lambda: deque(maxlen=self.pose_buffer_size))
        self.camera_info_publisher = self.create_publisher(CameraInfo, '/camera_info', 10)


    def camera_info_callback(self, msg):
        # Extract camera matrix and distortion coefficients from CameraInfo message
        K = np.array(msg.k).reshape(3, 3)  # Intrinsic parameters
        D = np.array(msg.d)  # Distortion coefficients

        self.camera_matrix = K
        self.dist_coeffs = D

        self.camera_info_received = True  # Set the flag
        self.get_logger().info("Camera info received and parameters set.")

        # Unsubscribe after receiving the information (optional, but good practice)
        self.destroy_subscription(
            self.camera_info_subscription)  # No need to keep listening
        self.camera_info_msg = msg  # Save the latest one
        self.camera_info_publisher.publish(msg)


    def image_callback(self, msg):
        if not self.camera_info_received:  # Don't process until we have camera info
            self.get_logger().warn("Waiting for camera info...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict)

        if ids is not None:


            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.1, self.camera_matrix,
                self.dist_coeffs)  # 0.1 is marker size
            
            # for i, marker_id in enumerate(ids.flatten()):
            #     # This is the 3D center of the marker in camera coordinate system
            #     center_3d = tvecs[i][0]  # shape: (3,)
            #     x, y, z = center_3d
            #     self.get_logger().info(f"ID {marker_id} cen:x={x:.3f},y={y:.3f},z={z:.3f}")

            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                cv2.drawFrameAxes(cv_image, self.camera_matrix,
                                  self.dist_coeffs, rvecs[i], tvecs[i], 0.1)
                                  

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.image_publisher.publish(overlay_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")


    def image_callback2(self, msg):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for camera info...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.1, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                self.tvecs_buffer[marker_id].append(tvecs[i][0])
                self.rvecs_buffer[marker_id].append(rvecs[i][0])

                avg_tvec = np.mean(self.tvecs_buffer[marker_id], axis=0)
                avg_rvec = np.mean(self.rvecs_buffer[marker_id], axis=0)

                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, avg_rvec, avg_tvec, 0.1)

                point_msg = Point(x=float(avg_tvec[0]), y=float(avg_tvec[1]), z=float(avg_tvec[2]))
                self.pose_publisher.publish(point_msg)

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            overlay_msg.header.stamp = msg.header.stamp
            overlay_msg.header.frame_id = msg.header.frame_id or "camera_color_optical_frame"  # fallback default
            self.image_publisher.publish(overlay_msg)

            if hasattr(self, 'camera_info_msg'):
                # ✅ Deep copy to avoid mutating original
                camera_info_copy = copy.deepcopy(self.camera_info_msg)
                camera_info_copy.header.stamp = msg.header.stamp
                camera_info_copy.header.frame_id = msg.header.frame_id or "camera_color_optical_frame"
                self.camera_info_publisher.publish(camera_info_copy)

        except Exception as e:
            self.get_logger().error(f"Error publishing overlay image or camera info: {e}")



def main(args=None):
    # debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # Ensures the debugger connects before continuing
    # print("Debugger connected.")

    rclpy.init(args=args)
    aruco_pose_estimator = ArucoPoseEstimator()
    rclpy.spin(aruco_pose_estimator)
    aruco_pose_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
