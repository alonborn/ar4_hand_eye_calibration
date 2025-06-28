#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from collections import defaultdict, deque
from geometry_msgs.msg import Point
import copy

class ArucoPoseEstimator(Node):
    """Node to estimate the pose of ArUco markers in the camera image."""

    def __init__(self):
        super().__init__('aruco_pose_estimator')

        self.bridge = CvBridge()

        # Image subscriber
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            20)

        # CameraInfo subscriber
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10)
        self.camera_info_received = False

        # Pose publisher
        self.pose_publisher = self.create_publisher(Point, '/aruco_pose', 10)

        # Camera parameters (filled from CameraInfo)
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Image publisher
        self.image_publisher = self.create_publisher(Image, '/aruco_image', 10)
        self.get_logger().info("Initialization complete.")

        self.pose_buffer_size = 5
        self.tvecs_buffer = defaultdict(lambda: deque(maxlen=self.pose_buffer_size))
        self.rvecs_buffer = defaultdict(lambda: deque(maxlen=self.pose_buffer_size))
        self.camera_info_publisher = self.create_publisher(CameraInfo, '/camera_info', 10)

        # Define marker size (in meters)
        self.marker_size = 0.1

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float32)

        self.camera_info_received = True
        self.get_logger().info("Camera info received and parameters set.")

        self.destroy_subscription(self.camera_info_subscription)
        self.camera_info_msg = msg
        self.camera_info_publisher.publish(msg)

    def image_callback(self, msg):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for camera info...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        corners, ids, _ = self.aruco_detector.detectMarkers(cv_image)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i]

                # Prepare object points (marker corners in 3D)
                obj_points = np.array([
                    [-self.marker_size / 2,  self.marker_size / 2, 0],
                    [ self.marker_size / 2,  self.marker_size / 2, 0],
                    [ self.marker_size / 2, -self.marker_size / 2, 0],
                    [-self.marker_size / 2, -self.marker_size / 2, 0]
                ], dtype=np.float32)

                # Reshape image points
                img_points = corner.reshape((4, 2)).astype(np.float32)

                retval, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    img_points,
                    self.camera_matrix,
                    self.dist_coeffs
                )

                if not retval:
                    self.get_logger().warn(f"Pose estimation failed for marker {marker_id}")
                    continue

                # Flatten for consistency
                tvec = tvec.flatten()
                rvec = rvec.flatten()

                self.tvecs_buffer[marker_id].append(tvec)
                self.rvecs_buffer[marker_id].append(rvec)

                avg_tvec = np.mean(self.tvecs_buffer[marker_id], axis=0)
                avg_rvec = np.mean(self.rvecs_buffer[marker_id], axis=0)

                # Draw detected markers and pose
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, avg_rvec, avg_tvec, 0.1)

                # Publish smoothed position
                point_msg = Point(
                    x=float(avg_tvec[0]),
                    y=float(avg_tvec[1]),
                    z=float(avg_tvec[2])
                )
                self.pose_publisher.publish(point_msg)

        # Publish overlay image and camera info with correct timestamp
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            overlay_msg.header.stamp = msg.header.stamp
            overlay_msg.header.frame_id = msg.header.frame_id or "camera_color_optical_frame"
            self.image_publisher.publish(overlay_msg)

            if hasattr(self, 'camera_info_msg'):
                camera_info_copy = copy.deepcopy(self.camera_info_msg)
                camera_info_copy.header.stamp = msg.header.stamp
                camera_info_copy.header.frame_id = msg.header.frame_id or "camera_color_optical_frame"
                self.camera_info_publisher.publish(camera_info_copy)

        except Exception as e:
            self.get_logger().error(f"Error publishing overlay image or camera info: {e}")


def main(args=None):
    rclpy.init(args=args)
    aruco_pose_estimator = ArucoPoseEstimator()
    rclpy.spin(aruco_pose_estimator)
    aruco_pose_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
