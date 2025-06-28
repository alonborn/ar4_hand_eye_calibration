#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from ros2_aruco_interfaces.msg import ArucoMarkers
import copy
import tf_transformations


class ArucoPoseVisualizer(Node):
    """Node to visualize ArUco marker poses published by aruco_node."""

    def __init__(self):
        super().__init__('aruco_pose_visualizer')

        self.bridge = CvBridge()
        self.latest_markers = None
        self.camera_matrix = None
        self.dist_coeffs = None

        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(ArucoMarkers, '/aruco_markers', self.aruco_markers_callback, 10)

        self.image_publisher = self.create_publisher(Image, '/aruco_image', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, '/camera_info', 10)

        self.get_logger().info("ArucoPoseVisualizer initialized.")

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d, dtype=np.float32)
        self.camera_info_msg = msg
        # self.get_logger().info("Camera intrinsics received.")
        self.destroy_subscription(self.camera_info_callback)

    def aruco_markers_callback(self, msg):
        self.latest_markers = msg
        self.get_logger().info(f"Received {len(msg.poses)} markers.")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # If we have marker data, draw axes
        if self.latest_markers is not None and len(self.latest_markers.poses) > 0:
            for i, pose in enumerate(self.latest_markers.poses):
                try:
                    # Approximate rvec from quaternion (not strictly correct, see note)
                    rot_matrix = tf_transformations.quaternion_matrix([
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w
                    ])[:3, :3]
                    rvec, _ = cv2.Rodrigues(rot_matrix)

                    tvec = np.array([
                        [pose.position.x],
                        [pose.position.y],
                        [pose.position.z]
                    ])

                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                except Exception as e:
                    self.get_logger().warn(f"drawFrameAxes failed: {e}")

        # Publish the image (with or without markers)
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            overlay_msg.header = msg.header
            self.image_publisher.publish(overlay_msg)

            if hasattr(self, 'camera_info_msg'):
                cam_info = copy.deepcopy(self.camera_info_msg)
                cam_info.header = msg.header
                self.camera_info_publisher.publish(cam_info)

        except Exception as e:
            self.get_logger().error(f"Failed to publish visualization: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
