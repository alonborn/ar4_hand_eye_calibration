#!/usr/bin/env python3
import numpy as np
import rclpy
import tf2_ros
from easy_handeye2.handeye_calibration import load_calibration
from geometry_msgs.msg import Transform, TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from rclpy.executors import ExternalShutdownException
from rclpy.node import ParameterType, ParameterDescriptor
from std_srvs.srv import Trigger
from transforms3d.quaternions import quat2mat, mat2quat


def transform_to_matrix(transform):
    """Converts a Transform message to a 4x4 transformation matrix using transforms3d."""
    # Extract translation
    translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    # Extract quaternion (note that transforms3d uses [w, x, y, z] order)
    rotation = np.array([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z])
    # Convert quaternion to rotation matrix
    rotation_matrix = quat2mat(rotation)
    # Create the 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def matrix_to_transform(matrix):
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    rotation = mat2quat(rotation_matrix)
    transform = Transform()
    transform.translation.x = translation[0]
    transform.translation.y = translation[1]
    transform.translation.z = translation[2]
    transform.rotation.w = rotation[0]
    transform.rotation.x = rotation[1]
    transform.rotation.y = rotation[2]
    transform.rotation.z = rotation[3]
    return transform


class HandeyePublisher(Node):
    def __init__(self):
        super().__init__('handeye_publisher')

        # Load calibration name from parameter
        self.calibration_name = self.declare_parameter(
            'calibration_name', '',
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        ).get_parameter_value().string_value

        self.get_logger().info(f'Loading the calibration with name {self.calibration_name}')
        self.calibration = load_calibration(self.calibration_name)
        self.parameters = self.calibration.parameters

        if self.parameters.calibration_type == 'eye_in_hand':
            self.orig = self.parameters.robot_effector_frame
        else:
            self.orig = self.parameters.robot_base_frame
        self.dest = "camera_link"  # fixed to "camera_link"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.calibration_tf = TransformStamped()
        self.calibration_tf.header.frame_id = self.orig
        self.calibration_tf.child_frame_id = self.dest

        # Initialize timers and services
        self.compute_transform_timer = self.create_timer(0.1, self.compute_transform)

        self.srv = self.create_service(
            Trigger,
            'refresh_handeye_transform',
            self.refresh_transform_callback
        )

    def compute_transform(self):
        try:
            color_to_link_tf = self.tf_buffer.lookup_transform(
                target_frame=self.parameters.tracking_base_frame,
                source_frame="camera_link",
                time=Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().error('Could not get transform from tracking_base_frame to camera_link')
            return

        base_to_color_mat = transform_to_matrix(self.calibration.transform)
        color_to_link_mat = transform_to_matrix(color_to_link_tf.transform)
        base_to_link_mat = np.dot(base_to_color_mat, color_to_link_mat)

        self.calibration_tf.transform = matrix_to_transform(base_to_link_mat)
        self.get_logger().info('Computed the robot base to camera transform')

        self.compute_transform_timer.cancel()
        self.publish_transform_timer = self.create_timer(0.1, self.publish_transform)

    def publish_transform(self):
        self.calibration_tf.header.stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform(self.calibration_tf)

    def refresh_transform_callback(self, request, response):
        try:
            self.get_logger().info("Reloading calibration file and refreshing transform")
            self.calibration = load_calibration(self.calibration_name)
            self.parameters = self.calibration.parameters
            # Recompute and republish the transform on next timer tick
            self.compute_transform()
            response.success = True
            response.message = "Transform successfully refreshed from updated calibration file."
        except Exception as e:
            response.success = False
            response.message = f"Failed to refresh transform: {e}"
        return response


def main(args=None):
    rclpy.init(args=args)
    handeye_publisher = HandeyePublisher()

    try:
        rclpy.spin(handeye_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        handeye_publisher.destroy_node()


if __name__ == '__main__':
    main()
