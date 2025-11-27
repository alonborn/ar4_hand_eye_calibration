import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
  
    visualize_aruco = Node(
        package="ar4_hand_eye_calibration",
        executable="visualize_aruco_marker.py",
        name="visualize_aruco_marker",
        output="screen",
    )
    
   
    ld = LaunchDescription()
    #ld.add_action(realsense)
    #ld.add_action(static_tf_publisher)
    #ld.add_action(ar_moveit)
    #ld.add_action(aruco_recognition_node)
    ld.add_action(visualize_aruco)
    #ld.add_action(easy_handeye2)
    return ld
