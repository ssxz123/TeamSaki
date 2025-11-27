from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('team_challenge'),
        'config',
        'params.yaml'
    )
    vision_node = Node(
        package="team_challenge",
        executable="vision_node",
        name='vision_node',
        parameters=[config_file]
    )
    shooter_node = Node(
        package="team_challenge",
        executable="shooter_node",
        name='shooter_node', 
        parameters=[config_file]
    )
    launch_description = LaunchDescription([
        vision_node,
        shooter_node
    ])
    return launch_description