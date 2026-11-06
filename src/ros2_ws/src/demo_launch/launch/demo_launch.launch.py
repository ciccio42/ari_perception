from launch import LaunchDescription
from ament_index_python import get_package_share_directory
import os
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    
    launch_directory = get_package_share_directory("demo_launch")
    
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_directory, 'launch', 'data_nodes.launch.py'))
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_directory, 'launch', 'ai_nodes.launch.py'))
        ),
    ])