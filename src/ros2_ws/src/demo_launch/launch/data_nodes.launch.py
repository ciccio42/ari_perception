from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os
from pathlib import Path

def generate_launch_description():
    
    launch_directory = get_package_share_directory("demo_launch")
    params_yaml = os.path.join(launch_directory, "configs", "data_nodes_params.yaml")
    assert Path(params_yaml).exists()

    return LaunchDescription([

        #Realsense
        # Node(
        #     package="data_gathering",
        #      executable='image_input_node', 
        #      name='image_input_node', 
        #      output='screen',
        #      parameters=[params_yaml]
        # ),

        #Respeaker
        Node(
            package="data_gathering",
             executable='image_ari_input_node', 
             name='image_ari_input_node', 
             output='screen',
             parameters=[params_yaml]
        ),
        # #Mapping
        # Node(
        #     package= "respeaker_ros2", 
        #     executable='audio_conversion_node',
        #     name='audio_conversion_node',
        #     parameters=[params_yaml], 
        #     output='screen',
        # ),

        # #Respeaker
        # Node(
        #     package='respeaker_ros2',
        #     executable='respeaker_node',
        #     name='respeaker_node',
        #     output='screen',
        #     parameters=[params_yaml],
        #      remappings=[
        #          ('/audio', '/audio/respeaker'),
        #          ('/speech_audio', '/speech_audio/respeaker'),
        #          ('/sound_direction', '/sound_direction/respeaker'),
        #      ],
        #      respawn=True,
        #      respawn_delay=2.0
        # ),

        #Models Data Sync
        Node(
            package="data_gathering",
            executable='model_data_sync_node',
            name='model_data_sync_node',
            output='screen',
            parameters=[params_yaml]
        ),

        #View Node
         Node(
            package="data_gathering",
            executable='view_node',
            name='view_node',
            output='screen',
            parameters=[params_yaml]
        ),

    ])