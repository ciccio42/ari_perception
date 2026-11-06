from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os
from pathlib import Path


def generate_launch_description():
    
    launch_directory = get_package_share_directory("demo_launch")
    ai_params_yaml = os.path.join(launch_directory, "configs", "ai_nodes_params.yaml")
    assert Path(ai_params_yaml).exists()

    return LaunchDescription([

        #Face Detection Node
        Node(
            package="asd",
            executable='face_detection_node',
            name='face_detection_node',
            output='screen',
            parameters=[ai_params_yaml]
        ),
        
        #Asd Node
        Node(
            package="asd",
            executable='asd_node',
            name='asd_node',
            output='screen',
            parameters=[ai_params_yaml]
        ),

        #Gaze
        Node(
            package="gaze",
            executable='gaze_estimation_node',
            name='gaze_estimation_node',
            output='screen',
            parameters=[ai_params_yaml]
        ),

        #Soft Biometrics
        Node(
            package="soft_biometrics",
            executable='soft_biometrics_node',
            name='soft_biometrics_node',
            output='screen',
            parameters=[ai_params_yaml]
        ),

        #Object Detector
        Node(
           package="object_detector",
           executable='obj_detector_node',
           name='obj_detector_node',
           output='screen',
           parameters=[ai_params_yaml]
        ),

        # Connect Rock Paper Scissor Hand Node
        Node(
           package="game_rock_paper_scissor",
            executable='image_hand_node',
            name='image_hand_node',
            output='screen',
            parameters=[ai_params_yaml],
        ),

        #Connect Four Hand Node
        Node(
            package="game_connect_four",
            executable='image_hand_fingers_node',
            name='image_hand_fingers_node',
            output='screen',
            parameters=[ai_params_yaml],  
        ),

        #ASR Node
        Node(
            package="asr",
            executable='asr_node',
            name='asr_node',
            output='screen',
            parameters=[ai_params_yaml],
        )
    ])