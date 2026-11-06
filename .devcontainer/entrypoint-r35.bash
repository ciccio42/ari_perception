#!/bin/bash
set -e

# Nell'immagine dustynv ROS sta in install/, a differenza dell'installazione da apt
source /opt/ros/humble/install/setup.bash

# if [ ! -d /workspace/ros2_ws/install ]; then
#     cd /workspace/ros2_ws
#     colcon build --symlink-install --packages-up-to \
#         messages demo_launch data_gathering asd gaze soft_biometrics \
#         object_detector game_rock_paper_scissor game_connect_four asr
# fi

# source /workspace/ros2_ws/install/setup.bash
# echo "Starting ROS2 demo launch file..."
# cd /workspace
# exec ros2 launch demo_launch demo_launch.launch.py

exec "$@"
