# Build Ari-Perception Docker

Build the image on the Jetson (arm64) with internet access, from the repository root:

```bash
docker build -f .devcontainer/Dockerfile-jetson-r35 -t pepper_demo:r35 .devcontainer
```

# Run Ari-Perception Docker

Start a shell in the container from the repository root:

```bash
docker run -it \
        --rm \
        --name='ari_perception' \
        --runtime=nvidia \
        --gpus=all \
        --net=host \
        --ipc=host \
        -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
        -e DEMO_LANGUAGE='Italian' \
        -v "$(pwd)/src":/workspace \
        pepper_demo:r35 \
        bash
```

`--runtime nvidia` is required: without it, `cv2` and `cv_bridge` crash on import and CUDA is not available.

The entrypoint sources ROS 2, then runs the given command.

# Run Demo

```bash
cd /workspace/ros2_ws
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
ros2 daemon stop
ros2 launch demo_launch demo_launch.launch.py
```

# Useful commands
```bash
# Remove ```bashall build cache, including entries still reusable by images
docker builder prune -a

# Also remove all images not used by a container
docker system prune -a
```