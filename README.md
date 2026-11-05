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
        --runtime nvidia \
        --network host \
        -v "$(pwd)":/workspace \
        pepper_demo:r35 \
        bash
```

`--runtime nvidia` is required: without it, `cv2` and `cv_bridge` crash on import and CUDA is not available.

ROS 2 is not sourced by default, so run this inside the container:

```bash
source /opt/ros/humble/install/setup.bash
```

# Run Demo


# Useful commands
```bash
# Remove ```bashall build cache, including entries still reusable by images
docker builder prune -a

# Also remove all images not used by a container
docker system prune -a
```