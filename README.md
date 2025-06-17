# Real Time Ray Tracing Engine

## Get Started

### Local (Simple)

To download and build all dependencies, and compile and build the final project, use the provided script:

[build.sh](https://github.com/prabhask5/real-time-ray-tracing-engine/blob/main/build.sh)

Then to run the compiled binary, use the following command:

```bash
bin/raytracer
```

This engine comes with a CLI to select different rendering options. This includes either rendering a static scene to a .ppm file (static camera), or rendering a dynamic scene that you can move around in to a SDL3 window. This also includes either rendering via CPU or GPU. To view all the CLI options, run the following command:

```bash
bin/raytracer --help
```

### On GPU Instance with Docker (Complex)

My local development environment (Mac) does not come with access to a Nvidia GPU to play around with CUDA on. So I needed to use WSL2 on a Windows computer to be able to use CUDA. Here are the following commands I ran on a WSL2 instance to be able to set up the project via a Docker container for GPU compilation and execution:

```bash
# Update system and install Docker
sudo apt update && sudo apt install -y docker.io
sudo apt upgrade -y

# Install Docker Compose plugin
sudo apt install -y docker-compose-plugin

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-container-toolkit

# Configure NVIDIA runtime for Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Add yourself to Docker group (if not already)
sudo usermod -aG docker $USER
newgrp docker

# Set $DISPLAY inside WSL2 to point to Windows host.
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
export LIBGL_ALWAYS_INDIRECT=1

# Clone repo and build
git clone https://github.com/prabhask5/real-time-ray-tracing-engine.git
cd real-time-ray-tracing-engine

# Build image
docker compose build

# Run container with GPU
docker compose run --gpus all raytracer bash

# Inside container:
xeyes  # To test X11 forwarding.
./build.sh
bin/raytracer [args]
```

To be able to view the SDL3 window on the Windows computer, install [VcXsrv](https://sourceforge.net/projects/vcxsrv/), and start it with "Disable access control" and "OpenGL or native acceleration" ON. 

## Development

This project uses custom githooks to format .cpp/.hpp files, to change the folder used to run githook scripts from use the following commands:

```bash
# Install dependency clang-format (Mac).
brew install clang-format

# Install dependency clang-format (Linux).
sudo apt install -y clang-format

chmod +x .githooks/pre-commit
git config --local core.hooksPath .githooks/
```