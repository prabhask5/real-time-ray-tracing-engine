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

### On Service Instance with Docker (Complex)

My local development environment does not come with access to a Nvidia GPU to play around with CUDA on. So I needed to provision a AWS cloud instance with a GPU, and needed to use Docker to setup the environment. Here are the following commands I ran on a fresh cloud instance to be able to set up the project for GPU compilation and execution:

```bash
# Update system and install Docker.
sudo apt update && sudo apt install -y docker.io

# Set up NVIDIA Container Toolkit.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Add yourself to docker group to avoid sudo.
sudo usermod -aG docker $USER
newgrp docker

# Clone the Github repo (requires setting up and figuring out Git beforehand).
git clone https://github.com/prabhask5/real-time-ray-tracing-engine.git
cd real-time-ray-tracing-engine

# Build the Docker image with CUDA and your dependencies, and start an interactive container terminal to compile and execute the project through.
docker-compose build
docker-compose run raytracer bash

# Inside the resulting container shell:
./build.sh
bin/raytracer [args]
```

To be able to view the SDL3 window from your ssh terminal session on MacOS (you need to have XQuartz installed, running, and disable access control first):

```bash
ssh -X -i <.PEM FILE> ubuntu@<EC2_PUBLIC_IP>
```

## Development

This project uses custom githooks to format .cpp/.hpp files, to change the folder used to run githook scripts from use the following commands:

```bash
# Install dependency clang-format
brew install clang-format

chmod +x .githooks/pre-commit
git config --local core.hooksPath .githooks/
```