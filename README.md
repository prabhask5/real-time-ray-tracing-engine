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
sudo apt upgrade -y

# Add the graphics driver PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install gcc (NVIDIA driver dependency).
sudo apt install -y gcc

# Install the latest NVIDIA driver.
sudo apt install -y nvidia-driver-570
sudo reboot

# Set up NVIDIA Container Toolkit.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-container-toolkit

# Add NVIDIA runtime config to Docker.
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker

# Add yourself to docker group to avoid sudo.
sudo usermod -aG docker $USER
newgrp docker

# Clone the Github repo (requires setting up and figuring out Git beforehand).
git clone https://github.com/prabhask5/real-time-ray-tracing-engine.git
cd real-time-ray-tracing-engine

# Install docker-compose
sudo apt install docker-compose

# Build the Docker image with CUDA and your dependencies.
docker-compose build

# Clean up docker.
docker container prune -f
docker image prune -f
docker volume prune -f
docker network prune -f

# Start an interactive container terminal to compile and execute the project through.
docker-compose run raytracer bash


# Inside the resulting container shell:
./build.sh
bin/raytracer [args]
```

To be able to view the SDL3 window from your ssh terminal session on MacOS (you need to have XQuartz installed, running, and disable access control first) run the following command from the XQuartz terminal:

```bash
ssh -Y -i <.PEM FILE> ubuntu@<EC2_PUBLIC_IP>
```

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