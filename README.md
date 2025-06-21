# Real Time Ray Tracing Engine

## Get Started

### Local (Simple)

To download and build all dependencies, and compile and build the final project, use the provided script:

[build.sh](https://github.com/prabhask5/real-time-ray-tracing-engine/blob/main/build.sh)

Then to run the compiled binary, use the following command (more information on why there's a bash file wrapper here in the section "Using GPU Instance JUST For Builds and Execution Testing"):

```bash
./run.sh
```

This engine comes with a CLI to select different rendering options. This includes either rendering a static scene to a .ppm file (static camera), or rendering a dynamic scene that you can move around in to a SDL3 window. This also includes either rendering via CPU or GPU. To view all the CLI options, run the following command:

```bash
./run.sh --help
```

### On GPU Instance with Docker (Complex)

My local development environment does not come with access to a Nvidia GPU to play around with CUDA on. So I needed to provision a cloud instance with a GPU, and needed to use Docker to setup the environment. Here are the following commands I ran on a fresh cloud instance to be able to set up the project for GPU compilation and execution:

```bash
# Update system and install Docker.
sudo apt update && sudo apt install -y docker.io
sudo apt upgrade -y

# Add Docker's official GPG key.
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources.
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker Compose plugin.
sudo apt update && sudo apt install -y docker-compose-plugin

# Clone repo and build.
git clone https://github.com/prabhask5/real-time-ray-tracing-engine.git
cd real-time-ray-tracing-engine

# Grant X11 Access to Docker (Optional if trying to make an SDL3 window work- VERY HARD).
xhost +local:root

# Make sure the daemon is running.
sudo dockerd &

# Build image.
docker compose build

# Launch container.
docker compose up -d

# Enter the container.
docker exec -it raytracer bash

# Inside container:
chmod +x build.sh
./build.sh
bin/raytracer [args]
```

OR if you cannot use Docker (in my case this is because the VM was already booted into a Docker container and Docker in Docker was not enabled), use this command to install all the dependencies needed for the project:

```bash
sudo apt update && sudo apt install -y cmake build-essential git curl libx11-dev libgl1-mesa-dev libfreetype6-dev libsdl2-dev
```

#### Using GPU Instance JUST For Builds and Execution Testing

To save money on usages for the GPU cloud instance, and to stop me having to go all the way into VSCode and the Git repository just to build and compile and run the project, I included fallback workflows to only use the cloud instance when the GPU is needed. To set this up for your use case, do the following (I assume you have found a cloud instance to use and set it up with the commands above):
- In \~/.ssh/config, create a new alias called "cloudgpu".
- In your cloud instance, clone the repository at the directory "\~/workspace/real-time-ray-tracing-engine".

#### Building

I integrated a cloud fallback into build.sh in the case that CUDA compiling was ON (if cuda or no argument was provided) and nvcc was not available on the local machine. The build script then ssh's into the "cloudgpu" alias, cds into the specific repo directory "/workspace/real-time-ray-tracing-engine", pulls all the new changes from Github, and runs build.sh. Then the output logs from build.sh and pipes into the local repostiory at logs/cloudlog.txt. 

#### Running

I made a new unified workflow bash file, run.sh to replace directly calling bin/raytracer. Call ./run.sh [args] with the regular bin/raytracer arguments to execute the local version of bin/raytracer. Call ./run.sh cloud [args] with the regular bin/raytracer arguments to execute the cloud version of bin/raytracer, then pipe the generated .ppm file (if created) onto the local workspace.

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