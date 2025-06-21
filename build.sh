#!/bin/bash

# This script auto-offloads CUDA builds to cloudgpu if nvcc is not present locally.
# If run on the cloud, make sure IS_REMOTE=true is set (automatically done by SSH call).

# Detect if running on remote to avoid recursion.
IS_REMOTE="${IS_REMOTE:-false}"

mkdir -p logs
exec &>logs/log.txt
set -e

# Determine if CUDA should be enabled.
if [[ "$1" == "cuda" ]]; then
    ENABLE_CUDA=ON
elif [[ "$1" == "nocuda" ]]; then
    ENABLE_CUDA=OFF
else
    ENABLE_CUDA=OFF
fi

echo "CUDA support: $ENABLE_CUDA"

if [[ "$ENABLE_CUDA" == "ON" && "$IS_REMOTE" != "true" ]]; then
    # If CUDA is requested but nvcc isn't available, offload to cloud.
    if ! command -v nvcc &> /dev/null; then
        echo "CUDA requested but nvcc not found. Offloading build to cloudgpu..."

        # Run the build remotely.
        set +e # Turn off exiting on any error temporarily in case the ssh command fails.
        ssh cloudgpu "cd /workspace/real-time-ray-tracing-engine && git pull && export IS_REMOTE=true && ./build.sh cuda"
        REMOTE_EXIT_CODE=$?
        set -e

        echo "Syncing log from remote to local logs/cloudlog.txt..."
        mkdir -p logs
        scp cloudgpu:/workspace/real-time-ray-tracing-engine/logs/log.txt logs/cloudlog.txt || echo "Failed to copy logs from cloud."

        if [[ $REMOTE_EXIT_CODE -ne 0 ]]; then
            echo "Remote build failed with exit code $REMOTE_EXIT_CODE."
            exit $REMOTE_EXIT_CODE
        fi

        exit 0
    fi
fi


RTRT_SOURCE_DIR=`pwd`
DEPS_BUILD_DIR=$RTRT_SOURCE_DIR/deps

mkdir -p $DEPS_BUILD_DIR 

INSTALL_DIR=$DEPS_BUILD_DIR

SDL_PKG=3.2.16
SDL_PKG_NAME=SDL3-$SDL_PKG
SDL_TTF_PKG=3.2.2
SDL_TTF_PKG_NAME=SDL3_ttf-$SDL_TTF_PKG

[[ ! -f $DEPS_BUILD_DIR/$SDL_PKG_NAME.tar.gz ]] && curl -L "https://github.com/libsdl-org/SDL/releases/download/release-$SDL_PKG/$SDL_PKG_NAME.tar.gz"  -o $DEPS_BUILD_DIR/$SDL_PKG_NAME.tar.gz
[[ ! -f $DEPS_BUILD_DIR/$SDL_TTF_PKG_NAME.tar.gz ]] && curl -L "https://github.com/libsdl-org/SDL_ttf/releases/download/release-$SDL_TTF_PKG/$SDL_TTF_PKG_NAME.tar.gz" -o $DEPS_BUILD_DIR/$SDL_TTF_PKG_NAME.tar.gz

export INSTALL_PREFIX=$INSTALL_DIR

if [[ ! -f $INSTALL_DIR/include/SDL3/SDL.h ]]
then
cd $DEPS_BUILD_DIR
[[ ! -d $SDL_PKG_NAME ]] && tar xvf $SDL_PKG_NAME.tar.gz
cd $SDL_PKG_NAME
mkdir -p build; cd build;
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" -DBUILD_SHARED_LIBS=ON -DSDL_TEST=OFF -DSDL_INSTALL_TESTS=OFF
cmake --build . --target install
fi

if [[ ! -f $INSTALL_DIR/include/SDL3_ttf/SDL_ttf.h ]]
then
cd $DEPS_BUILD_DIR
[[ ! -d $SDL_TTF_PKG_NAME ]] && tar xvf $SDL_TTF_PKG_NAME.tar.gz
cd $SDL_TTF_PKG_NAME
mkdir -p build; cd build;
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" -DSDL3TTF_SAMPLES=OFF -DSDL3TTF_TESTS=OFF
cmake --build . --target install
fi

cd $RTRT_SOURCE_DIR

mkdir -p build
cd build

CMAKE_GENERATOR=""
CMAKE_CMD_ARGS=(
  -DSDL3_DIR=$INSTALL_DIR/lib/cmake/SDL3/
  -DSDL3_ttf_DIR=$INSTALL_DIR/lib/cmake/SDL3_ttf/
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
  -DCMAKE_PREFIX_PATH=$INSTALL_DIR
  -DENABLE_CUDA=$ENABLE_CUDA
)

if [[ "$(uname)" == "Darwin" ]]; then
    CMAKE_GENERATOR="-G Xcode"
fi

cmake $CMAKE_GENERATOR .. "${CMAKE_CMD_ARGS[@]}"

cmake --build . --config Release
cmake --install . --config Release