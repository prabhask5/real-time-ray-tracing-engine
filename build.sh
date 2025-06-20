#!/bin/bash

mkdir -p logs
exec &>logs/log.txt
set -e
set -x 

# Determine if CUDA should be enabled.
if [[ "$1" == "cuda" ]]; then
    ENABLE_CUDA=ON
elif [[ "$1" == "nocuda" ]]; then
    ENABLE_CUDA=OFF
else
    if command -v nvcc &> /dev/null; then
        ENABLE_CUDA=ON
    else
        ENABLE_CUDA=OFF
    fi
fi

echo "CUDA support: $ENABLE_CUDA"

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