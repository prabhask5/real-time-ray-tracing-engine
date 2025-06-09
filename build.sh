#!/bin/bash

exec &>logs/log.txt
set -x 

RTRT_SOURCE_DIR=`pwd`
DEPS_BUILD_DIR=$RTRT_SOURCE_DIR/deps

mkdir -p $DEPS_BUILD_DIR 

if [[ "$OSTYPE" == "darwin"* ]]; then
	CXXFLAGS="-DWAZOO_64_BIT -std=c++11 -stdlib=libc++"
fi

if [ -z "$1" ] 
then
    INSTALL_DIR=$DEPS_BUILD_DIR
    echo "Install directory is not supplied, installing in $INSTALL_DIR/"
else
    INSTALL_DIR=$1
fi

SDL_PKG=3.2.16
SDL_PKG_NAME=SDL3-$SDL_PKG

[[ ! -f $DEPS_BUILD_DIR/$SDL_PKG_NAME.tar.gz ]] && curl -L "https://github.com/libsdl-org/SDL/releases/download/release-$SDL_PKG/$SDL_PKG_NAME.tar.gz"  -o $DEPS_BUILD_DIR/$SDL_PKG_NAME.tar.gz

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

cd $RTRT_SOURCE_DIR

mkdir -p build
cd build

# For MacOS target
cmake -G Xcode .. -DSDL3_DIR=$INSTALL_DIR/lib/cmake/SDL3/ -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

# For Linux target
# cmake .. -DSDL3_DIR=$INSTALL_DIR/lib/cmake/SDL3/ -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

cmake --build . --config Release
cmake --install . --config Release