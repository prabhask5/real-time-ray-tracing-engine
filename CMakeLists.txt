option(ENABLE_CUDA "Enable CUDA support" ON)

cmake_minimum_required(VERSION 3.20)
project(RealTimeRayTracingEngine LANGUAGES CXX)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}")

if (ENABLE_CUDA)
    enable_language(CUDA)
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard
if (ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set the output directory for built objects and rpath for dylib files.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Enable recommended warnings for Clang and GCC
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wold-style-cast
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
        -Wcast-align
        -Wmissing-declarations
        -Wzero-as-null-pointer-constant
        -Wuseless-cast
        -Wno-unused-parameter
    )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wold-style-cast
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough=3
        -Wcast-align
        -Wmissing-declarations
        -Wzero-as-null-pointer-constant
        -Wunused
        -Wno-unused-parameter
    )
endif()

include_directories(${CMAKE_INSTALL_PREFIX}/include)

set(CMAKE_XCODE_ATTRIBUTE_OTHER_CODE_SIGN_FLAGS "-o linker-signed")

file(GLOB_RECURSE CPP_SRC src/*.cpp src/*.hpp)
if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SRC src/*.cu src/*.cuh)
endif()

add_executable(RealTimeRayTracingEngine ${CPP_SRC} $<IF:$<BOOL:${ENABLE_CUDA}>,${CUDA_SRC},>)
if (ENABLE_CUDA)
    set_target_properties(RealTimeRayTracingEngine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
    )
endif()

if (ENABLE_CUDA)
    target_compile_definitions(RealTimeRayTracingEngine PUBLIC USE_CUDA)
endif()

# Find dependencies.
find_package(SDL3 REQUIRED)
find_package(SDL3_ttf REQUIRED)

target_link_libraries(RealTimeRayTracingEngine PUBLIC SDL3::SDL3 SDL3_ttf::SDL3_ttf)

set_target_properties(RealTimeRayTracingEngine PROPERTIES OUTPUT_NAME "raytracer")
install(TARGETS RealTimeRayTracingEngine RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)