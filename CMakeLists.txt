cmake_minimum_required(VERSION 3.15)
project(RealTimeRayTracingEngine LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
        -Wconversion
        -Wsign-conversion
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
        -Wconversion
        -Wsign-conversion
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

include_directories(library)
include_directories(${CMAKE_INSTALL_PREFIX}/include)

set(CMAKE_XCODE_ATTRIBUTE_OTHER_CODE_SIGN_FLAGS "-o linker-signed")

file(GLOB_RECURSE RAY_TRACING_SRC src/*.cpp src/*.hpp)
add_executable(RealTimeRayTracingEngine ${RAY_TRACING_SRC})

# Find dependencies.
find_package(SDL3 REQUIRED)
find_package(SDL3_ttf REQUIRED)

target_link_libraries(RealTimeRayTracingEngine PUBLIC SDL3::SDL3 SDL3_ttf::SDL3_ttf)

install(TARGETS RealTimeRayTracingEngine RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)