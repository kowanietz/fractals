cmake_minimum_required(VERSION 3.18)
project(fractals LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# OpenGL library
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

set(SOURCES
        src/main.cpp
        src/utils/glad.cpp
        src/kernels/mandelbrot.cu
        src/kernels/julia.cu
)

add_executable(fractal ${SOURCES})

target_link_libraries(fractal
        ${GLFW_LIBRARIES}
        GL
        dl
        pthread
        cuda
)
