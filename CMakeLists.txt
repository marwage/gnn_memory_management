set(CMAKE_VERBOSE_MAKEFILE on)

cmake_minimum_required(VERSION 3.17)

project(alzheimer VERSION 0.1
    LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME}
    src/alzheimer.cpp
    src/mmio_wrapper.cpp
    src/mmio.c)

target_include_directories(${PROJECT_NAME}
    PRIVATE ${PROJECT_SOURCE_DIR}/include/alzheimer
    PRIVATE /usr/local/cuda/include)

target_link_directories(${PROJECT_NAME}
    PRIVATE /usr/local/lib)

find_package(CUDA 10.2 REQUIRED)

target_link_libraries(${PROJECT_NAME}
    cnpy
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    cudnn)

