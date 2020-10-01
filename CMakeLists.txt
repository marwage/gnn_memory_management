set( CMAKE_VERBOSE_MAKEFILE on )

cmake_minimum_required(VERSION 3.17)

project(alzheimer VERSION 0.1
    LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    src/read_tensors.cpp
    src/mmio_wrapper.cpp
    src/mmio.c)

# include_directories(${PROJECT_NAME}
    # PRIVATE ${PROJECT_SOURCE_DIR}/include/alzheimer)

target_include_directories(${PROJECT_NAME}
    PRIVATE ${PROJECT_SOURCE_DIR}/include/alzheimer
    PRIVATE /usr/local/cuda/include)

find_package(CUDA 10.2 REQUIRED)

target_link_libraries(${PROJECT_NAME}
    cnpy
    ${CUDA_LIBRARIES})

