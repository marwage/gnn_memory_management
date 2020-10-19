set(CMAKE_VERBOSE_MAKEFILE on)

cmake_minimum_required(VERSION 3.17)

project(alzheimer VERSION 0.1
        LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA 10.2 REQUIRED)

set(HEADER_FILES
        include/alzheimer/tensors.hpp
        include/alzheimer/cuda_helper.hpp
        include/alzheimer/graph_convolution.hpp
        include/alzheimer/dropout.hpp
        include/alzheimer/linear.hpp
        include/alzheimer/activation.hpp
        include/alzheimer/loss.hpp
        include/alzheimer/sage_linear.hpp
        include/alzheimer/adam.hpp
        include/alzheimer/mmio_wrapper.hpp
        include/alzheimer/mmio.h
        include/alzheimer/divmv.h
        include/alzheimer/axpby.h
        include/alzheimer/invsqrt.h
        include/alzheimer/elesq.h)

set(SOURCE_FILES
        src/tensors.cpp
        src/cuda_helper.cpp
        src/graph_convolution.cpp
        src/dropout.cpp
        src/linear.cpp
        src/activation.cpp
        src/loss.cpp
        src/sage_linear.cpp
        src/adam.cpp
        src/mmio_wrapper.cpp
        src/mmio.c
        src/divmv.cu
        src/axpby.cu
        src/invsqrt.cu src/elesq.cu src/axdy.cu)

add_executable(${PROJECT_NAME}
        src/alzheimer.cpp
        ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}/include/alzheimer
        PRIVATE /usr/local/cuda/include)

target_link_directories(${PROJECT_NAME}
        PRIVATE /usr/local/lib)

target_link_libraries(${PROJECT_NAME}
        cnpy
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        cudnn
        ${CUDA_CUBLAS_LIBRARIES})

include(tests/CMakeLists.txt)
