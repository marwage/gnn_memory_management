cmake_minimum_required(VERSION 3.17)

project(alzheimer VERSION 0.1
        LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_CXX_FLAGS "-ansi -pedantic-errors -Wall -Wextra")
set(CMAKE_CXX_FLAGS "-ansi -Wall -Wextra")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA 10.2 REQUIRED)
find_package(benchmark REQUIRED)

set(SOURCE_FILES
        src/alzheimer.cpp
        src/tensors.cpp
        src/cuda_helper.cpp
        src/feature_aggregation.cpp
        src/dropout.cpp
        src/linear.cpp
        src/relu.cpp
        src/log_softmax.cpp
        src/loss.cpp
        src/sage_linear.cpp
        src/adam.cpp
        src/mmio_wrapper.cpp
        src/mmio.c
        src/divmv.cu
        src/axpby.cu
        src/invsqrt.cu
        src/elesq.cu
        src/axdy.cu
        src/add.cpp
        src/gpu_memory.cpp
        src/gpu_memory_logger.cpp
        src/sparse_computation.cpp
        src/dense_computation.cpp
        src/chunking.cpp
        src/pipeline.cpp
        src/dataset.cpp)


add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})
target_link_directories(${PROJECT_NAME}
        PRIVATE /usr/local/lib)
target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PUBLIC ${CUDA_INCLUDE_DIRS}
        PUBLIC /usr/include/python3.6m)
target_link_libraries(${PROJECT_NAME}
        cnpy
        cudnn
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include)


include(tests/CMakeLists.txt)


include(benchmark/CMakeLists.txt)
