cmake_minimum_required(VERSION 3.17)

project(alzheimer VERSION 0.1
        LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA 10.2 REQUIRED)
find_package(benchmark REQUIRED)

set(SOURCE_FILES
        src/alzheimer.cpp
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
        src/invsqrt.cu
        src/elesq.cu
        src/axdy.cu
        src/add.cpp
        src/gpu_memory.cpp
        src/gpu_memory_logger.cpp
        src/sparse_computation.cpp)


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


set(EXECUTABLE_NAME run_${PROJECT_NAME})
add_executable(${EXECUTABLE_NAME}
        src/alzheimer.cpp
        src/run_alzheimer.cpp)
target_link_libraries(${EXECUTABLE_NAME}
        ${PROJECT_NAME})


include(tests/CMakeLists.txt)


include(benchmark/CMakeLists.txt)
