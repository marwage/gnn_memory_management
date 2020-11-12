cmake_minimum_required(VERSION 3.17)

project(alzheimer VERSION 0.1
        LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_VERBOSE_MAKEFILE ON)

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
        include/alzheimer/sage_linear_adam.hpp
        include/alzheimer/mmio_wrapper.hpp
        include/alzheimer/mmio.h
        include/alzheimer/divmv.h
        include/alzheimer/axpby.h
        include/alzheimer/invsqrt.h
        include/alzheimer/elesq.h
        include/alzheimer/axdy.h)

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
        src/invsqrt.cu
        src/elesq.cu
        src/axdy.cu)


add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}/include/alzheimer
        PUBLIC ${CUDA_INCLUDE_DIRS}
        PUBLIC /usr/include/python3.6m)

target_link_directories(${PROJECT_NAME}
        PRIVATE /usr/local/lib)

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
        src/alzheimer.cpp)

target_link_libraries(${EXECUTABLE_NAME}
        ${PROJECT_NAME})


set(EXECUTABLE_NAME run_${PROJECT_NAME}_chunked)

add_executable(${EXECUTABLE_NAME}
        src/alzheimer_chunked.cpp)

target_link_libraries(${EXECUTABLE_NAME}
        ${PROJECT_NAME})


include(tests/CMakeLists.txt)
