set(TEST_NAME test)

add_executable(${TEST_NAME}
        tests/tests.cpp
        tests/divmv.cpp
        ${SOURCE_FILES})

target_include_directories(${TEST_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}/include/alzheimer
        PRIVATE /usr/local/cuda/include)

target_link_directories(${TEST_NAME}
        PRIVATE /usr/local/lib)

target_link_libraries(${TEST_NAME}
        cnpy
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        cudnn
        ${CUDA_CUBLAS_LIBRARIES})
