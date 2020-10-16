set(TEST_NAME test_integration)

add_executable(${TEST_NAME}
        tests/test_integration.cpp
        tests/divmv.cpp
        tests/log_softmax.cpp
        tests/loss.cpp
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


set(TEST_NAME test_integration_subset)

add_executable(${TEST_NAME}
        tests/test_integration_subset.cpp
        tests/divmv.cpp
        tests/log_softmax.cpp
        tests/loss.cpp
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