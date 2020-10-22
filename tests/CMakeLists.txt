set(TEST_NAME test_integration)

add_executable(${TEST_NAME}
        tests/integration.cpp
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
        tests/integration_subset.cpp
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


set(TEST_NAME test_divmv)

add_executable(${TEST_NAME}
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


set(TEST_NAME test_log_softmax)

add_executable(${TEST_NAME}
        tests/log_softmax.cpp
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


set(TEST_NAME test_loss)

add_executable(${TEST_NAME}
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


set(TEST_NAME test_dropout)

add_executable(${TEST_NAME}
        tests/dropout.cpp
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


set(TEST_NAME test_relu)

add_executable(${TEST_NAME}
        tests/relu.cpp
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

set(TEST_NAME test_axpby)

add_executable(${TEST_NAME}
        tests/axpby.cpp
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


set(TEST_NAME test_invsqrt)

add_executable(${TEST_NAME}
        tests/invsqrt.cpp
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


set(TEST_NAME test_elesq)

add_executable(${TEST_NAME}
        tests/elesq.cpp
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


set(TEST_NAME test_tensors)

add_executable(${TEST_NAME}
        tests/tensors.cpp
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

