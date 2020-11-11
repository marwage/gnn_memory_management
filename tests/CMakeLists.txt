set(TEST_NAME test_integration)
add_executable(${TEST_NAME}
        tests/integration.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_integration_subset)
add_executable(${TEST_NAME}
        tests/integration_subset.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_divmv)
add_executable(${TEST_NAME}
        tests/divmv.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_log_softmax)
add_executable(${TEST_NAME}
        tests/log_softmax.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_loss)
add_executable(${TEST_NAME}
        tests/loss.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_dropout)
add_executable(${TEST_NAME}
        tests/dropout.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_relu)
add_executable(${TEST_NAME}
        tests/relu.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_axpby)
add_executable(${TEST_NAME}
        tests/axpby.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_invsqrt)
add_executable(${TEST_NAME}
        tests/invsqrt.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_elesq)
add_executable(${TEST_NAME}
        tests/elesq.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_tensors)
add_executable(${TEST_NAME}
        tests/tensors.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_graph_conv)
add_executable(${TEST_NAME}
        tests/graph_convolution.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_integration_chunked)
add_executable(${TEST_NAME}
        tests/integration_chunked.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_integration_subset_chunked)
add_executable(${TEST_NAME}
        tests/integration_subset_chunked.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_adam)
add_executable(${TEST_NAME}
        tests/adam.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_sage_linear)
add_executable(${TEST_NAME}
        tests/sage_linear.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})

set(TEST_NAME test_sage_linear_adam)
add_executable(${TEST_NAME}
        tests/sage_linear_adam.cpp)
target_link_libraries(${TEST_NAME}
        ${PROJECT_NAME})
