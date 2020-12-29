find_package(Catch2 REQUIRED)
find_package(Python3 REQUIRED
        COMPONENTS Interpreter Development)

set(TEST_FILES
        tests/adam.cpp
        tests/axpby.cpp
        tests/divmv.cpp
        tests/dropout.cpp
        tests/elesq.cpp
        tests/graph_convolution.cpp
        tests/integration.cpp
        tests/invsqrt.cpp
        tests/linear.cpp
        tests/log_softmax.cpp
        tests/loss.cpp
        tests/relu.cpp
        tests/sage_linear.cpp
        tests/sage_linear_adam.cpp
        tests/sum_rows.cpp
        tests/transpose.cpp
        tests/sparse.cpp
        tests/layer.cpp
        tests/sp_mat_mat_mult.cpp
        tests/pipeline.cpp
        tests/add.cpp
        tests/cuda_helper.cpp)

set(HELPER_FILES
        tests/helper.cpp)

set(LIBS
        ${PROJECT_NAME}
        Catch2::Catch2
        ${Python3_LIBRARIES})

set(TEST_NAME tests)
add_executable(${TEST_NAME}
        tests/tests.cpp
        ${TEST_FILES}
        ${HELPER_FILES})
target_link_libraries(${TEST_NAME}
        ${LIBS})
