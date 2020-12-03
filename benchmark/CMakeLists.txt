find_package(Catch2 REQUIRED)
find_package(Python3 REQUIRED
        COMPONENTS Interpreter Development)

set(BENCHMARK_FILES
        benchmark/alzheimer.cpp
        benchmark/dropout.cpp
        benchmark/graph_convolution.cpp
        benchmark/log_softmax.cpp
        benchmark/relu.cpp
        benchmark/sage_linear.cpp)

set(EXECUTABLE_NAME benchmark)
add_executable(${EXECUTABLE_NAME}
        benchmark/benchmark.cpp
        ${BENCHMARK_FILES}
        )
target_link_libraries(${EXECUTABLE_NAME}
        ${PROJECT_NAME}
        benchmark::benchmark)
