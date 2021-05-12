find_package(Catch2 REQUIRED)

set(BENCHMARK_FILES
        benchmark/alzheimer.cpp
        benchmark/dropout.cpp
        benchmark/feature_aggregation.cpp
        benchmark/log_softmax.cpp
        benchmark/relu.cpp
        benchmark/sparse_computation.cpp
        benchmark/layer.cpp
        benchmark/linear.cpp
        benchmark/add.cpp)

set(EXECUTABLE_NAME benchmark)
add_executable(${EXECUTABLE_NAME}
        benchmark/benchmark.cpp
        ${BENCHMARK_FILES}
        )
target_link_libraries(${EXECUTABLE_NAME}
        ${PROJECT_NAME}
        benchmark::benchmark)
