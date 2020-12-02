find_package(Catch2 REQUIRED)
find_package(Python3 REQUIRED
        COMPONENTS Interpreter Development)

set(BENCHMARK_FILES
        benchmark/graph_convolution.cpp
        benchmark/alzheimer.cpp)

set(EXECUTABLE_NAME benchmark)
add_executable(${EXECUTABLE_NAME}
        benchmark/benchmark.cpp
        ${BENCHMARK_FILES}
        )
target_link_libraries(${EXECUTABLE_NAME}
        ${PROJECT_NAME}
        benchmark::benchmark)
