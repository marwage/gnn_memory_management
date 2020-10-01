cmake_minimum_required(VERSION 3.17)

project(Alzheimer VERSION 0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(alzheimer read_tensors.cpp)

target_include_directories(alzheimer PUBLIC /usr/local/include)
target_link_directories(alzheimer PUBLIC /usr/local/lib)
target_link_libraries(alzheimer PUBLIC cnpy)

