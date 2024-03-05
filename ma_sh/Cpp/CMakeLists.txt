cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mash-cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development)
add_subdirectory(pybind11)

pybind11_add_module(mash-cpp src/core.cpp)

target_link_libraries(mash-cpp PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET mash-cpp PROPERTY CXX_STANDARD 17)
