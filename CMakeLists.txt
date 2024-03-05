cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(mash_cpp)

set(PYBIND11_ROOT_PATH ../ma-sh/ma_sh/Lib/pybind11)
set(CPP_ROOT_PATH ../ma-sh/ma_sh/Cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development)
add_subdirectory(${PYBIND11_ROOT_PATH})

set(SRC ${CPP_ROOT_PATH}/src/core.cpp)

pybind11_add_module(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
