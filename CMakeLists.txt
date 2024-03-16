cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(mash_cpp)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(D_GLIBCXX_USE_CXX11_ABI 0)

set(CPP_ROOT_PATH ../ma-sh/ma_sh/Cpp)
set(CPP_SRC_PATH ${CPP_ROOT_PATH}/src)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

set(SRC
  ${CPP_SRC_PATH}/filter.cpp
  ${CPP_SRC_PATH}/idx.cpp
  ${CPP_SRC_PATH}/inv.cpp
  ${CPP_SRC_PATH}/mash.cpp
  ${CPP_SRC_PATH}/mash_unit.cpp
  ${CPP_SRC_PATH}/mask.cpp
  ${CPP_SRC_PATH}/rotate.cpp
  ${CPP_SRC_PATH}/sample.cpp
  ${CPP_SRC_PATH}/sh.cpp
  ${CPP_SRC_PATH}/value.cpp
  ${CPP_ROOT_PATH}/main.cpp
)

include_directories(${CPP_ROOT_PATH}/include)

pybind11_add_module(${PROJECT_NAME} ${SRC})

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
