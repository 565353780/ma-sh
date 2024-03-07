cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 23)
project(mash_cpp)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(D_GLIBCXX_USE_CXX11_ABI 0)

set(PYBIND11_ROOT_PATH ../ma-sh/ma_sh/Lib/pybind11)
set(CPP_ROOT_PATH ../ma-sh/ma_sh/Cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development)
add_subdirectory(${PYBIND11_ROOT_PATH})

set(SRC
  ${CPP_ROOT_PATH}/src/add.cpp
  ${CPP_ROOT_PATH}/src/idx.cpp
  ${CPP_ROOT_PATH}/src/mask.cpp
  ${CPP_ROOT_PATH}/src/sample.cpp
  ${CPP_ROOT_PATH}/src/filter.cpp
  ${CPP_ROOT_PATH}/main.cpp
)

include_directories(${CPP_ROOT_PATH}/include)

pybind11_add_module(${PROJECT_NAME} ${SRC})

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
