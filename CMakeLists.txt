cmake_minimum_required(VERSION 3.19)
project(nth_lbm)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{Torch_DIR})
    message(STATUS "torch installation prefix=$ENV{Torch_DIR}")
else()
    message(FATAL_ERROR "Torch_DIR not defined")
endif()

if(DEFINED ENV{TOMLCPP_DIR})
    message(STATUS "tomlcplusplus include directory=$ENV{TOMLCPP_DIR}")
else()
    message(FATAL_ERROR "tomlcplusplus not defined")
endif()

include_directories("$ENV{TOMLCPP_DIR}")

find_package(Torch REQUIRED)

# Libraries
add_library(neutron_transport OBJECT src/neutron_transport.cpp)
set_property(TARGET neutron_transport PROPERTY CXX_STANDARD 17)
target_link_libraries(neutron_transport "${TORCH_LIBRARIES}")
target_include_directories(neutron_transport ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(d2q9 OBJECT src/d2q9.cpp)
set_property(TARGET d2q9 PROPERTY CXX_STANDARD 17)
target_link_libraries(d2q9 "${TORCH_LIBRARIES}")
target_include_directories(d2q9 ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

# Executables
add_executable(nth_lbm main.cpp)
set_property(TARGET nth_lbm PROPERTY CXX_STANDARD 17)
target_link_libraries(nth_lbm
        PRIVATE "${TORCH_LIBRARIES}" neutron_transport)
