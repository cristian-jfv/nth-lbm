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
add_library(params OBJECT src/params.cpp src/params.hpp)
set_property(TARGET params PROPERTY CXX_STANDARD 17)
target_link_libraries(params "${TORCH_LIBRARIES}")
target_include_directories(params ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(angular_directions OBJECT src/angular_directions.cpp src/angular_directions.hpp)
set_property(TARGET angular_directions PROPERTY CXX_STANDARD 17)
target_link_libraries(angular_directions "${TORCH_LIBRARIES}")
target_include_directories(angular_directions ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(d2q9 OBJECT src/d2q9.cpp src/d2q9.hpp)
set_property(TARGET d2q9 PROPERTY CXX_STANDARD 17)
target_link_libraries(d2q9 "${TORCH_LIBRARIES}")
target_include_directories(d2q9 ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(fluid_flow OBJECT src/fluid_flow.cpp src/fluid_flow.hpp)
set_property(TARGET fluid_flow PROPERTY CXX_STANDARD 17)
target_link_libraries(fluid_flow
        "${TORCH_LIBRARIES}" angular_directions d2q9 params)
target_include_directories(fluid_flow ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(delayed_precursors OBJECT src/delayed_precursors.cpp src/delayed_precursors.hpp)
set_property(TARGET delayed_precursors PROPERTY CXX_STANDARD 17)
target_link_libraries(delayed_precursors
        "${TORCH_LIBRARIES}" angular_directions d2q9 fluid_flow neutron_transport params)
target_include_directories(delayed_precursors ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(neutron_transport OBJECT src/neutron_transport.cpp src/neutron_transport.hpp)
set_property(TARGET neutron_transport PROPERTY CXX_STANDARD 17)
target_link_libraries(neutron_transport
        "${TORCH_LIBRARIES}" angular_directions d2q9 params)
target_include_directories(neutron_transport ${warning_guard} PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")


# Executables
add_executable(nth_lbm main.cpp)
set_property(TARGET nth_lbm PROPERTY CXX_STANDARD 17)
target_link_libraries(nth_lbm
        PRIVATE "${TORCH_LIBRARIES}" params neutron_transport delayed_precursors angular_directions d2q9)
