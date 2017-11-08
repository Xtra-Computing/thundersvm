cmake_minimum_required(VERSION 2.8.12)
set(PROJECT_NAME_STR thundersvm)
project(${PROJECT_NAME_STR} C CXX)

set(USE_CUDA ON CACHE BOOL "Compile with CUDA")

find_package(Threads REQUIRED)
<<<<<<< HEAD
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
=======
if (USE_CUDA)
    message("Compile with CUDA")
    find_package(CUDA REQUIRED QUIET)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
    set(LINK_LIBRARY ${CUDA_cusparse_LIBRARY})
else ()
    message("Compile without CUDA")
endif ()


>>>>>>> upstream/master
if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})


set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
configure_file(include/thundersvm/config.h.in config.h)

set(PROJECT_LIB_NAME ${PROJECT_NAME}-lib)
include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/src/thundersvm)


set(PROJECT_TEST_NAME ${PROJECT_NAME}-test)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/test)
add_custom_target(runtest
        COMMAND ${PROJECT_TEST_NAME})

