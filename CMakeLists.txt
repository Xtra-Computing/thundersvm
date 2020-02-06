cmake_minimum_required(VERSION 3.4)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
project(thundersvm VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

option(BUILD_SHARED_LIBS "Build as a shared library" ON)
option(USE_CUDA "Compile with CUDA" ON)
option(BUILD_TESTS "Build Tests" OFF)
option(USE_DOUBLE "Use double as kernel_type" OFF)

if(USE_DOUBLE)
    message("Use double as kernel_type")
endif()
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
if (MSVC AND BUILD_SHARED_LIBS)
    set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

find_package(Threads)
find_package(OpenMP REQUIRED)
add_definitions("-DELPP_NO_LOG_TO_FILE")

if (USE_CUDA)
    message("Compile with CUDA")
    find_package(CUDA REQUIRED QUIET)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
    list(APPEND LINK_LIBRARY ${CUDA_cusparse_LIBRARY})

    # Correct error for GCC 5 and CUDA 7.5
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
else ()
    message("Compile without CUDA")
    find_package (Eigen3 3.3 NO_MODULE)
    if (Eigen3_FOUND)
        message("using external Eigen3 for acceleration")
        include_directories(${EIGEN3_INCLUDE_DIR})
        list(APPEND LINK_LIBRARY Eigen3::Eigen)
    else ()
        message("using internal Eigen3 for acceleration")
        include_directories(${PROJECT_SOURCE_DIR}/eigen)
    endif ()
    if (NOT CMAKE_CXX_FLAGS MATCHES "-march")
        add_compile_options("-march=native")
    endif ()
endif ()

if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})


set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
configure_file(include/thundersvm/config.h.in thundersvm/config.h)

set(PROJECT_LIB_NAME ${PROJECT_NAME})
include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/src/thundersvm)

if (BUILD_TESTS)
    message("Building tests")
    set(PROJECT_TEST_NAME ${PROJECT_NAME}-test)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/test)
    add_custom_target(runtest
            COMMAND ${PROJECT_TEST_NAME})
endif ()
