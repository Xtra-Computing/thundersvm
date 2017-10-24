cmake_minimum_required(VERSION 2.8.8)
set(PROJECT_NAME_STR thundersvm)
project(${PROJECT_NAME_STR} C CXX)

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

#-------------------
# set common include folder for module
#-------------------
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
#-------------------
# Module source
#-------------------
#set(PROJECT_LIB_NAME ${PROJECT_NAME_STR}_impl)
#include_directories(${COMMON_INCLUDES})
#file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
#add_library(${PROJECT_LIB_NAME} ${SRC_FILES})

set(LINK_LIBRARY ${CUDA_cusparse_LIBRARY})

set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
configure_file(include/test/dataset.h.in dataset.h)

include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/src/thundersvm)


set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/test)
add_custom_target(runtest
        COMMAND ${PROJECT_TEST_NAME})

#enable_testing()
#add_test(test src/test/thundersvm_test)