cmake_minimum_required(VERSION 2.8.8)
set(PROJECT_NAME_STR thundersvm)
project(${PROJECT_NAME_STR} C CXX)

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
set(CMAKE_CXX_STANDARD 11)

#-------------------
# set common include folder for module
#-------------------
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
#-------------------
# Module source
#-------------------
#set(PROJECT_LIB_NAME ${PROJECT_NAME_STR}_impl)
#include_directories(${COMMON_INCLUDES})
#file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
#add_library(${PROJECT_LIB_NAME} ${SRC_FILES})



include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/src/thundersvm)


set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/test)
add_custom_target(runtest
        COMMAND ${PROJECT_TEST_NAME}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test)

#enable_testing()
#add_test(test src/test/thundersvm_test)