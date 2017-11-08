if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -Wno-unused-variable -ansi -Wno-deprecated -pthread)
endif()

if(MSVC)
    #vc 2012 fix for vararg templates
    set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()
add_subdirectory(gtest)
add_definitions(${MSVC_COMPILER_DEFS})
include_directories(${COMMON_INCLUDES} ${GTEST_INCLUDE_DIRS})
#enable_testing()
file(GLOB TEST_SRC_FILES *.cpp)
if (USE_CUDA)
    cuda_add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${SRC})
else ()
    add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${SRC})
endif ()
add_dependencies(${PROJECT_TEST_NAME} googletest ${PROJECT_LIB_NAME})

if(NOT WIN32 OR MINGW)
    target_link_libraries(${PROJECT_TEST_NAME}
            ${GTEST_LIBS_DIR}/libgtest.a
            ${GTEST_LIBS_DIR}/libgtest_main.a
            )
else()
    target_link_libraries(${PROJECT_TEST_NAME}
            debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
            optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
            )
    target_link_libraries(${PROJECT_TEST_NAME}
            debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
            optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
            )
endif()

target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT} ${PROJECT_LIB_NAME} ${LINK_LIBRARY})
