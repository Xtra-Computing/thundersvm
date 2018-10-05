if (UNIX AND CMAKE_VERSION VERSION_GREATER 3.10)
	include(GoogleTest)
else ()
	add_subdirectory(googletest)
	include_directories(googletest/googletest/include)
	include_directories(googletest/googlemock/include)
endif ()

file(GLOB TEST_SRC *)

if (USE_CUDA)
    cuda_add_executable(${PROJECT_TEST_NAME} ${TEST_SRC})
else ()
    add_executable(${PROJECT_TEST_NAME} ${TEST_SRC})
endif ()
target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_LIB_NAME} ${LINK_LIBRARY} gtest)

