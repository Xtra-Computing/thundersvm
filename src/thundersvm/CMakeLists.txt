set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELPP_NO_DEFAULT_LOG_FILE")

#file(GLOB_RECURSE SRC *)
file(GLOB SRC model/*.cpp solver/*.cpp util/*.cpp *.cpp)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}-train.cpp")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}-predict.cpp")
if (USE_CUDA)
    file(GLOB SRC_KERNEL kernel/*.cu)
    list(APPEND SRC ${SRC_KERNEL})
    cuda_add_library(${PROJECT_LIB_NAME} ${SRC})
    target_link_libraries(${PROJECT_LIB_NAME} ${CUDA_cusparse_LIBRARY} OpenMP::OpenMP_CXX)
    cuda_add_executable(${PROJECT_NAME}-train thundersvm-train.cpp ${COMMON_INCLUDES})
    cuda_add_executable(${PROJECT_NAME}-predict thundersvm-predict.cpp ${COMMON_INCLUDES})
else ()
    file(GLOB SRC_KERNEL kernel/*.cpp)
    list(APPEND SRC ${SRC_KERNEL})
    add_library(${PROJECT_LIB_NAME} ${SRC})
    target_link_libraries(${PROJECT_LIB_NAME} OpenMP::OpenMP_CXX)
    add_executable(${PROJECT_NAME}-train thundersvm-train.cpp ${COMMON_INCLUDES})
    add_executable(${PROJECT_NAME}-predict thundersvm-predict.cpp ${COMMON_INCLUDES})
endif ()

target_link_libraries(${PROJECT_NAME}-train ${LINK_LIBRARY} ${PROJECT_LIB_NAME})
target_link_libraries(${PROJECT_NAME}-predict ${LINK_LIBRARY} ${PROJECT_LIB_NAME})

# Export the package for use from the build tree
export(TARGETS ${PROJECT_NAME} NAMESPACE XComp:: FILE cmake/${PROJECT_NAME}Targets.cmake)
export(PACKAGE ${PROJECT_NAME})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    cmake/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/)

install(TARGETS ${PROJECT_NAME}-train ${PROJECT_NAME}-predict
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION bin)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/thundersvm DESTINATION include/)
install(FILES ${CMAKE_BINARY_DIR}/thundersvm/config.h DESTINATION include/thundersvm/)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake/)

install(EXPORT ${PROJECT_NAME}Targets
    NAMESPACE XComp:: DESTINATION lib/cmake/)
