
file(GLOB_RECURSE SRC *.c*)
list(REMOVE_ITEM SRC ${CMAKE_CURRENT_SOURCE_DIR}/thundersvm.cu)
cuda_add_library(${PROJECT_NAME}_lib ${SRC})
cuda_add_executable(${PROJECT_NAME}  thundersvm.cu ${COMMON_INCLUDES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)
