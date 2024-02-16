cmake_minimum_required(VERSION 3.6)

file(GLOB SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx" "*.cc")
SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE CXX )

add_library(nn_mnist ${SRC})

target_include_directories(
    nn_mnist PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MAIN_PROJECT_DIR}/mod/basic"
)

target_link_libraries(nn_mnist PUBLIC)
