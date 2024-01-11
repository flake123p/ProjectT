cmake_minimum_required(VERSION 3.6)

file(GLOB SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx" "*.cc")
SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE C )

add_library(mnist_for_c ${SRC})

target_include_directories(
    mnist_for_c PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MAIN_PROJECT_DIR}/mod/basic"
)

target_link_libraries(mnist_for_c PUBLIC)
