cmake_minimum_required(VERSION 3.12)

set (CMAKE_BUILD_TYPE Debug)

project(main)

#
# GLOB_RECURSE without build/ : https://stackoverflow.com/questions/27992346/how-to-use-cmake-glob-recurse-for-only-some-subdirectories
#
file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR} "*.c" "*.cpp")
list(FILTER SRC EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/.*" )
#file(GLOB SRC CONFIGURE_DEPENDS "*.c" "*.cpp")

SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE CXX )

add_executable(a.out ${SRC})

target_link_libraries(a.out)