cmake_minimum_required(VERSION 3.6)
project(test)
set(MAIN_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

#set(BUILD_SHARED_LIBS ON)

message("CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message("USE_EXTERNAL_INSTALL_PREFIX = ${USE_EXTERNAL_INSTALL_PREFIX}")

if(DEFINED USE_EXTERNAL_INSTALL_PREFIX)
else()
    # set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/buda)
    # set(CMD_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/cmd)
    # set(SWC_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/swc)
endif()

file(GLOB SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx" "*.cc")
SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE CXX )
set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")
add_executable(a.out ${SRC})

target_include_directories(
    a.out PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"

    "${MAIN_PROJECT_DIR}/mod/eigen34/include/eigen3"
    # "${PROJECT_SOURCE_DIR}/../embeig/inc"
)

target_link_libraries(a.out PUBLIC)
