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

#set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")

file(GLOB SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx" "*.cc")
SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE CXX )
set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")
add_executable(a.out ${SRC})

include(${MAIN_PROJECT_DIR}/modlists/mnist_for_c/mnist_for_c.cmake)
mnist_for_c_cmake(a.out) # include, subdirectory, link
# include(${MAIN_PROJECT_DIR}/modlists/ocv_util/ocv_util.cmake)
# ocv_util_cmake(a.out) # include, subdirectory, link
include(${MAIN_PROJECT_DIR}/modlists/nn/nn.cmake)
nn_cmake(a.out) # include, subdirectory, link

target_link_libraries(a.out PUBLIC)
