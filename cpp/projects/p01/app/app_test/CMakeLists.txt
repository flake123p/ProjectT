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

target_include_directories(
    a.out PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MAIN_PROJECT_DIR}/mod/basic"
    "${MAIN_PROJECT_DIR}/mod/stat"
)

#
# mnist_for_c
#
include(${MAIN_PROJECT_DIR}/modlists/mnist_for_c/mnist_for_c.cmake)
mnist_for_c_cmake(a.out) # include, subdirectory, link

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../mod/cmd/cmd_chan_fx3 ${CMAKE_BINARY_DIR}/cmd/cmd_chan_fx3)
# target_link_libraries(fx3 PUBLIC cmd_chan_fx3)

target_link_libraries(a.out PUBLIC)

# set_target_properties(
#     budart PROPERTIES
#     PUBLIC_HEADER 
#         hello_world.h
# )
    
# install(
#     TARGETS fx3
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
#     PUBLIC_HEADER DESTINATION inc
# )

# if(DEFINED ENV{_USE_FX3_})
# add_custom_command(
#     TARGET fx3
#     POST_BUILD
#     #COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/lib/libbudart.so ${CMAKE_INSTALL_PREFIX}/lib/libcudart.so.11.0  -f
#     #COMMAND cp ${CMAKE_BINARY_DIR}/cmd/cmd_device/*.so ${CMD_INSTALL_PREFIX}/lib -f
#     COMMAND cp ${CMAKE_BINARY_DIR}/cmd/cmd_chan_fx3/*.so ${CMD_INSTALL_PREFIX}/lib -f
#     #COMMAND cp ${CMAKE_BINARY_DIR}/cmd/cmd_host/*.so ${CMD_INSTALL_PREFIX}/lib -f
#     #COMMAND cp ${CMAKE_BINARY_DIR}/swc/*.so ${SWC_INSTALL_PREFIX}/lib -f
#     COMMENT "ln libcudart.so.11.0 ... Done"
# )
# else()
# # add_custom_command(
# #     TARGET fx3
# #     POST_BUILD
# #     COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/lib/libbudart.so ${CMAKE_INSTALL_PREFIX}/lib/libcudart.so.11.0  -f
# #     COMMAND cp ${CMAKE_BINARY_DIR}/cmd/cmd_device/*.so ${CMD_INSTALL_PREFIX}/lib -f
# #     COMMAND cp ${CMAKE_BINARY_DIR}/cmd/cmd_chan_sim/*.so ${CMD_INSTALL_PREFIX}/lib -f
# #     COMMAND cp ${CMAKE_BINARY_DIR}/cmd/cmd_host/*.so ${CMD_INSTALL_PREFIX}/lib -f
# #     COMMAND cp ${CMAKE_BINARY_DIR}/swc/*.so ${SWC_INSTALL_PREFIX}/lib -f
# #     COMMENT "ln libcudart.so.11.0 ... Done"
# # )
# endif()