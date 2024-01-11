cmake_minimum_required(VERSION 3.6)
set (CMAKE_CXX_STANDARD 14)
# set (CMAKE_BUILD_TYPE Debug)
set(MAIN_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

project(main)

###### CMake Define #############################################
# set(ENV{using_c_lang} 1)
# set(ENV{using_float_32} 1)
# set(ENV{using_sys_r5} 1)
# set(ENV{using_embcvqq} 1)

###### GCC/Make Define###########################################
# add_compile_options ( -D_NO_OS_ )
# add_compile_options ( -D_DISCARD_ACCU_ )
# add_compile_options ( -D_NO_ASSERT_ )
# add_compile_options ( -D_NO_PRINT_ )
# add_compile_options ( -D_NO_MEM_TRACER_ )

#################################################################
file(GLOB main_SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx")
SET_SOURCE_FILES_PROPERTIES(${main_SRC} PROPERTIES LANGUAGE CXX )
add_executable(a.out ${main_SRC})

if(DEFINED ENV{using_float_32})
    set(CMAKE_CXX_FLAGS "-O0 -Wall -Wextra -fsingle-precision-constant")
else()
    set(CMAKE_CXX_FLAGS "-O0 -Wall -Wextra")
endif()

#################################################################
target_include_directories(
    a.out PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MAIN_PROJECT_DIR}/mod/basic"
)

if(${LOCAL_OPENCV})
    message("Use local OpenCV ...")
    find_package( OpenCV REQUIRED PATHS "${PROJECT_SOURCE_DIR}/../cv")
else()
    message("Use system OpenCV ...")
    find_package( OpenCV REQUIRED )
endif()
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )

include(${MAIN_PROJECT_DIR}/modlists/ocv_util/ocv_util.cmake)
ocv_util_cmake(a.out) # include, subdirectory, link

#################################################################
target_link_libraries(a.out PUBLIC ${OpenCV_LIBS})
