cmake_minimum_required(VERSION 3.6)
set (CMAKE_CXX_STANDARD 14)
# set (CMAKE_BUILD_TYPE Debug)

project(main)

###### CMake Define #############################################
#set(CMAKE_SYSTEM_PROCESSOR riscv)
#set(RISCV_HOST_TAG linux)
#set(RISCV_TOOL_PATH "$ENV{HOME}/riscv" CACHE PATH "RISC-V tool path")
#set(RISCV_TOOL_PATH "/nui_home/flake.yeh/toolchain" CACHE PATH "RISC-V tool path")
set(RISCV_TOOL_PATH "/nui_home/flake.yeh/rvtc/bd_rv32imaf_ilp32f" CACHE PATH "RISC-V tool path")
set(RISCV_TOOLCHAIN_ROOT "${RISCV_TOOL_PATH}/bin" CACHE PATH "RISC-V compiler path")
set(CMAKE_FIND_ROOT_PATH ${RISCV_TOOLCHAIN_ROOT})
#list(APPEND CMAKE_PREFIX_PATH "${RISCV_TOOLCHAIN_ROOT}")

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/riscv32-unknown-elf-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/riscv32-unknown-elf-g++")
set(CMAKE_AR "${RISCV_TOOLCHAIN_ROOT}/riscv32-unknown-elf-ar")
set(CMAKE_LINKER "${RISCV_TOOLCHAIN_ROOT}/riscv32-unknown-elf-ld")
#set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER>")

# set(ENV{using_c_lang} 1)
# set(ENV{using_float_32} 1)
set(ENV{using_sys_r5} 1)
# set(ENV{using_embcvqq} 1)

set(ENV{DISABLE_OPENCV} 1)

###### GCC/Make Define###########################################
add_compile_options ( -D_NO_OS_ )
# add_compile_options ( -D_DISCARD_ACCU_ )
add_compile_options ( -D_NO_ASSERT_ )
add_compile_options ( -D_NO_PRINT_ )
add_compile_options ( -D_NO_MEM_TRACER_ )

add_compile_options ( -D_USE_EMBCV_BLUR_ )
add_compile_options ( -D_USE_EMBCV_THRESHOLD_ )
add_compile_options ( -D_USE_EMBCV_FIT_ELLIPSE_ )
add_compile_options ( -D_USE_EMBCV_FIND_CONTOURS_ )

#add_compile_options ( -D_TEMP_2CORES_VEX_R5_SMP_ )
add_compile_options ( -D_TEMP_R5_MCYCLE_DUMP_ )
add_compile_options( -D_USING_MY_MALLOC_ )

#add_compile_options ( -D_PROF_3_USE_ROI_IMG_1_ )
#add_compile_options ( -D_PROF_3_USE_ROI_IMG_3_ )

#add_compile_options (  -g -O2 -march=rv32imaf )
#set(CMAKE_CXX_FLAGS "-O0 -Wall -Wextra")
#set(CMAKE_CXX_FLAGS "-Ofast")

# Flake : to VSCode json file
#set(CMAKE_C_FLAGS "-O2 -march=rv32imaf")
# -specs=nano.specs ??
# set(CMAKE_CXX_FLAGS " -pg -g -O2 -march=rv32imaf -T /home/pupil/temp/p1/pupil_p1/app_prof_renode/elf32lriscv.lds" CACHE INTERNAL "")
# set(CMAKE_C_FLAGS " -pg -g -O2 -march=rv32imaf -T /home/pupil/temp/p1/pupil_p1/app_prof_renode/elf32lriscv.lds" CACHE INTERNAL "")
# set(CMAKE_EXE_LINKER_FLAGS " -pg" CACHE INTERNAL "")

#set(CMAKE_CXX_FLAGS " -g -O2 -march=rv32imaf -T /home/pupil/temp/p1/pupil_p1/app_prof_renode/elf32lriscv.lds")
#
# -v = verbose
# -O = 0 1 2(default) 3 s
#
#-mabi=ilp32f 

if(DEFINED ENV{using_sys_r5})
    add_compile_options ( -D_USING_FLOAT_32_ )
    add_compile_options ( -D_USING_MEM_BUS_32_ )
endif()

#################################################################
file(GLOB main_SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx" "*.s")
if(DEFINED ENV{using_c_lang})
    SET_SOURCE_FILES_PROPERTIES(${main_SRC} PROPERTIES LANGUAGE C )
    set(CMAKE_C_FLAGS " -fsingle-precision-constant -ffast-math -g -O2 -march=rv32imaf -T ${PROJECT_SOURCE_DIR}/elf32lriscv.lds")
else()
    SET_SOURCE_FILES_PROPERTIES(${main_SRC} PROPERTIES LANGUAGE CXX )
    set(CMAKE_CXX_FLAGS " -fsingle-precision-constant -ffast-math -g -O0 -march=rv32imaf -T ${PROJECT_SOURCE_DIR}/elf32lriscv.lds")
endif()
add_executable(a ${main_SRC})

#################################################################
target_include_directories(
    a PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

# if(DEFINED ENV{using_sys_r5})
#     target_include_directories(a PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../sys_r5/inc")
#     if(NOT DEFINED ENV{sys_added})
#         add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sys_r5 ${CMAKE_CURRENT_BINARY_DIR}/sys_r5)
#         set(ENV{sys_added} 1)
#     endif()
# else()
#     target_include_directories(a PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../sys_pc/inc")
#     if(NOT DEFINED ENV{sys_added})
#         add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sys_pc ${CMAKE_CURRENT_BINARY_DIR}/sys_pc)
#         set(ENV{sys_added} 1)
#     endif()
# endif()

# if(DEFINED ENV{sys_added})
#     message("skip add sys ...")
# else()
#     set(ENV{sys_added} 1)

#     if(DEFINED ENV{using_sys_r5})
#         add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sys_r5 ${CMAKE_CURRENT_BINARY_DIR}/sys_r5)
#     else()
#         add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sys_pc ${CMAKE_CURRENT_BINARY_DIR}/sys_pc)
#     endif()
# endif()

target_include_directories(a PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../demo_cpp")
if(NOT DEFINED ENV{demo_cpp_added})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../demo_cpp ${CMAKE_CURRENT_BINARY_DIR}/demo_cpp)
    set(ENV{demo_cpp_added} 1)
endif()

#################################################################
target_link_libraries(a PUBLIC demo_cpp)

add_custom_command(
    TARGET a 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy a /nui_home/flake.yeh/renode_portable/scripts/a
    COMMENT "Copied /nui_home/flake.yeh/renode_portable/scripts/a"
    COMMAND cp ../../gdbrc .
)

add_custom_command(
    TARGET a 
    POST_BUILD
    COMMAND "${RISCV_TOOL_PATH}/bin/riscv32-unknown-elf-objdump" -S a>a.S
    COMMENT "objdump to a.S ... Done"
    COMMAND "${RISCV_TOOL_PATH}/bin/riscv32-unknown-elf-size" a
    #COMMAND cd sys_r5 && riscv32-unknown-elf-objdump -S libsys.a>libsys.S
)
