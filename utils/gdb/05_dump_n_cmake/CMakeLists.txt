#
# https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html
#
cmake_minimum_required(VERSION 3.10)

# disable optimization for gdb
set(CMAKE_CXX_FLAGS "-g -O0 -Wall -Wextra")
set(CMAKE_BUILD_TYPE "Debug")

# set the project name
project(a01)

# add the executable
add_executable(the.exe hello.cpp)