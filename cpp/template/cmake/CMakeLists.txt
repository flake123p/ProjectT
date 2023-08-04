cmake_minimum_required(VERSION 3.12)

project(main)

file(GLOB main_SRC CONFIGURE_DEPENDS "*.c" "*.cpp")
add_executable(a.out ${main_SRC})

target_link_libraries(a.out)