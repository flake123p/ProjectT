function(mnist_for_c_cmake curr_target)
#message("curr_target = " ${curr_target})
target_include_directories(
    ${curr_target} PUBLIC
    "${MAIN_PROJECT_DIR}/mod/mnist_for_c"
)

if (NOT TARGET mnist_for_c)
add_subdirectory(${MAIN_PROJECT_DIR}/mod/mnist_for_c ${CMAKE_BINARY_DIR}/mod/mnist_for_c)
endif()

target_link_libraries(${curr_target} PUBLIC mnist_for_c)

endfunction()
