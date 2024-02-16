function(nn_mnist_cmake curr_target)
#message("curr_target = " ${curr_target})
target_include_directories(
    ${curr_target} PUBLIC
    "${MAIN_PROJECT_DIR}/mod"
    "${MAIN_PROJECT_DIR}/mod/nn"
    "${MAIN_PROJECT_DIR}/mod/nn_mnist"
)

if (NOT TARGET nn_mnist)
add_subdirectory(${MAIN_PROJECT_DIR}/mod/nn_mnist ${CMAKE_BINARY_DIR}/mod/nn_mnist)
endif()

target_link_libraries(${curr_target} PUBLIC nn_mnist)

endfunction()
