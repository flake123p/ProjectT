function(ocv_util_cmake curr_target)
#message("curr_target = " ${curr_target})
target_include_directories(
    ${curr_target} PUBLIC
    "${MAIN_PROJECT_DIR}/mod/ocv_util"
)

if (NOT TARGET ocv_util)
add_subdirectory(${MAIN_PROJECT_DIR}/mod/ocv_util ${CMAKE_BINARY_DIR}/mod/ocv_util)
endif()

target_link_libraries(${curr_target} PUBLIC ocv_util ${OpenCV_LIBS})

endfunction()
