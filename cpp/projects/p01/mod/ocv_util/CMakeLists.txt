cmake_minimum_required(VERSION 3.6)

file(GLOB SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx" "*.cc")
SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE CXX )

add_library(ocv_util ${SRC})

target_include_directories(
    ocv_util PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MAIN_PROJECT_DIR}/mod/basic"
)

include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries(ocv_util PUBLIC ${OpenCV_LIBS})




# #################################################################
# target_include_directories(
#     a.out PUBLIC
#     "${CMAKE_CURRENT_SOURCE_DIR}"
#     "${MAIN_PROJECT_DIR}/mod/basic"
# )

# if(${LOCAL_OPENCV})
#     message("Use local OpenCV ...")
#     find_package( OpenCV REQUIRED PATHS "${PROJECT_SOURCE_DIR}/../cv")
# else()
#     message("Use system OpenCV ...")
#     find_package( OpenCV REQUIRED )
# endif()
# # find_package( OpenCV REQUIRED )
# # include_directories( ${OpenCV_INCLUDE_DIRS} )

# #################################################################
# target_link_libraries(a.out PUBLIC ${OpenCV_LIBS})
