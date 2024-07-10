cmake_minimum_required(VERSION 3.6)

#################################################################
file(GLOB SRC CONFIGURE_DEPENDS "*.c" "*.cpp" "*.cxx")
if(DEFINED ENV{using_c_lang})
    SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE C )
else()
    SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES LANGUAGE CXX )
endif()
add_library(sys ${SRC})

#################################################################
target_include_directories(
    sys PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#################################################################
target_link_libraries(sys PUBLIC)

install(
    TARGETS sys
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
