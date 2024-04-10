cmake_minimum_required(VERSION 3.22)

project(bitSTIV CXX)

MESSAGE(STATUS "current platform: ${CMAKE_SYSTEM_NAME}")

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
	if(DEFINED  ENV{OPENCV_ROOT})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{OPENCV_ROOT}")
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    else()
        message(FATAL_ERROR"NOT DEFINED OPENCV_ROOT VARIABLES\nOPENCV_ROOT is set to ...\\opencv\\build")
    endif()
endif ()

find_package(OpenCV REQUIRED)
include_directories(include)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
