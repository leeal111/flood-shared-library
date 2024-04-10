cmake_minimum_required(VERSION 3.22)

project(bitSTIV CXX)

include_directories(include thirdparty/opencv4.9.0)
add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
