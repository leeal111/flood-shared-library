cmake_minimum_required(VERSION 3.22)

project(bitSTIV CXX)

find_package(OpenCV REQUIRED)

include_directories(include)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
