if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

add_library(${PROJECT_NAME} SHARED bitSTIV.cpp)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  if(DEFINED ENV{OPENCV_ROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{OPENCV_ROOT}")
    find_package(OpenCV REQUIRED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DLL_EXPORTS)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
  else()
    message(
      FATAL_ERROR"NOT DEFINED OPENCV_ROOT VARIABLES\nOPENCV_ROOT is set to ...\\opencv\\build"
    )
  endif()
endif()
