add_library(${PROJECT_NAME} SHARED bitSTIV.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})