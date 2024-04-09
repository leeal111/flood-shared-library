add_executable(unit_tests unit_tests.cpp bitSTIV_test.cpp)
target_link_libraries(unit_tests ${PROJECT_NAME})
add_test(NAME unit_tests COMMAND unit_tests)
