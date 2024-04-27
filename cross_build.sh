rm -rf build
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/arm_linux_setup.cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build
file build/src/libbitSTIV.so