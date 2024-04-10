set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CROSS_COMPILE_TOOL_PATH /home/leeal/workspace/bitSTIV/thirdparty/crosscompile)
set(CMAKE_C_COMPILER ${CROSS_COMPILE_TOOL_PATH}/bin/aarch64-none-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE_TOOL_PATH}/bin/aarch64-none-linux-gnu-g++)