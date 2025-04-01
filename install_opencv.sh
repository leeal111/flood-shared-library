curl -Lo opencv.tar.gz  https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz
tar -xzvf opencv.tar.gz
rm opencv.tar.gz
cd opencv-*
mkdir -p build && cd build
cmake  ..
cmake --build .
make install
cd ../..
rm -rf opencv-*