cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Vision_Kernel_AVX-2 VERSION 0.1.0)

# set(CMAKE_CXX_COMPILER "C:/ProgramData/chocolatey/bin/x86_64-w64-mingw32-g++")

include(CTest)
enable_testing()

set(CMAKE_CXX_FLAGS "-mavx2") # enable AVX-2 intrinsics

# find_package(OpenCV REQUIRED PATHS "C:/opencv/mingw_build") # find OpenCV libraries in a specific path
find_package( OpenCV REQUIRED ) # find OpenCV libraries

include_directories( ${OpenCV_INCLUDE_DIRS} ) # include OpenCV headers



add_executable(main main.cpp) # compile source code
# target_link_libraries(main ${OpenCV_LIBS} -lopencv_core -lopencv_highgui -lopencv_imgcodecs)
target_link_libraries( main ${OpenCV_LIBS} ) # link with OpenCV libraries

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)