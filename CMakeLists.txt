cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
#set(CMAKE_MODULE_PATH "/home/ayudha/Downloads/opencv-3.4.1")

find_package(OpenCV 
PATHS /home/ayudha/Downloads/opencv-3.4.1/build
REQUIRED xfeatures2d core features2d cudafeatures2d
NO_DEFAULT_PATH
)

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "includes/*.h")
#target_include_directories(DisplayImage PRIVATE ${HEADERS})
include_directories(includes ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage ${SOURCES})
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
