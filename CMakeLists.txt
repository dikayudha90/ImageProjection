cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

find_package(CUDA REQUIRED cuda cudart cudadevrt)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")

find_package(OpenCV 
PATHS /home/ayudha/Downloads/opencv-3.4.1/build
REQUIRED xfeatures2d core features2d cudafeatures2d
NO_DEFAULT_PATH
)

file(GLOB SOURCES "src/*.cpp")
file(GLOB SOURCES_CU "src/*.cu")
file(GLOB HEADERS "includes/*.h")

set(AKAZE_LIBS "/usr/local/lib" )
set(AKAZE_INCLUDES_DIRS "/usr/local/include/cuda_akaze")

include_directories(includes ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${AKAZE_INCLUDES_DIRS})
#cuda_compile(CUDA_KAZE ${SOURCES_CU})
add_executable( DisplayImage ${SOURCES})# ${CUDA_KAZE})
target_link_libraries( DisplayImage ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${AKAZE_LIBS} png)
