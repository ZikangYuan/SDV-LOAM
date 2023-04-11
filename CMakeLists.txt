cmake_minimum_required(VERSION 2.8.3)
project(sdv_loam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  nav_msgs
  pcl_ros
  pcl_conversions
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 

find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED QUIET)

# flags
add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS
#   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
#)

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

set(PROJECT_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Reprojector.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
) 

if (Pangolin_FOUND)
	message("--- found PANGOLIN, compiling PROJECT_PANGOLIN library.")
	include_directories( ${catkin_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS} ) 
	set(PROJECT_PANGOLIN_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinViewer.cpp)
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling PROJECT_PANGOLIN library.")
	message("    this means there will be no 3D display / GUI available for sdv_loam.")
	set(PROJECT_PANGOLIN_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

if (OpenCV_FOUND)
	message("--- found OpenCV, compiling PROJECT_OPENCV library.")
        message("--- OpenCV_VERSION: ${OpenCV_VERSION}")
        message("--- OpenCV_INCLUDE_PATH: ${OpenCV_INCLUDE_DIRS}")
        message("--- OpenCV_LIB_PATH: ${OpenCV_LIBRARIES}")
	include_directories( ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
	set(PROJECT_OPENCV_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling PROJECT_OPENCV library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(PROJECT_OPENCV_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp)
	set(HAS_OPENCV 0)
endif ()

if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${catkin_INCLUDE_DIRS} ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()

# compile main library.
include_directories( ${catkin_INCLUDE_DIRS} ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(vision_module ${PROJECT_SOURCE_FILES} ${PROJECT_OPENCV_SOURCE_FILES} ${PROJECT_PANGOLIN_SOURCE_FILES})
target_link_libraries(vision_module ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

#set_property( TARGET vision_module APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling sdv_loam.")
	add_executable(sdv_loam ${PROJECT_SOURCE_DIR}/src/main.cpp )
    target_link_libraries(sdv_loam vision_module boost_system cxsparse ${catkin_LIBRARIES} ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
else()
	message("--- not building sdv_loam, since either don't have openCV or Pangolin.")
endif()
