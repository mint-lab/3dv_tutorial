cmake_minimum_required(VERSION 3.8)
project(3dv_tutorial)

# Find external packages and specify their locations
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# Specify project files and locations
set(EXAMPLE_DIR	"${CMAKE_SOURCE_DIR}/examples")
file(GLOB EXAMPLE_FILES ${EXAMPLE_DIR}/*.cpp)

# Set common configuration
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Ceres_INCLUDE_DIRS}
)
link_directories(
    ${OpenCV_LIB_DIRS}
    ${Ceres_LIB_DIRS} 
)
link_libraries(
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES} 
)
add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES)

# Add an executable
foreach(example_file ${EXAMPLE_FILES})
    string(REPLACE ".cpp"            "" example_name ${example_file})
    string(REPLACE "${EXAMPLE_DIR}/" "" example_name ${example_name})
    add_executable(${example_name} ${example_file})
endforeach()