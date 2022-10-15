cmake_minimum_required(VERSION 3.8)
project(dune_log_2_rosbag)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(brov2_interfaces REQUIRED)

add_executable(dune_log_2_rosbag_exe src/sonar_data_2_rosbag.cpp src/DSSPParser.cpp)
ament_target_dependencies(dune_log_2_rosbag_exe rclcpp rosbag2_cpp brov2_interfaces)

install(TARGETS
dune_log_2_rosbag_exe
  DESTINATION lib/${PROJECT_NAME}
)

include_directories(
  include
  include/${PROJECT_NAME}
  ${CMAKE_INCLUDE_DIRECTORIES_BEFORE}
  ${colcon_INCLUDE_DIRS}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
