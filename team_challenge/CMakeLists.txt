cmake_minimum_required(VERSION 3.8)
project(team_challenge)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(referee_pkg REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(pose_msg REQUIRED)

# list(APPEND CMAKE_PREFIX_PATH "/home/ssxz/ymh/TeamSaki_challenge/TeamSaki_challenge/Vision_Arena_2025/install/referee_pkg")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(include)
add_executable(vision_node src/vision_node.cpp)
ament_target_dependencies(vision_node rclcpp OpenCV cv_bridge sensor_msgs ament_index_cpp referee_pkg std_msgs pose_msg)

add_executable(shooter_node src/shooter_node.cpp)
ament_target_dependencies(shooter_node rclcpp OpenCV cv_bridge sensor_msgs ament_index_cpp referee_pkg std_msgs pose_msg)

# add_executable(test_node src/test.cpp)
# ament_target_dependencies(test_node rclcpp OpenCV cv_bridge sensor_msgs ament_index_cpp referee_pkg std_msgs)

# target_link_libraries(vision_node ${typesupport_target})

install(TARGETS
  vision_node
  shooter_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  resources
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
