cmake_minimum_required(VERSION 3.20)
project(lidar_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters search kdtree segmentation)

# Core: ROS-free detector library (PCL is a private dependency).
add_library(lt_core SHARED src/box_fit.cpp src/detector.cpp src/hungarian.cpp src/tracker.cpp src/grid_cluster.cpp)
target_include_directories(lt_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(lt_core SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(lt_core PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(lt_core PRIVATE ${PCL_LIBRARIES} Eigen3::Eigen)

# ROS conversions.
add_library(lt_ros SHARED src/ros_conversions.cpp)
target_link_libraries(lt_ros PUBLIC lt_core ${sensor_msgs_TARGETS} ${std_msgs_TARGETS} ${vision_msgs_TARGETS})

# Live detector + tracker node (component + standalone executable).
add_library(tracker_component SHARED src/tracker_node.cpp)
target_link_libraries(tracker_component PUBLIC lt_ros rclcpp::rclcpp rclcpp_components::component
  tf2_ros::tf2_ros message_filters::message_filters)
rclcpp_components_register_node(tracker_component
  PLUGIN "lidar_tracking::TrackerNode" EXECUTABLE tracker_node)

# Offline bag runner used for scoring.
add_executable(process_bag src/process_bag.cpp)
target_link_libraries(process_bag PRIVATE lt_ros rclcpp::rclcpp rosbag2_cpp::rosbag2_cpp ${tf2_msgs_TARGETS})

install(TARGETS lt_core lt_ros tracker_component
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS process_bag RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_box_fit test/test_box_fit.cpp)
  target_link_libraries(test_box_fit lt_core)
  ament_add_gtest(test_detector test/test_detector.cpp)
  target_link_libraries(test_detector lt_core)
  ament_add_gtest(test_grid_cluster test/test_grid_cluster.cpp)
  target_link_libraries(test_grid_cluster lt_core)
  ament_add_gtest(test_tracker test/test_tracker.cpp)
  target_link_libraries(test_tracker lt_core)
endif()

ament_package()
