aux_source_directory(. MATHEMATICALTOOL_LIST)
add_library(MATHEMATICALTOOL_LIST ${MATHEMATICALTOOL_LIST})

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(urdf REQUIRED)
find_package(rcl_action REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tf2_msgs REQUIRED)

target_include_directories (MATHEMATICALTOOL_LIST
    PRIVATE
    ${moveit_ros_planning_interface_INCLUDE_DIRS}
    ${rosidl_runtime_cpp_INCLUDE_DIRS}
    ${moveit_msgs_INCLUDE_DIRS}
)

ament_target_dependencies(MATHEMATICALTOOL_LIST 
    Eigen3
    rclcpp_action
    rcl_action
    tf2_msgs
    urdf
    rclcpp
    std_msgs
    visualization_msgs
    unique_identifier_msgs
)
