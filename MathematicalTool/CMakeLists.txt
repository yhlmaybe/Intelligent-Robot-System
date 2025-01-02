aux_source_directory(. MATHEMATICALTOOL_LIST)
add_library(MATHEMATICALTOOL_LIST ${MATHEMATICALTOOL_LIST})

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
find_package(rcl_action REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_move_group REQUIRED)
find_package(moveit_kinematics REQUIRED)
find_package(moveit_planners REQUIRED)
find_package(moveit_simple_controller_manager REQUIRED)
find_package(controller_manager REQUIRED)
find_package(moveit_ros_visualization REQUIRED)
find_package(moveit_ros_warehouse REQUIRED)

#message(STATUS "planning_scene path: ${planning_scene_DIR}")

target_include_directories (MATHEMATICALTOOL_LIST
    PRIVATE
    ${moveit_core_INCLUDE_DIRS}
    #${moveit_ros_planning_interface_INCLUDE_DIRS}
    #${moveit_ros_planning_INCLUDE_DIRS}
    #${rosidl_runtime_cpp_INCLUDE_DIRS}
    #${moveit_ros_move_group_INCLUDE_DIRS}
    #${moveit_kinematics_INCLUDE_DIRS}
    #${moveit_planners_INCLUDE_DIRS}
    #${moveit_simple_controller_manager_INCLUDE_DIRS}
    #${controller_manager_INCLUDE_DIRS}
    #${moveit_ros_visualization_INCLUDE_DIRS}
    #${moveit_ros_warehouse_INCLUDE_DIRS}
)

target_link_libraries(MATHEMATICALTOOL_LIST
    ${moveit_core_LIBRARIES}
    #${moveit_ros_planning_interface_LIBRARIES}
    #${moveit_ros_planning_LIBRARIES}
    #${moveit_ros_move_group_LIBRARIES}
    #${moveit_kinematics_LIBRARIES}
    #${moveit_planners_LIBRARIES}
    #${moveit_simple_controller_manager_LIBRARIES}
    #${moveit_ros_visualization_LIBRARIES}
    #${moveit_ros_warehouse_LIBRARIES}
)

ament_target_dependencies(MATHEMATICALTOOL_LIST 
    urdf
    geometric_shapes
    unique_identifier_msgs
    rcl_action
    tf2_msgs
    moveit_msgs
    rclcpp_action
    tf2_ros)
