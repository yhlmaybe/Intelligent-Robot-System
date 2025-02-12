aux_source_directory(. MATHEMATICALTOOL_LIST)
add_library(MATHEMATICALTOOL_LIST ${MATHEMATICALTOOL_LIST})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
find_package(rcl_action REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(kdl_parser REQUIRED)

target_link_libraries(MATHEMATICALTOOL_LIST 
    /opt/ros/humble/lib/libmoveit_robot_model.so
    /opt/ros/humble/lib/libsrdfdom.so
    /opt/ros/humble/lib/libmoveit_robot_state.so
    /opt/ros/humble/lib/libmoveit_exceptions.so
    /usr/local/lib/libnlopt.so)

include_directories(/usr/include/ompl-1.5)

ament_target_dependencies(MATHEMATICALTOOL_LIST 
    urdf
    geometric_shapes
    unique_identifier_msgs
    rcl_action
    tf2_msgs
    moveit_msgs
    rclcpp_action
    tf2_ros
    kdl_parser)
