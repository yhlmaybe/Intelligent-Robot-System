aux_source_directory(. IRSMANAGER_LIST)
add_library(IRSMANAGER_LIST ${IRSMANAGER_LIST})

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rviz_common REQUIRED)
find_package(urdf REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(moveit_msgs REQUIRED)

find_package(rclcpp_components REQUIRED)
find_package(robot_state_publisher REQUIRED)

message("CMAKElujing: ${CMAKE_MODULE_PATH}")


ament_target_dependencies(IRSMANAGER_LIST 
    rclcpp
    std_msgs
    rviz_common
    urdf
    visualization_msgs
    robot_state_publisher
    moveit_msgs)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    #ament_lint_auto_find_test_dependencies()
endif()

#ament_package()

target_link_libraries(IRSMANAGER_LIST SC_LIST)

target_link_libraries(IRSMANAGER_LIST pugixml)

target_link_libraries(IRSMANAGER_LIST rviz_common::rviz_common)

target_link_libraries(IRSMANAGER_LIST ${orocos_kdl_LIBRARIES} ${kdl_parser_LIBRARIES})

target_link_libraries(IRSMANAGER_LIST MATHEMATICALTOOL_LIST)

#install dir, use cmake.. and make install
install(DIRECTORY rviz urdf
    DESTINATION ${INSTALL_DIR}/install
)   