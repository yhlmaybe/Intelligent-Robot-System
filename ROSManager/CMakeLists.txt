aux_source_directory(. ROSMANAGER_LIST)
add_library(ROSMANAGER_LIST ${ROSMANAGER_LIST})

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

message("CMAKElujing: ${CMAKE_MODULE_PATH}")

include_directories(/usr/lib/aarch64-linux-gnu)

ament_target_dependencies(ROSMANAGER_LIST "rclcpp" "std_msgs")

install(TARGETS
    ROSMANAGER_LIST  
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()

target_link_libraries(ROSMANAGER_LIST SC_LIST)

target_link_libraries(ROSMANAGER_LIST pugixml)