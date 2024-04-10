cmake_minimum_required(VERSION 3.0.0)
project(intelligent_robot_system VERSION 0.1.0 LANGUAGES C CXX)

cmake_policy(SET CMP0057 NEW)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/usr/share/cmake-3.10/Modules")

set(INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install")

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()  

include(CTest)
enable_testing()

aux_source_directory(. IRS_LIST)

include_directories(
    ${CMAKE_SOURCE_DIR}/ServoControl
    ${CMAKE_SOURCE_DIR}/IRSGUI
    ${CMAKE_SOURCE_DIR}/ROSMANAGER_LIST
)

add_executable(IRS_CMake ${IRS_LIST})

target_link_libraries(IRS_CMake SC_LIST)
target_link_libraries(IRS_CMake IRSGUI_LIST)
target_link_libraries(IRS_CMake ROSMANAGER_LIST)

  
add_subdirectory(ServoControl)
add_subdirectory(IRSGUI)
add_subdirectory(ROSManager)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)

#install dir, use cmake.. and make install
install(DIRECTORY ROSManager
    DESTINATION ${INSTALL_DIR}
    FILES_MATCHING PATTERN "*"
    PATTERN "package.xml" EXCLUDE)   
