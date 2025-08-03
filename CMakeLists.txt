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
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()  

include(CTest)
enable_testing()

aux_source_directory(. IRS_LIST)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/IRSGUI
)

find_package(ompl REQUIRED)
include_directories(${OMPL_INCLUDE_DIRS})

#include_directories(/usr/locate/include/ompl-1.6)

find_program(PYTHON_EXECUTABLE python)

if(PYTHON_EXECUTABLE)
    message(STATUS "python executable found at ${PYTHON_EXECUTABLE}")
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${PYTHON_INCLUDE_DIR})

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])" OUTPUT_VARIABLE PYTHON_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    link_directories(${PYTHON_LIBRARY_DIR})

    message("lib : ${PYTHON_LIBRARY_DIR}")
    message("include : ${PYTHON_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "python not found")
endif()


add_executable(IRS_CMake ${IRS_LIST})

target_link_libraries(IRS_CMake /usr/lib/aarch64-linux-gnu/libccd.so.2)

target_link_libraries(IRS_CMake pugixml)

target_link_libraries(IRS_CMake SC_LIST)
target_link_libraries(IRS_CMake BRAINDEEPLEARN_LIST)
target_link_libraries(IRS_CMake IRSGUI_LIST)
target_link_libraries(IRS_CMake IRSMANAGER_LIST)
target_link_libraries(IRS_CMake MATHEMATICALTOOL_LIST)
  
add_subdirectory(ServoControl)
add_subdirectory(BrainDeepLearn)
add_subdirectory(IRSGUI)
add_subdirectory(IRSManager)
add_subdirectory(MathematicalTool)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
