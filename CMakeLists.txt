cmake_minimum_required(VERSION 3.0.0)
project(Intelligent-Robot-System VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/usr/share/cmake-3.10/Modules")

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
${CMAKE_SOURCE_DIR}/include
${CMAKE_SOURCE_DIR}/ServoControl
${CMAKE_SOURCE_DIR}/IRSGUI
)

include(CTest)
enable_testing()

aux_source_directory(. IRS_LIST)

find_program(PYTHON_EXECUTABLE python)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

message(STATUS "Qt5_DIR = ${Qt5_DIR}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(PYTHON_EXECUTABLE)
    message(STATUS "python executable found at ${PYTHON_EXECUTABLE}")
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${PYTHON_INCLUDE_DIR})

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])" OUTPUT_VARIABLE PYTHON_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    link_directories(${PYTHON_LIBRARY_DIR})

    add_executable(IRS_CMake ${IRS_LIST})

    target_link_libraries(IRS_CMake python)
#   target_link_libraries(IRS_CMake /usr/lib/python3.6/config-3.6m-aarch64-linux-gnu/libpython3.6.so)

    target_include_directories(IRS_CMake PRIVATE ${Python_INCLUDE_DIRS})
   
    target_link_libraries(IRS_CMake SC_LIST)
    target_link_libraries(IRS_CMake IRSGUI_LIST)

    target_link_libraries(IRS_CMake Qt5::Core Qt5::Gui Qt5::Widgets)
    
    target_compile_options(IRS_CMake PRIVATE -Wall)

else()
    message(FATAL_ERROR "python not found")
endif()




message("lib : ${PYTHON_LIBRARY_DIR}")
message("include : ${PYTHON_INCLUDE_DIR}")

add_subdirectory(ServoControl)
add_subdirectory(IRSGUI)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
