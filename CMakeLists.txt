cmake_minimum_required(VERSION 3.0.0)
project(Intelligent-Robot-System VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/usr/share/cmake-3.10/Modules")

set(CMAKE_BUILD_TYPE Debug)

include_directories(
${CMAKE_SOURCE_DIR}/include
${CMAKE_SOURCE_DIR}/ServoControl
)

include(CTest)
enable_testing()

aux_source_directory(. IRS_LIST)

find_program(PYTHON_EXECUTABLE python)

if(PYTHON_EXECUTABLE)
    message(STATUS "python executable found at ${PYTHON_EXECUTABLE}")
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${PYTHON_INCLUDE_DIR})

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])" OUTPUT_VARIABLE PYTHON_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    link_directories(${PYTHON_LIBRARY_DIR})

    add_executable(IRS_CMake ${IRS_LIST})

    target_link_libraries(IRS_CMake python)

else()
    message(FATAL_ERROR "python not found")
endif()




message("lib : ${PYTHON_LIBRARY_DIR}")
message("include : ${PYTHON_INCLUDE_DIR}")

add_subdirectory(ServoControl)


#if(PYTHON_EXECUTABLE)
#    message(STATUS "Python executable found at: ${PYTHON_EXECUTABLE}")
#    
#    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
#    include_directories(${PYTHON_INCLUDE_DIR})
#
#    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])" OUTPUT_VARIABLE PYTHON_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
#    link_directories(${PYTHON_LIBRARY_DIR})
#    
#    add_executable(IRS_CMake ${IRS_LIST})
#
#    #target_link_libraries(IRS_CMake PRIVATE ${Python_LIBRARIES})
#
#    target_compile_options(IRS_CMake PRIVATE -Wall)
#    target_link_libraries(IRS_CMake SC_LIST)
#    target_link_libraries(IRS_CMake ${PYTHON_LIBRARY_DIR}/libpython3.so)
#    else()
#    message(FATAL_ERROR "python not found.")
#endif()


#target_include_directories(IRS_CMake PRIVATE ${Python_INCLUDE_DIRS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
