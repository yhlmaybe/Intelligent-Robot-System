cmake_minimum_required(VERSION 3.0.0)
project(Intelligent-Robot-System VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(Intelligent-Robot-System_CMake main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
