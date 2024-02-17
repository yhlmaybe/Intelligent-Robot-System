cmake_minimum_required(VERSION 3.0.0)
project(Intelligent-Robot-System VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Debug)

include_directories(
${CMAKE_SOURCE_DIR}/include
${CMAKE_SOURCE_DIR}/SteeringEngineControl
)

include(CTest)
enable_testing()

aux_source_directory(. IRS_LIST)

add_subdirectory(SteeringEngineControl)

add_executable(IRS_CMake ${IRS_LIST})

target_link_libraries(IRS_CMake SEC_LIST)

target_compile_options(IRS_CMake -wall)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
