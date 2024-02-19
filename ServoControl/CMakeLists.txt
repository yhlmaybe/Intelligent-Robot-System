set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/usr/share/cmake-3.10/Modules")
find_package(PythonInterp REQUIRED COMPONENTS Interpreter Development)

aux_source_directory(. SC_LIST)
add_library(SC_LIST ${SC_LIST})

target_include_directories(SC_LIST PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(SC_LIST ${Python_LIBRARIES})