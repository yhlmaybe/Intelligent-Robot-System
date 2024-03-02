set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/usr/share/cmake-3.10/Modules")

find_program(PYTHON_EXECUTABLE python)

if(PYTHON_EXECUTABLE)
    message(STATUS "python executable found at ${PYTHON_EXECUTABLE}")
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${PYTHON_INCLUDE_DIR})

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])" OUTPUT_VARIABLE PYTHON_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    link_directories(${PYTHON_LIBRARY_DIR})


    aux_source_directory(. SC_LIST)
    add_library(SC_LIST ${SC_LIST})

    target_link_libraries(SC_LIST python)
#   target_link_libraries(SC_LIST /usr/lib/python3.6/config-3.6m-aarch64-linux-gnu/libpython3.6.so)

    target_include_directories(SC_LIST PRIVATE ${Python_INCLUDE_DIRS})
    
    target_compile_options(IRS_CMake PRIVATE -Wall)

else()
    message(FATAL_ERROR "python not found")
endif()
