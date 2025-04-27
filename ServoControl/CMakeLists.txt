#[[
find_program(PYTHON_EXECUTABLE python)

if(PYTHON_EXECUTABLE)
    message(STATUS "python executable found at ${PYTHON_EXECUTABLE}")
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${PYTHON_INCLUDE_DIR})

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])" OUTPUT_VARIABLE PYTHON_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    link_directories(${PYTHON_LIBRARY_DIR})

    aux_source_directory(. SC_LIST)
    add_library(SC_LIST ${SC_LIST})

    #target_link_libraries(SC_LIST PUBLIC /usr/lib/python3.8/config-3.8-aarch64-linux-gnu/libpython3.8.so)
    target_link_libraries(SC_LIST PUBLIC python)
    target_include_directories(SC_LIST PUBLIC ${Python_INCLUDE_DIRS})

    message("lib : ${PYTHON_LIBRARY_DIR}")
    message("include : ${PYTHON_INCLUDE_DIR}")

    find_path(TINYXML2_INCLUDE_DIR NAMES tinyxml2.h)
    find_library(TINYXML2_LIBRARY NAMES tinyxml2)
    target_include_directories(SC_LIST PUBLIC ${TINYXML2_INCLUDE_DIR})
    target_link_libraries(SC_LIST PUBLIC ${TINYXML2_LIBRARY})

else()
    message(FATAL_ERROR "python not found")
endif()
]]


aux_source_directory(. SC_LIST)
add_library(SC_LIST ${SC_LIST})

#target_link_libraries(SC_LIST PUBLIC /usr/lib/python3.8/config-3.8-aarch64-linux-gnu/libpython3.8.so)
target_link_libraries(SC_LIST PUBLIC python3.8)
target_include_directories(SC_LIST PUBLIC ${Python_INCLUDE_DIRS})

find_path(TINYXML2_INCLUDE_DIR NAMES tinyxml2.h)
find_library(TINYXML2_LIBRARY NAMES tinyxml2)
target_include_directories(SC_LIST PUBLIC ${TINYXML2_INCLUDE_DIR})
target_link_libraries(SC_LIST PUBLIC ${TINYXML2_LIBRARY})
