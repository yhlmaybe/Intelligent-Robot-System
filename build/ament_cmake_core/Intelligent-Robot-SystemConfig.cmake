# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_Intelligent-Robot-System_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED Intelligent-Robot-System_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(Intelligent-Robot-System_FOUND FALSE)
  elseif(NOT Intelligent-Robot-System_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(Intelligent-Robot-System_FOUND FALSE)
  endif()
  return()
endif()
set(_Intelligent-Robot-System_CONFIG_INCLUDED TRUE)

# output package information
if(NOT Intelligent-Robot-System_FIND_QUIETLY)
  message(STATUS "Found Intelligent-Robot-System: 0.0.0 (${Intelligent-Robot-System_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'Intelligent-Robot-System' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  message(WARNING "${_msg}")
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(Intelligent-Robot-System_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${Intelligent-Robot-System_DIR}/${_extra}")
endforeach()
