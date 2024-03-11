# CMake generated Testfile for 
# Source directory: /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/ROSManager
# Build directory: /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/ROSManager
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(cppcheck "/usr/bin/python" "-u" "/opt/ros/eloquent/share/ament_cmake_test/cmake/run_test.py" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/test_results/Intelligent-Robot-System/cppcheck.xunit.xml" "--package-name" "Intelligent-Robot-System" "--output-file" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/ament_cppcheck/cppcheck.txt" "--command" "/opt/ros/eloquent/bin/ament_cppcheck" "--xunit-file" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/test_results/Intelligent-Robot-System/cppcheck.xunit.xml" "--include_dirs" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/include" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/ServoControl" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/IRSGUI" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/ROSMANAGER_LIST")
set_tests_properties(cppcheck PROPERTIES  LABELS "cppcheck;linter" TIMEOUT "120" WORKING_DIRECTORY "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/ROSManager")
add_test(lint_cmake "/usr/bin/python" "-u" "/opt/ros/eloquent/share/ament_cmake_test/cmake/run_test.py" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/test_results/Intelligent-Robot-System/lint_cmake.xunit.xml" "--package-name" "Intelligent-Robot-System" "--output-file" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/ament_lint_cmake/lint_cmake.txt" "--command" "/opt/ros/eloquent/bin/ament_lint_cmake" "--xunit-file" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/test_results/Intelligent-Robot-System/lint_cmake.xunit.xml")
set_tests_properties(lint_cmake PROPERTIES  LABELS "lint_cmake;linter" TIMEOUT "60" WORKING_DIRECTORY "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/ROSManager")
add_test(uncrustify "/usr/bin/python" "-u" "/opt/ros/eloquent/share/ament_cmake_test/cmake/run_test.py" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/test_results/Intelligent-Robot-System/uncrustify.xunit.xml" "--package-name" "Intelligent-Robot-System" "--output-file" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/ament_uncrustify/uncrustify.txt" "--command" "/opt/ros/eloquent/bin/ament_uncrustify" "--xunit-file" "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/test_results/Intelligent-Robot-System/uncrustify.xunit.xml")
set_tests_properties(uncrustify PROPERTIES  LABELS "uncrustify;linter" TIMEOUT "60" WORKING_DIRECTORY "/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/ROSManager")