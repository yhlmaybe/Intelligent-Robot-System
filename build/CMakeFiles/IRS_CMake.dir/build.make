# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build

# Include any dependencies generated for this target.
include CMakeFiles/IRS_CMake.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/IRS_CMake.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/IRS_CMake.dir/flags.make

CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o: CMakeFiles/IRS_CMake.dir/flags.make
CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o: ../IRSInitiate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o -c /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/IRSInitiate.cpp

CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/IRSInitiate.cpp > CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.i

CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/IRSInitiate.cpp -o CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.s

CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.requires:

.PHONY : CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.requires

CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.provides: CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.requires
	$(MAKE) -f CMakeFiles/IRS_CMake.dir/build.make CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.provides.build
.PHONY : CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.provides

CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.provides.build: CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o


CMakeFiles/IRS_CMake.dir/Manager.cpp.o: CMakeFiles/IRS_CMake.dir/flags.make
CMakeFiles/IRS_CMake.dir/Manager.cpp.o: ../Manager.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/IRS_CMake.dir/Manager.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/IRS_CMake.dir/Manager.cpp.o -c /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/Manager.cpp

CMakeFiles/IRS_CMake.dir/Manager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/IRS_CMake.dir/Manager.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/Manager.cpp > CMakeFiles/IRS_CMake.dir/Manager.cpp.i

CMakeFiles/IRS_CMake.dir/Manager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/IRS_CMake.dir/Manager.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/Manager.cpp -o CMakeFiles/IRS_CMake.dir/Manager.cpp.s

CMakeFiles/IRS_CMake.dir/Manager.cpp.o.requires:

.PHONY : CMakeFiles/IRS_CMake.dir/Manager.cpp.o.requires

CMakeFiles/IRS_CMake.dir/Manager.cpp.o.provides: CMakeFiles/IRS_CMake.dir/Manager.cpp.o.requires
	$(MAKE) -f CMakeFiles/IRS_CMake.dir/build.make CMakeFiles/IRS_CMake.dir/Manager.cpp.o.provides.build
.PHONY : CMakeFiles/IRS_CMake.dir/Manager.cpp.o.provides

CMakeFiles/IRS_CMake.dir/Manager.cpp.o.provides.build: CMakeFiles/IRS_CMake.dir/Manager.cpp.o


CMakeFiles/IRS_CMake.dir/main.cpp.o: CMakeFiles/IRS_CMake.dir/flags.make
CMakeFiles/IRS_CMake.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/IRS_CMake.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/IRS_CMake.dir/main.cpp.o -c /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/main.cpp

CMakeFiles/IRS_CMake.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/IRS_CMake.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/main.cpp > CMakeFiles/IRS_CMake.dir/main.cpp.i

CMakeFiles/IRS_CMake.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/IRS_CMake.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/main.cpp -o CMakeFiles/IRS_CMake.dir/main.cpp.s

CMakeFiles/IRS_CMake.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/IRS_CMake.dir/main.cpp.o.requires

CMakeFiles/IRS_CMake.dir/main.cpp.o.provides: CMakeFiles/IRS_CMake.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/IRS_CMake.dir/build.make CMakeFiles/IRS_CMake.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/IRS_CMake.dir/main.cpp.o.provides

CMakeFiles/IRS_CMake.dir/main.cpp.o.provides.build: CMakeFiles/IRS_CMake.dir/main.cpp.o


CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o: CMakeFiles/IRS_CMake.dir/flags.make
CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o: IRS_CMake_autogen/mocs_compilation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o -c /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/IRS_CMake_autogen/mocs_compilation.cpp

CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/IRS_CMake_autogen/mocs_compilation.cpp > CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.i

CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/IRS_CMake_autogen/mocs_compilation.cpp -o CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.s

CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.requires:

.PHONY : CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.requires

CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.provides: CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.requires
	$(MAKE) -f CMakeFiles/IRS_CMake.dir/build.make CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.provides.build
.PHONY : CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.provides

CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.provides.build: CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o


# Object files for target IRS_CMake
IRS_CMake_OBJECTS = \
"CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o" \
"CMakeFiles/IRS_CMake.dir/Manager.cpp.o" \
"CMakeFiles/IRS_CMake.dir/main.cpp.o" \
"CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o"

# External object files for target IRS_CMake
IRS_CMake_EXTERNAL_OBJECTS =

IRS_CMake: CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o
IRS_CMake: CMakeFiles/IRS_CMake.dir/Manager.cpp.o
IRS_CMake: CMakeFiles/IRS_CMake.dir/main.cpp.o
IRS_CMake: CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o
IRS_CMake: CMakeFiles/IRS_CMake.dir/build.make
IRS_CMake: ServoControl/libSC_LIST.a
IRS_CMake: IRSGUI/libIRSGUI_LIST.a
IRS_CMake: /usr/lib/aarch64-linux-gnu/libQt5Widgets.so.5.9.5
IRS_CMake: /usr/lib/aarch64-linux-gnu/libQt5Gui.so.5.9.5
IRS_CMake: /usr/lib/aarch64-linux-gnu/libQt5Core.so.5.9.5
IRS_CMake: CMakeFiles/IRS_CMake.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable IRS_CMake"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/IRS_CMake.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/IRS_CMake.dir/build: IRS_CMake

.PHONY : CMakeFiles/IRS_CMake.dir/build

CMakeFiles/IRS_CMake.dir/requires: CMakeFiles/IRS_CMake.dir/IRSInitiate.cpp.o.requires
CMakeFiles/IRS_CMake.dir/requires: CMakeFiles/IRS_CMake.dir/Manager.cpp.o.requires
CMakeFiles/IRS_CMake.dir/requires: CMakeFiles/IRS_CMake.dir/main.cpp.o.requires
CMakeFiles/IRS_CMake.dir/requires: CMakeFiles/IRS_CMake.dir/IRS_CMake_autogen/mocs_compilation.cpp.o.requires

.PHONY : CMakeFiles/IRS_CMake.dir/requires

CMakeFiles/IRS_CMake.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/IRS_CMake.dir/cmake_clean.cmake
.PHONY : CMakeFiles/IRS_CMake.dir/clean

CMakeFiles/IRS_CMake.dir/depend:
	cd /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build /home/yhlmaybe/Documents/HLIRS/Intelligent-Robot-System/build/CMakeFiles/IRS_CMake.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/IRS_CMake.dir/depend

