# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\Maathess\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\212.5080.54\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\Maathess\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\212.5080.54\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\PMC3.dir\depend.make
# Include the progress variables for this target.
include CMakeFiles\PMC3.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\PMC3.dir\flags.make

CMakeFiles\PMC3.dir\library.cpp.obj: CMakeFiles\PMC3.dir\flags.make
CMakeFiles\PMC3.dir\library.cpp.obj: ..\library.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PMC3.dir/library.cpp.obj"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\PMC3.dir\library.cpp.obj /FdCMakeFiles\PMC3.dir\ /FS -c C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\library.cpp
<<

CMakeFiles\PMC3.dir\library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PMC3.dir/library.cpp.i"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\PMC3.dir\library.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\library.cpp
<<

CMakeFiles\PMC3.dir\library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PMC3.dir/library.cpp.s"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\PMC3.dir\library.cpp.s /c C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\library.cpp
<<

# Object files for target PMC3
PMC3_OBJECTS = \
"CMakeFiles\PMC3.dir\library.cpp.obj"

# External object files for target PMC3
PMC3_EXTERNAL_OBJECTS =

PMC3.dll: CMakeFiles\PMC3.dir\library.cpp.obj
PMC3.dll: CMakeFiles\PMC3.dir\build.make
PMC3.dll: CMakeFiles\PMC3.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library PMC3.dll"
	C:\Users\Maathess\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\212.5080.54\bin\cmake\win\bin\cmake.exe -E vs_link_dll --intdir=CMakeFiles\PMC3.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\mt.exe --manifests -- C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\link.exe /nologo @CMakeFiles\PMC3.dir\objects1.rsp @<<
 /out:PMC3.dll /implib:PMC3.lib /pdb:C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\cmake-build-debug\PMC3.pdb /dll /version:0.0 /machine:x64 /debug /INCREMENTAL  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  
<<

# Rule to build all files generated by this target.
CMakeFiles\PMC3.dir\build: PMC3.dll
.PHONY : CMakeFiles\PMC3.dir\build

CMakeFiles\PMC3.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\PMC3.dir\cmake_clean.cmake
.PHONY : CMakeFiles\PMC3.dir\clean

CMakeFiles\PMC3.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\cmake-build-debug C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\cmake-build-debug C:\Users\Maathess\Desktop\Projet_annuel_flag\PMC\cmake-build-debug\CMakeFiles\PMC3.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\PMC3.dir\depend

