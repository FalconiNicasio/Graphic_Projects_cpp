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
CMAKE_SOURCE_DIR = /mnt/d/4th_year/4_2CIS441/project1/project1D

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/4th_year/4_2CIS441/project1/project1D

# Include any dependencies generated for this target.
include CMakeFiles/project1D.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/project1D.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/project1D.dir/flags.make

CMakeFiles/project1D.dir/project1D.cxx.o: CMakeFiles/project1D.dir/flags.make
CMakeFiles/project1D.dir/project1D.cxx.o: project1D.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/4th_year/4_2CIS441/project1/project1D/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/project1D.dir/project1D.cxx.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/project1D.dir/project1D.cxx.o -c /mnt/d/4th_year/4_2CIS441/project1/project1D/project1D.cxx

CMakeFiles/project1D.dir/project1D.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/project1D.dir/project1D.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/4th_year/4_2CIS441/project1/project1D/project1D.cxx > CMakeFiles/project1D.dir/project1D.cxx.i

CMakeFiles/project1D.dir/project1D.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/project1D.dir/project1D.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/4th_year/4_2CIS441/project1/project1D/project1D.cxx -o CMakeFiles/project1D.dir/project1D.cxx.s

CMakeFiles/project1D.dir/project1D.cxx.o.requires:

.PHONY : CMakeFiles/project1D.dir/project1D.cxx.o.requires

CMakeFiles/project1D.dir/project1D.cxx.o.provides: CMakeFiles/project1D.dir/project1D.cxx.o.requires
	$(MAKE) -f CMakeFiles/project1D.dir/build.make CMakeFiles/project1D.dir/project1D.cxx.o.provides.build
.PHONY : CMakeFiles/project1D.dir/project1D.cxx.o.provides

CMakeFiles/project1D.dir/project1D.cxx.o.provides.build: CMakeFiles/project1D.dir/project1D.cxx.o


# Object files for target project1D
project1D_OBJECTS = \
"CMakeFiles/project1D.dir/project1D.cxx.o"

# External object files for target project1D
project1D_EXTERNAL_OBJECTS =

project1D: CMakeFiles/project1D.dir/project1D.cxx.o
project1D: CMakeFiles/project1D.dir/build.make
project1D: /usr/bin/VTK-8.1.2/lib/libvtkDomainsChemistryOpenGL2-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersFlowPaths-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersGeneric-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersHyperTree-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersParallelImaging-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersPoints-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersProgrammable-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersSMP-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersSelection-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersTexture-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersTopology-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersVerdict-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkGeovisCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOAMR-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOEnSight-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOExodus-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOExportOpenGL2-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOImport-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOInfovis-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOLSDyna-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOMINC-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOMovie-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOPLY-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOParallel-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOParallelXML-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOSQL-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOTecplotTable-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOVideo-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingMorphological-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingStatistics-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingStencil-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkInteractionImage-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingContextOpenGL2-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingImage-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingLOD-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingVolumeOpenGL2-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkViewsContext2D-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkViewsInfovis-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkDomainsChemistry-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkverdict-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkproj4-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersAMR-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOExport-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingGL2PSOpenGL2-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkgl2ps-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtklibharu-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtklibxml2-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkoggtheora-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersParallel-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkexoIIc-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOGeometry-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIONetCDF-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtknetcdfcpp-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkNetCDF-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkhdf5_hl-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkhdf5-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkjsoncpp-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkParallelCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOLegacy-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtksqlite-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingOpenGL2-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkglew-8.1.so.1
project1D: /usr/lib/x86_64-linux-gnu/libSM.so
project1D: /usr/lib/x86_64-linux-gnu/libICE.so
project1D: /usr/lib/x86_64-linux-gnu/libX11.so
project1D: /usr/lib/x86_64-linux-gnu/libXext.so
project1D: /usr/lib/x86_64-linux-gnu/libXt.so
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingMath-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkChartsCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingContext2D-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersImaging-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkInfovisLayout-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkInfovisCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkViewsCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkInteractionWidgets-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersHybrid-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingGeneral-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingSources-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersModeling-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingHybrid-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOImage-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkDICOMParser-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkmetaio-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkpng-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtktiff-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkjpeg-8.1.so.1
project1D: /usr/lib/x86_64-linux-gnu/libm.so
project1D: /usr/bin/VTK-8.1.2/lib/libvtkInteractionStyle-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersExtraction-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersStatistics-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingFourier-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkalglib-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingAnnotation-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingColor-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingVolume-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkImagingCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOXML-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOXMLParser-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkIOCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtklz4-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkexpat-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingLabel-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingFreeType-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkRenderingCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonColor-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersGeometry-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersSources-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersGeneral-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonComputationalGeometry-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkFiltersCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonExecutionModel-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonDataModel-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonMisc-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonSystem-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtksys-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonTransforms-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonMath-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkCommonCore-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkfreetype-8.1.so.1
project1D: /usr/bin/VTK-8.1.2/lib/libvtkzlib-8.1.so.1
project1D: CMakeFiles/project1D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/4th_year/4_2CIS441/project1/project1D/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable project1D"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/project1D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/project1D.dir/build: project1D

.PHONY : CMakeFiles/project1D.dir/build

CMakeFiles/project1D.dir/requires: CMakeFiles/project1D.dir/project1D.cxx.o.requires

.PHONY : CMakeFiles/project1D.dir/requires

CMakeFiles/project1D.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/project1D.dir/cmake_clean.cmake
.PHONY : CMakeFiles/project1D.dir/clean

CMakeFiles/project1D.dir/depend:
	cd /mnt/d/4th_year/4_2CIS441/project1/project1D && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/4th_year/4_2CIS441/project1/project1D /mnt/d/4th_year/4_2CIS441/project1/project1D /mnt/d/4th_year/4_2CIS441/project1/project1D /mnt/d/4th_year/4_2CIS441/project1/project1D /mnt/d/4th_year/4_2CIS441/project1/project1D/CMakeFiles/project1D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/project1D.dir/depend

