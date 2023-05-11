#-------------------------------------------------------------------------------
#	CMAKE SETTINGS
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8.12)
set(QT_MINIMUM_VERSION 5.9.0)

#-------------------------------------------------------------------------------
#	SOURCE CODE AND USER SETTINGS
#-------------------------------------------------------------------------------

set(SOURCE_FILES_LIB src/covidsim.cpp
    src/Configuration.cpp
    src/Convert.cpp
    src/infection.cpp
    src/motion.cpp
    src/path_planning.cpp
    src/Population_trackers.cpp
    src/RandomDevice.cpp
    src/simulation.cpp
    src/utilities.cpp
)

set(SOURCE_FILES_CUDA src/kernels.cu
    src/CUDA_functions.cu
    src/Utilities.cu
)

set(SOURCE_FILES src/mainwindow.cpp
	src/covidsim_ui.cpp
    src/qcustomplot.cpp
    src/Worker.cpp
	src/io_blackbox_functions.cpp
)

set(HEADER_FILES header/mainwindow.h
    header/qcustomplot.h
    header/Worker.h
	header/io_blackbox_functions.h
	header/kernels.cuh
	header/CUDA_functions.cuh
)

set(BUILD_WTIH_CUDA ON) # whether to build GPU accelerated application

#-------------------------------------------------------------------------------
#	PROJECT SETTINGS
#-------------------------------------------------------------------------------

set(PROJECT_NAME "COVID_SIM_UI")
cmake_policy(SET CMP0048 NEW)
project(${PROJECT_NAME} VERSION 0.1.0)

set(TARGET_NAME "COVID_SIM_UI")

#-------------------------------------------------------------------------------
#	CUDA Settings
#-------------------------------------------------------------------------------

if(${BUILD_WTIH_CUDA} STREQUAL ON)
	find_package(CUDA REQUIRED)

	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)

	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		set(CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS};
			-g -G -w --keep -gencode arch=compute_75,code=sm_75
		)
	else()
		# Pass options to NVCC
		set(CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS};
			-O3 -w -gencode arch=compute_75,code=sm_75
		)
	endif()

endif()

#-------------------------------------------------------------------------------
#	Qt Settings
#-------------------------------------------------------------------------------

# Required Qt settings
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(QT_USE_QTMAIN TRUE)

# Setup the cmake prefix path so Qt can be found. Use the QTDIR variable if defined
if(NOT DEFINED ENV{QTDIR})
	MESSAGE(STATUS "QTDIR environment variable not set. Be sure to set the QTDIR
	cmake cache variable to the Qt root install location")
endif()

set(QTDIR $ENV{QTDIR} CACHE STRING "Qt install path")
list(APPEND CMAKE_PREFIX_PATH ${QTDIR})

# variable_watch(CMAKE_PREFIX_PATH)

# Find the necessary Qt5 modules
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5PrintSupport REQUIRED)
find_package(Qt5Sql REQUIRED)

# Test for supported Qt version
find_program(QMAKE_EXECUTABLE NAMES qmake HINTS ${QTDIR} ENV QTDIR PATH_SUFFIXES bin)
execute_process(COMMAND ${QMAKE_EXECUTABLE} -query QT_VERSION OUTPUT_VARIABLE QT_VERSION)
if(QT_VERSION VERSION_LESS QT_MINIMUM_VERSION)
	MESSAGE(FATAL_ERROR "Minimum supported Qt version: ${QT_MINIMUM_VERSION}. Installed version: ${QT_VERSION}")
endif()

# find thread library 
find_package(Threads REQUIRED)

#-------------------------------------------------------------------------------
#	COMPILE FLAGS
#-------------------------------------------------------------------------------

# Common compiler flags for all build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

# Debug build flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
# Release build flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	set(GUI_TYPE WIN32)
	add_compile_options(/MP)
elseif(CMAKE_COMPILER_IS_GNUCXX)
	add_compile_options(-std=c++17)
endif(MSVC)


#-------------------------------------------------------------------------------
#	ADDITIONAL LIBRARIES
#-------------------------------------------------------------------------------

list(APPEND LIBRARIES
	Qt5::Core
	Qt5::Gui
	Qt5::Widgets
	Qt5::PrintSupport
	Qt5::Sql
	Threads::Threads
	${CUDA_LIBRARIES}
	${CUDA_CUBLAS_LIBRARIES}
)

# Remove duplicate library dependencies. 
list(REMOVE_DUPLICATES LIBRARIES)

#-------------------------------------------------------------------------------
#	GENERATE EXECUTABLE
#-------------------------------------------------------------------------------

# Set additional include directories
include_directories(${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Targets for compilation (COVID simulation library (covidsim))
# add_executable(covidsim ${SOURCE_FILES})
add_library(covidsim ${SOURCE_FILES_LIB})
target_include_directories(covidsim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/header
)

if(${BUILD_WTIH_CUDA} STREQUAL ON)
	# Targets for compilation (CUDA acceleration (cudasim)
	# add_executable(covidsim ${SOURCE_FILES})
	cuda_add_library(cudasim ${SOURCE_FILES_CUDA})
	target_include_directories(cudasim PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/Include
		${CMAKE_CURRENT_SOURCE_DIR}/header
	)
endif()

# Generate the executable
add_executable(${TARGET_NAME} ${GUI_TYPE} ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/header
)
# link libraries
# set_property(TARGET cudasim PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(${BUILD_WTIH_CUDA} STREQUAL ON)
	target_link_libraries(${TARGET_NAME} PRIVATE covidsim cudasim ${LIBRARIES})
else()
	target_link_libraries(${TARGET_NAME} PRIVATE covidsim ${LIBRARIES})
endif()

# Windows specific build steps
if(WIN32)
	# Run winddeployqt if it can be found
	find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt HINTS ${QTDIR} ENV QTDIR PATH_SUFFIXES bin)
	add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
	COMMAND ${WINDEPLOYQT_EXECUTABLE} $<TARGET_FILE:${TARGET_NAME}>)
	# Gather up all the dependencies in the bin dir
	# Lack of a trailing slash means install this directory, trailing slash means install everything in this directory.
	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Release/
			RUNTIME DESTINATION bin
			FILES_MATCHING
			PATTERN "_CPack*" EXCLUDE
			PATTERN "*.dll"
			)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)