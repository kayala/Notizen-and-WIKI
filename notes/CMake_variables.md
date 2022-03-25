* CMAKE_BINARY_DIR
PROJECT_BINARY_DIR
<projectname>_BINARY_DIR

These three variables refer to the same content. If it is in source compilation, it refers to the top-level directory of the project. If it is out-of-source compilation, it refers to the directory where the project compilation occurs.

* CMAKE_SOURCE_DIR
PROJECT_SOURCE_DIR
<projectname>_SOURCE_DIR

No matter which compilation method is adopted, it is the top-level directory of the project.

* CMAKE_CURRENT_SOURCE_DIR

Refers to the path where the currently processed CMakeLists.txt is located, such as the src subdirectory we mentioned above.

* CMAKE_CURRRENT_BINARY_DIR 

If it is in-source compilation, it is consistent with CMAKE_CURRENT_SOURCE_DIR. If it is out-of-source compilation, it refers to the target compilation directory.
Use the ADD_SUBDIRECTORY (src bin) to change the value of this variable.
Using SET(EXECUTABLE_OUTPUT_PATH <new path>) will not affect this variable, it only modifies the path where the final target file is stored.

* CMAKE_CURRENT_LIST_FILE

Output the full path of CMakeLists.txt that calls this variable

* CMAKE_CURRENT_LIST_LINE

Output the line where this variable is located

* CMAKE_MODULE_PATH

This variable is used to define the path where your cmake module is located. If your project is more complicated, you may write some cmake modules yourself. These cmake modules are released with your project. In order for cmake to find these modules when processing CMakeLists.txt, you need to pass the SET command to change your cmake Set the module path.
for example
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
At this time you can use INCLUDE instruction to call your own module.

* EXECUTABLE_OUTPUT_PATH 
 LIBRARY_OUTPUT_PATH

They are used to redefine the storage directory of the final result. 

* PROJECT_NAME

Returns the name of the project defined by the PROJECT command

* CMAKE_BUILD_TYPE

Generate debug version and release version of the program.

Possible values are Debug Release RelWithDebInfo and MinSizeRel. When the value of this variable is Debug, CMake will use the strings in the variables CMAKE_CXX_FLAGS_DEBUG and CMAKE_C_FLAGS_DEBUG as compilation options to generate Makefile. When the value of this variable is Release, the project will use the variables CMAKE_CXX_FLAGS_RELEASE and CMAKE_C_FLAGS_RELEASE to generate Makefile.

Now suppose there is only one file main.cpp in the project, the following is a CMakeList.txt that can choose to generate debug version and release version of the program:

PROJECT(main)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_SOURCE_DIR .)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
AUX_SOURCE_DIRECTORY(. DIR_SRCS)
ADD_EXECUTABLE(main ${DIR_SRCS})

Lines 5 and 6 set two variables CMAKE_CXX_FLAGS_DEBUG and

CMAKE_CXX_FLAGS_RELEASE, these two variables are the compilation options for debug and release respectively.

* CMAKE_C_FLAGS

Specify the compilation options when compiling C files, such as -g to specify debugging information. You can also add compilation options through the add_definitions command.

* EXECUTABLE_OUTPUT_PATH
LIBRARY_OUTPUT_PATH

Specify the path where the executable file is stored and the path where the library file is placed