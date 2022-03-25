# CmakeLists WIKI

## Variables reference

${}

## Customize variables

### Implicit definition

<projectname>_BINARY_DIR

<projectname>_SOURCE_DIR

### Explicit definition

SET(HELLO_SRC main.SOURCE_PATHc) 

reference variable inside PROJECT_BINARY_DIR with ${HELLO_SRC}

## cmake calls environment variables

$ENV{NAME}

exp: MESSAGE(STATUS “HOME dir: $ENV{HOME}”)

SET(ENV{VAR} value)

* CMAKE_INCLUDE_CURRENT_DIR

Automatically add CMAKE_CURRENT_BINARY_DIR and CMAKE_CURRENT_SOURCE_DIR to the current process
CMakeLists.txt. It is equivalent to adding in each CMakeLists.txt:
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
${CMAKE_CURRENT_SOURCE_DIR})

* CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE

The header file directory provided by the project is always in front of the system header file directory. When the header file you define does conflict with the system, you can provide some help

* CMAKE_INCLUDE_PATH 
CMAKE_LIBRARY_PATH