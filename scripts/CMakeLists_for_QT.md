#BUILD QT with CMake
project(Helloworld)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_PREFIX_PATH $ENV{QTDIR512MINGW})
find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

set(HEADER_LIST
    mainwindow.h
)

set(SOURCE_LIST
    main.cpp
    mainwindow.cpp
)

set(LIB_SRC)

add_library(libHello ${LIB_SRC})

add_executable(${PROJECT_NAME} ${HEADER_LIST} ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Widgets QT5::Gui Qt5::Network)

set(CMAKE_VERBOSE_MAKEFILE ON)
