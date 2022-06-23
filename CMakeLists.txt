cmake_minimum_required(VERSION 2.8.4)

PROJECT(MatchOrNoMatch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
find_package(OpenCV REQUIRED)
find_package (PkgConfig REQUIRED)
pkg_check_modules (GTK3 REQUIRED gtk+-3.0)

add_definitions (${GTK3_CFLAGS_OTHER})
include_directories(${GTK3_INCLUDE_DIRS})
link_directories (${GTK3_LIBRARY_DIRS})
add_executable(matchornomatch main.cpp App.cpp Matching.cpp VariablesCalculation.cpp FileHandling.cpp)
target_link_libraries(matchornomatch ${OpenCV_LIBS} ${GTK3_LIBRARIES})
