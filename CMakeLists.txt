cmake_minimum_required(VERSION 3.10)
project(SWI_KNMI)
enable_testing()

set(Python_EXECUTABLE python)
message("Using Python_EXECUTABLE = 'python', because you should build from a Python Virtual Environment that KNOWS which python you want.")

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(swi_knmi SHARED ${SOURCES})
include_directories("include")
target_compile_options(swi_knmi PRIVATE "-Wall" "-Wconversion" "-Wextra" "-O3" "-g" "-fopenmp")
target_link_libraries(swi_knmi gomp)
 
