cmake_minimum_required(VERSION 3.10)
project(SWI_KNMI)
enable_testing()

set(Python_EXECUTABLE python)
message("Using Python_EXECUTABLE = 'python', because you should build from a Python Virtual Environment that KNOWS which python you want.")

if(APPLE)
    message("Install homebrew and run this: brew install gcc libomp")
    set(HOMEBREW_PREFIX "/opt/homebrew"
    CACHE PATH "Path to Homebrew installation")

set(CMAKE_C_COMPILER "${HOMEBREW_PREFIX}/bin/gcc-12")
set(CMAKE_CXX_COMPILER "${HOMEBREW_PREFIX}/bin/g++-12")

set(CMAKE_PREFIX_PATH
    "${HOMEBREW_PREFIX}"
    # These libraries are keg-only and not loaded into
    # the root prefix by default (to avoid clashes).
    "${HOMEBREW_PREFIX}/opt/lapack"
    "${HOMEBREW_PREFIX}/opt/openblas"
    "${HOMEBREW_PREFIX}/opt/gcc/lib/gcc/11"
    "${HOMEBREW_PREFIX}/opt/libomp"
)

list(TRANSFORM CMAKE_PREFIX_PATH APPEND "/include"
     OUTPUT_VARIABLE CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES)
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

set(CMAKE_FIND_FRAMEWORK NEVER)
set(CMAKE_FIND_APPBUNDLE NEVER)

set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
endif()


file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(swi_knmi SHARED ${SOURCES})
include_directories("include")
target_compile_options(swi_knmi PRIVATE "-Wall" "-Wconversion" "-Wextra" "-O3" "-g" "-fopenmp")
target_link_libraries(swi_knmi gomp)
 
