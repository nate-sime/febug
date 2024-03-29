cmake_minimum_required(VERSION 3.10)

PROJECT(febug_pybind11)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create the binding library
pybind11_add_module(cpp SHARED
   febug/cpp_/meshquality.cpp
   febug/cpp_/febug.cpp)

target_link_libraries(cpp PUBLIC dolfinx)
target_include_directories(cpp PUBLIC dolfinx)

# Get python include-dirs
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import dolfinx.wrappers, sys; sys.stdout.write(str(dolfinx.wrappers.get_include_path()))"
  OUTPUT_VARIABLE DOLFINX_PY_DIR
  RESULT_VARIABLE DOLFINX_PY_COMMAND_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)

if (DOLFINX_PY_DIR)
  message(STATUS "Adding ${DOLFINX_PY_DIR} to include directories")
  target_include_directories(cpp PRIVATE ${DOLFINX_PY_DIR})
endif()

# Add strict compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(cpp PRIVATE -Wall;-Werror;-pedantic)
endif()


# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# Add DOLFINX libraries and other config
find_package(DOLFINX REQUIRED)
if (DOLFINX_FOUND)
   target_include_directories(cpp PRIVATE ${DOLFINX_INCLUDE_DIR})

# Find petsc4py through python
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})

endif()

