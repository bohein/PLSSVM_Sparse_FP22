## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was plssvmConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
find_dependency(OpenMP QUIET)

# check if the OpenMP backend is required
set(PLSSVM_HAS_OPENMP_BACKEND plssvm-OpenMP)
if(PLSSVM_HAS_OPENMP_BACKEND)
    find_dependency(OpenMP REQUIRED)
endif()

# check if fmt has been installed via FetchContent
set(PLSSVM_FOUND_FMT 0)
if(PLSSVM_FOUND_FMT)
    find_dependency(fmt REQUIRED)
endif()

# check if the CUDA backend is required
set(PLSSVM_HAS_CUDA_BACKEND )
if(PLSSVM_HAS_CUDA_BACKEND)
    enable_language(CUDA)
endif()

# check if the OpenCL backend is required
set(PLSSVM_HAS_OPENCL_BACKEND )
if(PLSSVM_HAS_OPENCL_BACKEND)
    find_dependency(OpenCL REQUIRED)
endif()

# check if the SYCL implementation hipSYCL backend is required
set(PLSSVM_HAS_SYCL_BACKEND_HIPSYCL )
if(PLSSVM_HAS_SYCL_BACKEND_HIPSYCL)
    set(HIPSYCL_TARGETS cpu:avx512;nvidia:sm_86)
    find_dependency(hipSYCL CONFIG REQUIRED)
    message(STATUS "Found hipSYCL with ${HIPSYCL_TARGETS}")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/plssvmTargets.cmake")
check_required_components("plssvm")
