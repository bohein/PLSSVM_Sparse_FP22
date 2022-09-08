# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(fast_float-populate NONE)

include(ExternalProject)
ExternalProject_Add(fast_float-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/fastfloat/fast_float" "GIT_TAG" "v3.4.0" "GIT_SHALLOW" "TRUE" "set" "(" "FASTFLOAT_TEST" "OFF" "CACHE" "BOOL" "FORCE" ")" "set" "(" "FASTFLOAT_SANITIZE" "OFF" "CACHE" "BOOL" "FORCE" ")"
                    SOURCE_DIR          "/home/boschmhh/PLSSVM/_deps/fast_float-src"
                    BINARY_DIR          "/home/boschmhh/PLSSVM/_deps/fast_float-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
