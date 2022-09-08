# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(fmt-populate NONE)

include(ExternalProject)
ExternalProject_Add(fmt-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/fmtlib/fmt.git" "GIT_TAG" "8.1.1" "GIT_SHALLOW" "TRUE" "set" "(" "FMT_TEST" "OFF" "CACHE" "BOOL" "FORCE" ")" "set" "(" "FMT_DOC" "OFF" "CACHE" "BOOL" "FORCE" ")" "set" "(" "FMT_INSTALL" "OFF" "CACHE" "BOOL" "FORCE" ")" "set" "(" "FMT_SYSTEM_HEADERS" "ON" "CACHE" "BOOL" "FORCE" ")"
                    SOURCE_DIR          "/home/boschmhh/PLSSVM/_deps/fmt-src"
                    BINARY_DIR          "/home/boschmhh/PLSSVM/_deps/fmt-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
