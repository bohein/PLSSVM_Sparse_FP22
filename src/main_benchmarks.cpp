/**
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-benchmark` executable used for benchmarking possible benefits and shortcomings of storing training data in sparse matrix formats.
 */

#include "plssvm/core.hpp"

#include "plssvm/benchmarks/benchmark_read_data.hpp"

#include "fmt/core.h"     // std::format
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl

// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

int main(int argc, char *argv[]) {
    plssvm::benchmarks::benchmark_read_data read_data;
    read_data.run();
    fmt::print(read_data.get_name() + "\n" + read_data.get_data() + "\n");
}
