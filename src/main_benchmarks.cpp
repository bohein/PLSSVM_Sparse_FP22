/**
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-benchmark` executable used for benchmarking possible benefits and shortcomings of storing training data in sparse matrix formats.
 */

#include "plssvm/core.hpp"

#include "plssvm/benchmarks/benchmark.hpp"             // plssvm::benchmarks::benchmark
#include "plssvm/benchmarks/benchmark_read_data.hpp"   // plssvm::benchmarks::benchmark_read_data
#include "plssvm/benchmarks/benchmark_q_kernel_openmp.hpp" // plssvm::benchmarks::benchmark_q_kernel_openmp
#include "plssvm/benchmarks/benchmark_svm_kernel_openmp.hpp" // plssvm::benchmarks::benchmark_svm_kernel_openmp
#include "plssvm/benchmarks/CUDA/benchmark_q_kernel_cuda.cuh" // plssvm::benchmarks::benchmark_q_kernel_cuda

#include "fmt/core.h"     // std::format
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <fstream>    // std::ofstream
#include <filesystem> // fs::create_directory

// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

std::string OUTPUT_DIR = "./benchmark_data/results"; // TODO: figure out relative paths somehow

int main(int argc, char *argv[]) {
    using namespace plssvm::benchmarks;

    std::vector<benchmark*> benchmarks;

    // Create Benchmarks
    //benchmarks.push_back(new benchmark_read_data);
    //benchmarks.push_back(new benchmark_q_kernel_openmp);
    //benchmarks.push_back(new benchmark_svm_kernel_openmp);
    benchmarks.push_back(new benchmark_q_kernel_cuda);

    for (benchmark* b : benchmarks) { // DO NOT PARALLELIZE THIS!
            b->run();
            if (!std::filesystem::is_directory(OUTPUT_DIR) || !std::filesystem::exists(OUTPUT_DIR)) {
                std::filesystem::create_directory(OUTPUT_DIR);
            }
            std::ofstream results_file(OUTPUT_DIR + "/"+ b->get_name() + ".csv");
            results_file << b->get_data();
            results_file.close();
    }
}