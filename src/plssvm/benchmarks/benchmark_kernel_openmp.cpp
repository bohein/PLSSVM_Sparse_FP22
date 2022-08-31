/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks
 */

#include "plssvm/benchmarks/benchmark_kernel_openmp.hpp"

#include <numeric>
#include <iostream>

namespace plssvm::benchmarks {

benchmark_kernel_openmp::benchmark_kernel_openmp() : benchmark{"Kernel functions (OpenMP)"} {}

void benchmark_kernel_openmp::run() {
    using real_type = double;

    evaluate_dataset("tiny (~150)", DATASET_TINY);
    evaluate_dataset("small (~5000)", DATASET_SMALL);
    evaluate_dataset("medium (~50000)", DATASET_MEDIUM);
    evaluate_dataset("large (~250000)", DATASET_LARGE);
}

void benchmark_kernel_openmp::evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {
    using real_type = double;

    plssvm::parameter<real_type> params;

    std::vector<std::vector<real_type>> data_dense;
    plssvm::openmp::coo<real_type> data_coo{};
    plssvm::openmp::csr<real_type> data_csr{};

    auto data_ptr_dense = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data_dense));
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo));
    auto data_ptr_csr = std::make_shared<const plssvm::openmp::csr<real_type>>(std::move(data_csr));

    // dense
    std::vector<std::chrono::nanoseconds> raw_runtimes_dense;
    // TODO: implement 

    // coo
    std::vector<std::chrono::nanoseconds> raw_runtimes_coo;
    // TODO: implement 

    // csr
    std::vector<std::chrono::nanoseconds> raw_runtimes_csr;
    // TODO: implement 
    
    sub_benchmark_names.push_back(sub_benchmark_name + " dense");
    sub_benchmark_names.push_back(sub_benchmark_name + " COO");
    sub_benchmark_names.push_back(sub_benchmark_name + " CSR");
    
   // mean
    runtimes_mean.push_back(mean(raw_runtimes_dense));
    runtimes_mean.push_back(mean(raw_runtimes_coo));
    runtimes_mean.push_back(mean(raw_runtimes_csr));

    // median
    runtimes_median.push_back(median(raw_runtimes_dense));
    runtimes_median.push_back(median(raw_runtimes_coo));
    runtimes_median.push_back(median(raw_runtimes_csr));

    // min
    runtimes_min.push_back(min(raw_runtimes_dense));
    runtimes_min.push_back(min(raw_runtimes_coo));
    runtimes_min.push_back(min(raw_runtimes_csr));

    // standard deviation
    runtimes_std_deviation.push_back(std_deviation(raw_runtimes_dense));
    runtimes_std_deviation.push_back(std_deviation(raw_runtimes_coo));
    runtimes_std_deviation.push_back(std_deviation(raw_runtimes_csr));
}

}  // namespace plssvm::benchmarks
