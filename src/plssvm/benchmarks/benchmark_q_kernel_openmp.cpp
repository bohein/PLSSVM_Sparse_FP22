/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks
 */

#include "plssvm/benchmarks/benchmark_q_kernel_openmp.hpp"

#include "plssvm/backends/OpenMP/q_kernel.hpp"

#include <numeric>
#include <iostream>

namespace plssvm::benchmarks {

benchmark_q_kernel_openmp::benchmark_q_kernel_openmp() : benchmark{"Q-Kernels (OpenMP)"} {}

void benchmark_q_kernel_openmp::run() {
    using real_type = double;

    evaluate_dataset("tiny (~150)", DATASET_TINY);
    evaluate_dataset("small (~5000)", DATASET_SMALL);
    evaluate_dataset("medium (~50000)", DATASET_MEDIUM);
    evaluate_dataset("large (~250000)", DATASET_LARGE);
}

void benchmark_q_kernel_openmp::evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {
    using real_type = double;

    std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();

    plssvm::parameter<real_type> params;

    std::vector<std::vector<real_type>> data_dense;
    plssvm::openmp::coo<real_type> data_coo{};
    plssvm::openmp::csr<real_type> data_csr{};

    auto data_ptr_dense = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data_dense));
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo));
    auto data_ptr_csr = std::make_shared<const plssvm::openmp::csr<real_type>>(std::move(data_csr));

    // dense
    std::vector<ns> raw_runtimes_dense_linear;
    std::vector<ns> raw_runtimes_dense_poly;
    std::vector<ns> raw_runtimes_dense_radial;
    params.parse_libsvm_file(path_to_dataset, data_ptr_dense);
    fmt::print(std::to_string(data_ptr_dense->size()));
    for(size_t i = 0; i < cycles; i++) {
        std::vector<real_type> q(data_ptr_dense->size() - 1); // q-Vector

        // linear
        fmt::print("dense (linear) " + std::to_string(i) + "/" + std::to_string(cycles) + "\n");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_linear<real_type>(q, *data_ptr_dense);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i) + "/" + std::to_string(cycles) + "\n");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_poly<real_type>(q, *data_ptr_dense, degree, gamma, coef0);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));

        // radial
        fmt::print("dense (radial) " + std::to_string(i) + "/" + std::to_string(cycles) + "\n");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_radial<real_type>(q, *data_ptr_dense, gamma);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
    }
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    // TODO: implement 

    // csr
    std::vector<ns> raw_runtimes_csr_linear;
    std::vector<ns> raw_runtimes_csr_poly;
    std::vector<ns> raw_runtimes_csr_radial;
    // TODO: implement 
    
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (linear)");
    //sub_benchmark_names.push_back(sub_benchmark_name + " COO (linear)");
    //sub_benchmark_names.push_back(sub_benchmark_name + " CSR (linear)");
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (polynomial)");
    //sub_benchmark_names.push_back(sub_benchmark_name + " COO (polynomial)");
    //sub_benchmark_names.push_back(sub_benchmark_name + " CSR (polynomial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (radial)");
    //sub_benchmark_names.push_back(sub_benchmark_name + " COO (radial)");
    //sub_benchmark_names.push_back(sub_benchmark_name + " CSR (radial)");
    auto sub_benchmark_runtimes = std::vector<std::vector<ns>>{
        raw_runtimes_dense_linear,
        raw_runtimes_dense_poly,
        raw_runtimes_dense_radial};
    perform_statistics(sub_benchmark_runtimes);
}

}  // namespace plssvm::benchmarks
