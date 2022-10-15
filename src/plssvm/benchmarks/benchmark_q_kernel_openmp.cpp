/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks reagrding q-kernel functions.
 */

#include "plssvm/benchmarks/benchmark_q_kernel_openmp.hpp"

#include "plssvm/backends/OpenMP/q_kernel.hpp"

#include <numeric>
#include <iostream>

namespace plssvm::benchmarks {

benchmark_q_kernel_openmp::benchmark_q_kernel_openmp() : benchmark{"Q-Kernels (OpenMP)"} {}

void benchmark_q_kernel_openmp::run() {
    using real_type = double;

    datasets.insert(datasets.end(), DATAPOINT.begin(), DATAPOINT.end());
    datasets.insert(datasets.end(), FEATURE.begin(), FEATURE.end());
    datasets.insert(datasets.end(), DENSITY.begin(), DENSITY.end());
    //datasets.insert(datasets.end(), REAL_WORLD.begin(), REAL_WORLD.end());

    for (auto& ds : datasets) evaluate_dataset(ds);
}

void benchmark_q_kernel_openmp::evaluate_dataset(const dataset &ds) {
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
    params.parse_libsvm_file(ds.path, data_ptr_dense);
    for(size_t i = 0; i < cycles; i++) {
        std::vector<real_type> q(data_ptr_dense->size() - 1); // q-Vector

        // linear
        std::cout << fmt::format("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_linear<real_type>(q, *data_ptr_dense);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        std::cout << fmt::format("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_poly<real_type>(q, *data_ptr_dense, degree, gamma, coef0);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        std::cout << fmt::format("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_radial<real_type>(q, *data_ptr_dense, gamma);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");
    }
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);
    for(size_t i = 0; i < cycles; i++) {
        std::vector<real_type> q(data_ptr_coo->get_height() - 1); // q-Vector

        // linear
        std::cout << fmt::format("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_linear<real_type>(q, *data_ptr_coo);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_coo_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        std::cout << fmt::format("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_poly<real_type>(q, *data_ptr_coo, degree, gamma, coef0);
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        std::cout << fmt::format("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_radial<real_type>(q, *data_ptr_coo, gamma);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");
    }

    // csr
    std::vector<ns> raw_runtimes_csr_linear;
    std::vector<ns> raw_runtimes_csr_poly;
    std::vector<ns> raw_runtimes_csr_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_csr);
    for(size_t i = 0; i < cycles; i++) {
        std::vector<real_type> q(data_ptr_csr->get_height() - 1); // q-Vector

        // linear
        std::cout << fmt::format("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_linear<real_type>(q, *data_ptr_csr);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_csr_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        std::cout << fmt::format("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_poly<real_type>(q, *data_ptr_csr, degree, gamma, coef0);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_csr_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        std::cout << fmt::format("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        //fmt::print("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_q_radial<real_type>(q, *data_ptr_csr, gamma);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        std::cout << fmt::format(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)") << std::endl;
        //fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");
    }

    
    sub_benchmark_names.push_back("dense (linear)");
    sub_benchmark_names.push_back("COO (linear)");
    sub_benchmark_names.push_back("CSR (linear)");
    sub_benchmark_names.push_back("dense (polynomial)");
    sub_benchmark_names.push_back("COO (polynomial)");
    sub_benchmark_names.push_back("CSR (polynomial)");
    sub_benchmark_names.push_back("dense (radial)");
    sub_benchmark_names.push_back("COO (radial)");
    sub_benchmark_names.push_back("CSR (radial)");
    auto sub_benchmark_runtimes = std::vector<std::vector<ns>>{
        raw_runtimes_dense_linear,
        raw_runtimes_coo_linear,
        raw_runtimes_csr_linear,
        raw_runtimes_dense_poly,
        raw_runtimes_coo_poly,
        raw_runtimes_csr_poly,
        raw_runtimes_dense_radial,
        raw_runtimes_coo_radial,
        raw_runtimes_csr_radial};
    perform_statistics(sub_benchmark_runtimes);
}

}  // namespace plssvm::benchmarks
