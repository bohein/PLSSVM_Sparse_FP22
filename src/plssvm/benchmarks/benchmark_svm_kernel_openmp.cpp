/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks reagrding svm-kernel functions.
 */

#include "plssvm/benchmarks/benchmark_svm_kernel_openmp.hpp"

#include "plssvm/backends/OpenMP/q_kernel.hpp"
#include "plssvm/backends/OpenMP/svm_kernel.hpp"

#include <numeric>
#include <iostream>

namespace plssvm::benchmarks {

benchmark_svm_kernel_openmp::benchmark_svm_kernel_openmp() : benchmark{"SVM-Kernels (OpenMP)"} {}

void benchmark_svm_kernel_openmp::run() {
    using real_type = double;

    evaluate_dataset("tiny (~150)", DATASET_TINY);
    evaluate_dataset("small (~5000)", DATASET_SMALL);
    //evaluate_dataset("medium (~50000)", DATASET_MEDIUM);
    //evaluate_dataset("large (~250000)", DATASET_LARGE);
}

void benchmark_svm_kernel_openmp::evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {
    using real_type = double;

    std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();

    plssvm::parameter<real_type> params;
    std::vector<real_type> q; // q-Vector
    real_type QA_cost;
    std::vector<real_type> ret; // result Vector
    std::vector<real_type> d; // ""right-hand side of the equation"

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
    for(size_t i = 0; i < cycles; i++) {
        q = std::vector<real_type>(data_ptr_dense->size() - 1); // q-Vector
        
        // linear
        fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.); 
        plssvm::openmp::device_kernel_q_linear<real_type>(q, *data_ptr_dense);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_linear<real_type>(q, ret, d, *data_ptr_dense, QA_cost, cost, add);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.); 
        plssvm::openmp::device_kernel_q_poly<real_type>(q, *data_ptr_dense, degree, gamma, coef0);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_poly<real_type>(q, ret, d, *data_ptr_dense, QA_cost, cost, add, degree, gamma, coef0);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.); 
        plssvm::openmp::device_kernel_q_radial<real_type>(q, *data_ptr_dense, gamma);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_radial<real_type>(q, ret, d, *data_ptr_dense, QA_cost, cost, add, gamma);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");
    }
    
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(path_to_dataset, data_ptr_coo);
    for(size_t i = 0; i < cycles; i++) {
        q = std::vector<real_type>(data_ptr_coo->get_height() - 1); // q-Vector

        // linear
        fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.); 
        plssvm::openmp::device_kernel_q_linear<real_type>(q, *data_ptr_coo);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_linear<real_type>(q, ret, d, *data_ptr_coo, QA_cost, cost, add);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_coo_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.); 
        plssvm::openmp::device_kernel_q_poly<real_type>(q, *data_ptr_coo, degree, gamma, coef0);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_poly<real_type>(q, ret, d, *data_ptr_coo, QA_cost, cost, add, degree, gamma, coef0);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_coo_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.); 
        plssvm::openmp::device_kernel_q_radial<real_type>(q, *data_ptr_coo, gamma);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_radial<real_type>(q, ret, d, *data_ptr_coo, QA_cost, cost, add, gamma);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");
    }
    
    // coo
    std::vector<ns> raw_runtimes_csr_linear;
    std::vector<ns> raw_runtimes_csr_poly;
    std::vector<ns> raw_runtimes_csr_radial;
    params.parse_libsvm_file_sparse(path_to_dataset, data_ptr_csr);
    for(size_t i = 0; i < cycles; i++) {
        q = std::vector<real_type>(data_ptr_csr->get_height() - 1); // q-Vector

        // linear
        fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        plssvm::openmp::device_kernel_q_linear<real_type>(q, *data_ptr_csr);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_linear<real_type>(q, ret, d, *data_ptr_csr, QA_cost, cost, add);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_csr_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        plssvm::openmp::device_kernel_q_poly<real_type>(q, *data_ptr_csr, degree, gamma, coef0);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_poly<real_type>(q, ret, d, *data_ptr_csr, QA_cost, cost, add, degree, gamma, coef0);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_csr_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        plssvm::openmp::device_kernel_q_radial<real_type>(q, *data_ptr_csr, gamma);
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::openmp::device_kernel_radial<real_type>(q, ret, d, *data_ptr_csr, QA_cost, cost, add, gamma);
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");
    }
    
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (linear)");
    sub_benchmark_names.push_back(sub_benchmark_name + "COO (linear)");
    sub_benchmark_names.push_back(sub_benchmark_name + "CSR (linear)");
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (polynomial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "COO (polynomial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "CSR (polynomial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (radial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "COO (radial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "CSR (radial)");
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
