/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks
 */

#include "plssvm/benchmarks/benchmark_read_data.hpp"

#include <numeric>
#include <iostream>

namespace plssvm::benchmarks {

benchmark_read_data::benchmark_read_data() : benchmark{"Read Data"} {}

void benchmark_read_data::run() {
    using real_type = double;

    evaluate_dataset("tiny (~150)", DATASET_TINY);
    evaluate_dataset("small (~5000)", DATASET_SMALL);
    evaluate_dataset("medium (~50000)", DATASET_MEDIUM);
    evaluate_dataset("large (~250000)", DATASET_LARGE);
}

void benchmark_read_data::evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {
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
    std::vector<std::chrono::nanoseconds> raw_runtimes_dense_vectorized;
    size_t num_features;
    size_t num_points;
    for(size_t i = 0; i < cycles; i++) {
        std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
        params.parse_libsvm_file(path_to_dataset, data_ptr_dense);
        std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense.push_back(std::chrono::round<std::chrono::nanoseconds>(end_time - start_time));
        
        // 2D -> 1D transformation (borrowed from csvm.cpp from protected function transform_data())
        num_features = (*data_ptr_dense)[0].size();
        num_points = data_ptr_dense->size();
        std::vector<real_type> vec(num_features * (num_points));
        #pragma omp parallel for collapse(2)
        for (typename std::vector<real_type>::size_type col = 0; col < num_features; ++col) {
            for (size_t row = 0; row < num_points; ++row) {
                vec[col * (num_points) + row] = (*data_ptr_dense)[row][col];
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_dense_vectorized.push_back(std::chrono::round<std::chrono::nanoseconds>(end_time - start_time));
    }

    // coo
    std::vector<std::chrono::nanoseconds> raw_runtimes_coo;
    for(size_t i = 0; i < cycles; i++) {
        std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
        params.parse_libsvm_file_sparse(path_to_dataset, data_ptr_coo);
        std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_coo.push_back(std::chrono::round<std::chrono::nanoseconds>(end_time - start_time));
    }

    // csr
    std::vector<std::chrono::nanoseconds> raw_runtimes_csr;
    for(size_t i = 0; i < cycles; i++) {
        std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
        params.parse_libsvm_file_sparse_csr(path_to_dataset, data_ptr_csr);
        std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_csr.push_back(std::chrono::round<std::chrono::nanoseconds>(end_time - start_time));
    }
    
    sub_benchmark_names.push_back(sub_benchmark_name + " dense");
    sub_benchmark_names.push_back(sub_benchmark_name + " dense (+ vectorization)");
    sub_benchmark_names.push_back(sub_benchmark_name + " COO");
    sub_benchmark_names.push_back(sub_benchmark_name + " CSR");
    
    // mean
    runtimes_mean.push_back(std::reduce(raw_runtimes_dense.begin(), raw_runtimes_dense.end()) / raw_runtimes_dense.size());
    runtimes_mean.push_back(std::reduce(raw_runtimes_dense_vectorized.begin(), raw_runtimes_dense_vectorized.end()) / raw_runtimes_dense_vectorized.size());
    runtimes_mean.push_back(std::reduce(raw_runtimes_coo.begin(), raw_runtimes_coo.end()) / raw_runtimes_coo.size());
    runtimes_mean.push_back(std::reduce(raw_runtimes_csr.begin(), raw_runtimes_csr.end()) / raw_runtimes_csr.size());

    // median
    std::nth_element(raw_runtimes_dense.begin(), raw_runtimes_dense.begin() + raw_runtimes_dense.size()/2, raw_runtimes_dense.end());
    runtimes_median.push_back(raw_runtimes_dense[raw_runtimes_dense.size()/2]);
    std::nth_element(raw_runtimes_dense_vectorized.begin(), raw_runtimes_dense_vectorized.begin() + raw_runtimes_dense_vectorized.size()/2, raw_runtimes_dense_vectorized.end());
    runtimes_median.push_back(raw_runtimes_dense_vectorized[raw_runtimes_dense_vectorized.size()/2]);
    std::nth_element(raw_runtimes_coo.begin(), raw_runtimes_coo.begin() + raw_runtimes_coo.size()/2, raw_runtimes_coo.end());
    runtimes_median.push_back(raw_runtimes_coo[raw_runtimes_coo.size()/2]);
    std::nth_element(raw_runtimes_csr.begin(), raw_runtimes_csr.begin() + raw_runtimes_csr.size()/2, raw_runtimes_csr.end());
    runtimes_median.push_back(raw_runtimes_csr[raw_runtimes_csr.size()/2]);


    // min
    runtimes_min.push_back(*std::min_element(raw_runtimes_dense.begin(), raw_runtimes_dense.end()));
    runtimes_min.push_back(*std::min_element(raw_runtimes_dense_vectorized.begin(), raw_runtimes_dense_vectorized.end()));
    runtimes_min.push_back(*std::min_element(raw_runtimes_coo.begin(), raw_runtimes_coo.end()));
    runtimes_min.push_back(*std::min_element(raw_runtimes_csr.begin(), raw_runtimes_csr.end()));
}

}  // namespace plssvm::benchmarks
