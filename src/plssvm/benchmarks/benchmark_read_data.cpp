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

namespace plssvm::benchmarks {

benchmark_read_data::benchmark_read_data() : benchmark{"Read Data"} {}

void benchmark_read_data::run() {
    using real_type = double;

    evaluate_dataset("tiny (~150)", DATASET_TINY);
    evaluate_dataset("small (~5000)", DATASET_SMALL);
}

void benchmark_read_data::evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {
    using real_type = double;

    plssvm::parameter<real_type> params;

    std::vector<std::vector<real_type>> data_dense;
    plssvm::openmp::coo<real_type> data_coo{};

    auto data_ptr_dense = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data_dense));
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo));

    // dense
    std::vector<std::chrono::nanoseconds> raw_runtimes_dense;
    for(size_t i = 0; i < cycles; i++) {
        std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
        params.parse_libsvm_file(path_to_dataset, data_ptr_dense);
        std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();
        // TODO: somehow factor in vectorization of data matrix
        raw_runtimes_dense.push_back(std::chrono::round<std::chrono::nanoseconds>(end_time - start_time));
    }

    // coo
    std::vector<std::chrono::nanoseconds> raw_runtimes_coo;
    for(size_t i = 0; i < cycles; i++) {
        std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
        params.parse_libsvm_file_sparse(path_to_dataset, data_ptr_coo);
        std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes_coo.push_back(std::chrono::round<std::chrono::nanoseconds>(end_time - start_time));
    }
    
    sub_benchmark_names.push_back(sub_benchmark_name + " dense");
    sub_benchmark_names.push_back(sub_benchmark_name + " COO");

    // mean
    runtimes_mean.push_back(std::reduce(raw_runtimes_dense.begin(), raw_runtimes_dense.end()) / raw_runtimes_dense.size());
    runtimes_mean.push_back(std::reduce(raw_runtimes_coo.begin(), raw_runtimes_coo.end()) / raw_runtimes_coo.size());

    // median
    std::nth_element(raw_runtimes_dense.begin(), raw_runtimes_dense.begin() + raw_runtimes_dense.size()/2, raw_runtimes_dense.end());
    runtimes_median.push_back(raw_runtimes_dense[raw_runtimes_dense.size()/2]);
    std::nth_element(raw_runtimes_coo.begin(), raw_runtimes_coo.begin() + raw_runtimes_coo.size()/2, raw_runtimes_coo.end());
    runtimes_median.push_back(raw_runtimes_coo[raw_runtimes_coo.size()/2]);

    // max
    runtimes_max.push_back(*std::max_element(raw_runtimes_dense.begin(), raw_runtimes_dense.end()));
    runtimes_max.push_back(*std::max_element(raw_runtimes_coo.begin(), raw_runtimes_coo.end()));
}

}  // namespace plssvm::benchmarks