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
}

void benchmark_read_data::evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {
    using real_type = double;

    plssvm::parameter<real_type> params;
    plssvm::openmp::coo<real_type> data{};
    std::shared_ptr<const plssvm::openmp::coo<real_type>> data_ptr = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data));

    std::vector<std::chrono::nanoseconds> raw_runtimes;
    // run benchmark for many iterations
    for(size_t i = 0; i < cycles; i++) {
        std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
        params.parse_libsvm_file_sparse(path_to_dataset, data_ptr);
        std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();
        raw_runtimes.push_back(std::chrono::round<std::chrono::nanoseconds>(end_time - start_time));
    }
    
    sub_benchmark_names.push_back(sub_benchmark_name);

    // mean
    runtimes_mean.push_back(std::reduce(raw_runtimes.begin(), raw_runtimes.end()) / raw_runtimes.size());

    // median
    std::nth_element(raw_runtimes.begin(), raw_runtimes.begin() + raw_runtimes.size()/2, raw_runtimes.end());
    runtimes_median.push_back(raw_runtimes[raw_runtimes.size()/2]);

    // max
    runtimes_max.push_back(*std::max_element(raw_runtimes.begin(), raw_runtimes.end()));
}

}  // namespace plssvm::benchmarks