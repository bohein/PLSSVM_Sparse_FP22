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

namespace plssvm::benchmarks {

benchmark_read_data::benchmark_read_data() : benchmark{"Read Data"} {}

void benchmark_read_data::run() {
    using real_type = double;

    plssvm::parameter<real_type> params;
    plssvm::openmp::coo<real_type> actual_data{};
    std::shared_ptr<const plssvm::openmp::coo<real_type>> actual_data_ptr = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(actual_data));

    // TODO: actual implementation, not this short proof-of-concept
    std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
    params.parse_libsvm_file_sparse(DATASET_TINY, actual_data_ptr);
    std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();
    fmt::print("{:%S}\n", end_time - start_time);
}

std::string benchmark_read_data::data_to_csv() {
    std::string csv = "";
    if (sub_benchmark_names.size() == runtimes.size()) {
        for (size_t i = 0; i < sub_benchmark_names.size(); i++) {
            csv += sub_benchmark_names[i] + ',' + std::to_string(runtimes[i]) + "\n";
        }
    }
    return csv;
}

}  // namespace plssvm::benchmarks