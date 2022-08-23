/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks
 */

#include "plssvm/benchmarks/benchmark.hpp"

namespace plssvm::benchmarks {

benchmark::benchmark(const std::string benchmark_name) : name{benchmark_name} {}

std::string benchmark::data_to_csv() {
    std::string csv = name + "\n" + "sub-benchmarks" + ',' + "mean (ms)" + ',' + "median (ms)" + ',' + "min (ms)" + "\n";
    if (sub_benchmark_names.size() == runtimes_mean.size()
        && sub_benchmark_names.size() == runtimes_median.size()
        && sub_benchmark_names.size() == runtimes_min.size()) {
        for (size_t i = 0; i < sub_benchmark_names.size(); i++) {
            csv += sub_benchmark_names[i] + ',' 
            + std::to_string(runtimes_mean[i].count() / 1000000.0) + ',' 
            + std::to_string(runtimes_median[i].count() / 1000000.0) + ',' 
            + std::to_string(runtimes_min[i].count() / 1000000.0)+ "\n";
        }
    }
    return csv;
}

}  // namespace plssvm::benchmarks