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
    std::string csv = name + "\n" 
        + "sub-benchmarks" + ',' 
        + "mean (ms)" + ',' 
        + "median (ms)" + ',' 
        + "min (ms)" + ',' 
        + "max (ms)" + ',' 
        + "variance (ms²)" + ','
        + "std. deviation (ms)" + "\n";
    if (sub_benchmark_names.size() == runtimes_mean.size()
        && sub_benchmark_names.size() == runtimes_median.size()
        && sub_benchmark_names.size() == runtimes_min.size()
        && sub_benchmark_names.size() == runtimes_max.size()
        && sub_benchmark_names.size() == runtimes_variance.size()
        && sub_benchmark_names.size() == runtimes_std_deviation.size()) {
        for (size_t i = 0; i < sub_benchmark_names.size(); i++) {
            csv += sub_benchmark_names[i] + ',' 
            + std::to_string(runtimes_mean[i].count() / 1000000.0) + ',' 
            + std::to_string(runtimes_median[i].count() / 1000000.0) + ','
            + std::to_string(runtimes_min[i].count() / 1000000.0) + ','  
            + std::to_string(runtimes_max[i].count() / 1000000.0) + ',' 
            + std::to_string(runtimes_variance[i].count() / 1000000000000.0) + ',' // 10^6 * 10^6
            + std::to_string(runtimes_std_deviation[i].count() / 1000000.0) + "\n";
        }
    }
    else {
        csv += "Error: data for mean/median/min have different lengths (" 
        + std::to_string(runtimes_mean.size()) + "/" 
        + std::to_string(runtimes_median.size()) + "/" 
        + std::to_string(runtimes_min.size())  + "/"
        + std::to_string(runtimes_max.size())  + "/"
        + std::to_string(runtimes_variance.size())  + "/" 
        + std::to_string(runtimes_std_deviation.size())  + ")";
    }
    return csv;
}

void benchmark::perform_statistics(std::vector<std::vector<ns>> &various_runtimes) {
    for (auto& individual_runtimes : various_runtimes) {
        runtimes_mean.push_back(mean(individual_runtimes));
        runtimes_median.push_back(median(individual_runtimes));
        runtimes_min.push_back(min(individual_runtimes));
        runtimes_max.push_back(max(individual_runtimes));
        runtimes_variance.push_back(variance(individual_runtimes));
        runtimes_std_deviation.push_back(std_deviation(individual_runtimes));
    }
    
}

}  // namespace plssvm::benchmarks