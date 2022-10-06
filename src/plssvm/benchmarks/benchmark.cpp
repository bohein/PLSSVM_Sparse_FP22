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
        + "dataset" + ',' 
        + "num. datapoints" + ',' 
        + "num. features" + ',' 
        + "approx. density" + ',' 
        + "sub-benchmark" + ',' 
        + "mean (ms)" + ',' 
        + "median (ms)" + ',' 
        + "min (ms)" + ',' 
        + "max (ms)" + ',' 
        + "variance (ms²)" + ','
        + "std. deviation (ms)" + "\n";
    if (datasets.size() * num_data_structures * num_kernel_types == sub_benchmark_names.size()
        && datasets.size() * num_data_structures * num_kernel_types == runtimes_mean.size()
        && datasets.size() * num_data_structures * num_kernel_types == runtimes_median.size()
        && datasets.size() * num_data_structures * num_kernel_types == runtimes_min.size()
        && datasets.size() * num_data_structures * num_kernel_types == runtimes_max.size()
        && datasets.size() * num_data_structures * num_kernel_types == runtimes_variance.size()
        && datasets.size() * num_data_structures * num_kernel_types == runtimes_std_deviation.size()) {
        for (size_t i = 0; i < datasets.size(); i++) {
            for (size_t j = 0; j < num_data_structures * num_kernel_types; j++) {
                csv += datasets[i].name + ',' 
                + std::to_string(datasets[i].numDatapoints) + ',' 
                + std::to_string(datasets[i].numFeatures) + ',' 
                + std::to_string(datasets[i].approxDensity) + ',' 
                + sub_benchmark_names[i+j] + ',' 
                + std::to_string(runtimes_mean[i+j].count() / 1000000.0) + ',' 
                + std::to_string(runtimes_median[i+j].count() / 1000000.0) + ','
                + std::to_string(runtimes_min[i+j].count() / 1000000.0) + ','  
                + std::to_string(runtimes_max[i+j].count() / 1000000.0) + ',' 
                + std::to_string(runtimes_variance[i+j].count() / 1000000000000.0) + ',' // 10^6 * 10^6
                + std::to_string(runtimes_std_deviation[i+j].count() / 1000000.0) + "\n";
            }
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