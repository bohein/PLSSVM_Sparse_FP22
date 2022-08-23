/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks
 */

#pragma once

#include "plssvm/core.hpp"

#include "fmt/core.h"
#include "fmt/ostream.h" 
#include "fmt/chrono.h"

#include <chrono>

namespace plssvm::benchmarks {

class benchmark {
    public:
        benchmark(const std::string benchmark_name);
        virtual ~benchmark() {}

        std::string get_name() const {return name;}
        std::string get_data() {return data_to_csv();}

        virtual void run() {}

    protected:
        std::string name;
        std::vector<std::string> sub_benchmark_names;
        std::vector<std::chrono::nanoseconds> runtimes_mean;
        std::vector<std::chrono::nanoseconds> runtimes_median;
        std::vector<std::chrono::nanoseconds> runtimes_min;

        uint64_t cycles = 100;

        benchmark();

        virtual void evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {sub_benchmark_name + path_to_dataset;}
        virtual std::string data_to_csv();

};

}  // namespace plssvm:benchmarks