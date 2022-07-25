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
        benchmark();

        std::string name;
        std::vector<std::string> sub_benchmark_names;
        std::vector<double> runtimes;

        virtual std::string data_to_csv() {return "";}

};

}  // namespace plssvm:benchmarks