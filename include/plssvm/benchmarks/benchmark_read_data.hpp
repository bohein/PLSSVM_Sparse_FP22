/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks reagrding reading and storing of test data into different (sparse) data structures.
 */

#pragma once

#include "plssvm/benchmarks/benchmark.hpp"

#include <filesystem> // std::filesystem::current_path

namespace plssvm::benchmarks {

class benchmark_read_data : public benchmark {
    public:
        benchmark_read_data();
        void run() override;

    protected:
        std::string data_to_csv() override;

    private:
        // TODO: change to relative path
        std::string DATASET_TINY = "/home/schmidtm/PLSSVM/benchmark_data/iris.libsvm"; // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale
};

}  // namespace plssvm::benchmarks