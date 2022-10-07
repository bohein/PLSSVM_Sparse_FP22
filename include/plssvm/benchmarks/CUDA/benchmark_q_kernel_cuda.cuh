/**
 * @file
 * @author Tim Schmidt
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks reagrding q-kernel functions.
 */

#pragma once

#include "plssvm/benchmarks/benchmark.hpp"

#include <filesystem> // std::filesystem::current_path

namespace plssvm::benchmarks {

class benchmark_q_kernel_cuda : public benchmark {
    public:
        benchmark_q_kernel_cuda();
        void run() override;

    protected:
        void evaluate_dataset(const dataset& ds) override;


        // params for poly/radial kernel functions
        int degree = 3; 
        real_type gamma = 3;
        real_type coef0 = 3;
};

}  // namespace plssvm::benchmarks