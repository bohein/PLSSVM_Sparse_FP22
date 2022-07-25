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

}  // namespace plssvm::benchmarks