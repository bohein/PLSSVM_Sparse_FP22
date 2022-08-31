/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief This header contains some statistics functions for vectors of std::chrono::nanoseconds
 */

#pragma once

#include <vector> // std::vector
#include <chrono> // std::chrono::nanoseconds
#include <numeric>
#include <algorithm>

namespace plssvm::benchmarks {

using ns = std::chrono::nanoseconds;

inline ns mean(std::vector<ns> &v) {
    return std::reduce(v.begin(), v.end()) / v.size();
}

inline ns median(std::vector<ns> &v) {
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    return v[v.size()/2];
}

inline ns max(std::vector<ns> &v) {
    return *std::max_element(v.begin(), v.end());
}

inline ns min(std::vector<ns> &v) {
    return *std::min_element(v.begin(), v.end());
}

inline ns variance(std::vector<ns> &v) {
    double mu = mean(v).count();
    double sq_sum = 0;
    for (auto it = v.begin(); it != v.end(); it++) {
        sq_sum += (it->count() - mu) * (it->count() - mu);
    }
    int64_t sigma_sq = std::round(sq_sum / v.size());
    return ns(sigma_sq);
}

inline ns std_deviation(std::vector<ns> &v) {
    int64_t sigma = std::round(std::sqrt(variance(v).count()));
    return ns(sigma);
}

}  // namespace plssvm::benchmarks