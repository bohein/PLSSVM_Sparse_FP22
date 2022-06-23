/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure(s) for sparse matrices
 */

#pragma once

#include <vector> // std::vector

namespace plssvm::openmp {

template <typename real_type>
struct coo {
    size_t nnz;
    const std::vector<size_t> col_ids;
    const std::vector<size_t> row_ids;
    const std::vector<real_type> data;
};

void insert_element(coo& matrix, size_t col_id, size_t row_id, real_type data) {};

}  // namespace plssvm::openmp