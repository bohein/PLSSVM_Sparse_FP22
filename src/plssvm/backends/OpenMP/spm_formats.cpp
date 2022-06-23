/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure(s) for sparse matrices
 */

#include "plssvm/backends/OpenMP/spm_formats.hpp"

namespace plssvm::openmp {

void insert_element(coo& matrix, size_t col_id, size_t row_id, real_type data) {
    matrix.nnz++;
    matrix.col_ids.insert(col_id);
    matrix.row_ids.insert(row_id);
    matrix.data.insert(data);
}

}  // namespace plssvm:openmp