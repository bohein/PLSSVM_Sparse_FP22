/**
 * @file
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines CUDA functions for generating the `q` vector from data in COO format.
 */

#pragma once

namespace plssvm::cuda {

/**
 * @brief Calculates the `q` vector using the linear C-SVM kernel on data in COO format.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] col_ids column indices of coo data structure
 * @param[in] row_ids row indices of coo data structure
 * @param[in] values data values of coo data structure
 * @param[in] last_row_begin index of first element of last data point
 */
template <typename real_type>
__global__ void device_kernel_q_linear(real_type *q, const size_t *col_ids, const size_t *row_ids, const real_type *values, const size_t last_row_begin);

}  // namespace plssvm::cuda
