/**
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the kernel functions for the C-SVM using the CUDA backend and CSR-data representation.
 */

#pragma once

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda::csr {

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function on data in CSR format.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] col_ids column indices of coo data structure
 * @param[in] row_ids row indices of coo data structure
 * @param[in] values data values of coo data structure
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] nnz the number of non-zero elements in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 */
template <typename real_type>
__global__ void device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type height, const real_type add);

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] nnz the number of non-zero elements in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
template <typename real_type>
__global__ void device_kernel_poly(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type height, const real_type add, const int degree, const real_type gamma, const real_type coef0);

/**
 * @brief Calculates the C-SVM kernel using the radial basis function kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] nnz the number of non-zero elements in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
__global__ void device_kernel_radial(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type height, const real_type add, const real_type gamma);

}  // namespace plssvm::cuda
