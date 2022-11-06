/**
 * @file
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the CUDA backend.
 */

#pragma once

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda::coo {

/**
 * @brief Predicts the labels for data points using the linear kernel function on data in COO format.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] out_d the calculated predictions
 * @param[in] col_ids column indices of coo data structure
 * @param[in] row_ids row indices of coo data structure
 * @param[in] values data values of coo data structure
 * @param[in] nnz the number of non-zero elements in the data matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] predict_col_ids column indices of coo predict matrix
 * @param[in] predict_row_ids row indices of coo predict matrix
 * @param[in] predict_values data values of coo predict matrix
 * @param[in] predict_nnz the number of non-zero elements in the predict matrix
 */
template <typename real_type>
__global__ void device_kernel_predict_linear(real_type *out_d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_ids, const real_type *predict_values, const kernel_index_type predict_nnz, const int degree, const real_type gamma, const real_type coef0);

/**
 * @brief Predicts the labels for data points using the polynomial kernel function on data in COO format.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] out_d the calculated predictions
 * @param[in] col_ids column indices of coo data structure
 * @param[in] row_ids row indices of coo data structure
 * @param[in] values data values of coo data structure
 * @param[in] nnz the number of non-zero elements in the data matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] predict_col_ids column indices of coo predict matrix
 * @param[in] predict_row_ids row indices of coo predict matrix
 * @param[in] predict_values data values of coo predict matrix
 * @param[in] predict_nnz the number of non-zero elements in the predict matrix
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
template <typename real_type>
__global__ void device_kernel_predict_poly(real_type *out_d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_ids, const real_type *predict_values, const kernel_index_type predict_nnz, const int degree, const real_type gamma, const real_type coef0);

/**
 * @brief Predicts the labels for data points using the radial basis functions kernel function on data in COO format.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] out_d the calculated predictions
 * @param[in] col_ids column indices of coo data structure
 * @param[in] row_ids row indices of coo data structure
 * @param[in] values data values of coo data structure
 * @param[in] nnz the number of non-zero elements in the data matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] predict_col_ids column indices of coo predict matrix
 * @param[in] predict_row_ids row indices of coo predict matrix
 * @param[in] predict_values data values of coo predict matrix
 * @param[in] predict_nnz the number of non-zero elements in the predict matrix
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
__global__ void device_kernel_predict_radial(real_type *out_d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_ids, const real_type *predict_values, const kernel_index_type predict_nnz, const real_type gamma);

}  // namespace plssvm::cuda