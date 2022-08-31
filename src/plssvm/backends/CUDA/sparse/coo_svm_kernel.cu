/**
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/coo_svm_kernel.cuh"

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const real_type add) {
    kernel_index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    kernel_index_type j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < j) {
        return;
    }

    kernel_index_type row_id_i = static_cast<kernel_index_type>(row_ids[i]);
    kernel_index_type row_id_j = static_cast<kernel_index_type>(row_ids[j]);

    const real_type temp = (values[i] * values[j] + QA_cost - q[row_id_i] - q[row_id_j]) * add;
    if (i == j) {
        ret[row_id_i] += (temp + cost * add) * d[row_id_i];
    } else {
        ret[row_id_i] += temp * d[row_id_j];
        ret[row_id_j] += temp * d[row_id_i];
    }
}
template __global__ void device_kernel_linear(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const float);
template __global__ void device_kernel_linear(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const double);

template <typename real_type>
__global__ void device_kernel_poly(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    kernel_index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    kernel_index_type j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < j) {
        return;
    }

    kernel_index_type row_id_i = static_cast<kernel_index_type>(row_ids[i]);
    kernel_index_type row_id_j = static_cast<kernel_index_type>(row_ids[j]);

    const real_type temp = (pow(gamma * values[i] * values[j] + coef0, degree) + QA_cost - q[row_id_i] - q[row_id_j]) * add;
    if (i == j) {
        ret[row_id_i] += (temp + cost * add) * d[row_id_i];
    } else {
        ret[row_id_i] += temp * d[row_id_j];
        ret[row_id_j] += temp * d[row_id_i];
    }
}
template __global__ void device_kernel_poly(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const float, const int, const float, const float);
template __global__ void device_kernel_poly(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const double, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_radial(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const real_type add, const real_type gamma) {
    kernel_index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    kernel_index_type j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < j) {
        return;
    }

    kernel_index_type row_id_i = static_cast<kernel_index_type>(row_ids[i]);
    kernel_index_type row_id_j = static_cast<kernel_index_type>(row_ids[j]);

    const real_type temp = (exp(-gamma * pow(values[i] - values[j], 2)) + QA_cost - q[row_id_i] - q[row_id_j]) * add;
    if (i == j) {
        ret[row_id_i] += (temp + cost * add) * d[row_id_i];
    } else {
        ret[row_id_i] += temp * d[row_id_j];
        ret[row_id_j] += temp * d[row_id_i];
    }
}
template __global__ void device_kernel_radial(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const float, const float);
template __global__ void device_kernel_radial(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const double, const double);

}  // namespace plssvm::cuda
