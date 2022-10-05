/**
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/coo_svm_kernel.cuh"

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd
#include "plssvm/constants.hpp"                     // plssvm::INTERNAL_BLOCK_SIZE, plssvm::kernel_index_type

// UNTESTED
namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type width, const kernel_index_type height, const real_type add) {
    kernel_index_type i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;

    if (i < j) {
        return;
    }

    kernel_index_type row_1_indices[INTERNAL_BLOCK_SIZE] = { 0 };
    kernel_index_type row_2_indices[INTERNAL_BLOCK_SIZE] = { 0 };

    #pragma unroll INTERNAL_BLOCK_SIZE
    for (kernel_index_type block_index = 0; block_index < INTERNAL_BLOCK_SIZE; ++block_index) {
        kernel_index_type row_1_index = nnz * (i + block_index) / height;
        kernel_index_type row_2_index = nnz * (j + block_index) / height;

        if (row_ids[row_1_index] < i + block_index) {
            for (; row_1_index < nnz && row_ids[row_1_index] < i + block_index; ++row_1_index);
        } else {
            for (; row_1_index >= 0 && row_ids[row_1_index] >= i + block_index; --row_1_index);
            row_1_index++;
        }

        if (row_ids[row_2_index] < j + block_index) {
            for (; row_2_index < nnz && row_ids[row_2_index] < j + block_index; ++row_2_index);
        } else {
            for (; row_2_index >= 0 && row_ids[row_2_index] >= j + block_index; --row_2_index);
            row_2_index++;
        }

        row_1_indices[block_index] = row_1_index;
        row_2_indices[block_index] = row_2_index;
    }

    #pragma unroll INTERNAL_BLOCK_SIZE
    for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
        real_type ret_jy = 0.0;
        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            kernel_index_type row_1_index = row_1_indices[x];
            kernel_index_type row_2_index = row_2_indices[y];
            real_type matr_ix_jy = 0.0;

            // multiply rows
            while (row_ids[row_1_index] == i + x && row_ids[row_2_index] == j + y) {
                if (col_ids[row_1_index] == col_ids[row_2_index]) {
                    matr_ix_jy += values[row_1_index] * values[row_2_index];
                    row_1_index++;
                    row_2_index++;
                } else if (col_ids[row_1_index] < col_ids[row_2_index]) {
                    row_1_index++;
                } else {
                    row_2_index++;
                }
            }

            real_type temp = (matr_ix_jy + QA_cost - q[i + x] - q[j + y]) * add;
            if (i + x > j + y) {
                atomicAdd(&ret[i + x], temp * d[j + y]);
                ret_jy += temp * d[i + x];
            } else if (i + x == j + y) {
                ret_jy += (temp + cost * add) * d[i + x];
            }
        }
        atomicAdd(&ret[j + y], ret_jy);
    }
}
template __global__ void device_kernel_linear(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const kernel_index_type, const float);
template __global__ void device_kernel_linear(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const kernel_index_type, const double);

template <typename real_type>
__global__ void device_kernel_poly(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type width, const kernel_index_type height, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    kernel_index_type i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;

    if (i < j) {
        return;
    }

    kernel_index_type row_1_indices[INTERNAL_BLOCK_SIZE] = { 0 };
    kernel_index_type row_2_indices[INTERNAL_BLOCK_SIZE] = { 0 };

    #pragma unroll INTERNAL_BLOCK_SIZE
    for (kernel_index_type block_index = 0; block_index < INTERNAL_BLOCK_SIZE; ++block_index) {
        kernel_index_type row_1_index = nnz * (i + block_index) / height;
        kernel_index_type row_2_index = nnz * (j + block_index) / height;

        if (row_ids[row_1_index] < i + block_index) {
            for (; row_1_index < nnz && row_ids[row_1_index] < i + block_index; ++row_1_index);
        } else {
            for (; row_1_index >= 0 && row_ids[row_1_index] >= i + block_index; --row_1_index);
            row_1_index++;
        }

        if (row_ids[row_2_index] < j + block_index) {
            for (; row_2_index < nnz && row_ids[row_2_index] < j + block_index; ++row_2_index);
        } else {
            for (; row_2_index >= 0 && row_ids[row_2_index] >= j + block_index; --row_2_index);
            row_2_index++;
        }

        row_1_indices[block_index] = row_1_index;
        row_2_indices[block_index] = row_2_index;
    }

    #pragma unroll INTERNAL_BLOCK_SIZE
    for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
        real_type ret_jy = 0.0;
        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            kernel_index_type row_1_index = row_1_indices[x];
            kernel_index_type row_2_index = row_2_indices[y];
            real_type matr_ix_jy = 0.0;

            // multiply rows
            while (row_ids[row_1_index] == i + x && row_ids[row_2_index] == j + y) {
                if (col_ids[row_1_index] == col_ids[row_2_index]) {
                    matr_ix_jy += values[row_1_index] * values[row_2_index];
                    row_1_index++;
                    row_2_index++;
                } else if (col_ids[row_1_index] < col_ids[row_2_index]) {
                    row_1_index++;
                } else {
                    row_2_index++;
                }
            }

            real_type temp = (pow(gamma * matr_ix_jy + coef0, degree) + QA_cost - q[i + x] - q[j + y]) * add;
            if (i + x > j + y) {
                atomicAdd(&ret[i + x], temp * d[j + y]);
                ret_jy += temp * d[i + x];
            } else if (i + x == j + y) {
                ret_jy += (temp + cost * add) * d[i + x];
            }
        }
        atomicAdd(&ret[j + y], ret_jy);
    }
}
template __global__ void device_kernel_poly(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const kernel_index_type, const float, const int, const float, const float);
template __global__ void device_kernel_poly(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const kernel_index_type, const double, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_radial(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type width, const kernel_index_type height, const real_type add, const real_type gamma) {
    kernel_index_type i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;

    if (i < j) {
        return;
    }

    kernel_index_type row_1_indices[INTERNAL_BLOCK_SIZE] = { 0 };
    kernel_index_type row_2_indices[INTERNAL_BLOCK_SIZE] = { 0 };

    for (kernel_index_type block_index = 0; block_index < INTERNAL_BLOCK_SIZE; ++block_index) {
        kernel_index_type row_1_index = nnz * (i + block_index) / height;
        kernel_index_type row_2_index = nnz * (j + block_index) / height;

        if (row_ids[row_1_index] < i + block_index) {
            for (; row_1_index < nnz && row_ids[row_1_index] < i + block_index; ++row_1_index);
        } else {
            for (; row_1_index >= 0 && row_ids[row_1_index] >= i + block_index; --row_1_index);
            row_1_index++;
        }

        if (row_ids[row_2_index] < j + block_index) {
            for (; row_2_index < nnz && row_ids[row_2_index] < j + block_index; ++row_2_index);
        } else {
            for (; row_2_index >= 0 && row_ids[row_2_index] >= j + block_index; --row_2_index);
            row_2_index++;
        }

        row_1_indices[block_index] = row_1_index;
        row_2_indices[block_index] = row_2_index;
    }

    for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
        real_type ret_jy = 0.0;
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            kernel_index_type row_1_index = row_1_indices[x];
            kernel_index_type row_2_index = row_2_indices[y];
            real_type matr_ix_jy = 0.0;

            // calc sq. e. dist
            while (row_ids[row_1_index] == i + x && row_ids[row_2_index] == j + y) {
                if (col_ids[row_1_index] == col_ids[row_2_index]) {
                    matr_ix_jy += (values[row_1_index] - values[row_2_index]) * (values[row_1_index] - values[row_2_index]);
                    row_1_index++;
                    row_2_index++;
                } else if (col_ids[row_1_index] < col_ids[row_2_index]) {
                    matr_ix_jy += values[row_1_index] * values[row_1_index];
                    row_1_index++;
                } else {
                    matr_ix_jy += values[row_2_index] * values[row_2_index];
                    row_2_index++;
                }
            }
            if (row_ids[row_1_index] != i + x) {
                for (; row_2_index < nnz && row_ids[row_2_index] == j + y; ++row_2_index) {
                    matr_ix_jy += values[row_2_index] * values[row_2_index];
                }
            }
            if (row_ids[row_2_index] != j + y) {
                for (; row_1_index < nnz && row_ids[row_1_index] == i + x; ++row_1_index) {
                    matr_ix_jy += values[row_1_index] * values[row_1_index];
                }
            }

            real_type temp = (exp(-gamma * matr_ix_jy) + QA_cost - q[i + x] - q[j + y]) * add;
            if (i + x > j + y) {
                atomicAdd(&ret[i + x], temp * d[j + y]);
                ret_jy += temp * d[i + x];
            } else if (i + x == j + y) {
                ret_jy += (temp + cost * add) * d[i + x];
            }
        }
        atomicAdd(&ret[j + y], ret_jy);
    }
}
template __global__ void device_kernel_radial(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const kernel_index_type, const float, const float);
template __global__ void device_kernel_radial(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const kernel_index_type, const double, const double);

}  // namespace plssvm::cuda
