/**
 * @author Vincent Duttle
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Creating and using sparse q kernels (coo and csr) for testing purposes.
 */

#include "plssvm/backends/CUDA/sparse/sparse_q_kernel.hpp"
#include "plssvm/backends/CUDA/q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/coo/coo_q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/csr/csr_q_kernel.cuh"
#include "plssvm/coo.hpp"
#include "plssvm/csr.hpp"
#include "plssvm/benchmarks/benchmark.hpp"
#include "plssvm/constants.hpp"
#include "plssvm/detail/execution_range.hpp"

namespace plssvm::benchmarks {

template <typename real_type>
void sparse_q_kernel<real_type>::test_coo_q_kernel_linear() {
    // needs to be filled
    const benchmarks::dataset &ds = {"datapoint_128_a", "../benchmark_data/datapoint/128_8192_005_A.libsvm", 128, 8192, .005};
    parameter<real_type> params;
    
    real_type *q_d;

    // coo
    plssvm::openmp::coo<real_type> coo;
    real_type *values_coo_d;
    size_t *row_coo_d;
    size_t *col_coo_d;
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(coo));
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    // parameters for cuda and padding
    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_coo -> get_height() - 1;
    plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    dim3 grid(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block(range_q.block[0], range_q.block[1], range_q.block[2]);

    // coo padded
    plssvm::openmp::coo<real_type> data_coo_padded = *(data_ptr_coo.get());
    data_coo_padded.add_padding(boundary_size, 0, 0, 0);
    auto data_ptr_coo_padded = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo_padded));

    // coo properties
    auto nnz_coo = data_ptr_coo -> get_nnz();
    auto last_row_begin_coo = data_ptr_coo -> get_last_row_begin();
    // q-Vector
    std::vector<real_type> q((data_ptr_coo -> get_height() - 1) + boundary_size);

    cudaMalloc((void**)&q_d, sizeof(real_type)*((data_ptr_coo -> get_height() - 1) + boundary_size));
    cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(nnz_coo + boundary_size));
    cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
    cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
    
    cudaMemcpy(values_coo_d, data_ptr_coo_padded -> get_values().data(), sizeof(real_type)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(row_coo_d, data_ptr_coo_padded -> get_row_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(col_coo_d, data_ptr_coo_padded -> get_col_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

    plssvm::cuda::coo::device_kernel_q_linear<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, nnz_coo, last_row_begin_coo);
    cudaDeviceSynchronize();

    cudaFree(q_d);
    cudaFree(values_coo_d);
    cudaFree(col_coo_d);
    cudaFree(row_coo_d);
}

template <typename real_type>
void sparse_q_kernel<real_type>::test_coo_q_kernel_polynomial() {
    // needs to be filled
    const benchmarks::dataset &ds = {"datapoint_128_a", "../benchmark_data/datapoint/128_8192_005_A.libsvm", 128, 8192, .005};
    parameter<real_type> params;
    
    real_type *q_d;

    // coo
    plssvm::openmp::coo<real_type> coo;
    real_type *values_coo_d;
    size_t *row_coo_d;
    size_t *col_coo_d;
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(coo));
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    // parameters for cuda and padding
    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_coo -> get_height() - 1;
    plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    dim3 grid(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block(range_q.block[0], range_q.block[1], range_q.block[2]);

    // coo padded
    plssvm::openmp::coo<real_type> data_coo_padded = *(data_ptr_coo.get());
    data_coo_padded.add_padding(boundary_size, 0, 0, 0);
    auto data_ptr_coo_padded = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo_padded));

    // coo properties
    auto nnz_coo = data_ptr_coo -> get_nnz();
    auto last_row_begin_coo = data_ptr_coo -> get_last_row_begin();
    // q-Vector
    std::vector<real_type> q((data_ptr_coo -> get_height() - 1) + boundary_size);

    cudaMalloc((void**)&q_d, sizeof(real_type)*((data_ptr_coo -> get_height() - 1) + boundary_size));
    cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(nnz_coo + boundary_size));
    cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
    cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
    
    cudaMemcpy(values_coo_d, data_ptr_coo_padded -> get_values().data(), sizeof(real_type)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(row_coo_d, data_ptr_coo_padded -> get_row_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(col_coo_d, data_ptr_coo_padded -> get_col_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

    plssvm::cuda::coo::device_kernel_q_poly<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, nnz_coo, last_row_begin_coo, degree, gamma, coef0);
    cudaDeviceSynchronize();

    cudaFree(q_d);
    cudaFree(values_coo_d);
    cudaFree(col_coo_d);
    cudaFree(row_coo_d);
}

template <typename real_type>
void sparse_q_kernel<real_type>::test_coo_q_kernel_radial() {
    // needs to be filled
    const benchmarks::dataset &ds = {"datapoint_128_a", "../benchmark_data/datapoint/128_8192_005_A.libsvm", 128, 8192, .005};
    parameter<real_type> params;
    
    real_type *q_d;

    // coo
    plssvm::openmp::coo<real_type> coo;
    real_type *values_coo_d;
    size_t *row_coo_d;
    size_t *col_coo_d;
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(coo));
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    // parameters for cuda and padding
    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_coo -> get_height() - 1;
    plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    dim3 grid(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block(range_q.block[0], range_q.block[1], range_q.block[2]);

    // coo padded
    plssvm::openmp::coo<real_type> data_coo_padded = *(data_ptr_coo.get());
    data_coo_padded.add_padding(boundary_size, 0, 0, 0);
    auto data_ptr_coo_padded = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo_padded));

    // coo properties
    auto nnz_coo = data_ptr_coo -> get_nnz();
    auto last_row_begin_coo = data_ptr_coo -> get_last_row_begin();
    // q-Vector
    std::vector<real_type> q((data_ptr_coo -> get_height() - 1) + boundary_size);

    cudaMalloc((void**)&q_d, sizeof(real_type)*((data_ptr_coo -> get_height() - 1) + boundary_size));
    cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(nnz_coo + boundary_size));
    cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
    cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
    
    cudaMemcpy(values_coo_d, data_ptr_coo_padded -> get_values().data(), sizeof(real_type)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(row_coo_d, data_ptr_coo_padded -> get_row_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(col_coo_d, data_ptr_coo_padded -> get_col_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
    cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

    plssvm::cuda::coo::device_kernel_q_radial<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, nnz_coo, last_row_begin_coo, gamma);
    cudaDeviceSynchronize();

    cudaFree(q_d);
    cudaFree(values_coo_d);
    cudaFree(col_coo_d);
    cudaFree(row_coo_d);
}

// template <typename real_type>
// void sparse_q_kernel<real_type>::test_csr_q_kernel(plssvm::openmp::csr<real_type> csr) {
//     real_type *q_d;

//     real_type *values_csr_d;
//     size_t *row_csr_d;
//     size_t *col_csr_d;
    
//     auto data_ptr_csr = std::make_shared<const plssvm::openmp::csr<real_type>>(std::move(csr));

//     size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);

// }
template class sparse_q_kernel<float>;
template class sparse_q_kernel<double>;

}