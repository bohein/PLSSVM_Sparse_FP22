/**
 * @file
 * @author Tim Schmidt
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks reagrding svm-kernel functions.
 */

#include "plssvm/benchmarks/CUDA/benchmark_svm_kernel_cuda.cuh"

#include "plssvm/backends/CUDA/q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/coo_q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/csr_q_kernel.cuh"

#include "plssvm/backends/CUDA/svm_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/coo_svm_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/csr_svm_kernel.cuh"

#include "plssvm/detail/execution_range.hpp"

#include <numeric>
#include <iostream>

namespace plssvm::benchmarks {

benchmark_svm_kernel_cuda::benchmark_svm_kernel_cuda() : benchmark{"SVM-Kernels (CUDA)"} {}

void benchmark_svm_kernel_cuda::run() {
    using real_type = double;

    //evaluate_dataset("tiny (~150)", DATASET_TINY);
    //evaluate_dataset("small (~5000)", DATASET_SMALL);
    //evaluate_dataset("medium (~50000)", DATASET_MEDIUM);
    //evaluate_dataset("large (~250000)", DATASET_LARGE);
}

void benchmark_svm_kernel_cuda::evaluate_dataset(const dataset& ds) {
    using real_type = double;

    std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();

    plssvm::parameter<real_type> params;
    std::vector<real_type> q; // q-Vector
    real_type *q_d; // q-Vector on device
    real_type QA_cost;
    std::vector<real_type> ret; // result Vector
    real_type *ret_d; // result Vector on device
    std::vector<real_type> d; // ""right-hand side of the equation"
    real_type *d_d; // ""right-hand side of the equation" on device

    std::vector<std::vector<real_type>> data_dense;
    real_type *data_dense_d;

    plssvm::openmp::coo<real_type> data_coo{};
    real_type *values_coo_d;
    size_t *row_coo_d;
    size_t *col_coo_d;
    

    plssvm::openmp::csr<real_type> data_csr{};
    real_type *values_csr_d;
    size_t *row_csr_d;
    size_t *col_csr_d;

    auto data_ptr_dense = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data_dense));
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo));
    auto data_ptr_csr = std::make_shared<const plssvm::openmp::csr<real_type>>(std::move(data_csr));

    // dense
    std::vector<ns> raw_runtimes_dense_linear;
    std::vector<ns> raw_runtimes_dense_poly;
    std::vector<ns> raw_runtimes_dense_radial;
    params.parse_libsvm_file(ds.path, data_ptr_dense);
    //auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(plssvm::csvm<real_type>::transform_data(data_ptr_dense.get(), 0, 
    // ((*data_ptr_dense.get())[0].size()) * (data_ptr_dense.get() -> size()))); //padding----------------------

    const size_t num = ((*data_ptr_dense.get())[0].size()) * (data_ptr_dense -> size());
    std::vector<real_type> vec_1D(num);
    
    for (typename std::vector<real_type>::size_type col = 0; col < data_ptr_dense.get()[0].size(); ++col) {
        for (std::size_t row = 0; row < data_ptr_dense -> size(); ++row) {
            vec_1D[col * data_ptr_dense -> size() + row] = data_ptr_dense->at(row)[col];
        }
    }

    auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(vec_1D);

    auto data_dense_last = std::make_shared<const std::vector<real_type>>((*data_ptr_dense.get())[data_ptr_dense.get() -> size() - 1]);
    real_type *data_dense_last_d;
    
    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_dense -> size() - 1;

    plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    dim3 grid_q(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block_q(range_q.block[0], range_q.block[1], range_q.block[2]); 

    auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    plssvm::detail::execution_range range_svm({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    dim3 grid_svm(range_svm.grid[0], range_svm.grid[1], range_svm.grid[2]);
    dim3 block_svm(range_svm.block[0], range_svm.block[1], range_svm.block[2]); 

    for(size_t i = 0; i < cycles; i++) {
        auto num_rows = data_ptr_dense -> size() + boundary_size;
        auto num_cols = data_ptr_dense -> at(0).size();
        
        cudaMalloc((void**)&q_d, sizeof(real_type)*(num_rows - 1));
        cudaMalloc((void**)&ret_d, sizeof(real_type)*num_rows);
        cudaMalloc((void**)&d_d, sizeof(real_type)*num_rows);

        cudaMalloc((void**)&data_dense_d, sizeof(real_type)*(data_ptr_dense_1D -> size()));
        //cudaMalloc((void**)&data_dense_last_d, sizeof(real_type)*(*data_ptr_dense_1D.get())[0].size());
        cudaMalloc((void**)&data_dense_last_d, sizeof(real_type)*num_cols);

        
        cudaMemcpy(data_dense_d, data_ptr_dense_1D->data(), sizeof(real_type)*(data_ptr_dense_1D -> size()),cudaMemcpyHostToDevice);
        cudaMemcpy(data_dense_last_d, data_dense_last -> data(), sizeof(real_type) * num_cols,cudaMemcpyHostToDevice);

        q = std::vector<real_type>(num_rows - 1); // q-Vector
        cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        ret = std::vector<real_type>(num_rows, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(num_rows, 1.); 
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_linear<<<grid_q, block_q>>>(q_d, data_dense_d, data_dense_last_d, num_rows, num_cols);
        cudaDeviceSynchronize();
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_linear<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, data_dense_d, QA_cost, cost, num_rows, num_cols, add, id); //id = 0;
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        ret = std::vector<real_type>(num_rows, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(num_rows, 1.);
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_poly<<<grid_q, block_q>>>(q_d, data_dense_d, data_dense_last_d, num_rows, num_cols, degree, gamma, coef0);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_poly<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, data_dense_d, QA_cost, cost, num_rows, num_cols, add, degree, gamma, coef0);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        ret = std::vector<real_type>(num_rows, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(num_rows, 1.); 
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, data_dense_d, data_dense_last_d, num_rows, num_cols, gamma);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, data_dense_d, QA_cost, cost, num_rows, num_cols, add, gamma);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(ret_d);
        cudaFree(d_d);
        cudaFree(data_dense_d);
        cudaFree(data_dense_last_d);
    }
    
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    num_rows_exc_last = data_ptr_coo -> get_height() - 1;
/*
    range_q = plssvm::detail::execution_range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });*/

    range_q = plssvm::detail::execution_range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });                                         
    grid_q = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block_q = dim3(range_q.block[0], range_q.block[1], range_q.block[2]); 

    grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    plssvm::detail::execution_range range_svm_coo({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    grid_svm = dim3(range_svm_coo.grid[0], range_svm_coo.grid[1], range_svm_coo.grid[2]);
    block_svm = dim3(range_svm_coo.block[0], range_svm_coo.block[1], range_svm_coo.block[2]); 

    plssvm::openmp::coo<real_type> data_coo_padded = *(data_ptr_coo.get());
    data_coo_padded.add_padding(boundary_size, 0, 0, 0);
    auto data_ptr_coo_padded = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo_padded));

    for(size_t i = 0; i < cycles; i++) {
        auto nnz_coo = data_ptr_coo -> get_nnz();
        auto last_row_begin_coo = data_ptr_coo -> get_last_row_begin();
        auto height_coo = data_ptr_coo -> get_height();
        auto width_coo = data_ptr_coo -> get_width();
        
        cudaMalloc((void**)&q_d, sizeof(real_type)*(height_coo - 1 + boundary_size));
        cudaMalloc((void**)&ret_d, sizeof(real_type)*(height_coo + boundary_size));
        cudaMalloc((void**)&d_d, sizeof(real_type)*(height_coo + boundary_size));

        cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(nnz_coo + boundary_size));
        cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
        cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));

        cudaMemcpy(values_coo_d, data_ptr_coo_padded -> get_values().data(), sizeof(real_type)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
        cudaMemcpy(row_coo_d, data_ptr_coo_padded -> get_row_ids().data(), sizeof(real_type)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
        cudaMemcpy(col_coo_d, data_ptr_coo_padded -> get_col_ids().data(), sizeof(real_type)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);

        q = std::vector<real_type>(height_coo - 1 + boundary_size); // q-Vector
        cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        ret = std::vector<real_type>(height_coo + boundary_size, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(height_coo + boundary_size, 1.); 
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::coo::device_kernel_q_linear<<<grid_q, block_q>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo, nnz_coo);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_linear<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_coo_d, row_coo_d, values_coo_d, QA_cost, cost, nnz_coo, width_coo, height_coo, add);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_coo_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        ret = std::vector<real_type>(height_coo + boundary_size, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(height_coo + boundary_size, 1.);
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::coo::device_kernel_q_poly<<<grid_q, block_q>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo, nnz_coo, degree, gamma, coef0);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_poly<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_coo_d, row_coo_d, values_coo_d, QA_cost, cost, nnz_coo, width_coo, height_coo, add, degree, gamma, coef0);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_coo_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        ret = std::vector<real_type>(height_coo + boundary_size, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(height_coo + boundary_size, 1.); 
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::coo::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo, nnz_coo, gamma);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_coo_d, row_coo_d, values_coo_d, QA_cost, cost, nnz_coo, width_coo, height_coo, add, gamma);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(ret_d);
        cudaFree(d_d);

        cudaFree(values_coo_d);
        cudaFree(col_coo_d);
        cudaFree(row_coo_d);
    }
    
    // csr
    std::vector<ns> raw_runtimes_csr_linear;
    std::vector<ns> raw_runtimes_csr_poly;
    std::vector<ns> raw_runtimes_csr_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_csr);

    num_rows_exc_last = data_ptr_csr -> get_height() - 1;
    
    range_q = plssvm::detail::execution_range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });                                         
    grid_q = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block_q = dim3(range_q.block[0], range_q.block[1], range_q.block[2]); 

    grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    plssvm::detail::execution_range range_svm_csr({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    grid_svm = dim3(range_svm_csr.grid[0], range_svm_csr.grid[1], range_svm_csr.grid[2]);
    block_svm = dim3(range_svm_csr.block[0], range_svm_csr.block[1], range_svm_csr.block[2]);

    plssvm::openmp::csr<real_type> data_csr_padded = *(data_ptr_csr.get());
    data_csr_padded.add_padding(boundary_size, data_ptr_csr -> get_nnz());
    auto data_ptr_csr_padded = std::make_shared<const plssvm::openmp::csr<real_type>>(std::move(data_csr_padded));

    for(size_t i = 0; i < cycles; i++) {
        auto height_csr = data_ptr_csr -> get_height();
        auto nnz_csr = data_ptr_csr -> get_nnz();
        
        cudaMalloc((void**)&q_d, sizeof(real_type)*(height_csr + boundary_size - 1));
        cudaMalloc((void**)&ret_d, sizeof(real_type)*(height_csr + boundary_size));
        cudaMalloc((void**)&d_d, sizeof(real_type)*(height_csr + boundary_size));

        cudaMalloc((void**)&values_csr_d, sizeof(real_type)*nnz_csr);
        cudaMalloc((void**)&col_csr_d, sizeof(size_t)*nnz_csr);
        cudaMalloc((void**)&row_csr_d, sizeof(size_t)*(height_csr + boundary_size));
       
        cudaMemcpy(values_csr_d, data_ptr_csr -> get_values().data(), sizeof(real_type)*nnz_csr, cudaMemcpyHostToDevice);
        cudaMemcpy(row_csr_d, data_ptr_csr_padded -> get_row_offset().data(), sizeof(size_t)*(height_csr + boundary_size), cudaMemcpyHostToDevice);
        cudaMemcpy(col_csr_d, data_ptr_csr -> get_col_ids().data(), sizeof(size_t)*nnz_csr, cudaMemcpyHostToDevice);

        q = std::vector<real_type>(height_csr + boundary_size - 1); // q-Vector
        cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        ret = std::vector<real_type>(height_csr + boundary_size, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(height_csr + boundary_size, 1.); 
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::csr::device_kernel_q_linear<<<grid_q, block_q>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr, height_csr);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_linear<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_csr_d, row_csr_d, values_csr_d, QA_cost, cost, nnz_csr, height_csr, add);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        ret = std::vector<real_type>(height_csr + boundary_size, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(height_csr + boundary_size, 1.); 
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::csr::device_kernel_q_poly<<<grid_q, block_q>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr, height_csr, degree, gamma, coef0);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_poly<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_csr_d, row_csr_d, values_csr_d, QA_cost, cost, nnz_csr, height_csr, add, degree, gamma, coef0);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        ret = std::vector<real_type>(height_csr + boundary_size, 0.);
        cudaMemcpy(ret_d, ret.data(), sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(height_csr + boundary_size, 1.); 
        cudaMemcpy(d_d, d.data(), sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::csr::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr, height_csr, gamma);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_csr_d, row_csr_d, values_csr_d, QA_cost, cost, nnz_csr, height_csr, add, gamma);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(ret_d);
        cudaFree(d_d);

        cudaFree(values_csr_d);
        cudaFree(col_csr_d);
        cudaFree(row_csr_d);
    }
    
    sub_benchmark_names.push_back("dense (linear)");
    sub_benchmark_names.push_back("COO (linear)");
    sub_benchmark_names.push_back("CSR (linear)");
    sub_benchmark_names.push_back("dense (polynomial)");
    sub_benchmark_names.push_back("COO (polynomial)");
    sub_benchmark_names.push_back("CSR (polynomial)");
    sub_benchmark_names.push_back("dense (radial)");
    sub_benchmark_names.push_back("COO (radial)");
    sub_benchmark_names.push_back("CSR (radial)");
    auto sub_benchmark_runtimes = std::vector<std::vector<ns>>{
        raw_runtimes_dense_linear,
        raw_runtimes_coo_linear,
        raw_runtimes_csr_linear,
        raw_runtimes_dense_poly,
        raw_runtimes_coo_poly,
        raw_runtimes_csr_poly,
        raw_runtimes_dense_radial,
        raw_runtimes_coo_radial,
        raw_runtimes_csr_radial};
    perform_statistics(sub_benchmark_runtimes);
}

}  // namespace plssvm::benchmarks
