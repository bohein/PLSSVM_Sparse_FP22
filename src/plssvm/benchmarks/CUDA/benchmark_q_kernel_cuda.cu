/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks reagrding q-kernel functions.
 */

#include "plssvm/benchmarks/CUDA/benchmark_q_kernel_cuda.cuh"

#include "plssvm/backends/CUDA/q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/coo_q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/csr_q_kernel.cuh"
#include "plssvm/detail/execution_range.hpp"


#include <numeric>
#include <iostream>

#include <stdio.h>
namespace plssvm::benchmarks {

benchmark_q_kernel_cuda::benchmark_q_kernel_cuda() : benchmark{"Q-Kernels (CUDA)"} {}

void benchmark_q_kernel_cuda::run() {
    /*
    int vector_size = 3;
    std::vector<int> vec{ 0, 2, 5 };
    std::vector<int> vec_d;

    cudaError_t cudaStatus = cudaMalloc((void**)&vec_d, sizeof(int) * vector_size);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %i\n", cudaStatus);
        return;
    }

    cudaStatus = cudaMemcpy((void*)&vec_d[0], (void*)&vec[0], sizeof(int) * vector_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %i\n", cudaStatus);
        return;
    }

    // vector_size works pass by value
    plssvm::cuda::coo::myKernel<<<1, 2>>>(vec_d.data(), vector_size);
    cudaDeviceSynchronize();

    auto *p_vec_d = &vec_d;
    cudaStatus = cudaFree((void*)p_vec_d);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree failed: %i\n", cudaStatus);
    }

    return;
    */

    using real_type = double;

    datasets.insert(datasets.end(), DATAPOINT.begin(), DATAPOINT.end());
    datasets.insert(datasets.end(), FEATURE.begin(), FEATURE.end());
    datasets.insert(datasets.end(), DENSITY.begin(), DENSITY.end());
    //datasets.insert(datasets.end(), REAL_WORLD.begin(), REAL_WORLD.end());

    for (auto& ds : datasets) evaluate_dataset(ds);
}

void benchmark_q_kernel_cuda::evaluate_dataset(const dataset &ds) {
    using real_type = double;

    std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();

    plssvm::parameter<real_type> params;

    real_type *q_d;

    std::vector<std::vector<real_type>> data_dense;
    
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

    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_dense -> size() - 1;

 //auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(transform_data(data_ptr_dense.get(), boundary_size, num_rows_exc_last));
    std::vector<real_type> vec_1D(data_ptr_dense -> at(0).size() * (num_rows_exc_last + boundary_size));
    
    for (typename std::vector<real_type>::size_type col = 0; col <  data_ptr_dense -> at(0).size(); ++col) {
        for (std::size_t row = 0; row < num_rows_exc_last; ++row) {
            vec_1D[col * (num_rows_exc_last + boundary_size)  + row] = data_ptr_dense->at(row).at(col);
        }
    }

    auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(vec_1D);

    auto data_dense_last = std::make_shared<const std::vector<real_type>>((*data_ptr_dense.get())[data_ptr_dense -> size() - 1]);
    real_type *data_dense_d;
    real_type *data_dense_last_d;


    plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });

    dim3 grid(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block(range_q.block[0], range_q.block[1], range_q.block[2]); 

    for(size_t i = 0; i < 0; i++) {
        size_t num_rows = num_rows_exc_last + boundary_size;
        size_t num_cols = data_ptr_dense -> at(0).size();

        cudaMalloc((void**)&q_d, sizeof(real_type)*num_rows);
        cudaMalloc((void**)&data_dense_d, sizeof(real_type)*(data_ptr_dense_1D -> size()));
        cudaMalloc((void**)&data_dense_last_d, sizeof(real_type)*num_cols);

        cudaMemcpy(data_dense_d, data_ptr_dense_1D -> data(), sizeof(real_type)*(data_ptr_dense_1D -> size()), cudaMemcpyHostToDevice);
        cudaMemcpy(data_dense_last_d, data_dense_last -> data(), sizeof(real_type)*num_cols, cudaMemcpyHostToDevice);

        std::vector<real_type> q(num_rows); // q-Vector
        cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        //Ist .data() hier richtig?
        plssvm::cuda::device_kernel_q_linear<<<grid, block>>>(q_d, data_dense_d, data_dense_last_d, num_rows, num_cols);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_poly<<<grid, block>>>(q_d, data_dense_d, data_dense_last_d, num_rows, num_cols, degree, gamma, coef0);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_radial<<<grid, block>>>(q_d, data_dense_d, data_dense_last_d, num_rows, num_cols, gamma);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
      
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(data_dense_d);
        cudaFree(data_dense_last_d);
    }
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    num_rows_exc_last = data_ptr_coo -> get_height() - 1;

    range_q = plssvm::detail::execution_range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    
    grid = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block = dim3(range_q.block[0], range_q.block[1], range_q.block[2]); 

    plssvm::openmp::coo<real_type> data_coo_padded = *(data_ptr_coo.get());
    data_coo_padded.add_padding(boundary_size, 0, 0, 0);
    auto data_ptr_coo_padded = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo_padded));

    for(size_t i = 0; i < cycles; i++) {
        auto nnz_coo = data_ptr_coo -> get_nnz();
        auto last_row_begin_coo = data_ptr_coo -> get_last_row_begin();

        cudaMalloc((void**)&q_d, sizeof(real_type)*((data_ptr_coo -> get_height() - 1) + boundary_size));
        cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(nnz_coo + boundary_size));
        cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
        cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(nnz_coo + boundary_size));
        
        cudaMemcpy(values_coo_d, data_ptr_coo_padded -> get_values().data(), sizeof(real_type)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
        cudaMemcpy(row_coo_d, data_ptr_coo_padded -> get_row_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);
        cudaMemcpy(col_coo_d, data_ptr_coo_padded -> get_col_ids().data(), sizeof(size_t)*(nnz_coo + boundary_size), cudaMemcpyHostToDevice);

        std::vector<real_type> q((data_ptr_coo -> get_height() - 1) + boundary_size); // q-Vector
        cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_q_linear<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, nnz_coo, last_row_begin_coo);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_q_poly<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, nnz_coo, last_row_begin_coo, degree, gamma, coef0);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_q_radial<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, nnz_coo, last_row_begin_coo, gamma);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
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

    grid = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block= dim3(range_q.block[0], range_q.block[1], range_q.block[2]);

    plssvm::openmp::csr<real_type> data_csr_padded = *(data_ptr_csr.get());
    data_csr_padded.add_padding(boundary_size, data_ptr_csr -> get_nnz());
    auto data_ptr_csr_padded = std::make_shared<const plssvm::openmp::csr<real_type>>(std::move(data_csr_padded));

    for(size_t i = 0; i < cycles; i++) {
        auto height_csr = data_ptr_csr -> get_height();
        auto nnz_csr = data_ptr_csr -> get_nnz();
        
        cudaMalloc((void**)&q_d, sizeof(real_type)*(height_csr - 1 + boundary_size));
        cudaMalloc((void**)&values_csr_d, sizeof(real_type)*nnz_csr);
        cudaMalloc((void**)&col_csr_d, sizeof(size_t)*nnz_csr);
        cudaMalloc((void**)&row_csr_d, sizeof(size_t)*(height_csr + boundary_size));
        
        cudaMemcpy(values_csr_d, data_ptr_csr -> get_values().data(), sizeof(real_type)*nnz_csr, cudaMemcpyHostToDevice);
        cudaMemcpy(row_csr_d, data_ptr_csr_padded -> get_row_offset().data(), sizeof(size_t)*(height_csr + boundary_size), cudaMemcpyHostToDevice);
        cudaMemcpy(col_csr_d, data_ptr_csr -> get_col_ids().data(), sizeof(size_t)*nnz_csr, cudaMemcpyHostToDevice);
        
        std::vector<real_type> q(height_csr - 1 + boundary_size); // q-Vector
        cudaMemcpy(q_d, q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_q_linear<<<grid, block>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr, height_csr);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_csr_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_q_poly<<<grid, block>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr, height_csr, degree, gamma, coef0);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_q_radial<<<grid, block>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr, height_csr, gamma);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
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