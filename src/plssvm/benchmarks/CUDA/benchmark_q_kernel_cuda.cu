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

    //datasets.insert(datasets.end(), DATAPOINT.begin(), DATAPOINT.end());
    //datasets.insert(datasets.end(), FEATURE.begin(), FEATURE.end());
    //datasets.insert(datasets.end(), DENSITY.begin(), DENSITY.end());
    //datasets.insert(datasets.end(), REAL_WORLD.begin(), REAL_WORLD.end());

    //for (auto& ds : datasets) evaluate_dataset(ds);
    datasets.push_back(DATAPOINT[12]);
    evaluate_dataset(DATAPOINT[12]);
}

void benchmark_q_kernel_cuda::evaluate_dataset(const dataset &ds) {
    using real_type = double;

    std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();

    plssvm::parameter<real_type> params;

    int degree_d;
    real_type gamma_d;
    real_type coef0_d;

    cudaMalloc((void**)&degree_d, sizeof(int)); //cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(value_type))
    cudaMemcpy((void*)&degree_d, (void*)&degree, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&gamma_d, sizeof(real_type));
    cudaMemcpy((void*)&gamma_d, (void*)&gamma, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&coef0_d, sizeof(real_type));
    cudaMemcpy((void*)&coef0_d, (void*)&coef0, sizeof(real_type), cudaMemcpyHostToDevice);

    std::vector<real_type> q_d;

    std::vector<std::vector<real_type>> data_dense;
    
    plssvm::openmp::coo<real_type> data_coo{};
    std::vector<real_type> values_coo_d;
    std::vector<size_t> row_coo_d;
    std::vector<size_t> col_coo_d;
    int nnz_coo_d;
    int last_row_begin_coo_d;
    
    plssvm::openmp::csr<real_type> data_csr{};
    std::vector<real_type> values_csr_d;
    std::vector<size_t> row_csr_d;
    std::vector<size_t> col_csr_d;
    int nnz_csr_d;
    int height_csr_d;

    auto data_ptr_dense = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data_dense));
    auto data_ptr_coo = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(data_coo));
    auto data_ptr_csr = std::make_shared<const plssvm::openmp::csr<real_type>>(std::move(data_csr));

    // dense
    std::vector<ns> raw_runtimes_dense_linear;
    std::vector<ns> raw_runtimes_dense_poly;
    std::vector<ns> raw_runtimes_dense_radial;
   
    params.parse_libsvm_file(ds.path, data_ptr_dense);

    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_dense.get() -> size() - 1;

 //auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(transform_data(data_ptr_dense.get(), boundary_size, num_rows_exc_last));
    std::vector<real_type> vec_1D(data_ptr_dense.get()[0].size() * (num_rows_exc_last + boundary_size));
    
    for (typename std::vector<real_type>::size_type col = 0; col < data_ptr_dense.get()[0].size(); ++col) {
        for (std::size_t row = 0; row < num_rows_exc_last; ++row) {
            vec_1D[col * (num_rows_exc_last + boundary_size) + row] = data_ptr_dense->at(row)[col];
        }
    }

    auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(vec_1D);

    auto data_dense_last = std::make_shared<const std::vector<real_type>>((*data_ptr_dense.get())[data_ptr_dense.get() -> size() - 1]);
    std::vector<real_type> data_dense_d;
    std::vector<real_type> data_dense_last_d;
    int num_rows_d;
    int num_cols_d;


    plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });

    dim3 grid(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block(range_q.block[0], range_q.block[1], range_q.block[2]); 

    for(size_t i = 0; i < 0; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_dense -> size() - 1));
        cudaMalloc((void**)&data_dense_d, sizeof(real_type)*(data_ptr_dense_1D.get() -> size()));
        cudaMalloc((void**)&num_rows_d, sizeof(int));
        cudaMalloc((void**)&num_cols_d, sizeof(int));
        cudaMalloc((void**)&data_dense_last_d, sizeof(real_type) * (data_ptr_dense.get()->at(0).size()));

        size_t num_rows = num_rows_exc_last + boundary_size;
        size_t num_cols = data_ptr_dense.get()->at(0).size();
        cudaMemcpy((void*)&data_dense_d[0], (void*)&data_ptr_dense_1D->at(0), sizeof(real_type)*(data_ptr_dense_1D.get() -> size()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&num_rows_d, (void*)&num_rows, sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&num_cols_d, (void*)&num_cols, sizeof(int) ,cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&data_dense_last_d[0], (void*)&data_dense_last->at(0), sizeof(real_type)*(data_ptr_dense.get()->at(0).size()),cudaMemcpyHostToDevice);

        std::vector<real_type> q(data_ptr_dense->size() - 1); // q-Vector
        cudaMemcpy((void*)&q_d[0], (void*)&q, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        //Ist .data() hier richtig?
        plssvm::cuda::device_kernel_q_linear<<<grid, block>>>(q_d.data(), data_dense_d.data(), data_dense_last_d.data(), num_rows_d, num_cols_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_poly<<<grid, block>>>(q_d.data(), data_dense_d.data(), data_dense_last_d.data(), num_rows_d, num_cols_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_radial<<<grid, block>>>(q_d.data(), data_dense_d.data(), data_dense_last_d.data(), num_rows_d, num_cols_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
      
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree((void*)&q_d);
        cudaFree((void*)&data_dense_d);
        cudaFree((void*)&num_rows_d);
        cudaFree((void*)&num_cols_d);
        cudaFree((void*)&data_dense_last_d);
    }
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    num_rows_exc_last = data_ptr_coo.get() -> get_height() - 1;

    range_q = plssvm::detail::execution_range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    
    grid = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block = dim3(range_q.block[0], range_q.block[1], range_q.block[2]); 

    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_coo -> get_height() - 1));

        cudaMalloc((void**)&nnz_coo_d, sizeof(int));
        cudaMalloc((void**)&last_row_begin_coo_d, sizeof(int));
        cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(data_ptr_coo -> get_nnz()));
        cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(data_ptr_coo -> get_nnz()));
        cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(data_ptr_coo -> get_nnz()));

        auto nnz_coo = data_ptr_coo.get() -> get_nnz();
        auto last_row_begin_coo = data_ptr_coo.get() -> get_last_row_begin();
        cudaMemcpy((void*)&nnz_coo_d, (void*)&nnz_coo, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&last_row_begin_coo_d, (void*)&last_row_begin_coo, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&values_coo_d[0], (void*)&data_ptr_coo.get() -> get_values().at(0), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&row_coo_d[0], (void*)&data_ptr_coo.get() ->get_row_ids().at(0), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&col_coo_d[0], (void*)&data_ptr_coo.get() -> get_col_ids().at(0), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);

        std::vector<real_type> q(data_ptr_coo->get_height() - 1); // q-Vector
        cudaMemcpy(q_d.data(), q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_q_linear<<<grid, block>>>(q_d.data(), col_coo_d.data(), row_coo_d.data(), values_coo_d.data(), nnz_coo_d, last_row_begin_coo_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_q_poly<<<grid, block>>>(q_d.data(), col_coo_d.data(), row_coo_d.data(), values_coo_d.data(), nnz_coo_d, last_row_begin_coo_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::coo::device_kernel_q_radial<<<grid, block>>>(q_d.data(), col_coo_d.data(), row_coo_d.data(), values_coo_d.data(), nnz_coo_d, last_row_begin_coo_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree((void*)&q_d);

        cudaFree((void*)&nnz_coo_d);
        cudaFree((void*)&last_row_begin_coo_d);
        cudaFree((void*)&values_coo_d);
        cudaFree((void*)&col_coo_d);
        cudaFree((void*)&row_coo_d);
    }

    // csr
    std::vector<ns> raw_runtimes_csr_linear;
    std::vector<ns> raw_runtimes_csr_poly;
    std::vector<ns> raw_runtimes_csr_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_csr);

    num_rows_exc_last = data_ptr_csr.get() -> get_height() - 1;

    range_q = plssvm::detail::execution_range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });

    grid = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block= dim3(range_q.block[0], range_q.block[1], range_q.block[2]);

    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_csr -> get_height() - 1));

        cudaMalloc((void**)&height_csr_d, sizeof(int));
        cudaMalloc((void**)&nnz_csr_d, sizeof(int));
        cudaMalloc((void**)&values_csr_d, sizeof(real_type)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&col_csr_d, sizeof(size_t)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&row_csr_d, sizeof(size_t)*(data_ptr_csr -> get_height()));


       
        auto height_csr = data_ptr_csr.get() -> get_height();
        cudaMemcpy((void*)&height_csr_d, (void*)&height_csr, sizeof(int), cudaMemcpyHostToDevice);
        auto nnz_csr = data_ptr_csr.get() -> get_nnz();
        cudaMemcpy((void*)&nnz_csr_d, (void*)&nnz_csr, sizeof(int), cudaMemcpyHostToDevice);
        
        cudaMemcpy((void*)&values_csr_d[0], (void*)&data_ptr_csr.get() -> get_values().at(0), sizeof(real_type)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&row_csr_d[0], (void*)&data_ptr_csr.get() -> get_row_offset().at(0), sizeof(size_t)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&col_csr_d[0], (void*)&data_ptr_csr.get() -> get_col_ids().at(0), sizeof(size_t)*(data_ptr_csr -> get_height()), cudaMemcpyHostToDevice);
        
        std::vector<real_type> q(data_ptr_csr->get_height() - 1); // q-Vector
        cudaMemcpy(q_d.data(), q.data(), sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_q_linear<<<grid, block>>>(q_d.data(), col_csr_d.data(), row_csr_d.data(), values_csr_d.data(), nnz_csr_d, height_csr_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_csr_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_q_poly<<<grid, block>>>(q_d.data(), col_csr_d.data(), row_csr_d.data(), values_csr_d.data(), nnz_csr_d, height_csr_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::csr::device_kernel_q_radial<<<grid, block>>>(q_d.data(), col_csr_d.data(), row_csr_d.data(), values_csr_d.data(), nnz_csr_d, height_csr_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree((void*)&q_d);

        cudaFree((void*)&height_csr_d);
        cudaFree((void*)&nnz_csr_d);
        cudaFree((void*)&values_csr_d);
        cudaFree((void*)&col_csr_d);
        cudaFree((void*)&row_csr_d);
    }

    cudaFree((void*)&degree_d);
    cudaFree((void*)&gamma_d);
    cudaFree((void*)&coef0_d);
    
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