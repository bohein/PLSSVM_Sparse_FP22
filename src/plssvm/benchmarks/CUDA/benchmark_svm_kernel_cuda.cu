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
    std::vector<real_type> q_d; // q-Vector on device
    real_type QA_cost;
    real_type QA_cost_d;
    std::vector<real_type> ret; // result Vector
    std::vector<real_type> ret_d; // result Vector on device
    std::vector<real_type> d; // ""right-hand side of the equation"
    std::vector<real_type> d_d; // ""right-hand side of the equation" on device

    real_type cost_d;
    real_type add_d;
    int degree_d;
    real_type gamma_d;
    real_type coef0_d;
    

    cudaMalloc((void**)&cost_d, sizeof(real_type));
    cudaMemcpy((void*)&cost_d, (void*)&cost, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&add_d, sizeof(real_type));
    cudaMemcpy((void*)&add_d, (void*)&add, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&degree_d, sizeof(int));
    cudaMemcpy((void*)&degree_d, (void*)&degree, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&gamma_d, sizeof(real_type));
    cudaMemcpy((void*)&gamma_d, (void*)&gamma, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&coef0_d, sizeof(real_type));
    cudaMemcpy((void*)&coef0_d, (void*)&coef0, sizeof(real_type), cudaMemcpyHostToDevice);

    std::vector<std::vector<real_type>> data_dense;
    std::vector<real_type> data_dense_d;

    plssvm::openmp::coo<real_type> data_coo{};
    std::vector<real_type> values_coo_d;
    std::vector<size_t> row_coo_d;
    std::vector<size_t> col_coo_d;
    size_t nnz_coo_d;
    size_t last_row_begin_coo_d;
    

    plssvm::openmp::csr<real_type> data_csr{};
    std::vector<real_type> values_csr_d;
    std::vector<size_t> row_csr_d;
    std::vector<size_t> col_csr_d;
    size_t nnz_csr_d;
    size_t height_csr_d;

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

    const size_t num = ((*data_ptr_dense.get())[0].size()) * (data_ptr_dense.get()->size());
    std::vector<real_type> vec_1D(num);
    
    for (typename std::vector<real_type>::size_type col = 0; col < data_ptr_dense.get()[0].size(); ++col) {
        for (std::size_t row = 0; row < data_ptr_dense.get()->size(); ++row) {
            vec_1D[col * data_ptr_dense.get()->size() + row] = data_ptr_dense->at(row)[col];
        }
    }

    auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(vec_1D);

    auto data_dense_last = std::make_shared<const std::vector<real_type>>((*data_ptr_dense.get())[data_ptr_dense.get() -> size() - 1]);
    std::vector<real_type> data_dense_last_d;
    int num_rows_d;
    int num_cols_d;
    int id_d;
    
    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_dense.get() -> size() - 1;

    plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    dim3 grid_q(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block_q(range_q.block[0], range_q.block[1], range_q.block[2]); 

    auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    plssvm::detail::execution_range range_svm({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    dim3 grid_svm(range_svm.grid[0], range_svm.grid[1], range_svm.grid[2]);
    dim3 block_svm(range_svm.block[0], range_svm.block[1], range_svm.block[2]); 

    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_dense -> size()));
        cudaMalloc((void**)&ret_d, sizeof(real_type)*(data_ptr_dense -> size()));
        cudaMalloc((void**)&d_d, sizeof(real_type)*(data_ptr_dense -> size()));
        cudaMalloc((void**)&QA_cost_d, sizeof(real_type));
        cudaMalloc((void**)&id_d, sizeof(int));

        cudaMalloc((void**)&data_dense_d, sizeof(real_type)*(data_ptr_dense_1D.get() -> size()));
        cudaMalloc((void**)&num_rows_d, sizeof(int));
        cudaMalloc((void**)&num_cols_d, sizeof(int));
        //cudaMalloc((void**)&data_dense_last_d, sizeof(real_type)*(*data_ptr_dense_1D.get())[0].size());
        cudaMalloc((void**)&data_dense_last_d, sizeof(real_type)*(data_ptr_dense.get()->at(0).size()));

        
        cudaMemcpy((void*)&data_dense_d, (void*)&data_ptr_dense_1D.get()->at(0), sizeof(real_type)*(data_ptr_dense_1D.get() -> size()),cudaMemcpyHostToDevice);
        auto num_rows = data_ptr_dense.get() -> size();
        cudaMemcpy((void*)&num_rows_d, (void*)&num_rows, sizeof(int),cudaMemcpyHostToDevice);
        auto num_cols = data_ptr_dense.get()->at(0).size();
        cudaMemcpy((void*)&num_cols_d, (void*)&num_cols, sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&data_dense_last_d, (void*)&data_dense_last, sizeof(real_type) * (data_ptr_dense.get()->at(0).size()),cudaMemcpyHostToDevice);

        
        cudaMemcpy((void*)&id_d, (void*)&id, sizeof(int),cudaMemcpyHostToDevice);

        q = std::vector<real_type>(data_ptr_dense->size() - 1); // q-Vector
        cudaMemcpy((void*)&q_d, (void*)&q, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.); 
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_linear<<<grid_q, block_q>>>(q_d.data(), data_dense_d.data(), data_dense_last_d.data(), num_rows_d, num_cols_d);
        cudaDeviceSynchronize();
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_linear<<<grid_svm, block_svm>>>(q_d.data(), ret_d.data(), d_d, data_dense_d.data(), QA_cost_d, cost_d, num_rows_d, num_cols_d, add_d, id_d); //id = 0;
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.);
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_poly<<<grid_q, block_q>>>(q_d, data_dense_d, data_dense_last_d, num_rows_d, num_cols_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_poly<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, data_dense_d, QA_cost_d, cost_d, num_rows_d, num_cols_d, add_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.); 
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, data_dense_d, data_dense_last_d, num_rows_d, num_cols_d, gamma_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, data_dense_d, QA_cost_d, cost_d, num_rows_d, num_cols_d, add_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree((void*)&q_d);
        cudaFree((void*)&QA_cost_d);
        cudaFree((void*)&ret_d);
        cudaFree((void*)&d_d);
        cudaFree((void*)&data_dense_d);
        cudaFree((void*)&num_rows_d);
        cudaFree((void*)&num_cols_d);
        cudaFree((void*)&data_dense_last_d);
        cudaFree((void*)&id_d);
    }
    
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    num_rows_exc_last = data_ptr_coo.get() -> get_height() - 1;
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

    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_coo -> get_height() - 1));
        cudaMalloc((void**)&ret_d, sizeof(real_type)*(data_ptr_coo -> get_height()));
        cudaMalloc((void**)&d_d, sizeof(real_type)*(data_ptr_coo -> get_height()));
        cudaMalloc((void**)&QA_cost_d, sizeof(real_type));


        cudaMalloc((void**)&nnz_coo_d, sizeof(size_t));
        cudaMalloc((void**)&last_row_begin_coo_d, sizeof(size_t));
        cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(data_ptr_coo -> get_nnz()));
        cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(data_ptr_coo -> get_nnz()));
        cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(data_ptr_coo -> get_nnz()));

        auto nnz_coo = data_ptr_coo.get() -> get_nnz();
        cudaMemcpy((void*)&nnz_coo_d, (void*)&nnz_coo, sizeof(size_t), cudaMemcpyHostToDevice);
        auto last_row_begin_coo = data_ptr_coo.get() -> get_last_row_begin();
        cudaMemcpy((void*)&last_row_begin_coo_d, (void*)&last_row_begin_coo, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&values_coo_d[0], (void*)&data_ptr_coo.get() -> get_values().at(0), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&row_coo_d[0], (void*)&data_ptr_coo.get() -> get_row_ids().at(0), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&col_coo_d[0], (void*)&data_ptr_coo.get() -> get_col_ids().at(0), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);

        q = std::vector<real_type>(data_ptr_coo->get_height() - 1); // q-Vector
        cudaMemcpy((void*)&q, (void*)&q_d, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.); 
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_linear<<<grid_q, block_q>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo_d, nnz_coo_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_linear<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_coo_d, row_coo_d, values_coo_d, QA_cost_d, cost_d, nnz_coo_d, add_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_coo_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.);
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_poly<<<grid_q, block_q>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo_d, nnz_coo_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_poly<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_coo_d, row_coo_d, values_coo_d, QA_cost_d, cost_d, nnz_coo_d, add_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_coo_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.); 
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo_d, nnz_coo_d, gamma_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_coo_d, row_coo_d, values_coo_d, QA_cost_d, cost_d, nnz_coo_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree((void*)&q_d);
        cudaFree((void*)&QA_cost_d);
        cudaFree((void*)&ret_d);
        cudaFree((void*)&d_d);

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
    grid_q = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block_q = dim3(range_q.block[0], range_q.block[1], range_q.block[2]); 

    grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    plssvm::detail::execution_range range_svm_csr({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    grid_svm = dim3(range_svm_csr.grid[0], range_svm_csr.grid[1], range_svm_csr.grid[2]);
    block_svm = dim3(range_svm_csr.block[0], range_svm_csr.block[1], range_svm_csr.block[2]);

    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_dense -> size() - 1));
        cudaMalloc((void**)&ret_d, sizeof(real_type)*(data_ptr_csr -> get_height()));
        cudaMalloc((void**)&d_d, sizeof(real_type)*(data_ptr_csr -> get_height()));
        cudaMalloc((void**)&QA_cost_d, sizeof(real_type));

        cudaMalloc((void**)&height_csr_d, sizeof(size_t));
        cudaMalloc((void**)&nnz_csr_d, sizeof(size_t));
        cudaMalloc((void**)&values_csr_d, sizeof(real_type)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&col_csr_d, sizeof(size_t)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&row_csr_d, sizeof(size_t)*(data_ptr_csr -> get_height()));

        auto height_csr = data_ptr_csr.get() -> get_height();
        cudaMemcpy((void*)&height_csr_d, (void*)&height_csr, sizeof(size_t), cudaMemcpyHostToDevice);
        auto nnz_csr = data_ptr_csr.get() -> get_nnz();
        cudaMemcpy((void*)&nnz_csr_d, (void*)&nnz_csr, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&values_csr_d, (void*)&data_ptr_csr.get() -> get_values().at(0), sizeof(real_type)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&row_csr_d, (void*)&data_ptr_csr.get() -> get_row_offset().at(0), sizeof(size_t)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)&col_csr_d, (void*)&data_ptr_csr.get() -> get_col_ids().at(0), sizeof(size_t)*(data_ptr_csr -> get_height()), cudaMemcpyHostToDevice);

        q = std::vector<real_type>(data_ptr_csr->get_height() - 1); // q-Vector
        cudaMemcpy((void*)&q, (void*)&q_d, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        cudaMemcpy((void*)&d_d,(void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_linear<<<grid_q, block_q>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr_d, height_csr_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_linear<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_csr_d, row_csr_d, values_csr_d, QA_cost_d, cost_d, nnz_csr_d, height_csr_d, add_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_poly<<<grid_q, block_q>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr_d, height_csr_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_poly<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_csr_d, row_csr_d, values_csr_d, QA_cost_d, cost_d, nnz_csr_d, height_csr_d, add_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        cudaMemcpy((void*)&QA_cost_d, (void*)&QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        cudaMemcpy((void*)&ret_d, (void*)&ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        cudaMemcpy((void*)&d_d, (void*)&d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr_d, height_csr_d, gamma_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_csr_d, row_csr_d, values_csr_d, QA_cost_d, cost_d, nnz_csr_d, height_csr_d, add_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree((void*)&q_d);
        cudaFree((void*)&QA_cost_d);
        cudaFree((void*)&ret_d);
        cudaFree((void*)&d_d);

        cudaFree((void*)&height_csr_d);
        cudaFree((void*)&nnz_csr_d);
        cudaFree((void*)&values_csr_d);
        cudaFree((void*)&col_csr_d);
        cudaFree((void*)&row_csr_d);
    }

    cudaFree((void*)&cost_d);
    cudaFree((void*)&add_d);
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