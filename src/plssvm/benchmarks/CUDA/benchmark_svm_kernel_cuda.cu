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

#include "plssvm/benchmarks/benchmark_svm_kernel_cuda.cuh"

#include "plssvm/backends/CUDA/q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/coo_q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/csr_q_kernel.cuh"

#include "plssvm/backends/CUDA/svm_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/coo_svm_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/csr_svm_kernel.cuh"

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
    

    cudaMalloc((void*)&cost_d, sizeof(real_type));
    cudaMemcpy(cost_d, cost, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void*)&add_d, sizeof(real_type));
    cudaMemcpy(add_d, add, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void*)&degree_d, sizeof(int));
    cudaMemcpy(degree_d, degree, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void*)&gamma_d, sizeof(real_type));
    cudaMemcpy(gamma_d, gamma, sizeof(real_type), cudaMemcpyHostToDevice);
    
    cudaMalloc((void*)&coef0_d, sizeof(real_type));
    cudaMemcpy(coef0_d, coef0, sizeof(real_type), cudaMemcpyHostToDevice);

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
    auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(plssvm::csvm<real_type>::transform_data(data_ptr_dense.get(), 0, ((*data_ptr_dense.get())[0].size()) * (data_ptr_dense.get() -> size()))); //padding----------------------

    auto data_dense_last = std::make_shared<const std::vector<real_type>>((*data_ptr_dense.get())[data_ptr_dense.get() -> size() - 1]);
    std::vector<real_type> data_dense_d;
    std::vector<real_type> data_dense_last_d;
    int num_rows_d;
    int num_cols_d;
    int id_d;
    
    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last;

    size_t boundary_size = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    size_t num_rows_exc_last = data_ptr_dense.get() -> size() - 1;

    const plssvm::detail::execution_range range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    dim3 grid_q(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    dim3 block_q(range_q.block[0], range_q.block[1], range_q.block[2]); 

    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    const plssvm::detail::execution_range range_svm({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

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
        cudaMalloc((void**)&data_dense_last_d, sizeof(real_type)*(*data_ptr_dense_1D.get())[0].size());


        cudaMemcpy(data_dense_d, data_ptr_dense_1D.get(), sizeof(real_type)*(data_ptr_dense_1D.get() -> size()));
        cudaMemcpy(num_rows_d, data_ptr_dense.get() -> size(), sizeof(int));
        cudaMemcpy(num_cols_d, (*data_ptr_dense.get())[0].size(), sizeof(int));
        cudaMemcpy(data_dense_last_d, data_dense_last, sizeof(real_type) * (*data_ptr_dense.get())[0].size());
        cudaMemcpy(id_d, id, sizeof(int));

        q = std::vector<real_type>(data_ptr_dense->size() - 1); // q-Vector
        cudaMemcpy(q_d, q, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.); 
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_linear<<<grid_q, block_q>>>(q_d, data_dense_d, data_dense_last_d, num_rows_d, num_cols_d);
        cudaDeviceSynchronize();
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_linear<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, data_dense_d, QA_cost_d, cost_d, num_rows_d, num_cols_d, add_d, id_d); //id = 0;
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = (*data_ptr_dense)[data_ptr_dense->size() - 1][(*data_ptr_dense)[0].size() - 1] * cost;
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.);
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

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
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_dense->size(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_dense->size(), 1.); 
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, data_dense_d, data_dense_last_d, num_rows_d, num_cols_d, gamma_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, data_dense_d, QA_cost_d, cost_d, num_rows_d, num_cols_d, add_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(QA_cost_d);
        cudaFree(ret_d);
        cudaFree(d_d);
        cudaFree(data_dense_d);
        cudaFree(num_rows_d);
        cudaFree(num_cols_d);
        cudaFree(data_dense_last_d);
        cudaFree(id_d);
    }
    
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_coo);

    num_rows_exc_last = data_ptr_coo.get() -> get_heigth() - 1;

    range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    grid_q = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block_q = dim3(range_q.block[0], range_q.block[1], range_q.block[2]); 

    grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    range_svm({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    grid_svm = dim3(range_svm.grid[0], range_svm.grid[1], range_svm.grid[2]);
    block_svm = dim3(range_svm.block[0], range_svm.block[1], range_svm.block[2]); 

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

        cudaMemcpy(nnz_coo_d, data_ptr_coo.get() -> get_nnz(), sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(last_row_begin_coo_d, data_ptr_coo.get() -> get_last_row_begin(), sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(values_coo_d, data_ptr_coo.get() -> get_values(), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(row_coo_d, data_ptr_coo.get() -> get_rows(), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(column_coo_d, data_ptr_coo.get() -> get_columns(), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);

        q = std::vector<real_type>(data_ptr_coo->get_height() - 1); // q-Vector
        cudaMemcpy(q, q_d, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_coo->get_element(data_ptr_coo->get_height() - 1, data_ptr_coo->get_width() - 1) * cost;
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.); 
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

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
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.);
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

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
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_coo->get_height(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_coo->get_height(), 1.); 
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo_d, nnz_coo_d, gamma_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_coo_d, row_coo_d, values_coo_d, QA_cost_d, cost_d, nnz_coo_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(QA_cost_d);
        cudaFree(ret_d);
        cudaFree(d_d);

        cudaFree(nnz_coo_d);
        cudaFree(last_row_begin_coo_d);
        cudaFree(values_coo_d);
        cudaFree(col_coo_d);
        cudaFree(row_coo_d);
    }
    
    // coo
    std::vector<ns> raw_runtimes_csr_linear;
    std::vector<ns> raw_runtimes_csr_poly;
    std::vector<ns> raw_runtimes_csr_radial;
    params.parse_libsvm_file_sparse(ds.path, data_ptr_csr);

    num_rows_exc_last = data_ptr_csr.get() -> get_heigth() - 1;

    range_q({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_rows_exc_last) });
    grid_q = dim3(range_q.grid[0], range_q.grid[1], range_q.grid[2]);
    block_q = dim3(range_q.block[0], range_q.block[1], range_q.block[2]); 

    grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows_exc_last) / static_cast<real_type>(boundary_size)));
    range_svm({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    grid_svm = dim3(range_svm.grid[0], range_svm.grid[1], range_svm.grid[2]);
    block_svm = dim3(range_svm.block[0], range_svm.block[1], range_svm.block[2]);

    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_csr -> get_height()-1));
        cudaMalloc((void**)&ret_d, sizeof(real_type)*(data_ptr_dense -> get_height()));
        cudaMalloc((void**)&d_d, sizeof(real_type)*(data_ptr_dense -> get_height()));
        cudaMalloc((void**)&QA_cost_d, sizeof(real_type));

        cudaMalloc((void**)&height_csr_d, sizeof(size_t));
        cudaMalloc((void**)&nnz_csr_d, sizeof(size_t));
        cudaMalloc((void**)&values_csr_d, sizeof(real_type)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&col_csr_d, sizeof(size_t)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&row_csr_d, sizeof(size_t)*(data_ptr_csr -> get_height()));

        cudaMemcpy(height_csr_d, data_ptr_csr.get() -> get_height(), sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(nnz_csr_d, data_ptr_csr.get() -> get_nnz(), sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(values_csr_d, data_ptr_csr.get() -> get_values(), sizeof(real_type)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(row_csr_d, data_ptr_csr.get() -> get_rows(), sizeof(size_t)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(column_csr_d, data_ptr_csr.get() -> get_columns(), sizeof(size_t)*(data_ptr_csr -> get_height()), cudaMemcpyHostToDevice);

        q = std::vector<real_type>(data_ptr_csr->get_height() - 1); // q-Vector
        cudaMemcpy(q, q_d, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);
        // linear
        fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        QA_cost = data_ptr_csr->get_element(data_ptr_csr->get_height() - 1, data_ptr_csr->get_width() - 1) * cost;
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

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
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

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
        cudaMemcpy(QA_cost_d, QA_cost, sizeof(real_type), cudaMemcpyHostToDevice);
        ret = std::vector<real_type>(data_ptr_csr->get_height(), 0.);
        cudaMemcpy(ret_d, ret, sizeof(real_type)*ret.size(), cudaMemcpyHostToDevice);
        d = std::vector<real_type>(data_ptr_csr->get_height(), 1.); 
        cudaMemcpy(d_d, d, sizeof(real_type)*d.size(), cudaMemcpyHostToDevice);

        plssvm::cuda::device_kernel_q_radial<<<grid_q, block_q>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr_d, height_csr_d, gamma_d);
        cudaDeviceSynchronize();
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_radial<<<grid_svm, block_svm>>>(q_d, ret_d, d_d, col_csr_d, row_csr_d, values_csr_d, QA_cost_d, cost_d, nnz_csr_d, height_csr_d, add_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(QA_cost_d);
        cudaFree(ret_d);
        cudaFree(d_d);

        cudaFree(csr_height_d);
        cudaFree(nnz_csr_d);
        cudaFree(values_csr_d);
        cudaFree(col_csr_d);
        cudaFree(row_csr_d);
    }

    cudaFree(cost_d);
    cudaFree(add_d);
    cudaFree(degree_d);
    cudaFree(gamma_d);
    cudaFree(coef0_d);
    
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (linear)");
    sub_benchmark_names.push_back(sub_benchmark_name + "COO (linear)");
    sub_benchmark_names.push_back(sub_benchmark_name + "CSR (linear)");
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (polynomial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "COO (polynomial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "CSR (polynomial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "dense (radial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "COO (radial)");
    sub_benchmark_names.push_back(sub_benchmark_name + "CSR (radial)");
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
