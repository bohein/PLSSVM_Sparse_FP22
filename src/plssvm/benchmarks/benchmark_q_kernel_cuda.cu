/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks reagrding q-kernel functions.
 */

#include "plssvm/benchmarks/benchmark_q_kernel_cuda.cuh"

#include "plssvm/backends/CUDA/q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/coo_q_kernel.cuh"
#include "plssvm/backends/CUDA/sparse/csr_q_kernel.cuh"


#include <numeric>
#include <iostream>

namespace plssvm::benchmarks {

benchmark_q_kernel_cuda::benchmark_q_kernel_cuda() : benchmark{"Q-Kernels (CUDA)"} {}

void benchmark_q_kernel_openmp::run() {
    using real_type = double;

    evaluate_dataset("tiny (~150)", DATASET_TINY);
    evaluate_dataset("small (~5000)", DATASET_SMALL);
    evaluate_dataset("medium (~50000)", DATASET_MEDIUM);
    evaluate_dataset("large (~250000)", DATASET_LARGE);
}

void benchmark_q_kernel_openmp::evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {
    using real_type = double;

    std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end_time = std::chrono::high_resolution_clock::now();

    plssvm::parameter<real_type> params;

    int degree_d;
    real_type gamma_d;
    real_type coef0_d;

    cudaMalloc((void**)&degree_d, sizeof(int)); //cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(value_type))
    cudaMemcpy(degree_d, degree, sizeof(real_type), cudaMemcpyToDevice);
    
    cudaMalloc((void**)&gamma_d, sizeof(real_type));
    cudaMemcpy(gamma_d, gamma, sizeof(real_type), cudaMemcpyToDevice);
    
    cudaMalloc((void**)&coef0_d, sizeof(real_type));
    cudaMemcpy(coef0_d, coef0, sizeof(real_type), cudaMemcpyToDevice);

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
    params.parse_libsvm_file(path_to_dataset, data_ptr_dense);
    auto data_ptr_dense_1D = std::make_shared<const std::vector<real_type>>(base_type::transform_data(data_ptr_dense.get(), 0, ((*data_ptr_dense.get())[0].size()) * (data_ptr_dense.get() -> size())));

    auto data_dense_last = std::make_shared<const std::vector<real_type>>((*data_ptr_dense.get())[data_ptr_dense.get() -> size() - 1]);
    std::vector<real_type> data_dense_d;
    std::vector<real_type> data_dense_last_d;
    int num_rows_d;
    int num_cols_d;

    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_dense -> size() - 1));
        cudaMalloc((void**)&data_dense_d, sizeof(real_type)*(data_ptr_dense_1D.get() -> size()));
        cudaMalloc((void**)&num_rows_d, sizeof(int));
        cudaMalloc((void**)&num_cols_d, sizeof(int));
        cudaMalloc((void**)&data_dense_last_d, sizeof(real_type)*(*data_ptr_dense_1D.get())[0].size());


        cudaMemcpy(data_dense_d, data_ptr_dense_1D.get(), sizeof(real_type)*(data_ptr_dense_1D.get() -> size()));
        cudaMemcpy(num_rows_d, data_ptr_dense_1D.get() -> size(), sizeof(int));
        cudaMemcpy(num_cols_d, (*data_ptr_dense_1D.get())[0].size(), sizeof(int));
        cudaMemcpy(data_dense_last_d, data_dense_last, sizeof(real_type) * (*data_ptr_dense_1D.get())[0].size());

        std::vector<real_type> q(data_ptr_dense->size() - 1); // q-Vector
        cudaMemcpy(q_d, q, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("dense (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_linear<<<grid, block>>>(q_d, data_dense_d, data_dense_last_d, num_rows_d, num_cols_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("dense (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_poly<<<grid, block>>>(q_d, data_dense_d, data_dense_last_d, num_rows_d, num_cols_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_dense_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("dense (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_radial<<<grid, block>>>(q_d, data_dense_d, data_dense_last_d, num_rows_d, num_cols_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
      
        raw_runtimes_dense_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);
        cudaFree(data_dense_d);
        cudaFree(num_rows_d);
        cudaFree(num_cols_d);
        cudaFree(data_dense_last_d);
    }
    
    // coo
    std::vector<ns> raw_runtimes_coo_linear;
    std::vector<ns> raw_runtimes_coo_poly;
    std::vector<ns> raw_runtimes_coo_radial;
    params.parse_libsvm_file_sparse(path_to_dataset, data_ptr_coo);
    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_dense -> size() - 1));

        cudaMalloc((void**)&nnz_coo_d, sizeof(int));
        cudaMalloc((void**)&last_row_begin_coo_d, sizeof(int));
        cudaMalloc((void**)&values_coo_d, sizeof(real_type)*(data_ptr_coo -> get_nnz()));
        cudaMalloc((void**)&col_coo_d, sizeof(size_t)*(data_ptr_coo -> get_nnz()));
        cudaMalloc((void**)&row_coo_d, sizeof(size_t)*(data_ptr_coo -> get_nnz()));

        cudaMemcpy(nnz_coo_d, data_ptr_coo.get() -> get_nnz(), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(last_row_begin_coo_d, data_ptr_coo.get() -> get_last_row_begin(), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(values_coo_d, data_ptr_coo.get() -> get_values(), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(row_coo_d, data_ptr_coo.get() -> get_rows(), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(column_coo_d, data_ptr_coo.get() -> get_columns(), sizeof(real_type)*(data_ptr_coo -> get_nnz()), cudaMemcpyHostToDevice);

        std::vector<real_type> q(data_ptr_coo->get_height() - 1); // q-Vector
        cudaMemcpy(q_d, q, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("coo (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_linear<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo_d, nnz_coo_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("coo (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
       
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_poly<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo_d, nnz_coo_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("coo (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_radial<<<grid, block>>>(q_d, col_coo_d, row_coo_d, values_coo_d, last_row_begin_coo_d, nnz_coo_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_coo_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);

        cudaFree(nnz_coo_d);
        cudaFree(last_row_begin_coo_d);
        cudaFree(values_coo_d);
        cudaFree(col_coo_d);
        cudaFree(row_coo_d);
    }

    // csr
    std::vector<ns> raw_runtimes_csr_linear;
    std::vector<ns> raw_runtimes_csr_poly;
    std::vector<ns> raw_runtimes_csr_radial;
    params.parse_libsvm_file_sparse(path_to_dataset, data_ptr_csr);
    for(size_t i = 0; i < cycles; i++) {
        cudaMalloc((void**)&q_d, sizeof(real_type)*(data_ptr_dense -> size() - 1));

        cudaMalloc((void**)&height_csr_d, sizeof(int));
        cudaMalloc((void**)&nnz_csr_d, sizeof(int));
        cudaMalloc((void**)&values_csr_d, sizeof(real_type)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&col_csr_d, sizeof(size_t)*(data_ptr_csr -> get_nnz()));
        cudaMalloc((void**)&row_csr_d, sizeof(size_t)*(data_ptr_csr -> get_height()));

        cudaMemcpy(height_csr_d, data_ptr_csr.get() -> get_height(), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(nnz_csr_d, data_ptr_csr.get() -> get_nnz(), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(values_csr_d, data_ptr_csr.get() -> get_values(), sizeof(real_type)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(row_csr_d, data_ptr_csr.get() -> get_rows(), sizeof(size_t)*(data_ptr_csr -> get_nnz()), cudaMemcpyHostToDevice);
        cudaMemcpy(column_csr_d, data_ptr_csr.get() -> get_columns(), sizeof(size_t)*(data_ptr_csr -> get_height()), cudaMemcpyHostToDevice);
        
        std::vector<real_type> q(data_ptr_csr->get_height() - 1); // q-Vector
        cudaMemcpy(q_d, q, sizeof(real_type)*q.size(), cudaMemcpyHostToDevice);

        // linear
        fmt::print("csr (linear) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_linear<<<grid, block>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr_d, height_csr_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
       
        raw_runtimes_csr_linear.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // polynomial
        fmt::print("csr (polynomial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_poly<<<grid, block>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr_d, height_csr_d, degree_d, gamma_d, coef0_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_poly.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        // radial
        fmt::print("csr (radial) " + std::to_string(i + 1) + "/" + std::to_string(cycles) + " (");
        
        start_time = std::chrono::high_resolution_clock::now();
        plssvm::cuda::device_kernel_q_radial<<<grid, block>>>(q_d, col_csr_d, row_csr_d, values_csr_d, nnz_csr_d, height_csr_d, gamma_d);
        cudaDeviceSynchronize();
        end_time = std::chrono::high_resolution_clock::now();
        
        raw_runtimes_csr_radial.push_back(std::chrono::round<ns>(end_time - start_time));
        fmt::print(std::to_string(std::chrono::round<ns>(end_time - start_time).count()/1000000) + "ms)\n");

        cudaFree(q_d);

        cudaFree(csr_height_d);
        cudaFree(nnz_csr_d);
        cudaFree(values_csr_d);
        cudaFree(col_csr_d);
        cudaFree(row_csr_d);
    }

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