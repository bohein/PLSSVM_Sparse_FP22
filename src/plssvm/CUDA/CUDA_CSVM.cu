#include <plssvm/CUDA/CUDA_CSVM.hpp>
#include <plssvm/CUDA/cuda-kernel.cuh>
#include <plssvm/CUDA/cuda-kernel.hpp>
#include <plssvm/CUDA/svm-kernel.cuh>

namespace plssvm {

    int CUDADEVICE = 0;

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
            if (abort)
                exit(code);
        }
    }

    int count_devices = 1;

    CUDA_CSVM::CUDA_CSVM(real_t cost_, real_t epsilon_, unsigned kernel_, real_t degree_, real_t gamma_, real_t coef0_,
                         bool info_) : CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {
        gpuErrchk(cudaGetDeviceCount(&count_devices));
        datlast_d = std::vector<real_t *>(count_devices);
        data_d = std::vector<real_t *>(count_devices);

        std::cout << "GPUs found: " << count_devices << std::endl;
    }

    void CUDA_CSVM::loadDataDevice() {
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMalloc((void **) &datlast_d[device],
                                 (num_data_points - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_t)));
        }
        std::vector <real_t> datalast(data[num_data_points - 1]);
        datalast.resize(num_data_points - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
#pragma omp parallel for
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMemcpy(datlast_d[device], datalast.data(),
                                 (num_data_points - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_t),
                                 cudaMemcpyHostToDevice));
        }
        datalast.resize(num_data_points - 1);
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMalloc((void **) &data_d[device],
                                 num_features * (num_data_points + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) *
                                 sizeof(real_t)));
        }

        auto begin_transform = std::chrono::high_resolution_clock::now();
        const std::vector <real_t> transformet_data = transform_data(0, THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
        auto end_transform = std::chrono::high_resolution_clock::now();
        if (info) {
            std::clog << std::endl
                      << data.size() << " Datenpunkte mit Dimension " << num_features << " in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_transform - begin_transform).count()
                      << " ms transformiert" << std::endl;
        }
#pragma omp parallel for
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));

            gpuErrchk(cudaMemcpy(data_d[device], transformet_data.data(),
                                 num_features * (num_data_points - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) *
                                 sizeof(real_t), cudaMemcpyHostToDevice));
        }
    }

    std::vector <real_t> CUDA_CSVM::CG(const std::vector <real_t> &b, const int imax, const real_t eps) {
        const size_t dept = num_data_points - 1;
        const size_t boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
        const size_t dept_all = dept + boundary_size;
        std::vector <real_t> zeros(dept_all, 0.0);

        // dim3 grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
        dim3 block(THREADBLOCK_SIZE, THREADBLOCK_SIZE);

        real_t *d;
        std::vector <real_t> x(dept_all, 1.0);
        std::fill(x.end() - boundary_size, x.end(), 0.0);

        std::vector < real_t * > x_d(count_devices);
        std::vector <real_t> r(dept_all, 0.0);
        std::vector < real_t * > r_d(count_devices);
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMalloc((void **) &x_d[device], dept_all * sizeof(real_t)));
            gpuErrchk(cudaMemcpy(x_d[device], x.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMalloc((void **) &r_d[device], dept_all * sizeof(real_t)));
        }

        gpuErrchk(cudaSetDevice(0));
        gpuErrchk(cudaMemcpy(r_d[0], b.data(), dept * sizeof(real_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(r_d[0] + dept, 0, (dept_all - dept) * sizeof(real_t)));
#pragma omp parallel for
        for (int device = 1; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMemset(r_d[device], 0, dept_all * sizeof(real_t)));
        }
        d = new real_t[dept];

        std::vector < real_t * > q_d(count_devices);
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMalloc((void **) &q_d[device], dept_all * sizeof(real_t)));
            gpuErrchk(cudaMemset(q_d[device], 0, dept_all * sizeof(real_t)));
        }
        if (info)
            std::cout << "kernel_q" << std::endl;

        gpuErrchk(cudaDeviceSynchronize());
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            const int Ncols = num_features;
            const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;

            const int start = device * Ncols / count_devices;
            const int end = (device + 1) * Ncols / count_devices;
            kernel_q<<<((int) dept / CUDABLOCK_SIZE) + 1, std::min((size_t) CUDABLOCK_SIZE, dept)>>>(q_d[device],
                                                                                                     data_d[device],
                                                                                                     datlast_d[device],
                                                                                                     Nrows, start, end);
            gpuErrchk(cudaPeekAtLastError());
        }
        gpuErrchk(cudaDeviceSynchronize());
        {
            std::vector <real_t> buffer(dept_all);
            gpuErrchk(cudaSetDevice(0));
            gpuErrchk(cudaMemcpy(buffer.data(), q_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
            std::vector <real_t> ret(dept_all);
            for (int device = 1; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(cudaMemcpy(ret.data(), q_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
                for (int i = 0; i < dept_all; ++i)
                    buffer[i] += ret[i];
            }

#pragma omp parallel for
            for (int device = 0; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(cudaMemcpy(q_d[device], buffer.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
            }
        }

        switch (kernel) {
            case 0: {
#pragma omp parallel for
                for (int device = 0; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    const int Ncols = num_features;
                    const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
                    dim3 grid(static_cast<size_t>(ceil(
                            static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                              static_cast<size_t>(ceil(static_cast<real_t>(dept) /
                                                       static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
                    const int start = device * Ncols / count_devices;
                    const int end = (device + 1) * Ncols / count_devices;
                    kernel_linear<<<grid, block>>>(q_d[device], r_d[device], x_d[device], data_d[device], QA_cost,
                                                   1 / cost, Ncols, Nrows, -1, start, end);
                    gpuErrchk(cudaPeekAtLastError());
                }
                break;
            }
            case 1:
                // kernel_poly<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, num_features , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
                break;
            case 2:
                // kernel_radial<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, num_features , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma);
                break;
            default:
                throw std::runtime_error("Can not decide wich kernel!");
        }

        // cudaMemcpy(r, r_d, dept*sizeof(real_t), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaDeviceSynchronize());
        {
            gpuErrchk(cudaSetDevice(0));
            gpuErrchk(cudaMemcpy(r.data(), r_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
            for (int device = 1; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                std::vector <real_t> ret(dept_all);
                gpuErrchk(cudaMemcpy(ret.data(), r_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
                for (int j = 0; j <= dept; ++j) {
                    r[j] += ret[j];
                }
            }
        }
        real_t delta = mult(r.data(), r.data(), dept); //TODO:
        const real_t delta0 = delta;
        real_t alpha_cd, beta;
        std::vector <real_t> Ad(dept);

        std::vector < real_t * > Ad_d(count_devices);
        for (int device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMalloc((void **) &Ad_d[device], dept_all * sizeof(real_t)));
            gpuErrchk(cudaMemcpy(r_d[device], r.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
        }
        //cudaMallocHost((void **) &Ad, dept *sizeof(real_t));

        int run;
        for (run = 0; run < imax; ++run) {
            if (info)
                std::cout << "Start Iteration: " << run << std::endl;
            //Ad = A * d
            for (int device = 0; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(cudaMemset(Ad_d[device], 0, dept_all * sizeof(real_t)));
                gpuErrchk(cudaMemset(r_d[device] + dept, 0, (dept_all - dept) * sizeof(real_t)));
            }
            switch (kernel) {
                case 0: {
#pragma omp parallel for
                    for (int device = 0; device < count_devices; ++device) {
                        gpuErrchk(cudaSetDevice(device));
                        const int Ncols = num_features;
                        const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
                        dim3 grid(static_cast<size_t>(ceil(static_cast<real_t>(dept) /
                                                           static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                                  static_cast<size_t>(ceil(static_cast<real_t>(dept) /
                                                           static_cast<real_t>(THREADBLOCK_SIZE *
                                                                               INTERNALBLOCK_SIZE))));
                        const int start = device * Ncols / count_devices;
                        const int end = (device + 1) * Ncols / count_devices;
                        kernel_linear<<<grid, block>>>(q_d[device], Ad_d[device], r_d[device], data_d[device], QA_cost,
                                                       1 / cost, Ncols, Nrows, 1, start, end);
                        gpuErrchk(cudaPeekAtLastError());
                    }
                }
                    break;
                case 1:
                    // kernel_poly<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1, gamma, coef0, degree);
                    break;
                case 2:
                    // kernel_radial<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), 1, gamma);
                    break;
                default:
                    throw std::runtime_error("Can not decide wich kernel!");
            }

            for (int i = 0; i < dept; ++i)
                d[i] = r[i];

            gpuErrchk(cudaDeviceSynchronize());
            {
                std::vector <real_t> buffer(dept_all, 0);
                for (int device = 0; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    std::vector <real_t> ret(dept_all, 0);
                    gpuErrchk(cudaMemcpy(ret.data(), Ad_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
                    for (int j = 0; j <= dept; ++j) {
                        buffer[j] += ret[j];
                    }
                }
                std::copy(buffer.begin(), buffer.begin() + dept, Ad.data());
                for (int device = 0; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    gpuErrchk(
                            cudaMemcpy(Ad_d[device], buffer.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
                }
            }

            alpha_cd = delta / mult(d, Ad.data(), dept);
            // add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
            //TODO: auf GPU
            std::vector <real_t> buffer_r(dept_all);
            cudaSetDevice(0);
            gpuErrchk(cudaMemcpy(buffer_r.data(), r_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
            add_mult_(((int) dept / 1024) + 1, std::min(1024, (int) dept), x.data(), buffer_r.data(), alpha_cd, dept);

#pragma omp parallel for
            for (int device = 0; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(cudaMemcpy(x_d[device], x.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
            }
            if (run % 50 == 49) {
                std::vector <real_t> buffer(b);
                buffer.resize(dept_all);
                gpuErrchk(cudaSetDevice(0));
                gpuErrchk(cudaMemcpy(r_d[0], buffer.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
#pragma omp parallel for
                for (int device = 1; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    gpuErrchk(cudaMemset(r_d[device], 0, dept_all * sizeof(real_t)));
                }
                switch (kernel) {
                    case 0: {
#pragma omp parallel for
                        for (int device = 0; device < count_devices; ++device) {
                            gpuErrchk(cudaSetDevice(device));
                            const int Ncols = num_features;
                            const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
                            const int start = device * Ncols / count_devices;
                            const int end = (device + 1) * Ncols / count_devices;
                            dim3 grid(static_cast<size_t>(ceil(static_cast<real_t>(dept) /
                                                               static_cast<real_t>(THREADBLOCK_SIZE *
                                                                                   INTERNALBLOCK_SIZE))),
                                      static_cast<size_t>(ceil(static_cast<real_t>(dept) /
                                                               static_cast<real_t>(THREADBLOCK_SIZE *
                                                                                   INTERNALBLOCK_SIZE))));
                            kernel_linear<<<grid, block>>>(q_d[device], r_d[device], x_d[device], data_d[device],
                                                           QA_cost, 1 / cost, Ncols, Nrows, -1, start, end);
                            gpuErrchk(cudaPeekAtLastError());
                        }
                    }
                        break;
                    case 1:
                        // kernel_poly<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
                        break;
                    case 2:
                        // kernel_radial<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , -1, gamma);
                        break;
                    default:
                        throw std::runtime_error("Can not decide wich kernel!");
                }
                gpuErrchk(cudaDeviceSynchronize());
                // cudaMemcpy(r, r_d, dept*sizeof(real_t), cudaMemcpyDeviceToHost);

                {
                    gpuErrchk(cudaSetDevice(0));
                    gpuErrchk(cudaMemcpy(r.data(), r_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for
                    for (int device = 1; device < count_devices; ++device) {
                        gpuErrchk(cudaSetDevice(device));
                        std::vector <real_t> ret(dept_all, 0);
                        gpuErrchk(
                                cudaMemcpy(ret.data(), r_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
                        for (int j = 0; j <= dept; ++j) {
                            r[j] += ret[j];
                        }
                    }
#pragma omp parallel for
                    for (int device = 0; device < count_devices; ++device) {
                        gpuErrchk(cudaSetDevice(device));
                        gpuErrchk(cudaMemcpy(r_d[device], r.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
                    }
                }
            } else {
                for (int index = 0; index < dept; ++index) {
                    r[index] -= alpha_cd * Ad[index];
                }
            }

            delta = mult(r.data(), r.data(), dept); //TODO:
            if (delta < eps * eps * delta0)
                break;
            beta = -mult(r.data(), Ad.data(), dept) / mult(d, Ad.data(), dept); //TODO:
            add(mult(beta, d, dept), r.data(), d, dept);                        //TODO:

            {
                std::vector <real_t> buffer(dept_all, 0.0);
                std::copy(d, d + dept, buffer.begin());
#pragma omp parallel for
                for (int device = 0; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    gpuErrchk(
                            cudaMemcpy(r_d[device], buffer.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
                }
            }
        }
        if (run == imax)
            std::clog << "Regard reached maximum number of CG-iterations" << std::endl;

        alpha.resize(dept);
        std::vector <real_t> ret_q(dept);
        gpuErrchk(cudaDeviceSynchronize());
        {
            std::vector <real_t> buffer(dept_all);
            std::copy(x.begin(), x.begin() + dept, alpha.begin());
            gpuErrchk(cudaSetDevice(0));
            gpuErrchk(cudaMemcpy(buffer.data(), q_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
            std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
        }
        // cudaMemcpy(&alpha[0],x_d, dept * sizeof(real_t), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&ret_q[0],q_d, dept * sizeof(real_t), cudaMemcpyDeviceToHost);
        // cudaFree(Ad_d);
        // cudaFree(r_d);
        // cudaFree(datlast);
        // cudaFreeHost(Ad);
        // cudaFree(x_d);
        // cudaFreeHost(r);
        // cudaFreeHost(d);
        return ret_q;
    }

}