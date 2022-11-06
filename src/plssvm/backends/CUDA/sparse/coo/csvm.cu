/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/coo/csvm.hpp"

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/CUDA/detail/utility.cuh"     // plssvm::cuda::detail::device_synchronize, plssvm::detail::cuda::get_device_count, plssvm::detail::cuda::set_device, plssvm::detail::cuda::peek_at_last_error
#include "plssvm/backends/CUDA/exceptions.hpp"         // plssvm::cuda::backend_exception

//#include "plssvm/backends/CUDA/predict_kernel.cuh"     // plssvm::cuda::kernel_w, plssvm::cuda::predict_points_poly, plssvm::cuda::predict_points_rbf
#include "plssvm/backends/CUDA/sparse/coo/coo_q_kernel.cuh"           // plssvm::cuda::device_kernel_q_linear, plssvm::cuda::device_kernel_q_poly, plssvm::cuda::device_kernel_q_radial
#include "plssvm/backends/CUDA/sparse/coo/coo_svm_kernel.cuh"         // plssvm::cuda::device_kernel_linear, plssvm::cuda::device_kernel_poly, plssvm::cuda::device_kernel_radial
#include "plssvm/backends/gpu_csvm.hpp"                // plssvm::detail::gpu_csvm
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"           // plssvm::detail::execution_range
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform
#include "plssvm/sparse_types.hpp"

#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <exception>  // std::terminate
#include <numeric>    // std::iota
#include <utility>    // std::pair, std::make_pair
#include <vector>     // std::vector

namespace plssvm::cuda::coo {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    base_type{ params } {
    // check if supported target platform has been selected
    if (target_ != target_platform::automatic && target_ != target_platform::gpu_nvidia) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the CUDA backend!", target_) };
    } else {
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
        throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
    }

    if (print_info_) {
        fmt::print("Using CUDA as backend.\n");
    }

    // get all available devices wrt the requested target platform
    devices_.resize(std::min<std::size_t>(detail::get_device_count(), num_features_));
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no CUDA devices could be found
    if (devices_.empty()) {
        throw backend_exception{ "CUDA backend selected but no CUDA devices were found!" };
    }

    //only support single GPU execution
    devices_.resize(1);

    // resize vectors accordingly
    data_d_.resize(devices_.size());
    data_last_d_.resize(devices_.size());

    if (print_info_) {
        // print found CUDA devices
        fmt::print("Found {} CUDA device:\n", devices_.size());
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, devices_[device]);
            fmt::print("  [{}, {}, {}.{}]\n", devices_[device], prop.name, prop.major, prop.minor);
        }
        fmt::print("\n");
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the CUDA devices have finished before destruction
        for (const queue_type &device : devices_) {
            detail::device_synchronize(device);
        }
    } catch (const plssvm::exception &e) {
        fmt::print("{}\n", e.what_with_loc());
        std::terminate();
    }
}

template <typename T>
void csvm<T>::device_synchronize(queue_type &queue) {
 //nothing to do here
}

std::pair<dim3, dim3> execution_range_to_native(const ::plssvm::detail::execution_range &range) {
    dim3 grid(range.grid[0], range.grid[1], range.grid[2]);
    dim3 block(range.block[0], range.block[1], range.block[2]);
    return std::make_pair(grid, block);
}

template <typename T>
void csvm<T>::run_q_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &q_d, const std::size_t num_features) {
   
}

template <typename T>
void csvm<T>::run_svm_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const std::size_t num_features) {
    
}

template <typename T>
void csvm<T>::run_w_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, const std::size_t num_features) {
    
}

template <typename T>
void csvm<T>::run_predict_kernel(const ::plssvm::detail::execution_range &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const std::size_t num_predict_points) {
    
}

template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::cuda::coo
