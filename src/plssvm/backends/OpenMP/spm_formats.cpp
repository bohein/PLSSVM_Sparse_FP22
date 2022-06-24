/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure(s) for sparse matrices
 */

#include "plssvm/backends/OpenMP/spm_formats.hpp" 

namespace plssvm::openmp {

template <typename T>
coo<T>::coo() 
    : nnz(0), col_ids(new std::vector<size_t>()), row_ids(new std::vector<size_t>()), data(new std::vector<real_type>()) { }

template <typename real_type>
void coo<T>::insert_element(size_t col_id, size_t row_id, real_type value) {
    this->nnz++;
    this->col_ids.push_back(col_id);
    this->row_ids.push_back(row_id);
    this->values.push_back(value);
}

// explicitly instantiate template class
template class coo<float>;
template class coo<double>;

}  // namespace plssvm:openmp