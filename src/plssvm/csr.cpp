/**
 * @file
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure for sparse matrices in COO format
 */
#include <algorithm>
#include <iterator>
#include <exception>

#include "plssvm/csr.hpp"

namespace plssvm::openmp {

template <typename T>
csr<T>::csr() 
    : nnz(0)
    , height(0)
    , width(0)
    , rowptr(0)
    , row_offset{0}
{ }

//TODO -------------------------------------------------------------------------------------------------------------------------------------------------- offset correcting!
template <typename T>
void csr<T>::insert_element(const size_t col_id, const size_t row_id, const real_type value) {
    if(rowptr > row_id){
        throw std::invalid_argument("row_id was already added before!");
    }else if(rowptr < row_id){
        while(rowptr < row_id){
            row_offset.push_back(col_ids.size());
            rowptr++;
        }
    }
    
    nnz++;
    col_ids.push_back(col_id);
    values.push_back(value);

    height = std::max(height, row_id + 1);
    width = std::max(width, col_id + 1);
}

template <typename T>
void csr<T>::insert_element_no_column(){
    row_offset.push_back(col_ids.size());
    rowptr++;
}

template <typename T>
void csr<T>::append(const csr<real_type> &other) {
    nnz += other.nnz;
    rowptr += (other.rowptr + 1);

    size_t old_max_offset = col_ids.size();
    size_t old_row_offset_end = row_offset.size();

    // TODO: potentially parallelize
    col_ids.insert(col_ids.end(), other.col_ids.begin(), other.col_ids.end());
    row_offset.insert(row_offset.end(), other.row_offset.begin(), other.row_offset.end());
    values.insert(values.end(), other.values.begin(), other.values.end());

    for(size_t i = old_row_offset_end; i < row_offset.size(); i++){
        row_offset.at(i) = row_offset.at(i) + old_max_offset;
    }

    height = height + other.height;
    width = std::max(width, other.width);
}

template <typename T>
T csr<T>::get_element(const size_t col_id, const size_t row_id) {
    // case: out of bounds
    if (row_id > rowptr) {
        return static_cast<real_type>(0);
    } 
    // case: first occurence found
    else {
        size_t last_to_check = col_ids.size();
        if(row_id < rowptr){
            last_to_check = row_offset.at(row_id + 1);
        }
        // check col_ids / row_ids for valid (cold_id, row_id) pair until one is either found or confirmed nonexistent
        for (size_t i = row_offset.at(row_id); i < last_to_check; i++)
        {
            // case: valid (cold_id, row_id) pair found
            if (col_ids.at(i) == col_id) {
                return values[i];
            }
        }
        // case: (cold_id, row_id) does not exist
        return static_cast<real_type>(0);
    }

}

template <typename T>
bool csr<T>::operator==(const csr<T>& other) {
    return nnz == other.nnz
        && width == other.width
        && height == other.height
        && rowptr == other.rowptr
        && col_ids == other.col_ids
        && row_offset == other.row_offset
        && values == other.values;
}

// explicitly instantiate template class
template class csr<float>;
template class csr<double>;

}  // namespace plssvm:openmp