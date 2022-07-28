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
    , empty_row_buffer(0)
{ }

// 0/1 index
template <typename T>
void csr<T>::insert_element(const size_t col_id, const size_t row_id, const real_type value) {
    if(row_offset.size() > row_id + 1){
        throw std::invalid_argument("row_id was already added before!");
    }

    while(row_offset.size() < row_id + 1){
        row_offset.push_back(col_ids.size());
    }

    nnz++;
    col_ids.push_back(col_id);
    values.push_back(value);
    //TODO Groesse
    height = std::max(height, row_id + 1);
    width = std::max(width, col_id + 1);
}

template <typename T>
void csr<T>::append(const csr<real_type> &other) {
    //check whether row to append is empty
    if(other.nnz == 0){
         empty_row_buffer++;
         return;
    }
    //add row offset for all empty rows
    for(;empty_row_buffer > 0; empty_row_buffer--){
        row_offset.push_back(col_ids.size());
    }

    nnz += other.nnz;

    size_t old_max_offset = col_ids.size();
    size_t old_row_offset_end = row_offset.size();

    // TODO: potentially parallelize
    col_ids.insert(col_ids.end(), other.col_ids.begin(), other.col_ids.end());
    row_offset.insert(row_offset.end(), other.row_offset.begin(), other.row_offset.end());
    values.insert(values.end(), other.values.begin(), other.values.end());

    for(size_t i = old_row_offset_end; i < row_offset.size(); i++){
        row_offset.at(i) = row_offset.at(i) + old_max_offset;
    }

    height = old_row_offset_end + other.height;
    width = std::max(width, other.width);
}

template <typename T>
T csr<T>::get_element(const size_t col_id, const size_t row_id) {
    // case: out of bounds
    if (row_id + 1 > height) {
        return static_cast<real_type>(0);
    } 
    // case: first occurence found
    else {
        size_t last_to_check = nnz;
        if(row_id + 1 < height){
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
T csr<T>::get_row_dot_product(const size_t row_id_1, const size_t row_id_2) {
    T result = 0;

    // get borders of row 1
    size_t row_id_1_cur = row_offset[row_id_1];
    size_t last_to_check_row_1 = nnz;
    if(row_id_1 + 1 < height){
       last_to_check_row_1 = row_offset.at(row_id_1 + 1);
    }

    // get borders of row 2
    size_t row_id_2_cur = row_offset[row_id_2];
    size_t last_to_check_row_2 = nnz;
    if(row_id_2 + 1 < height){
        last_to_check_row_2 = row_offset.at(row_id_2 + 1);
    }
    
    // multiply matching col_ids
    while (row_id_1_cur < last_to_check_row_1 && row_id_2_cur < last_to_check_row_2) {
        // matching col_ids, else increment
        if (col_ids[row_id_1_cur] == col_ids[row_id_2_cur]) {
            result += values[++row_id_1_cur] * values[++row_id_2_cur];
        } else if (col_ids[row_id_1_cur] < col_ids[row_id_2_cur]) {
            row_id_1_cur++;
        } else {
           row_id_2_cur++;
        }
    }
    return result;
}


template <typename T>
bool csr<T>::operator==(const csr<T>& other) {
    return nnz == other.nnz
        && width == other.width
        && height == other.height
        && col_ids == other.col_ids
        && row_offset == other.row_offset 
       && values == other.values;
}

// explicitly instantiate template class
template class csr<float>;
template class csr<double>;

}  // namespace plssvm:openmp