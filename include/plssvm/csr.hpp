/**
 * @file
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure for sparse matrices in CSR format
 */

#pragma once

#include <vector> // std::vector

namespace plssvm::openmp {

template <typename T>
class csr {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

    public:
        /// The type of the data. Must be either `float` or `double`.
        using real_type = T;

        csr();


        ////////////////////
        // element access //
        ////////////////////

        /**
         * @brief Get the data value stored at given indices. Returns 0.0 if no value is stored there.
         * 
         * @param col_id column index of requested value
         * @param row_id row index of requested value
         * @return real_type data value stored at given indices
         */
        real_type get_element(const size_t col_id, const size_t row_id) const; // requires being sorted row-whise

        /**
         * @brief Returns the dot-product of the two specified rows in the matrix
         * 
         * @param row_id_1 index of the first row
         * @param row_id_2 index of the second row
         * @return real_type dot-product of the two rows
         */
        real_type get_row_dot_product(const size_t row_id_1, const size_t row_id_2) const;

         /**
         * @brief Returns the squared euclidean distance of the two specified rows in the matrix
         * 
         * @param row_id_1 index of the first row
         * @param row_id_2 index of the second row
         * @return real_type squared euclidean distance of the two rows
         */
        real_type get_row_squared_euclidean_dist(const size_t row_id_1, const size_t row_id_2) const;

        /**
         * @return size_t number of non-zero elements
         */
        size_t get_nnz() const {return nnz;}

        /**
         * @return size_t height of stored matrix
         */
        size_t get_height() const {return height;}

        /**
         * @return size_t width of stored matrix
         */
        size_t get_width() const {return width;}

        /**
         * @return std::vector<size_t> col_ids of stored matrix
         */
        std::vector<size_t> get_col_ids() const {return col_ids;}

        /**
         * @return std::vector<size_t> row_ids of stored matrix
         */
        std::vector<size_t> get_row_offset() const {return row_offset;}

        /**
         * @return std::vector<real_type> values of stored matrix
         */
        std::vector<real_type> get_values() const {return values;}

        /**
         * @return size_t first index of last row
         */
        size_t get_last_row_begin() const {return row_offset.back();}


        ///////////////
        // modifiers //
        ///////////////

        /**
         * @brief Insert data element into CSR matrix data structure
         * 
         * @param col_id column index of element to insert
         * @param row_id row index of element to insert
         * @param value data value of element to insert
         */
        void insert_element(const size_t col_id, const size_t row_id, const real_type value);

        /**
         * @brief Append another matrix stored in CSR format to this matrix
         * 
         * @param other another matrix in CSR format
         */
        void append(const csr<real_type> &other);

        /**
         * @brief inserts padding to row array
         * 
         * @param padding_size size of added padding
         * @param padding_value value in padding
         */
        void add_padding(const size_t padding_size, const size_t padding_value);


        //////////////////////////
        // non-member functions //
        //////////////////////////

        bool operator==(const csr<real_type> &other);

    private:
        /// number of non-zero elements
        size_t nnz;
        /// height of stored matrix
        size_t height;
        /// width of stored matrix
        size_t width;
        /// number of empty rows when appending
        size_t empty_row_buffer;
        /// column indices of stored data values
        std::vector<size_t> col_ids;
        /// row indices of stored data values
        std::vector<size_t> row_offset;
        /// stored data values
        std::vector<real_type> values;
};

}  // namespace plssvm::openmp
