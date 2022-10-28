/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Heinrich Boschmann
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines all possible sparse datastructures.
 */

#pragma once

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm {

/**
 * @brief Enum class for all possible datastructures.
 */
enum class sparse_type {
    automatic, //CSR
    /** COO Datastructure: TODO: WIKIPEDIA LINK */
    coo,
    /** CSR Datastructure: TODO: WIKIPEDIA LINK */
    csr
};

/**
 * TODO: Kommentar anpassen
 * @brief Output the @p sparse platform to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the target platform to
 * @param[in] target the target platform
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, sparse_type sparse);

/**
 * TODO: Kommentar anpassen
 * @brief Use the input-stream @p in to initialize the @p target platform.
 * @param[in,out] in input-stream to extract the target platform from
 * @param[in] target the target platform
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, sparse_type &sparse);

}  // namespace plssvm