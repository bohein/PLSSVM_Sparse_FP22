/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks
 */

#pragma once

#include "plssvm/core.hpp"
#include "plssvm/benchmarks/statistics.hpp"

#include "fmt/core.h"
#include "fmt/ostream.h" 
#include "fmt/chrono.h"

namespace plssvm::benchmarks {

class benchmark {
    public:
        using real_type = double;

        benchmark(const std::string benchmark_name);
        virtual ~benchmark() {}

        std::string get_name() const {return name;}
        std::string get_data() {return data_to_csv();}

        virtual void run() {}

        

    protected:
        std::string name;
        std::vector<std::string> sub_benchmark_names;
        std::vector<ns> runtimes_mean;
        std::vector<ns> runtimes_median;
        std::vector<ns> runtimes_min;
        std::vector<ns> runtimes_max;
        std::vector<ns> runtimes_variance;
        std::vector<ns> runtimes_std_deviation;

        uint64_t cycles = 100;

        benchmark();

        virtual void evaluate_dataset(const std::string sub_benchmark_name, const std::string path_to_dataset) {sub_benchmark_name + path_to_dataset;}
        virtual std::string data_to_csv();
        virtual void perform_statistics(std::vector<std::vector<ns>> &various_runtimes);

        // TODO: change to relative path
        const std::string DATASET_TINY = "/home/schmidtm/PLSSVM/benchmark_data/iris.libsvm"; // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale
        const std::string DATASET_SMALL = "/home/schmidtm/PLSSVM/benchmark_data/w3a.libsvm"; // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w3a
        const std::string DATASET_MEDIUM = "/home/schmidtm/PLSSVM/benchmark_data/ijcnn1.libsvm"; // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2
        const std::string DATASET_LARGE = "/home/schmidtm/PLSSVM/benchmark_data/skin_nonskin.libsvm"; // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/skin_nonskin
        const std::string DATASET_HUGE = "/home/schmidtm/PLSSVM/benchmark_data/SUSY.libsvm"; // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.xz

};

}  // namespace plssvm:benchmarks