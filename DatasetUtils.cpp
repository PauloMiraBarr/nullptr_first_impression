//
// Created by paulo on 5/07/2025.
//

#include "DatasetUtils.h"
#include <algorithm>
#include <chrono>
#include <random>

using namespace utec::algebra;
using namespace utec::data;

Tensor<float, 2> DatasetUtils::vector_to_tensor(const std::vector<TextExample> &dataset) {
    if (dataset.empty()) return Tensor<float, 2>(0, 0);

    size_t num_samples = dataset.size();
    size_t num_features = dataset[0].vectorized_text.size();

    Tensor<float, 2> tensor(num_samples,num_features);

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            tensor(i, j) = dataset[i].vectorized_text[j];
        }
    }

    return tensor;
}


utec::algebra::Tensor<float, 2> DatasetUtils::labels_to_tensor(const std::vector<TextExample> &dataset) {
    if (dataset.empty()) return Tensor<float, 2>(0, 0);

    size_t num_samples = dataset.size();

    Tensor<float, 2> labels(num_samples, 1);

    for (size_t i = 0; i < num_samples; ++i) {
        labels(i, 0) = static_cast<float>(dataset[i].label);
    }

    return labels;
}



void DatasetUtils::split_dataset(const std::vector<TextExample> &dataset, std::vector<TextExample> &train_set, std::vector<TextExample> &test_set, float train_ratio) {
    train_set.clear(); test_set.clear();

    std::vector<TextExample> shuffled = dataset;

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    size_t train_size = static_cast<size_t>(train_ratio * shuffled.size());

    train_set.insert(train_set.end(), shuffled.begin(), shuffled.begin() + train_size);
    test_set.insert(test_set.end(), shuffled.begin() + train_size, shuffled.end());
}


