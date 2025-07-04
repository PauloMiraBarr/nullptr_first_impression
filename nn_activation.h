//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T>
    class ReLU final : public ILayer<T> {
        Tensor<T, 2> z_;
    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
            z_ = z;
            Tensor<T, 2> result = z;
            for (size_t i = 0; i < z.shape()[0]; ++i)
                for (size_t j = 0; j < z.shape()[1]; ++j)
                    result(i, j) = std::max(static_cast<T>(0), z(i, j));
            return result;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& g) override {
            Tensor<T, 2> dz = g;
            for (size_t i = 0; i < z_.shape()[0]; ++i)
                for (size_t j = 0; j < z_.shape()[1]; ++j)
                    dz(i, j) = z_(i, j) > 0 ? g(i, j) : 0;
            return dz;
        }
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
        Tensor<T, 2> s_;
    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
            s_ = z;
            for (size_t i = 0; i < z.shape()[0]; ++i)
                for (size_t j = 0; j < z.shape()[1]; ++j)
                    s_(i, j) = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-z(i, j)));
            return s_;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& g) override {
            Tensor<T, 2> dz = g;
            for (size_t i = 0; i < s_.shape()[0]; ++i)
                for (size_t j = 0; j < s_.shape()[1]; ++j)
                    dz(i, j) = g(i, j) * s_(i, j) * (1 - s_(i, j));
            return dz;
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H

