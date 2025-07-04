//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H


#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec::neural_network {

    template<typename T>
    class MSELoss final : public ILoss<T, 2> {
        Tensor<T, 2> y_pred_;
        Tensor<T, 2> y_true_;
    public:
        MSELoss(const Tensor<T, 2>& y_predicted, const Tensor<T, 2>& y_true)
            : y_pred_(y_predicted), y_true_(y_true) {}

        T loss() const override {
            T sum = 0;
            auto total = y_pred_.size();
            for (size_t i = 0; i < total; ++i) {
                T diff = y_pred_.cbegin()[i] - y_true_.cbegin()[i];
                sum += diff * diff;
            }
            return sum / static_cast<T>(total);
        }

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> grad = y_pred_ - y_true_;
            return grad * (static_cast<T>(2) / static_cast<T>(y_pred_.size()));
        }
    };

    template<typename T>
    class BCELoss final : public ILoss<T, 2> {
        Tensor<T, 2> y_pred_;
        Tensor<T, 2> y_true_;
        constexpr static T epsilon = 1e-7;
    public:
        BCELoss(const Tensor<T, 2>& y_predicted, const Tensor<T, 2>& y_true)
            : y_pred_(y_predicted), y_true_(y_true) {}

        T loss() const override {
            T sum = 0;
            auto total = y_pred_.size();
            for (size_t i = 0; i < total; ++i) {
                T y = y_true_.cbegin()[i];
                T p = std::clamp(y_pred_.cbegin()[i], epsilon, 1 - epsilon);
                sum += - (y * std::log(p) + (1 - y) * std::log(1 - p));
            }
            return sum / static_cast<T>(total);
        }

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> grad = y_pred_;
            auto total = y_pred_.size();
            for (size_t i = 0; i < total; ++i) {
                T y = y_true_.cbegin()[i];
                T p = std::clamp(y_pred_.cbegin()[i], epsilon, 1 - epsilon);
                grad.begin()[i] = (p - y) / (p * (1 - p) * static_cast<T>(total));
            }
            return grad;
        }
    };

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
