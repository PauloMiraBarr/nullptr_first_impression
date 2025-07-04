//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H


#include "nn_interfaces.h"
#include <cmath>
#include <unordered_map>

namespace utec::neural_network {

    // --- SGD ---
    template<typename T>
    class SGD final : public IOptimizer<T> {
        T lr_;
    public:
        explicit SGD(T learning_rate = 0.01) : lr_(learning_rate) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            for (size_t i = 0; i < params.size(); ++i) {
                params.begin()[i] -= lr_ * grads.cbegin()[i];
            }
        }
    };

    // --- Adam ---
    template<typename T>
    class Adam final : public IOptimizer<T> {
        T lr_;
        T beta1_;
        T beta2_;
        T epsilon_;
        size_t t_ = 0;

        // Almacenan momentos para cada tensor
        std::unordered_map<void*, Tensor<T, 2>> m_;
        std::unordered_map<void*, Tensor<T, 2>> v_;

        Tensor<T, 2>& get_or_init(std::unordered_map<void*, Tensor<T, 2>>& map, Tensor<T, 2>& ref) {
            void* key = static_cast<void*>(&ref);
            if (!map.count(key)) {
                map[key] = Tensor<T, 2>(ref.shape()[0], ref.shape()[1]);
                map[key].fill(0.0);
            }
            return map[key];
        }

    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

        void step() override {
            ++t_;
        }

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            ++t_;
            auto& m_t = get_or_init(m_, params);
            auto& v_t = get_or_init(v_, params);

            for (size_t i = 0; i < params.size(); ++i) {
                // mt = β1·mt + (1−β1)·gt
                m_t.begin()[i] = beta1_ * m_t.begin()[i] + (1 - beta1_) * grads.cbegin()[i];
                // vt = β2·vt + (1−β2)·(gt²)
                v_t.begin()[i] = beta2_ * v_t.begin()[i] + (1 - beta2_) * grads.cbegin()[i] * grads.cbegin()[i];

                // Bias correction
                T m_hat = m_t.begin()[i] / (1 - std::pow(beta1_, static_cast<T>(t_)));
                T v_hat = v_t.begin()[i] / (1 - std::pow(beta2_, static_cast<T>(t_)));

                // Update rule
                params.begin()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    };

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
