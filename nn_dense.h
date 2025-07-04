//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H


#include "nn_interfaces.h"
#include <functional>

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T> {
        Tensor<T, 2> W_, dW_;
        Tensor<T, 1> b_, db_;
        Tensor<T, 2> last_input_;

    public:
        // Constructor genérico con funciones de inicialización
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
                : W_(in_f, out_f), dW_(in_f, out_f),
                  b_(out_f), db_(out_f) {
            init_w_fun(W_);
            // Adaptar b_ como Tensor<T,2> para que init_b_fun pueda usarlo
            Tensor<T, 2> b_as_2d(1, out_f);
            for (size_t j = 0; j < out_f; ++j)
                b_as_2d(0, j) = b_(j);
            init_b_fun(b_as_2d);
            for (size_t j = 0; j < out_f; ++j)
                b_(j) = b_as_2d(0, j);
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            last_input_ = x;
            auto output = matrix_product(x, W_); // (batch_size × out_features)
            const size_t batch_size = x.shape()[0];
            for (size_t i = 0; i < batch_size; ++i)
                for (size_t j = 0; j < b_.shape()[0]; ++j)
                    output(i, j) += b_(j);
            return output;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
            // dW = Xᵗ * dZ
            dW_ = matrix_product(transpose_2d(last_input_), dZ);

            // db = sum over batch
            db_.fill(0);
            const size_t batch_size = dZ.shape()[0];
            const size_t out_features = dZ.shape()[1];
            for (size_t j = 0; j < out_features; ++j) {
                for (size_t i = 0; i < batch_size; ++i) {
                    db_(j) += dZ(i, j);
                }
            }

            // dX = dZ * Wᵗ
            return matrix_product(dZ, transpose_2d(W_));
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(W_, dW_);
            Tensor<T, 2> b2d(1, db_.shape()[0]);
            Tensor<T, 2> db2d(1, db_.shape()[0]);
            for (size_t j = 0; j < db_.shape()[0]; ++j) {
                b2d(0, j) = b_(j);
                db2d(0, j) = db_(j);
            }
            optimizer.update(b2d, db2d);
            for (size_t j = 0; j < db_.shape()[0]; ++j) {
                b_(j) = b2d(0, j);
            }
        }
    };

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
