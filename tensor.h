//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <initializer_list>
#include <functional>
#include <numeric>

namespace utec::algebra {

    template <typename T = int, std::size_t Rank = 0>
    class Tensor {
    private:
        std::array<std::size_t, Rank> dim{};
        std::vector<T> arr;

        static void ThrowDimensionMatchException(const std::string& e) {
            throw std::invalid_argument("Number of dimensions do not match with " + e);
        }
        static void ThrowDataSizeMatchException() {
            throw std::invalid_argument("Data size does not match tensor size");
        }
        static void ThrowBroadcastShapeCompatibilityException() {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        // Computes total number of elements
        std::size_t get_total_dim() const {
            std::size_t n = 1;
            for (const auto& d : dim) n *= d;
            return n;
        }

        // Correct multidimensional index calculation using strides
        T& get_element_by_list(const std::size_t* idx_list) {
            std::size_t index = 0;
            std::size_t stride = 1;
            for (int i = Rank - 1; i >= 0; --i) {
                index += idx_list[i] * stride;
                stride *= dim[i];
            }
            return arr[index];
        }
        const T& get_element_by_list(const std::size_t* idx_list) const {
            std::size_t index = 0;
            std::size_t stride = 1;
            for (int i = Rank - 1; i >= 0; --i) {
                index += idx_list[i] * stride;
                stride *= dim[i];
            }
            return arr[index];
        }

        // Recursive print helper
        void print_recursive(std::ostream& os, std::size_t level, std::size_t offset) const {
            std::size_t d = dim[level];
            if (level == Rank - 1) {
                os << std::string(level * 4, ' ');
                for (std::size_t i = 0; i < d; ++i) {
                    os << arr[offset + i];
                    if (i + 1 < d) os << " ";
                }
            } else {
                os << std::string(level * 4, ' ') << "{\n";
                std::size_t block_size = 1;
                for (std::size_t i = level + 1; i < Rank; ++i) block_size *= dim[i];
                for (std::size_t i = 0; i < d; ++i) {
                    print_recursive(os, level + 1, offset + i * block_size);
                    if (i + 1 < d) os << "\n";
                }
                os << "\n" << std::string(level * 4, ' ') << "}";
            }
        }

    public:
        // Default constructor: creates 1x1x...x1 tensor
        Tensor() {
            dim.fill(1);
            arr.resize(get_total_dim(), T{});
        }

        // Construct with given dimensions
        template<class... Args>
        explicit Tensor(Args... args) {
            if (sizeof...(Args) != Rank) ThrowDimensionMatchException(std::to_string(Rank));
            std::size_t tmp[] = {static_cast<std::size_t>(args)...};
            std::copy(tmp, tmp + Rank, dim.begin());
            arr.resize(get_total_dim(), T{});
        }

        // Copy and move semantics
        Tensor(const Tensor& other) = default;
        Tensor& operator=(const Tensor& other) = default;
        Tensor(Tensor&& other) noexcept = default;
        Tensor& operator=(Tensor&& other) noexcept = default;

        void fill(const T& value) noexcept {
            std::fill(arr.begin(), arr.end(), value);
        }

        // Assign from initializer_list
        Tensor& operator=(const std::initializer_list<T>& values) {
            if (values.size() != arr.size()) ThrowDataSizeMatchException();
            std::copy(values.begin(), values.end(), arr.begin());
            return *this;
        }

        // Reshape without changing total elements
        template<class... Args>
        void reshape(Args... arg) {
            if(sizeof...(Args) != Rank)
                ThrowDimensionMatchException(std::to_string(Rank));
            std::size_t temp[] = {static_cast<std::size_t>(arg)...};
            std::copy(temp, temp + Rank, dim.begin());
            arr.resize(get_total_dim());
        }

        const std::array<std::size_t, Rank>& shape() const noexcept { return dim; }
        std::size_t size() const noexcept { return arr.size(); }
        std::vector<T> data() const { return arr; }

        // Access bounds
        auto begin()  { return arr.begin(); }
        auto end() { return arr.end(); }
        // Safe access bounds
        auto cbegin() const { return arr.begin(); }
        auto cend() const { return arr.end(); }

        // Element access
        template<class... Args>
        T& operator()(Args... args) {
            if (sizeof...(Args) != Rank) ThrowDimensionMatchException(std::to_string(Rank));
            std::size_t tmp[] = {static_cast<std::size_t>(args)...};
            return get_element_by_list(tmp);
        }
        template<class... Args>
        const T& operator()(Args... args) const {
            if (sizeof...(Args) != Rank) ThrowDimensionMatchException(std::to_string(Rank));
            std::size_t tmp[] = {static_cast<std::size_t>(args)...};
            return get_element_by_list(tmp);
        }

        // Compute broadcasted shape
        static std::array<std::size_t, Rank> broadcast_shape(
            const std::array<std::size_t, Rank>& A,
            const std::array<std::size_t, Rank>& B) {
            std::array<std::size_t, Rank> result{};
            for (std::size_t i = 0; i < Rank; ++i) {
                if (A[i] == B[i] || B[i] == 1) result[i] = A[i];
                else if (A[i] == 1) result[i] = B[i];
                else ThrowBroadcastShapeCompatibilityException();
            }
            return result;
        }

        // Apply binary operation with broadcasting
        void apply(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B, const std::function<T(T, T)>& op) {
            auto shape_A = A.shape();
            auto shape_B = B.shape();
            auto shape_C = broadcast_shape(shape_A, shape_B);

            dim = shape_C;
            arr.resize(get_total_dim());
            std::size_t total = get_total_dim();

            for (std::size_t i = 0; i < total; ++i) {
                std::array<std::size_t, Rank> idx{};
                std::size_t rem = i;
                for (int r = Rank - 1; r >= 0; --r) {
                    idx[r] = rem % dim[r];
                    rem /= dim[r];
                }
                std::array<std::size_t, Rank> idxA{}, idxB{};
                for (std::size_t r = 0; r < Rank; ++r) {
                    idxA[r] = (shape_A[r] == 1 ? 0 : idx[r]);
                    idxB[r] = (shape_B[r] == 1 ? 0 : idx[r]);
                }
                get_element_by_list(idx.data()) =
                    op(A.get_element_by_list(idxA.data()), B.get_element_by_list(idxB.data()));
            }
        }

        // Apply scalar operation
        template<typename T_scalar, class operation>
        void apply(T_scalar scalar, operation op) {
            for (auto& a : arr) a = op(a, scalar);
        }

        // Print
        friend std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& t) {
            t.print_recursive(os, 0, 0);
            return os;
        }

        // Grant free functions access to private members
        template<typename U, std::size_t R>
        friend Tensor<U, R> transpose_2d(const Tensor<U, R>& t);
        template<typename U, std::size_t R>
        friend Tensor<U, R> matrix_product(const Tensor<U, R>& A, const Tensor<U, R>& B);
    };

    // Tensor-tensor operators
    template<typename T, std::size_t Rank>
    Tensor<T, Rank> operator+(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {
        Tensor<T, Rank> result;
        result.apply(A, B, [](T a, T b) { return a + b; });
        return result;
    }
    template<typename T, std::size_t Rank>
    Tensor<T, Rank> operator-(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {
        Tensor<T, Rank> result;
        result.apply(A, B, [](T a, T b) { return a - b; });
        return result;
    }
    template<typename T, std::size_t Rank>
    Tensor<T, Rank> operator*(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {
        Tensor<T, Rank> result;
        result.apply(A, B, [](T a, T b) { return a * b; });
        return result;
    }

    // Tensor-scalar operators
    template<typename T, std::size_t Rank, typename T_scalar>
    Tensor<T, Rank> operator+(const Tensor<T, Rank>& t, T_scalar scalar) {
        Tensor<T, Rank> result = t;
        result.apply(scalar, [](T a, T_scalar e) { return a + e; });
        return result;
    }
    template<typename T, std::size_t Rank, typename T_scalar>
    Tensor<T, Rank> operator+(T_scalar scalar, const Tensor<T, Rank>& t) {
        return t + scalar;
    }
    template<typename T, std::size_t Rank, typename T_scalar>
    Tensor<T, Rank> operator*(const Tensor<T, Rank>& t, T_scalar scalar) {
        Tensor<T, Rank> result = t;
        result.apply(scalar, [](T a, T_scalar e) { return a * e; });
        return result;
    }
    template<typename T, std::size_t Rank, typename T_scalar>
    Tensor<T, Rank> operator*(T_scalar scalar, const Tensor<T, Rank>& t) {
        Tensor<T, Rank> result = t * scalar;
        return result;
    }
    template<typename T, std::size_t Rank, typename T_scalar>
    Tensor<T, Rank> operator-(const Tensor<T, Rank>& t, T_scalar scalar) {
        Tensor<T, Rank> result = t;
        result.apply(scalar, [](T a, T_scalar e) { return a - e; });
        return result;
    }
    template<typename T, std::size_t Rank, typename T_scalar>
    Tensor<T, Rank> operator-(T_scalar scalar, const Tensor<T, Rank>& t) {
        Tensor<T, Rank> result = t - scalar;
        return -1 * result;
    }
    template<typename T, std::size_t Rank, typename T_scalar>
    Tensor<T, Rank> operator/(const Tensor<T, Rank>& t, T_scalar scalar) {
        Tensor<T, Rank> result = t;
        result.apply(scalar, [](T a, T_scalar e) { return a / e; });
        return result;
    }

    // Transpose 2D: swaps the last two dimensions, keeping batch dims
    template<typename T, std::size_t Rank>
    Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& t) {
        if constexpr (Rank < 2) {
            throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        }
        Tensor<T, Rank> r = t;
        std::size_t h = t.dim[Rank - 2];
        std::size_t v = t.dim[Rank - 1];
        std::size_t special = h * v;
        std::swap(r.dim[Rank - 2], r.dim[Rank - 1]);
        for (std::size_t i = 0; i < r.arr.size(); ++i) {
            std::size_t block = (i / special) * special;
            std::size_t p = i % special;
            std::size_t col_new = p % h;
            std::size_t row_new = p / h;
            std::size_t idx_old = block + col_new * v + row_new;
            r.arr[i] = t.arr[idx_old];
        }
        return r;
    }

    // Matrix product on last two dimensions; batch dims must match
    template<typename T, std::size_t Rank>
    Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {
        if constexpr (Rank < 2) {
            throw std::invalid_argument("Need at least 2D tensors for matrix multiplication");
        }
        auto sA = A.shape(), sB = B.shape();
        const std::size_t M = sA[Rank-2], K = sA[Rank-1];
        const std::size_t K2 = sB[Rank-2], N = sB[Rank-1];
        if (K != K2) throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        for (std::size_t i = 0; i < Rank-2; ++i)
            if (sA[i] != sB[i])
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        Tensor<T, Rank> C;
        for (std::size_t i = 0; i < Rank-2; ++i) C.dim[i] = sA[i];
        C.dim[Rank-2] = M; C.dim[Rank-1] = N;
        C.arr.resize(C.get_total_dim());
        std::size_t total = C.get_total_dim();
        for (std::size_t i = 0; i < total; ++i) {
            std::array<std::size_t, Rank> idx, idxA, idxB;
            std::size_t rem = i;
            for (int d = Rank-1; d >= 0; --d) { idx[d] = rem % C.dim[d]; rem /= C.dim[d]; }
            // setup batch indices
            for (std::size_t d = 0; d < Rank-2; ++d) idxA[d] = idxB[d] = idx[d];
            T sum = T{};
            for (std::size_t k = 0; k < K; ++k) {
                idxA[Rank-2] = idx[Rank-2]; idxA[Rank-1] = k;
                idxB[Rank-2] = k;           idxB[Rank-1] = idx[Rank-1];
                sum += A.get_element_by_list(idxA.data()) * B.get_element_by_list(idxB.data());
            }
            C.get_element_by_list(idx.data()) = sum;
        }
        return C;
    }
}

// Utils.h

template<typename T, size_t DIMS, typename F>
utec::algebra::Tensor<T, DIMS> apply(const utec::algebra::Tensor<T, DIMS>& input, F func) {
    utec::algebra::Tensor<T, DIMS> output = input;
    for (size_t i = 0; i < output.size(); ++i) {
        output.begin()[i] = func(input.cbegin()[i]);
    }
    return output;
}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
