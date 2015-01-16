#pragma once

#include <cmath>


namespace nntlib {

/* Contains possible activation functions for layers.
 *
 * They are intended to be used as template arguments of classes found in <layer>.
 *
 * During the forward phase the function gets executed twice. First f1 is called
 * using the netto input. After all f1 values are calculated, f2 is called using
 * the result of f1. This behaviour might be used to implement normalization.
 *
 * During the backward phase, the same object is used to call df on.
 */
namespace activation {

/* Identity function.
 * @T Floating point type which is used for the entire neural network.
 *
 * f(x) = x
 */
template <typename T = double>
struct identity {
    /* f1(x) = x
     */
    static constexpr T f1(T x) {
        return x;
    }

    /* f2(x) = x
     */
    static constexpr T f2(T x) {
        return x;
    }

    /* df(x) = 1
     */
    static constexpr T df(T _x) {
        return 1;
    }
};

/* Sigmoid function.
 * @T Floating point type which is used for the entire neural network.
 *
 * f(x) = 1 / (1 + exp(-x))
 */
template <typename T = double>
struct sigmoid {
    /* f1(x) = 1 / (1 + exp(-x))
     */
    static constexpr T f1(T x) {
        return 1 / (1 + std::exp(-x));
    }

    /* f2(x) = x
     */
    static constexpr T f2(T x) {
        return x;
    }

    /* df(x) = f(x) * (1 - f(x))
     */
    static constexpr T df(T x) {
        return f1(x) * (1 - f1(x));
    }
};

/* Softmax function.
 * @T Floating point type which is used for the entire neural network.
 *
 * f(x) = exp(x) / (sum_{i=1}^k exp(x_i))
 */
template <typename T = double>
struct softmax {
    /* f1(x) = exp(x)
     *
     * Add value to internal sum,
     */
    constexpr T f1(T x) {
        T y = std::exp(x);
        sum += y;
        return y;
    }

    /* f2(x) = x / sum
     */
    constexpr T f2(T x) {
        return x / sum;
    }

    /* df(x) = f(x) * (1 - f(x))
     */
    constexpr T df(T x) {
        T y = std::exp(x) / sum;
        return y * (1.0 - y);
    }

    T sum = 0.0;
};

/* Softplus function.
 * @T Floating point type which is used for the entire neural network.
 *
 * f(x) = log(1 + exp(x))
 */
template <typename T = double>
struct softplus {
    /* f1(x) = log(1 + exp(x))
     */
    static constexpr T f1(T x) {
        return std::log(1.0 + std::exp(x));
    }

    /* f2(x) = x
     */
    static constexpr T f2(T x) {
        return x;
    }

    /* df(x) = 1 / (1 + exp(-x))
     */
    static constexpr T df(T x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

/* tanh function.
 * @T Floating point type which is used for the entire neural network.
 *
 * f(x) = tanh(x)
 */
template <typename T = double>
struct tanh {
    /* f1(x) = tanh(x)
     */
    static constexpr T f1(T x) {
        return std::tanh(x);
    }

    /* f2(x) = x
     */
    static constexpr T f2(T x) {
        return x;
    }

    /* df(x) = 1 - f(x)^2
     */
    static constexpr T df(T x) {
        return 1.0 - f1(x) * f1(x);
    }
};

}
}

