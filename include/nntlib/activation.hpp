#pragma once

#include <cmath>


namespace nntlib {

/* Containes possible activation functions for layers.
 *
 * They are inteted to be used as template arguments of classes found in <layer>.
 */
namespace activation {

/* Identity function.
 * @T Floating point type which is used for the entire neural network.
 */
template <typename T = double>
struct identity {
    /* f(x) = x
     */
    static constexpr T f(T x) {
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
 */
template <typename T = double>
struct sigmoid {
    /* f(x) = 1 / (1 + exp(-x))
     */
    static constexpr T f(T x) {
        return 1 / (1 + std::exp(-x));
    }

    /* df(x) = f(x) * (1 - f(x))
     */
    static constexpr T df(T x) {
        return f(x) * (1 - f(x));
    }
};

/* tanh function.
 * @T Floating point type which is used for the entire neural network.
 */
template <typename T = double>
struct tanh {
    /* f(x) = tanh(x)
     */
    static constexpr T f(T x) {
        return std::tanh(x);
    }

    /* df(x) = 1 - f(x)^2
     */
    static constexpr T df(T x) {
        return 1.0 - f(x) * f(x);
    }
};

}
}

