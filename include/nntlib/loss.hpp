#pragma once

#include <cmath>


namespace nntlib {
namespace loss {

template <typename T = double>
struct mse {
    static constexpr T f(T y, T t) {
        constexpr T d = y - t;
        return d * d / 2.0;
    }

    static constexpr T df(T y, T t) {
        return y - t;
    }
};

template <typename T = double>
struct cross_entropy {
    static constexpr T f(T y, T t) {
        return -t * std::log(y) - (1.0 - t) * std::log(1.0 - y);
    }

    static constexpr T df(T y, T t) {
        return (y - t) / (y * (1.0 - y));
    }
};

}
}

