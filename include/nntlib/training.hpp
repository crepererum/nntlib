#pragma once

#include <cmath>

#include <functional>

#include "utils.hpp"


namespace nntlib {

namespace training {

template <typename T = double>
class batch {
    public:
        typedef std::function<T(std::size_t)> func_factor_t;
        typedef std::function<void(std::size_t)> func_callback_round_t;

        static func_factor_t func_factor_const(T factor) {
            return [factor](std::size_t _i) -> T {
                return factor;
            };
        }

        static func_factor_t func_factor_exp(T factor, T base) {
            return [factor, base](std::size_t i) -> T {
                return factor * std::pow(base, static_cast<T>(i));
            };
        }

        batch(func_factor_t func_factor, std::size_t batch_size, std::size_t n_rounds) : ffactor(func_factor), fround([](std::size_t _r){}), bsize(batch_size), rounds(n_rounds) {}

        void callback_round(func_callback_round_t callback) {
            fround = callback;
        }

        template <typename Net, typename InputIt1, typename InputIt2>
        void train(Net& net, InputIt1 x_first, InputIt1 x_last, InputIt2 y_first, InputIt2 y_last) {
            for (std::size_t round = 0; round < rounds; ++round) {
                T factor = ffactor(round);
                typename Net::weights_t update;
                std::size_t batchcounter = 0;
                bool first_batch = true;
                auto x_iter = x_first;
                auto y_iter = y_first;

                while ((x_iter != x_last) && (y_iter != y_last)) {
                    auto error_and_weights = net.backward(
                        x_iter->begin(), x_iter->end(),
                        y_iter->begin(), y_iter->end()
                    );
                    auto& weights = error_and_weights.second;
                    nntlib::utils::tuple_apply(weights, [factor](auto& w){
                        for (auto& wj : w) {
                            for (auto& wji : wj) {
                                wji *= factor;
                            }
                        }
                    });
                    if (batchcounter == 0) {
                        if (!first_batch) {
                            net.update(update);
                        } else {
                            first_batch = false;
                        }
                        update = weights;
                    } else {
                        nntlib::utils::tuple_join([](auto& lhs, auto& rhs){
                            auto first1 = lhs.begin();
                            auto last1 = lhs.end();
                            auto first2 = rhs.begin();
                            auto last2 = rhs.end();

                            while ((first1 != last1) && (first2 != last2)) {
                                auto inner_first1 = first1->begin();
                                auto inner_last1 = first1->end();
                                auto inner_first2 = first2->begin();
                                auto inner_last2 = first2->end();

                                while ((inner_first1 != inner_last1) && (inner_first2 != inner_last2)) {
                                    (*inner_first1) += *inner_first2;

                                    ++inner_first1;
                                    ++inner_first2;
                                }

                                ++first1;
                                ++first2;
                            }
                        }, update, weights);
                    }
                    batchcounter = (batchcounter + 1) % bsize;

                    ++x_iter;
                    ++y_iter;
                }

                if (batchcounter != 0) {
                    net.update(update);
                }

                fround(round);
            }
        }

    private:
        func_factor_t ffactor;
        func_callback_round_t fround;
        std::size_t bsize;
        std::size_t rounds;
};

}
}

