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

        batch(func_factor_t func_factor, std::size_t batch_size, std::size_t n_rounds, T l2 = 0.0) : ffactor(func_factor), fround([](std::size_t _r){}), bsize(batch_size), rounds(n_rounds), l2_factor(l2) {}

        void callback_round(func_callback_round_t callback) {
            fround = callback;
        }

        template <typename Net, typename InputIt1, typename InputIt2>
        void train(Net& net, InputIt1 x_first, InputIt1 x_last, InputIt2 y_first, InputIt2 y_last) {
            std::size_t n = static_cast<std::size_t>(std::distance(x_first, x_last));

            for (std::size_t round = 0; round < rounds; ++round) {
                T factor = ffactor(round);
                typename Net::weights_t update;
                std::size_t batchcounter = 0;
                bool first_batch = true;
                InputIt1 x_iter = x_first;
                InputIt2 y_iter = y_first;

                // iterate over entire training set
                while ((x_iter != x_last) && (y_iter != y_last)) {
                    // calc gradients
                    auto error_and_gradients = net.backward(
                        x_iter->begin(), x_iter->end(),
                        y_iter->begin(), y_iter->end()
                    );
                    auto& gradients = error_and_gradients.second;

                    // multiple gradients with learning rate AND mutliply by -1 (opposite direction)
                    nntlib::utils::tuple_apply(gradients, [factor](auto& w){
                        for (auto& wj : w) {
                            for (auto& wji : wj) {
                                wji *= -factor;
                            }
                        }
                    });

                    // should we update? (= end of batch)
                    if (batchcounter == 0) {
                        // yes => reinit update vector
                        if (!first_batch) {
                            update_with_l2(net, update, n);
                        } else {
                            first_batch = false;
                        }
                        update = gradients;
                    } else {
                        // no => add gradient to update
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
                        }, update, gradients);
                    }
                    batchcounter = (batchcounter + 1) % bsize;

                    ++x_iter;
                    ++y_iter;
                }

                // also commit last partial batch
                if (batchcounter != 0) {
                    update_with_l2(net, update, n);
                }

                // call round callback
                fround(round);
            }
        }

    private:
        func_factor_t ffactor;
        func_callback_round_t fround;
        std::size_t bsize;
        std::size_t rounds;
        T l2_factor;

        template <typename Net>
        void update_with_l2(Net& net, typename Net::weights_t& update, std::size_t n) {
            if (l2_factor > 0.0) {
                auto weights = net.get_weights();
                nntlib::utils::tuple_join([&](auto& lhs, auto& rhs){
                    auto first1 = lhs.begin();
                    auto last1 = lhs.end();
                    auto first2 = rhs.begin();
                    auto last2 = rhs.end();

                    while ((first1 != last1) && (first2 != last2)) {
                        auto inner_first1 = first1->begin();
                        auto inner_last1 = first1->end();
                        auto inner_first2 = first2->begin();
                        auto inner_last2 = first2->end();
                        bool first = true;

                        while ((inner_first1 != inner_last1) && (inner_first2 != inner_last2)) {
                            // skip first weight (= bias value)
                            if (first) {
                                first = false;
                            } else {
                                (*inner_first1) -= *inner_first2 * l2_factor / n;
                            }

                            ++inner_first1;
                            ++inner_first2;
                        }

                        ++first1;
                        ++first2;
                    }
                }, update, weights);
            }

            net.update(update);
        }
};

}
}

