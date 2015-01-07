#pragma once

#include "utils.hpp"

#include <etl/etl.hpp>
#include <etl/multiplication.hpp>

#include <cmath>

#include <functional>
#include <list>


namespace nntlib {

namespace training {

namespace _ {
template <typename T>
class batch_template {
    public:
        typedef std::function<T(std::size_t)> func_factor_t;
        typedef std::function<void(std::size_t)> func_callback_round_t;
        typedef std::function<void()> func_callback_batch_t;

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

        batch_template(func_factor_t func_factor, std::size_t batch_size, std::size_t n_rounds, T l2 = 0.0) : ffactor(func_factor), fround([](std::size_t _r){}), fbatch([](){}), bsize(batch_size), rounds(n_rounds), l2_factor(l2) {}
        virtual ~batch_template() = default;

        virtual void callback_round(func_callback_round_t callback) {
            fround = callback;
        }

    protected:
        template <typename Net, typename InputIt1, typename InputIt2, typename UpdateHook>
        void train_impl(Net& net, InputIt1 x_first, InputIt1 x_last, InputIt2 y_first, InputIt2 y_last, UpdateHook& update_hook) {
            std::size_t n = static_cast<std::size_t>(std::distance(x_first, x_last));

            for (std::size_t round = 0; round < rounds; ++round) {
                T round_factor = ffactor(round);
                typename Net::weights_t gradients_sum;
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

                    // should we update? (= end of batch)
                    if (batchcounter == 0) {
                        // yes => reinit update vector
                        if (!first_batch) {
                            prepare_and_commit_update(net, gradients_sum, n, round_factor, bsize, update_hook);

                            // call batch callback
                            fbatch();
                        } else {
                            first_batch = false;
                        }

                        gradients_sum = gradients;
                    } else {
                        // no => add gradient to update
                        nntlib::utils::tuple_join([](auto& lhs, auto& rhs){
                            nntlib::utils::multi_foreach([](auto& lhs2, auto& rhs2){
                                nntlib::utils::multi_foreach([](auto& lhs3, auto& rhs3){
                                    lhs3 += rhs3;
                                }, lhs2.begin(), lhs2.end(), rhs2.begin(), rhs2.end());
                            }, lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
                        }, gradients_sum, gradients);
                    }
                    batchcounter = (batchcounter + 1) % bsize;

                    ++x_iter;
                    ++y_iter;
                }

                // also commit last partial batch
                if (!first_batch) {
                    // also use bsize here to avoid over-rating of the remaining samples
                    prepare_and_commit_update(net, gradients_sum, n, round_factor, bsize, update_hook);
                }

                // call round callback
                fround(round);
            }
        }

    private:
        func_factor_t ffactor;
        func_callback_round_t fround;
        func_callback_batch_t fbatch;
        std::size_t bsize;
        std::size_t rounds;
        T l2_factor;

        template <typename Net, typename UpdateHook>
        void prepare_and_commit_update(Net& net, typename Net::weights_t& gradients_sum, std::size_t n, T round_factor, std::size_t batch_size, UpdateHook& update_hook) {
            // multiple gradients with learning rate AND mutliply by -1 (opposite direction)
            nntlib::utils::tuple_apply(gradients_sum, [round_factor, batch_size](auto& w){
                for (auto& wj : w) {
                    for (auto& wji : wj) {
                        wji *= -round_factor / batch_size;
                    }
                }
            });

            // optional l2 regularization
            if (l2_factor > 0.0) {
                auto weights = net.get_weights();
                nntlib::utils::tuple_join([&](auto& lhs, auto& rhs){
                    nntlib::utils::multi_foreach([&](auto& lhs2, auto& rhs2){
                        bool first = true;
                        nntlib::utils::multi_foreach([&](auto& lhs3, auto& rhs3){
                            // skip first weight (= bias value)
                            if (first) {
                                first = false;
                            } else {
                                lhs3 -= rhs3 * l2_factor / n;
                            }
                        }, lhs2.begin(), lhs2.end(), rhs2.begin(), rhs2.end());
                    }, lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
                }, gradients_sum, weights);
            }

            // call the update hook
            update_hook(gradients_sum);

            // finally update the net
            net.update(gradients_sum);
        }
};
}

template <typename T = double>
class batch : public _::batch_template<T> {
    public:
        typedef typename _::batch_template<T>::func_factor_t func_factor_t;
        typedef typename _::batch_template<T>::func_callback_round_t func_callback_round_t;
        typedef typename _::batch_template<T>::func_callback_batch_t func_callback_batch_t;

        batch(func_factor_t func_factor, std::size_t batch_size, std::size_t n_rounds, T l2 = 0.0) : _::batch_template<T>(func_factor, batch_size, n_rounds, l2) {}

        template <typename Net, typename InputIt1, typename InputIt2>
        void train(Net& net, InputIt1 x_first, InputIt1 x_last, InputIt2 y_first, InputIt2 y_last) {
            auto hook = [](typename Net::weights_t& _update){};
            _::batch_template<T>::train_impl(net, x_first, x_last, y_first, y_last, hook);
        }
};

template <typename T = double>
class lbfgs : public _::batch_template<T> {
    public:
        typedef typename _::batch_template<T>::func_factor_t func_factor_t;
        typedef typename _::batch_template<T>::func_callback_round_t func_callback_round_t;
        typedef typename _::batch_template<T>::func_callback_batch_t func_callback_batch_t;

        lbfgs(std::size_t history_size, func_factor_t func_factor, std::size_t batch_size, std::size_t n_rounds, T l2 = 0.0) :
                _::batch_template<T>([](std::size_t _i){return 1.0;}, batch_size, n_rounds, l2),
                ffactor(func_factor), fround([](std::size_t _i){}),
                histsize(history_size) {
            _::batch_template<T>::callback_round([&](std::size_t round){
                nround = round + 1;
                fround(round);
            });
        }

        virtual void callback_round(func_callback_round_t callback) override {
            fround = callback;
        }

        template <typename Net, typename InputIt1, typename InputIt2>
        void train(Net& net, InputIt1 x_first, InputIt1 x_last, InputIt2 y_first, InputIt2 y_last) {
            nround = 0;
            std::unique_ptr<etl::dyn_matrix<T, 2>> update_last;
            std::unique_ptr<etl::dyn_matrix<T, 2>> weights_last;
            bool first = true;
            std::list<history_entry> history;

            auto hook = [&](typename Net::weights_t& update){
                auto update_current = update2vector<typename Net::weights_t>(update, -1.0);
                auto weights_current = update2vector<typename Net::weights_t>(net.get_weights(), 1.0);

                if (first) {
                    first = false;
                } else {
                    history.emplace_back(*weights_current - *weights_last, *update_current - *update_last);
                }

                etl::dyn_matrix<T> id = gen_id(update_current->size());
                etl::dyn_matrix<T> bk = id;
                for (const auto& entry : history) {
                    T norm = etl::mmul(transpose(entry.yk), entry.sk)[0];
                    bk = etl::mmul(
                            etl::mmul(
                                (id - etl::mmul(entry.sk, transpose(entry.yk)) / norm),
                                bk
                            ),
                            (id - etl::mmul(entry.yk, transpose(entry.sk)) / norm)
                        ) + etl::mmul(entry.sk, transpose(entry.sk)) / norm;
                }

                vector2update<typename Net::weights_t>(update, etl::mmul(bk, *update_current));

                while (history.size() > histsize) {
                    history.pop_front();
                }

                update_last = std::move(update_current);
                weights_last = std::move(weights_current);
            };
            _::batch_template<T>::train_impl(net, x_first, x_last, y_first, y_last, hook);
        }

    private:
        func_factor_t ffactor;
        func_callback_round_t fround;
        std::size_t histsize;
        std::size_t nround;

        template <typename Weights>
        std::unique_ptr<etl::dyn_matrix<T, 2>> update2vector(const Weights& weights, T factor) {
            std::vector<T> vector;
            nntlib::utils::tuple_apply(weights, [&](const auto& part){
                for (const auto& x : part) {
                    for (T y : x) {
                        vector.push_back(y * factor);
                    }
                }
            });
            std::unique_ptr<etl::dyn_matrix<T, 2>> result(new etl::dyn_matrix<T, 2>(vector.size(), static_cast<std::size_t>(1)));
            for (std::size_t i = 0; i < vector.size(); ++i) {
                (*result)[i] = vector[i];
            }
            return result;
        }

        template <typename Weights>
        Weights vector2update(Weights& update, const etl::dyn_matrix<T, 2>& vector) {
            std::size_t pos = 0;
            T factor = -ffactor(nround);
            nntlib::utils::tuple_apply(update, [&](auto& part){
                for (auto& x : part) {
                    for (T& y : x) {
                        y = vector[pos++] * factor;
                    }
                }
            });
            return update;
        }

        etl::dyn_matrix<T, 2> gen_id(std::size_t size) {
            etl::dyn_matrix<T, 2> id(size, size);
            for (std::size_t i = 0; i < size; ++i) {
                for (std::size_t j = 0; j < size; ++j) {
                    id(i, j) = (i == j) ? 1.0 : 0.0;
                }
            }
            return id;
        }

        struct history_entry {
            etl::dyn_matrix<T, 2> sk;
            etl::dyn_matrix<T, 2> yk;

            history_entry(etl::dyn_matrix<T, 2>&& sk_move, etl::dyn_matrix<T, 2>&& yk_move) : sk(std::move(sk_move)), yk(std::move(yk_move)) {}
        };
};

}
}

