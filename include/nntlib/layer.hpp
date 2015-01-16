#pragma once

#include "utils.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <random>


namespace nntlib {

/* Containes possible layer types.
 *
 * They are intended to be used as template parameters of <net>.
 */
namespace layer {

/* Fully connected layer.
 * @Activation Activation function.
 * @T Floating point type which is used for the entire neural network.
 * @Rng Random number generator used to initalize the weights.
 */
template <typename Activation, typename T = double, typename Rng = std::mt19937>
class fully_connected {
    public:
        /* Weight matrix.
         */
        typedef std::vector<std::vector<T>> weights_t;
        typedef std::vector<T> state_t;

        fully_connected(std::size_t n_input, std::size_t n_output, Rng& rng) : weights(n_output) {
            T width = 0.2 / static_cast<T>(n_input + 1);
            std::uniform_real_distribution<T> dist(-width, width);
            auto rfunc = std::bind(dist, std::ref(rng));

            std::generate(weights.begin(), weights.end(), [&]{
                std::vector<T> wj(n_input + 1);
                std::generate(wj.begin(), wj.end(), rfunc);

                return wj;
            });
        }

        fully_connected(const fully_connected& other) = default;
        fully_connected(fully_connected&& other) = default;

        fully_connected& operator=(const fully_connected& other) = default;
        fully_connected& operator=(fully_connected&& other) = default;

        std::size_t size_in() const {
            return weights[0].size() - 1;
        }

        std::size_t size_out() const {
            return weights.size();
        }

        state_t allocate_state() const {
            return state_t(size_out());
        }

        weights_t allocate_delta_storage() const {
            weights_t storage;
            for (const auto& w : weights) {
                storage.emplace_back(w.size());
            }
            return storage;
        }

        state_t allocate_error_storage() const {
            return state_t(size_in());
        }

        template <typename InputIt>
        void forward(InputIt x_first, InputIt x_last, state_t& state) const {
            std::transform(weights.begin(), weights.end(), state.begin(), [&](const std::vector<T> wj){
                return Activation::f(calc_netj(x_first, x_last, wj)); // = oj
            });
        }

        template <typename InputIt>
        void backward(InputIt x_first, InputIt x_last, const std::vector<T>& prev_error, state_t& error_mem, weights_t& gradient) const {
            std::fill(error_mem.begin(), error_mem.end(), 0.0);

            for (std::size_t j = 0; j < size_out(); ++j) {
                T de_doj = prev_error[j];
                T doj_dnetj = Activation::df(calc_netj(x_first, x_last, weights[j]));
                T dj = de_doj * doj_dnetj;

                std::vector<T>& gradientj = gradient[j];
                gradientj[0] = dj;
                std::transform(x_first, x_last, std::next(gradientj.begin()), [&](T xi){
                    return dj * xi;
                });

                for (std::size_t i = 0; i < error_mem.size(); ++i) {
                    error_mem[i] += dj * weights[j][i + 1];
                }
            }
        }

        /* Update layer using a delta.
         * @delta Delta matrix, should be premultiplied with learning rate.
         */
        void update(const weights_t& delta) {
            nntlib::utils::multi_foreach([](auto& wj1, const auto& wj2){
                nntlib::utils::multi_foreach([](auto& wji1, const auto& wji2){
                    wji1 += wji2;
                }, wj1.begin(), wj1.end(), wj2.begin(), wj2.end());
            }, weights.begin(), weights.end(), delta.begin(), delta.end());
        }

        const weights_t& get_weights() const {
            return weights;
        }

    private:
        weights_t weights;

        template <typename InputIt>
        static T calc_netj(InputIt x_first, InputIt x_last, const std::vector<T>& wj) {
            T netj = wj[0];
            std::size_t k = 1;
            for (; (x_first != x_last) && (k < wj.size()); ++x_first) {
                netj += (*x_first) * wj[k];
                ++k;
            }
            return netj;
        }
};

template <typename T = double, typename Rng = std::mt19937>
class dropout {
    public:
        /* Weight matrix. Will be empty.
         */
        typedef std::vector<std::vector<T>> weights_t;
        typedef std::vector<T> state_t;

        dropout(std::size_t iosize, double probability, const Rng& rng_lvalue, T dropout_value = 0.0) : size(iosize), rng(rng_lvalue), prob(probability), dist(0.0, 1.0), value(dropout_value) {}
        dropout(std::size_t iosize, double probability, Rng&& rng_rvalue, T dropout_value = 0.0) : size(iosize), rng(std::move(rng_rvalue)), prob(probability), dist(0.0, 1.0), value(dropout_value) {}

        dropout(const dropout& other) = default;
        dropout(dropout&& other) = default;

        dropout& operator=(const dropout& other) = default;
        dropout& operator=(dropout&& other) = default;

        std::size_t size_in() const {
            return size;
        }

        std::size_t size_out() const {
            return size;
        }

        state_t allocate_state() const {
            return state_t(size);
        }

        weights_t allocate_delta_storage() const {
            return weights_t{};
        }

        state_t allocate_error_storage() const {
            return state_t(size);
        }

        template <typename InputIt>
        void forward(InputIt x_first, InputIt x_last, state_t& state) const {
            std::transform(x_first, x_last, state.begin(), [&](T xi){
                return dist(rng) >= prob ? xi : value;
            });
        }

        template <typename InputIt>
        void backward(InputIt _x_first, InputIt _x_last, const std::vector<T>& prev_error, state_t& error_mem, weights_t& _gradient) const {
            std::copy(prev_error.begin(), prev_error.end(), error_mem.begin());
        }

        void update(const weights_t& _delta) {/* noop */}

        weights_t get_weights() const {
            return {};
        }

    private:
        std::size_t size;
        mutable Rng rng;
        double prob;
        mutable std::uniform_real_distribution<double> dist;
        T value;
};

}
}

