#pragma once

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

        template <typename InputIt>
        std::vector<T> forward(InputIt x_first, InputIt x_last) const {
            std::vector<T> output(weights.size());
            std::transform(weights.begin(), weights.end(), output.begin(), [&](const std::vector<T> wj){
                return Activation::f(calc_netj(x_first, x_last, wj)); // = oj
            });

            return output;
        }

        template <typename InputIt>
        std::pair<std::vector<T>, weights_t> backward(InputIt x_first, InputIt x_last, const std::vector<T>& prev_error) const {
            weights_t gradient;
            std::vector<T> error(weights[0].size() - 1);

            for (std::size_t j = 0; j < weights.size(); ++j) {
                T de_doj = prev_error[j];
                T doj_dnetj = Activation::df(calc_netj(x_first, x_last, weights[j]));
                T dj = de_doj * doj_dnetj;

                std::vector<T> gradientj(weights[j].size());
                gradientj[0] = dj;
                std::transform(x_first, x_last, std::next(gradientj.begin()), [&](T xi){
                    return dj * xi;
                });

                gradient.emplace_back(std::move(gradientj));

                for (std::size_t i = 0; i < error.size(); ++i) {
                    error[i] += dj * weights[j][i + 1];
                }
            }

            return std::make_pair(std::move(error), std::move(gradient));
        }

        /* Update layer using a delta.
         * @delta Delta matrix, should be premultiplied with learning rate.
         */
        void update(const weights_t& delta) {
            auto it1 = weights.begin();
            auto it2 = delta.begin();
            auto end1 = weights.end();
            auto end2 = delta.end();

            while ((it1 != end1) && (it2 != end2)) {
                auto& wj1 = *it1;
                const auto& wj2 = *it2;
                auto inner_it1 = wj1.begin();
                auto inner_it2 = wj2.begin();
                auto inner_end1 = wj1.end();
                auto inner_end2 = wj2.end();

                while((inner_it1 != inner_end1) && (inner_it2 != inner_end2)) {
                    (*inner_it1) += (*inner_it2);

                    ++inner_it1;
                    ++inner_it2;
                }

                ++it1;
                ++it2;
            }
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

        dropout(double probability, const Rng& rng_lvalue, T dropout_value = 0.0) : prob(probability), rng(rng_lvalue), dist(0.0, 1.0), value(dropout_value) {}
        dropout(double probability, Rng&& rng_rvalue, T dropout_value = 0.0) : prob(probability), rng(std::move(rng_rvalue)), dist(0.0, 1.0), value(dropout_value) {}

        dropout(const dropout& other) = default;
        dropout(dropout&& other) = default;

        dropout& operator=(const dropout& other) = default;
        dropout& operator=(dropout&& other) = default;

        template <typename InputIt>
        std::vector<T> forward(InputIt x_first, InputIt x_last) const {
            std::vector<T> output{};
            std::transform(x_first, x_last, std::back_inserter(output), [&](T xi){
                return dist(rng) >= prob ? xi : value;
            });

            return output;
        }

        template <typename InputIt>
        std::pair<std::vector<T>, weights_t> backward(InputIt _x_first, InputIt _x_last, const std::vector<T>& prev_error) const {
            return std::make_pair(prev_error, weights_t{});
        }

        void update(const weights_t& _delta) {/* noop */}

        weights_t get_weights() const {
            return {};
        }

    private:
        mutable Rng rng;
        double prob;
        mutable std::uniform_real_distribution<double> dist;
        T value;
};

}
}

