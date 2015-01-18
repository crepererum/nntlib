#pragma once

#include "utils.hpp"

#include <functional>
#include <tuple>
#include <vector>


namespace nntlib {

template <typename T, typename Loss, typename... Layers>
class net {
    public:
        typedef nntlib::utils::undef weights_t;

        net(Layers&... layers);

        template <typename InputIt>
        std::vector<T> forward(InputIt x_first, InputIt x_last) const;

        template <typename InputIt1, typename InputIt2>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last) const;

        template <typename Tuple, int N = 0>
        void update(const Tuple& weights);
};

template <typename T, typename Loss, typename LayersLast>
class net<T, Loss, LayersLast> {
    public:
        typedef std::tuple<typename LayersLast::weights_t> weights_t;
        typedef std::tuple<std::vector<T>> state_t;
        typedef std::tuple<std::vector<T>, std::vector<T>> error_mem_t;

        net(LayersLast& layers_last) : last(layers_last) {}

        state_t allocate_state() const {
            return std::make_tuple(last.allocate_state());
        }

        error_mem_t allocate_error_storage() const {
            return std::make_tuple(last.allocate_error_storage(), std::vector<T>(last.size_out()));
        }

        weights_t allocate_delta_storage() const {
            return std::make_tuple(last.allocate_delta_storage());
        }

        template <typename InputIt>
        std::vector<T> forward(InputIt x_first, InputIt x_last) const {
            state_t state = allocate_state();
            forward(x_first, x_last, state);
            return std::get<0>(state);
        }

        template <typename InputIt, typename State, int N = 0>
        std::vector<T>& forward(InputIt x_first, InputIt x_last, State& state) const {
            std::vector<T>& y = std::get<N>(state);
            last.forward(x_first, x_last, y, false);
            return y;
        }

        template <typename InputIt1, typename InputIt2>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last) const {
            state_t state = allocate_state();
            error_mem_t error_mem = allocate_error_storage();
            weights_t gradient = allocate_delta_storage();

            backward(x_first, x_last, t_first, t_last, state, error_mem, gradient);

            return std::make_pair(std::move(error_mem), std::move(gradient));
        }

        template <typename InputIt1, typename InputIt2, typename State, typename Error, typename Weights, int N = 0>
        std::pair<std::vector<T>&, Weights&> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last, State& state, Error& error_mem, Weights& gradient) const {
            std::vector<T>& y = std::get<N>(state);
            auto cache = last.forward(x_first, x_last, y, true);

            std::vector<T>& error = std::get<N + 1>(error_mem);
            auto it = y.begin();
            auto end = y.end();
            std::size_t pos(0);
            while ((it != end) && (t_first != t_last)) {
                error[pos++] = Loss::df(*it, *t_first);

                ++it;
                ++t_first;
            }

            last.backward(x_first, x_last, error, std::get<N>(error_mem), std::get<N>(gradient), cache);
            return std::make_pair(std::ref(std::get<N>(error_mem)), std::ref(gradient));
        }

        template <typename Tuple, int N = 0>
        void update(const Tuple& weights) {
            last.update(std::get<N>(weights));
        }

        auto get_weights() const {
            return std::make_tuple(last.get_weights());
        }

    private:
        LayersLast& last;
};

template <typename T, typename Loss, typename LayersHead, typename... LayersTail>
class net<T, Loss, LayersHead, LayersTail...> {
    public:
        typedef decltype(std::tuple_cat(std::tuple<typename LayersHead::weights_t>(), typename net<T, Loss, LayersTail...>::weights_t())) weights_t;
        typedef decltype(std::tuple_cat(std::tuple<std::vector<T>>(), typename net<T, Loss, LayersTail...>::state_t())) state_t;
        typedef decltype(std::tuple_cat(std::tuple<std::vector<T>>(), typename net<T, Loss, LayersTail...>::error_mem_t())) error_mem_t;

        net(LayersHead& layers_head, LayersTail&... layers_tail) : head(layers_head), tail(layers_tail...) {}

        state_t allocate_state() const {
            return std::tuple_cat(std::make_tuple(head.allocate_state()), tail.allocate_state());
        }

        error_mem_t allocate_error_storage() const {
            return std::tuple_cat(std::make_tuple(head.allocate_error_storage()), tail.allocate_error_storage());
        }

        weights_t allocate_delta_storage() const {
            return std::tuple_cat(std::make_tuple(head.allocate_delta_storage()), tail.allocate_delta_storage());
        }

        template <typename InputIt>
        std::vector<T> forward(InputIt x_first, InputIt x_last) const {
            state_t state = allocate_state();
            forward(x_first, x_last, state);
            return std::get<std::tuple_size<state_t>::value - 1>(state);
        }

        template <typename InputIt, typename State, int N = 0>
        std::vector<T>& forward(InputIt x_first, InputIt x_last, State& state) const {
            std::vector<T>& x_next = std::get<N>(state);
            head.forward(x_first, x_last, x_next, false);
            return tail.template forward<decltype(x_next.begin()), State, N + 1>(x_next.begin(), x_next.end(), state);
        }

        template <typename InputIt1, typename InputIt2>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last) const {
            state_t state = allocate_state();
            error_mem_t error_mem = allocate_error_storage();
            weights_t gradient = allocate_delta_storage();

            backward(x_first, x_last, t_first, t_last, state, error_mem, gradient);

            return std::make_pair(std::move(error_mem), std::move(gradient));
        }

        template <typename InputIt1, typename InputIt2, typename State, typename Error, typename Weights, int N = 0>
        std::pair<std::vector<T>&, Weights&> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last, State& state, Error& error_mem, Weights& gradient) const {
            std::vector<T>& x_next = std::get<N>(state);
            auto cache = head.forward(x_first, x_last, x_next, true);

            auto fix_tail = tail.template backward<decltype(x_next.begin()), InputIt2, State, Error, Weights, N + 1>(x_next.begin(), x_next.end(), t_first, t_last, state, error_mem, gradient);
            head.backward(x_first, x_last, fix_tail.first, std::get<N>(error_mem), std::get<N>(gradient), cache);
            return std::make_pair(std::ref(std::get<N>(error_mem)), std::ref(gradient));
        }

        template <typename Tuple, int N = 0>
        void update(const Tuple& weights) {
            head.update(std::get<N>(weights));
            tail.template update<Tuple, N + 1>(weights);
        }

        auto get_weights() const {
            return std::tuple_cat(std::make_tuple(head.get_weights()), tail.get_weights());
        }

    private:
        LayersHead& head;
        net<T, Loss, LayersTail...> tail;
};

template <typename T, typename Loss, typename... Layers>
net<T, Loss, Layers...> make_net(Layers&... layers) {
    return net<T, Loss, Layers...>(layers...);
}

}

