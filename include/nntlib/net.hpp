#pragma once

#include "utils.hpp"

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

        net(LayersLast& layers_last) : last(layers_last) {}

        state_t allocate_state() const {
            return std::make_tuple(last.allocate_state());
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
            last.forward(x_first, x_last, y);
            return y;
        }

        template <typename InputIt1, typename InputIt2>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last) const {
            state_t state = allocate_state();
            return backward(x_first, x_last, t_first, t_last, state);
        }

        template <typename InputIt1, typename InputIt2, typename State, int N = 0>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last, State& state) const {
            std::vector<T>& y = std::get<N>(state);
            last.forward(x_first, x_last, y);

            std::vector<T> error{};
            auto it = y.begin();
            auto end = y.end();
            while ((it != end) && (t_first != t_last)) {
                error.push_back(Loss::df(*it, *t_first));

                ++it;
                ++t_first;
            }

            auto fix_last = last.backward(x_first, x_last, error);
            return std::make_pair(std::move(fix_last.first), std::make_tuple(std::move(fix_last.second)));
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

        net(LayersHead& layers_head, LayersTail&... layers_tail) : head(layers_head), tail(layers_tail...) {}

        state_t allocate_state() const {
            return std::tuple_cat(std::make_tuple(head.allocate_state()), tail.allocate_state());
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
            head.forward(x_first, x_last, x_next);
            return tail.template forward<decltype(x_next.begin()), State, N + 1>(x_next.begin(), x_next.end(), state);
        }

        template <typename InputIt1, typename InputIt2>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last) const {
            state_t state = allocate_state();
            return backward(x_first, x_last, t_first, t_last, state);
        }

        template <typename InputIt1, typename InputIt2, typename State, int N = 0>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last, State& state) const {
            std::vector<T>& x_next = std::get<N>(state);
            head.forward(x_first, x_last, x_next);
            auto fix_tail = tail.backward(x_next.begin(), x_next.end(), t_first, t_last, state);
            auto fix_head = head.backward(x_first, x_last, fix_tail.first);
            return std::make_pair(std::move(fix_head.first), std::tuple_cat(std::make_tuple(std::move(fix_head.second)), std::move(fix_tail.second)));
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

