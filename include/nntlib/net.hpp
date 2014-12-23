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

        net(LayersLast& layers_last) : last(layers_last) {}

        template <typename InputIt>
        std::vector<T> forward(InputIt x_first, InputIt x_last) const {
            return last.forward(x_first, x_last);
        }

        template <typename InputIt1, typename InputIt2>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last) const {
            std::vector<T> x_next = last.forward(x_first, x_last);

            std::vector<T> error{};
            auto it = x_next.begin();
            auto end = x_next.end();
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

    private:
        LayersLast& last;
};

template <typename T, typename Loss, typename LayersHead, typename... LayersTail>
class net<T, Loss, LayersHead, LayersTail...> {
    public:
        typedef decltype(std::tuple_cat(std::tuple<typename LayersHead::weights_t>(), typename net<T, Loss, LayersTail...>::weights_t())) weights_t;

        net(LayersHead& layers_head, LayersTail&... layers_tail) : head(layers_head), tail(layers_tail...) {}

        template <typename InputIt>
        std::vector<T> forward(InputIt x_first, InputIt x_last) const {
            std::vector<T> x_next = head.forward(x_first, x_last);
            return tail.forward(x_next.begin(), x_next.end());
        }

        template <typename InputIt1, typename InputIt2>
        std::pair<std::vector<T>, weights_t> backward(InputIt1 x_first, InputIt1 x_last, InputIt2 t_first, InputIt2 t_last) const {
            std::vector<T> x_next = head.forward(x_first, x_last);
            auto fix_tail = tail.backward(x_next.begin(), x_next.end(), t_first, t_last);
            auto fix_head = head.backward(x_first, x_last, fix_tail.first);
            return std::make_pair(std::move(fix_head.first), std::tuple_cat(std::make_tuple(std::move(fix_head.second)), std::move(fix_tail.second)));
        }

        template <typename Tuple, int N = 0>
        void update(const Tuple& weights) {
            head.update(std::get<N>(weights));
            tail.template update<Tuple, N + 1>(weights);
        }

    private:
        LayersHead& head;
        net<T, Loss, LayersTail...> tail;
};

}

