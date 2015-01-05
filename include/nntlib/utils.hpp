#pragma once

#include <iterator>
#include <tuple>


namespace nntlib {

/* Utils, mainly different funny templates
 */
namespace utils {

/* Private implementation details.
 */
namespace _ {
template <typename Tuple, typename Function, std::size_t N>
struct tuple_apply_impl {
    static inline void f(Tuple& tuple, Function func) {
        func(std::get<N>(tuple));
        tuple_apply_impl<Tuple, Function, N - 1>::f(tuple, func);
    }
};

template <typename Tuple, typename Function>
struct tuple_apply_impl<Tuple, Function, 0> {
    static inline void f(Tuple& tuple, Function func) {
        func(std::get<0>(tuple));
    }
};

template <typename Function, std::size_t N, typename... Tuples>
struct tuple_join_impl {
    static inline void f(Function func, Tuples&... tuples) {
        func(std::get<N>(tuples)...);
        tuple_join_impl<Function, N - 1, Tuples...>::f(func, tuples...);
    }
};

template <typename Function, typename... Tuples>
struct tuple_join_impl<Function, 0, Tuples...> {
    static inline void f(Function func, Tuples&... tuples) {
        func(std::get<0>(tuples)...);
    }
};

template <typename... Iters>
struct multi_foreach_impl_eq {};

template <typename HeadCurrent, typename HeadEnd, typename... Tail>
struct multi_foreach_impl_eq<HeadCurrent, HeadEnd, Tail...> {
    static inline bool f(HeadCurrent head_current, HeadEnd head_end, Tail... tail) {
        return (head_current != head_end) && multi_foreach_impl_eq<Tail...>::f(tail...);
    }
};

template <typename LastCurrent, typename LastEnd>
struct multi_foreach_impl_eq<LastCurrent, LastEnd> {
    static inline bool f(LastCurrent last_current, LastEnd last_end) {
        return last_current != last_end;
    }
};

template <std::size_t N, typename Function, typename HeadCurrent, typename HeadEnd, typename... Tail>
struct multi_foreach_impl_apply_strip;

template <std::size_t N, typename Function, typename... Iters>
struct multi_foreach_impl_apply {
    static inline void f(Function function, Iters... iters) {
        multi_foreach_impl_apply_strip<N, Function, Iters...>::f(function, iters...);
    }
};

template <typename Function, typename... Iters>
struct multi_foreach_impl_apply<0, Function, Iters...> {
    static inline void f(Function function, Iters... iters) {
        function((*iters)...);
    }
};

template <std::size_t N, typename Function, typename HeadCurrent, typename HeadEnd, typename... Tail>
struct multi_foreach_impl_apply_strip {
    static inline void f(Function function, HeadCurrent head_current, HeadEnd _head_end, Tail... tail) {
        multi_foreach_impl_apply<N - 2, Function, Tail..., HeadCurrent>::f(function, tail..., head_current);
    }
};

template <typename... Iters>
struct multi_foreach_impl_incr {};

template <typename HeadCurrent, typename HeadEnd, typename... Tail>
struct multi_foreach_impl_incr<HeadCurrent, HeadEnd, Tail...> {
    static inline void f(HeadCurrent& head_current, HeadEnd& _head_end, Tail&... tail) {
        ++head_current;
        multi_foreach_impl_incr<Tail...>::f(tail...);
    }
};

template <typename LastCurrent, typename LastEnd>
struct multi_foreach_impl_incr<LastCurrent, LastEnd> {
    static inline void f(LastCurrent& last_current, LastEnd& _last_end) {
        ++last_current;
    }
};

}

/* Placeholder for implementation defined types.
 */
struct undef final {};

/* Helper that extract head and tail types of a variadic template.
 */
template <typename Head, typename... Tail>
struct head_tail {
    /* Head = type of the first template argument.
     */
    typedef Head type_head;

    /* Tail = <head_tail> structure with remaining arguments
     */
    typedef head_tail<Tail...> tail;
};

/* Applies function to all tuple elements.
 * @Tuple Tuple type, must be of form std::tuple<...>.
 * @Function Function that is applied to the elements of a tuple, can be a template.
 * @tuple tuple reference, can be modified by the function.
 * @function Function object.
 */
template <typename Tuple, typename Function>
void tuple_apply(Tuple& tuple, Function function) {
    _::tuple_apply_impl<Tuple, Function, std::tuple_size<Tuple>::value - 1>::f(tuple, function);
}

/* Applies function on joined tuples.
 * @Function Function type that acceptes the number of arguments equal to the number of tuples.
 * @Tuples Tuples types that should be joined, must have same length.
 * @function Function that is called on each joined tuple.
 * @tuples Tuple references, can be modified by <function>.
 */
template <typename Function, typename... Tuples>
void tuple_join(Function function, Tuples&... tuples) {
    _::tuple_join_impl<Function, std::tuple_size<typename head_tail<Tuples...>::type_head>::value - 1, Tuples...>::f(function, tuples...);
}

template <typename Function, typename... Iters>
void multi_foreach(Function function, Iters... iters) {
    static_assert(sizeof...(Iters) % 2 == 0, "Provide first and last for all iterator streams!");

    while (_::multi_foreach_impl_eq<Iters...>::f(iters...)) {
        _::multi_foreach_impl_apply<sizeof...(Iters), Function, Iters...>::f(function, iters...);
        _::multi_foreach_impl_incr<Iters...>::f(iters...);
    }
}

}
}

