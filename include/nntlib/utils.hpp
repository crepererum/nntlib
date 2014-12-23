#pragma once

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
    static void f(Tuple& tuple, Function func) {
        func(std::get<N>(tuple));
        tuple_apply_impl<Tuple, Function, N - 1>::f(tuple, func);
    }
};

template <typename Tuple, typename Function>
struct tuple_apply_impl<Tuple, Function, 0> {
    static void f(Tuple& tuple, Function func) {
        func(std::get<0>(tuple));
    }
};

template <typename Function, std::size_t N, typename... Tuples>
struct tuple_join_impl {
    static void f(Function func, Tuples&... tuples) {
        func(std::get<N>(tuples)...);
        tuple_join_impl<Function, N - 1, Tuples...>::f(func, tuples...);
    }
};

template <typename Function, typename... Tuples>
struct tuple_join_impl<Function, 0, Tuples...> {
    static void f(Function func, Tuples&... tuples) {
        func(std::get<0>(tuples)...);
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

}
}

