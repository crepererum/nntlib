#pragma once

#include <memory>
#include <type_traits>
#include <vector>

namespace nntlib {

/* Containes different iterator helpers to wire exsisting data structures and storage backends into the library.
 */
namespace iterator {

/* Private implementation details.
 */
namespace _ {
template <typename T>
class combine_helper_base {
    public:
        virtual ~combine_helper_base() = default;
        virtual void incr() = 0;
        virtual T& deref() = 0;
        virtual bool eq(const std::unique_ptr<combine_helper_base<T>>& other) const = 0;
        virtual std::unique_ptr<combine_helper_base<T>> copy() const = 0;
};

template <typename T, typename Iter>
class combine_helper : public combine_helper_base<T> {
    public:
        combine_helper(Iter iter) : it(iter) {}

        virtual void incr() override {
            ++it;
        }

        virtual T& deref() override {
            return *it;
        }

        virtual bool eq(const std::unique_ptr<combine_helper_base<T>>& other) const override {
            auto test = dynamic_cast<const combine_helper<T, Iter>*>(other.get());
            if (test != nullptr) {
                return this->it == test->it;
            } else {
                return false;
            }
        }

        virtual std::unique_ptr<combine_helper_base<T>> copy() const override {
            return std::unique_ptr<combine_helper_base<T>>{dynamic_cast<combine_helper_base<T>*>(new combine_helper<T, Iter>(it))};
        }

    private:
        Iter it;
};

template <typename T>
using combine_helper_vec = std::shared_ptr<std::vector<std::unique_ptr<combine_helper_base<T>>>>;

template <typename T>
combine_helper_vec<T> combine_helper_vec_cpy(const combine_helper_vec<T>& orig) {
    auto cpy = std::make_shared<std::vector<std::unique_ptr<combine_helper_base<T>>>>(orig->size());
    for (std::size_t i = 0; i < orig->size(); ++i) {
        (*cpy)[i] = (*orig)[i]->copy();
    }
    return std::move(cpy);
}

template <typename T>
class combine_mapper {
    public:
        combine_mapper(const combine_helper_vec<T>& hidden_helpers, std::size_t internal_pos) : helpers(hidden_helpers), pos(internal_pos) {}

        combine_mapper(const combine_mapper& other) : helpers(other.helpers), pos(other.pos) {}
        combine_mapper(combine_mapper&& other) = default;

        combine_mapper& operator=(const combine_mapper<T>& other) {
            this->helper = other.helpers;
            this->pos = other.pos;
        }
        combine_mapper& operator=(combine_mapper<T>&& other) = default;

        combine_mapper& operator++() {
            ++pos;
            return *this;
        }

        T& operator*() {
            return (*helpers)[pos]->deref();
        }

        T* operator->() {
            return &((*helpers)[pos]->deref());
        }

        bool operator==(const combine_mapper& other) const {
            if (this->is_end()) {
                return other.is_end();
            } else {
                return !other.is_end() && (*this->helpers)[this->pos]->eq((*other.helpers)[other.pos]);
            }
        }

        bool operator!=(const combine_mapper& other) const {
            return !(*this == other);
        }

    private:
        combine_helper_vec<T> helpers;
        std::size_t pos;

        bool is_end() const {
            return !helpers || (pos >= helpers->size());
        }
};

template <typename T>
class combine_container {
    public:
        combine_container() : helpers(new std::vector<std::unique_ptr<combine_helper_base<T>>>{}) {}

        template <typename... Iters>
        combine_container(Iters... iters) : helpers(new std::vector<std::unique_ptr<combine_helper_base<T>>>{}) {
            add_all<Iters...>::f(*this, iters...);
        }

        combine_container(const combine_container& other) : helpers(other.helpers) {}
        combine_container(combine_container&& other) = default;

        combine_container& operator=(const combine_container<T>& other) {
            this->helpers = other.helpers;
        }
        combine_container& operator=(combine_container<T>&& other) = default;

        combine_mapper<T> begin() const {
            return combine_mapper<T>(helpers, 0);
        }

        combine_mapper<T> end() const {
            return combine_mapper<T>({}, 0);
        }

        T& operator[](std::size_t i) const {
            return (*helpers)[i]->deref();
        }

        template <typename Iter>
        void push_back(Iter it) {
            // avoid effect on other iterators
            if (!helpers.unique()) {
                helpers = combine_helper_vec_cpy(helpers);
            }

            helpers->emplace_back(new combine_helper<T, Iter>(it));
        }

        void incr_all() {
            // avoid effect on other iterators
            if (!helpers.unique()) {
                helpers = combine_helper_vec_cpy(helpers);
            }

            for (auto& h : (*helpers)) {
                h->incr();
            }
        }

        bool eq(const combine_container& other) const {
            if (this->helpers->size() == other.helpers->size()) {
                for (std::size_t i = 0; i < this->helpers->size(); ++i) {
                    if (!(*this->helpers)[i]->eq((*other.helpers)[i])) {
                        return false;
                    }
                }
                return true;
            } else {
                return false;
            }
        }

    private:
        combine_helper_vec<T> helpers;

        template <typename... Iters>
        struct add_all {};

        template <typename ItersHead, typename... ItersTail>
        struct add_all<ItersHead, ItersTail...> {
            static void f(combine_container& self, ItersHead head, ItersTail... tail) {
                self.push_back(head);
                add_all<ItersTail...>::f(tail...);
            }
        };

        template <typename ItersLast>
        struct add_all<ItersLast> {
            static void f(combine_container& self, ItersLast last) {
                self.push_back(last);
            }
        };
};
}

/* Combines different iterators (=columns) to one iterator (=rows).
 * @T Value type of all iterators.
 *
 * This helper allows you to combine multiple iterators It1, It2, ..., ItN to
 * a single iterator CIt which increments It1, It2, ..., ItN at once using
 * operator++ and yields a container that enables iterating over the current
 * elements *It1, *It2, ..., *ItN.
 *
 * Equality is checked using the equality of all iterators and is true iff both
 * combined iterators combine the same number of iterators and when all
 * combinded iterators are pairwise equal, (It1_a == It2_b AND It2_a== It2_b
 * AND ... AND ItN_a == ItN_b). So it might be helpful if the containers which
 * yield the different iterators to combine have the same length.
 */
template <typename T>
class combine {
    public:
        typedef std::input_iterator_tag iterator_category;
        typedef std::size_t difference_type;
        typedef _::combine_container<T> value_type;
        typedef value_type* pointer;
        typedef value_type& reference;

        /* Constructs new combined iterator and add all iterators to it.
         */
        template <typename... Iters>
        combine(Iters... iters) : container(iters...) {}

        /* Adds new iterator to combined version.
         * @Iter Type of the iterator.
         */
        template <typename Iter>
        void push_back(Iter it) {
            container.push_back(it);
        }

        /* Increments all containing iterators.
         *
         * @return reference to self.
         */
        combine& operator++() {
            container.incr_all();
            return *this;
        }

        /* Yields a container reference that containes the current state of all iterators.
         */
        const _::combine_container<T>& operator*() const {
            return container;
        }

        /* Yields a container pointer that containes the current state of all iterators.
         */
        const _::combine_container<T>* operator->() const {
            return &container;
        }

        /* Checks for equality. See preample for details.
         */
        bool operator==(const combine& other) const {
            return this->container.eq(other.container);
        }

        /* Checks for inequality. See preample for details.
         */
        bool operator!=(const combine& other) const {
            return !(*this == other);
        }

    private:
        _::combine_container<T> container;
};

/* Transform the results of one iterator using a function
 * @Iter source iterator.
 * @Function function that maps *iter -> Target:
 * @Target return type of the function.
 * @CleanupPointers if Function returns pointers, should we delete them?
 */
template <typename Iter, typename Function, typename Target, bool CleanupPointers = false>
struct transform {
    static_assert(!std::is_reference<Target>::value && !std::is_pointer<Target>::value, "Target must not be a reference or pointer!");

    Iter iter;
    Function function;
    mutable Target* current;

    typedef decltype(std::declval<Function>()(*std::declval<Iter>())) fresult_t;
    typedef std::input_iterator_tag iterator_category;
    typedef std::size_t difference_type;
    typedef Target value_type;
    typedef Target* pointer;
    typedef Target& reference;

    /* Creates new transform iterator using a underlying iterator and a function.
     * @i Iterator copy.
     * @f Funciton copy.
     */
    transform(Iter i, Function f) : iter(i), function(f), current(nullptr) {}

    transform(const transform& other) : iter(other.iter), function(other.function), current(nullptr) {}
    transform(transform&& other) : iter(other.iter), function(other.function), current(other.current) {
        other.current = nullptr;
    }

    ~transform() {
        if (current != nullptr) {
            cleanup<fresult_t, CleanupPointers>();
        }
    }

    transform& operator=(const transform& other) {
        if (current != nullptr) {
            cleanup<fresult_t, CleanupPointers>();
        }

        iter = other.iter;
        function = other.function;
        current = nullptr;

        return *this;
    }

    transform& operator=(transform&& other) {
        if (current != nullptr) {
            cleanup<fresult_t, CleanupPointers>();
        }

        iter = other.iter;
        function = other.function;
        current = other.current;

        other.current = nullptr;

        return *this;
    }

    /* Increments underlying iterator.
     */
    transform& operator++() {
        if (current != nullptr) {
            cleanup<fresult_t, CleanupPointers>();
        }

        ++iter;

        return *this;
    }

    /* Call iter.operator* and converts the result using the stored function.
     */
    reference operator*() const {
        if (!current) {
            update_current<fresult_t>();
        }
        return *current;
    }

    /* Call iter.operator* and converts the result using the stored function.
     */
    pointer operator->() const {
        if (!current) {
            update_current<fresult_t>();
        }
        return current;
    }

    /* Returns true iff underlying iterators are equal.
     */
    bool operator==(const transform& other) const {
        return this->iter == other.iter;
    }

    /* Returns true iff underlying iterators are inequal.
     */
    bool operator!=(const transform& other) const {
        return this->iter != other.iter;
    }

    template <typename F>
    typename std::enable_if<std::is_pointer<F>::value>::type
    update_current() const {
        current = function(*iter);
    }

    template <typename F>
    typename std::enable_if<std::is_reference<F>::value>::type
    update_current() const {
        current = &function(*iter);
    }

    template <typename F>
    typename std::enable_if<!std::is_pointer<F>::value && !std::is_reference<F>::value>::type
    update_current() const {
        current = new Target(function(*iter));
    }

    template <typename F, bool P>
    typename std::enable_if<std::is_pointer<F>::value && P>::type
    cleanup() {
        delete current;
        current = nullptr;
    }

    template <typename F, bool P>
    typename std::enable_if<std::is_pointer<F>::value && !P>::type
    cleanup() {
        current = nullptr;
    }

    template <typename F, bool P>
    typename std::enable_if<std::is_reference<F>::value>::type
    cleanup() {
        current = nullptr;
    }

    template <typename F, bool P>
    typename std::enable_if<!std::is_pointer<F>::value && !std::is_reference<F>::value>::type
    cleanup() {
        delete current;
        current = nullptr;
    }
};

/* Creates new transform iterator. See <transform> for details about parameter.
 */
template <
    typename Iter,
    typename Function,
    typename Target = typename std::remove_pointer<
            typename std::remove_reference<
                    decltype(std::declval<Function>()(*std::declval<Iter>()))
                >::type
        >::type,
    bool CleanupPointers = false
>
transform<Iter, Function, Target, CleanupPointers> make_transform(Iter i, Function f) {
    return transform<Iter, Function, Target, CleanupPointers>(i, f);
}

}

}

