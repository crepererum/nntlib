#pragma once

#include <memory>
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
        virtual void incr() = 0;
        virtual T deref() = 0;
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

        virtual T deref() override {
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
using combine_helper_vec = std::vector<std::unique_ptr<combine_helper_base<T>>>;

template <typename T>
combine_helper_vec<T> combine_helper_vec_cpy(const combine_helper_vec<T>& orig) {
    combine_helper_vec<T> cpy;
    for (const auto& element : orig) {
        cpy.emplace_back(element->copy());
    }
    return std::move(cpy);
}

template <typename T>
class combine_mapper {
    public:
        combine_mapper(combine_helper_vec<T>&& hidden_helpers, std::size_t internal_pos) : helpers(std::move(hidden_helpers)), pos(internal_pos) {}

        combine_mapper(const combine_mapper& other) : helpers(combine_helper_vec_cpy(other.helpers)), pos(other.pos) {}
        combine_mapper(combine_mapper&& other) = default;

        combine_mapper& operator=(const combine_mapper<T>& other) {
            this->helpers = combine_helper_vec_cpy(other.helpers);
            this->pos = other.pos;
        }
        combine_mapper& operator=(combine_mapper<T>&& other) = default;

        combine_mapper& operator++() {
            ++pos;
            return *this;
        }

        T operator*() {
            return helpers[pos]->deref();
        }

        T operator->() {
            return helpers[pos]->deref();
        }

        bool operator==(const combine_mapper& other) const {
            if (this->is_end()) {
                return other.is_end();
            } else {
                return !other.is_end() && this->helpers[this->pos]->eq(other.helpers[other.pos]);
            }
        }

        bool operator!=(const combine_mapper& other) const {
            return !(*this == other);
        }

    private:
        combine_helper_vec<T> helpers;
        std::size_t pos;

        bool is_end() const {
            return pos >= helpers.size();
        }
};

template <typename T>
class combine_container {
    public:
        combine_container() = default;

        combine_container(const combine_container& other) : helpers(combine_helper_vec_cpy(other.helpers)) {}
        combine_container(combine_container&& other) = default;

        combine_container& operator=(const combine_container<T>& other) {
            this->helpers = combine_helper_vec_cpy(other.helpers);
        }
        combine_container& operator=(combine_container<T>&& other) = default;

        combine_mapper<T> begin() {
            return combine_mapper<T>(combine_helper_vec_cpy(helpers), 0);
        }

        combine_mapper<T> end() {
            return combine_mapper<T>({}, 0);
        }

        template <typename Iter>
        void push_back(Iter it) {
            helpers.emplace_back(new combine_helper<T, Iter>(it));
        }

        void incr_all() {
            for (auto& h : helpers) {
                h->incr();
            }
        }

        bool eq(const combine_container& other) const {
            if (this->helpers.size() == other.helpers.size()) {
                for (std::size_t i = 0; i < this->helpers.size(); ++i) {
                    if (!this->helpers[i]->eq(other.helpers[i])) {
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
        _::combine_container<T>& operator*() {
            return container;
        }

        /* Yields a container pointer that containes the current state of all iterators.
         */
        _::combine_container<T>* operator->() {
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
 */
template <typename Iter, typename Function, typename Target>
struct transform {
    Iter iter;
    Function function;

    /* Creates new transform iterator using a underlying iterator and a function.
     * @i Iterator copy.
     * @f Funciton copy.
     */
    transform(Iter i, Function f) : iter(i), function(f) {}

    /* Increments underlying iterator.
     */
    transform& operator++() {
        ++iter;
        return *this;
    }

    /* Call iter.operator* and converts the result using the stored function.
     */
    Target operator*() {
        return function(*iter);
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
};

}

}

