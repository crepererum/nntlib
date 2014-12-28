#pragma once

#include <memory>
#include <vector>

namespace nntlib {

namespace iterator {

struct iter_end final {};

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
        combine_helper<T, Iter>(Iter iter) : it(iter) {}

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
        combine_mapper<T>(combine_helper_vec<T>&& hidden_helpers, std::size_t internal_pos) : helpers(std::move(hidden_helpers)), pos(internal_pos) {}

        combine_mapper<T>(const combine_mapper<T>& other) : helpers(combine_helper_vec_cpy(other.helpers)), pos(other.pos) {}
        combine_mapper<T>(combine_mapper<T>&& other) = default;

        combine_mapper<T>& operator=(const combine_mapper<T>& other) {
            this->helpers = combine_helper_vec_cpy(other.helpers);
            this->pos = other.pos;
        }
        combine_mapper<T>& operator=(combine_mapper<T>&& other) = default;

        combine_mapper<T>& operator++() {
            ++pos;
            return *this;
        }

        T operator*() {
            return helpers[pos]->deref();
        }

        T operator->() {
            return helpers[pos]->deref();
        }

        bool operator==(const combine_mapper<T>& other) const {
            if (this->is_end()) {
                return other.is_end();
            } else {
                return !other.is_end() && this->helpers[this->pos]->eq(other.helpers[other.pos]);
            }
        }

        bool operator!=(const combine_mapper<T>& other) const {
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
        combine_container<T>() = default;

        combine_container<T>(const combine_container<T>& other) : helpers(combine_helper_vec_cpy(other.helpers)) {}
        combine_container<T>(combine_container<T>&& other) = default;

        combine_container<T>& operator=(const combine_container<T>& other) {
            this->helpers = combine_helper_vec_cpy(other.helpers);
        }
        combine_container<T>& operator=(combine_container<T>&& other) = default;

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

        bool eq(const combine_container<T>& other) const {
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

template <typename T>
class combine {
    public:
        template <typename Iter>
        void push_back(Iter it) {
            container.push_back(it);
        }

        combine<T>& operator++() {
            container.incr_all();
            return *this;
        }

        combine_container<T>& operator*() {
            return container;
        }

        combine_container<T>* operator->() {
            return &container;
        }

        bool operator==(const combine<T>& other) const {
            return this->container.eq(other.container);
        }

        bool operator!=(const combine<T>& other) const {
            return !(*this == other);
        }

    private:
        combine_container<T> container;
};

template <typename Iter, typename Function, typename Target>
struct transform {
    Iter iter;
    Function function;

    transform<Iter, Function, Target>(Iter i, Function f) : iter(i), function(f) {}

    transform<Iter, Function, Target>& operator++() {
        ++iter;
        return *this;
    }

    Target operator*() {
        return function(*iter);
    }

    bool operator==(const transform<Iter, Function, Target>& other) const {
        return this->iter == other.iter;
    }

    bool operator!=(const transform<Iter, Function, Target>& other) const {
        return this->iter != other.iter;
    }
};

}

}

