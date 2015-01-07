#include <nntlib/nntlib.hpp>

#include <cmath>

#include <iostream>

constexpr std::size_t N = 1000;

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(-1, 1);

    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l1(2, 8, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l2(8, 1, rng);

    auto net = nntlib::make_net<double, nntlib::loss::mse<double>>(l1, l2);
    std::vector<std::vector<double>> input;
    std::vector<std::vector<double>> output;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < N; ++i) {
        double x1 = dist(rng);
        double x2 = dist(rng);
        double y = std::abs(x1 - x2) / 2;

        input.emplace_back(std::vector<double>{x1, x2});
        output.emplace_back(std::vector<double>{y});
        indices.push_back(i);
    }

    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<std::size_t> test(indices.begin(), indices.begin() + static_cast<std::size_t>(N * 0.5));
    std::vector<std::size_t> train(indices.begin() + static_cast<std::size_t>(N * 0.5), indices.end());
    std::sort(test.begin(), test.end());

    auto iFunc = [](std::size_t i, const std::vector<std::vector<double>>& source) -> std::vector<double> {
        return source[i];
    };
    auto iFuncInput = std::bind(iFunc, std::placeholders::_1, std::ref(input));
    auto iFuncOutput = std::bind(iFunc, std::placeholders::_1, std::ref(output));

    auto testInputBegin = nntlib::iterator::make_transform(test.begin(), iFuncInput);
    auto testInputEnd = nntlib::iterator::make_transform(test.end(), iFuncInput);
    auto trainInputBegin = nntlib::iterator::make_transform(train.begin(), iFuncInput);
    auto trainInputEnd = nntlib::iterator::make_transform(train.end(), iFuncInput);

    auto testOutputBegin = nntlib::iterator::make_transform(test.begin(), iFuncOutput);
    auto testOutputEnd = nntlib::iterator::make_transform(test.end(), iFuncOutput);
    auto trainOutputBegin = nntlib::iterator::make_transform(train.begin(), iFuncOutput);
    auto trainOutputEnd = nntlib::iterator::make_transform(train.end(), iFuncOutput);

    std::cout << "Train:" << std::endl;
    typedef nntlib::training::lbfgs<double> train_method_t;
    train_method_t tm(30, train_method_t::func_factor_exp(0.7, 0.95), 100, 5, 0.2);
    tm.callback_round([&](std::size_t round){
        double error = 0.0;
        std::size_t n = 0;
        nntlib::utils::multi_foreach([&](const auto& in, const auto& out){
            auto predict = net.forward(in.begin(), in.end());
            double d = predict[0] - out[0];
            error += d * d;
            ++n;
        }, testInputBegin, testInputEnd, testOutputBegin, testOutputEnd);
        std::cout << "  round " << round << ": error=" << error / static_cast<double>(n) << std::endl;

        std::shuffle(train.begin(), train.end(), rng);
    });
    tm.train(net, trainInputBegin, trainInputEnd, trainOutputBegin, trainOutputEnd);
    std::cout << "DONE" << std::endl << std::endl;

    for (std::size_t i : test) {
        auto out = net.forward(input[i].begin(), input[i].end());
        std::cout << output[i][0] << " " << out[0] << std::endl;
    }
}

