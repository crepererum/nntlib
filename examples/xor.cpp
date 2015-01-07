#include <nntlib/nntlib.hpp>

#include <cmath>

#include <iostream>

constexpr std::size_t N = 100;

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(-1, 1);

    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l1(1, 4, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l2(4, 4, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l3(4, 1, rng);

    auto net = nntlib::make_net<double, nntlib::loss::mse<double>>(l1, l2, l3);
    std::vector<std::vector<double>> input;
    std::vector<std::vector<double>> output;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < N; ++i) {
        double x1 = dist(rng);
        double x2 = dist(rng);
        double y = x1 + x2 / 2;

        input.emplace_back(std::vector<double>{x1, x2});
        output.emplace_back(std::vector<double>{y});
        indices.push_back(i);
    }

    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<std::size_t> test(indices.begin(), indices.begin() + static_cast<std::size_t>(N * 0.5));
    std::vector<std::size_t> train(indices.begin() + static_cast<std::size_t>(N * 0.5), indices.end());
    std::sort(test.begin(), test.end());

    std::vector<std::vector<double>> inputTest;
    std::vector<std::vector<double>> outputTest;
    for (std::size_t i : test) {
        inputTest.push_back(input[i]);
        outputTest.push_back(output[i]);
    }
    std::vector<std::vector<double>> inputTrain;
    std::vector<std::vector<double>> outputTrain;
    for (std::size_t i : train) {
        inputTrain.push_back(input[i]);
        outputTrain.push_back(output[i]);
    }

    std::cout << "Train:" << std::endl;
    typedef nntlib::training::lbfgs<double> train_method_t;
    train_method_t tm(30, train_method_t::func_factor_const(1), 15, 5);
    tm.callback_round([&](std::size_t round){
        double error = 0.0;
        for (std::size_t i : test) {
            auto out = net.forward(input[i].begin(), input[i].end());
            double d = out[0] - output[i][0];
            error += d * d;
        }
        std::cout << "  round " << round << ": error=" << error / static_cast<double>(test.size()) << std::endl;
    });
    tm.train(net, inputTrain.begin(), inputTrain.end(), outputTrain.begin(), outputTrain.end());
    std::cout << "DONE" << std::endl << std::endl;

    for (std::size_t i : test) {
        auto out = net.forward(input[i].begin(), input[i].end());
        std::cout << output[i][0] << " " << out[0] << std::endl;
    }
}

