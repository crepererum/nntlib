#include <nntlib/nntlib.hpp>

#include <cmath>

#include <iostream>

constexpr std::size_t N = 10000;

constexpr double pi() {
    return std::atan2(0, -1);
}

void print_matrix(const std::vector<std::vector<double>>& matrix) {
    std::cout << "[" << std::endl;
    for (const auto& row : matrix) {
        std::cout << "  [";
        for (const auto& cell : row) {
            std::cout << cell << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l1(1, 30, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l2(30, 30, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l3(30, 1, rng);

    nntlib::net<double, nntlib::loss::mse<double>, decltype(l1), decltype(l2), decltype(l3)> net(l1, l2, l3);
    std::vector<std::vector<double>> input;
    std::vector<std::vector<double>> output;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / static_cast<double>(N) * 2 - 1;
        double y = std::sin(-x * pi());

        input.emplace_back(std::vector<double>{x});
        output.emplace_back(std::vector<double>{y});
        indices.push_back(i);
    }

    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<std::size_t> test(indices.begin(), indices.begin() + static_cast<std::size_t>(N * 0.01));
    std::vector<std::size_t> train(indices.begin() + static_cast<std::size_t>(N * 0.01), indices.end());
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
    typedef nntlib::training::batch<double> train_method_t;
    train_method_t tm(train_method_t::func_factor_exp(0.5, 0.95), 1, 100);
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

