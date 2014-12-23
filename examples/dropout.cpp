#include <nntlib/nntlib.hpp>

#include <cmath>

#include <iostream>
#include <vector>

constexpr std::size_t N = 10000;

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    nntlib::layer::dropout<double> l1(0.2, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l2(10, 30, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l3(30, 1, rng);

    nntlib::net<double, nntlib::loss::mse<double>, decltype(l1), decltype(l2), decltype(l3)> net(l1, l2, l3);
    std::uniform_real_distribution<double> dist(0, 1);
    std::vector<std::vector<double>> input;
    std::vector<std::vector<double>> output;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < N; ++i) {
        double x0 = dist(rng);
        double x1 = dist(rng);
        double x2 = dist(rng);
        double x3 = dist(rng);
        double x4 = dist(rng);
        double x5 = dist(rng);
        double x6 = dist(rng);
        double x7 = dist(rng);
        double x8 = dist(rng);
        double x9 = dist(rng);
        double y =
            (x0 * x1 * x2
            + 2.0 * x3 * x4
            + 3.0 * x5 * x6 * x7
            + 4.0 * x8 * x9 * x0) / 5.0 - 1.0;

        input.emplace_back(std::vector<double>{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9});
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
    train_method_t tm(train_method_t::func_factor_exp(0.5, 0.95), 1, 5);
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

