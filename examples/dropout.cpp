#include <nntlib/nntlib.hpp>

#include <cmath>

#include <functional>
#include <iostream>
#include <vector>

constexpr std::size_t N = 10000;

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    nntlib::layer::dropout<double> l1(0.2, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l2(10, 30, rng);
    nntlib::layer::fully_connected<nntlib::activation::tanh<double>> l3(30, 1, rng);

    auto net = nntlib::make_net<double, nntlib::loss::mse<double>>(l1, l2, l3);
    std::uniform_real_distribution<double> dist(0, 1);
    std::vector<std::vector<double>> inputs(10);

    std::vector<double> output;
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

        inputs[0].push_back(x0);
        inputs[1].push_back(x1);
        inputs[2].push_back(x2);
        inputs[3].push_back(x3);
        inputs[4].push_back(x4);
        inputs[5].push_back(x5);
        inputs[6].push_back(x6);
        inputs[7].push_back(x7);
        inputs[8].push_back(x8);
        inputs[9].push_back(x9);
        output.push_back(y);
        indices.push_back(i);
    }

    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<std::size_t> test(indices.begin(), indices.begin() + static_cast<std::size_t>(N * 0.01));
    std::vector<std::size_t> train(indices.begin() + static_cast<std::size_t>(N * 0.01), indices.end());
    std::sort(test.begin(), test.end());

    auto iFunc = [](std::size_t i, const std::vector<double>& source) -> double {
        return source[i];
    };

    nntlib::iterator::combine<double> combTestInputBegin;
    nntlib::iterator::combine<double> combTestInputEnd;
    nntlib::iterator::combine<double> combTrainInputBegin;
    nntlib::iterator::combine<double> combTrainInputEnd;
    for (auto& input : inputs) {
        auto iFuncInput = std::bind(iFunc, std::placeholders::_1, std::ref(input));
        combTestInputBegin.push_back(nntlib::iterator::make_transform(test.begin(), iFuncInput));
        combTestInputEnd.push_back(nntlib::iterator::make_transform(test.end(), iFuncInput));
        combTrainInputBegin.push_back(nntlib::iterator::make_transform(test.begin(), iFuncInput));
        combTrainInputEnd.push_back(nntlib::iterator::make_transform(test.end(), iFuncInput));
    }

    auto iFuncOutput = std::bind(iFunc, std::placeholders::_1, std::ref(output));
    nntlib::iterator::combine<double> combTestOutputBegin(nntlib::iterator::make_transform(test.begin(), iFuncOutput));
    nntlib::iterator::combine<double> combTestOutputEnd(nntlib::iterator::make_transform(test.end(), iFuncOutput));
    nntlib::iterator::combine<double> combTrainOutputBegin(nntlib::iterator::make_transform(test.begin(), iFuncOutput));
    nntlib::iterator::combine<double> combTrainOutputEnd(nntlib::iterator::make_transform(test.end(), iFuncOutput));

    std::cout << "Train:" << std::endl;
    typedef nntlib::training::batch<double> train_method_t;
    train_method_t tm(train_method_t::func_factor_exp(0.5, 0.95), 1, 5);
    tm.callback_round([&](std::size_t round){
        double error = 0.0;
        auto itIn = combTestInputBegin;
        auto itOut = combTestOutputBegin;
        while ((itIn != combTestInputEnd) && (itOut != combTestOutputEnd)) {
            auto out = net.forward(itIn->begin(), itIn->end());
            double d = out[0] - *(itOut->begin());
            error += d * d;

            ++itIn;
            ++itOut;
        }
        std::cout << "  round " << round << ": error=" << error / static_cast<double>(test.size()) << std::endl;
    });
    tm.train(net, combTrainInputBegin, combTrainInputEnd, combTrainOutputBegin, combTrainOutputEnd);
    std::cout << "DONE" << std::endl << std::endl;

    /*for (std::size_t i : test) {
        auto out = net.forward(input[i].begin(), input[i].end());
        std::cout << output[i][0] << " " << out[0] << std::endl;
    }*/
}

