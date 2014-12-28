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

    nntlib::net<double, nntlib::loss::mse<double>, decltype(l1), decltype(l2), decltype(l3)> net(l1, l2, l3);
    std::uniform_real_distribution<double> dist(0, 1);
    std::vector<double> input0;
    std::vector<double> input1;
    std::vector<double> input2;
    std::vector<double> input3;
    std::vector<double> input4;
    std::vector<double> input5;
    std::vector<double> input6;
    std::vector<double> input7;
    std::vector<double> input8;
    std::vector<double> input9;
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

        input0.push_back(x0);
        input1.push_back(x1);
        input2.push_back(x2);
        input3.push_back(x3);
        input4.push_back(x4);
        input5.push_back(x5);
        input6.push_back(x6);
        input7.push_back(x7);
        input8.push_back(x8);
        input9.push_back(x9);
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
    auto iFuncInput0 = std::bind(iFunc, std::placeholders::_1, std::ref(input0));
    auto iFuncInput1 = std::bind(iFunc, std::placeholders::_1, std::ref(input1));
    auto iFuncInput2 = std::bind(iFunc, std::placeholders::_1, std::ref(input2));
    auto iFuncInput3 = std::bind(iFunc, std::placeholders::_1, std::ref(input3));
    auto iFuncInput4 = std::bind(iFunc, std::placeholders::_1, std::ref(input4));
    auto iFuncInput5 = std::bind(iFunc, std::placeholders::_1, std::ref(input5));
    auto iFuncInput6 = std::bind(iFunc, std::placeholders::_1, std::ref(input6));
    auto iFuncInput7 = std::bind(iFunc, std::placeholders::_1, std::ref(input7));
    auto iFuncInput8 = std::bind(iFunc, std::placeholders::_1, std::ref(input8));
    auto iFuncInput9 = std::bind(iFunc, std::placeholders::_1, std::ref(input9));
    auto iFuncOutput = std::bind(iFunc, std::placeholders::_1, std::ref(output));

    nntlib::iterator::combine<double> combTestInputBegin;
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput0), double>(test.begin(), iFuncInput0));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput1), double>(test.begin(), iFuncInput1));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput2), double>(test.begin(), iFuncInput2));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput3), double>(test.begin(), iFuncInput3));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput4), double>(test.begin(), iFuncInput4));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput5), double>(test.begin(), iFuncInput5));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput6), double>(test.begin(), iFuncInput6));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput7), double>(test.begin(), iFuncInput7));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput8), double>(test.begin(), iFuncInput8));
    combTestInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput9), double>(test.begin(), iFuncInput9));

    nntlib::iterator::combine<double> combTestOutputBegin;
    combTestOutputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncOutput), double>(test.begin(), iFuncOutput));

    nntlib::iterator::combine<double> combTestInputEnd;
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput0), double>(test.end(), iFuncInput0));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput1), double>(test.end(), iFuncInput1));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput2), double>(test.end(), iFuncInput2));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput3), double>(test.end(), iFuncInput3));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput4), double>(test.end(), iFuncInput4));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput5), double>(test.end(), iFuncInput5));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput6), double>(test.end(), iFuncInput6));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput7), double>(test.end(), iFuncInput7));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput8), double>(test.end(), iFuncInput8));
    combTestInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput9), double>(test.end(), iFuncInput9));

    nntlib::iterator::combine<double> combTestOutputEnd;
    combTestOutputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncOutput), double>(test.end(), iFuncOutput));

    nntlib::iterator::combine<double> combTrainInputBegin;
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput0), double>(test.begin(), iFuncInput0));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput1), double>(test.begin(), iFuncInput1));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput2), double>(test.begin(), iFuncInput2));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput3), double>(test.begin(), iFuncInput3));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput4), double>(test.begin(), iFuncInput4));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput5), double>(test.begin(), iFuncInput5));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput6), double>(test.begin(), iFuncInput6));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput7), double>(test.begin(), iFuncInput7));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput8), double>(test.begin(), iFuncInput8));
    combTrainInputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncInput9), double>(test.begin(), iFuncInput9));

    nntlib::iterator::combine<double> combTrainOutputBegin;
    combTrainOutputBegin.push_back(nntlib::iterator::transform<decltype(test.begin()), decltype(iFuncOutput), double>(test.begin(), iFuncOutput));

    nntlib::iterator::combine<double> combTrainInputEnd;
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput0), double>(test.end(), iFuncInput0));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput1), double>(test.end(), iFuncInput1));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput2), double>(test.end(), iFuncInput2));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput3), double>(test.end(), iFuncInput3));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput4), double>(test.end(), iFuncInput4));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput5), double>(test.end(), iFuncInput5));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput6), double>(test.end(), iFuncInput6));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput7), double>(test.end(), iFuncInput7));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput8), double>(test.end(), iFuncInput8));
    combTrainInputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncInput9), double>(test.end(), iFuncInput9));

    nntlib::iterator::combine<double> combTrainOutputEnd;
    combTrainOutputEnd.push_back(nntlib::iterator::transform<decltype(test.end()), decltype(iFuncOutput), double>(test.end(), iFuncOutput));

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

