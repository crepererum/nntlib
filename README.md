# nntlib - a Neural Network Template Library
This library offers neural network templates for C++14. It is designed to be very flexible regarding to data formats and data storage.

## What's (not) included
Currently, nntlib offers the following features:

 - flexible forward neural network
 - activation functions:
   - identity
   - sigmoid
   - tanh
 - loss functions:
   - mean-squared-error
   - cross-entropy
 - layers:
   - fully connected layer
   - dropout layer
 - simple batch oriented training based on stochastic gradient descent (optional L2 regularization)
 - iterator adaptors to avoid copying of data during preparation (e.g. training set generation)

The following features are missing:

 - convolutional layers (unlikely to get implemented because I don't need those)
 - better training methods
 - multi-threading
 - tests

## Usage
nntlib does not need to be precompiled because it is a header-only library. There are no external dependencies apart from a working C++14 compiler and standard library. See `examples` for some small programs. Because of the heavy usage of templates and very generic code it is recommended to active compiler optimization to get performant code.

To build the examples, use:

    make examples

To build the docs, use:

    make doc

## Inspiration
This work is inspired by:

 - [tiny-cnn](https://github.com/nyanp/tiny-cnn)

