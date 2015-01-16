# nntlib - a Neural Network Template Library
This library offers neural network templates for C++14. It is designed to be very flexible regarding to data formats and data storage.

## What's (not) Included
Currently, nntlib offers the following features:

### Network Design
The library implements the following design principles:

 - Flexible Templated Forward Neural Network
 - Choice of Data Types and Input Iterators
 - Ability to Preallocate Memory (e.g. for network state, error state and training deltas)

### Activation Functions
The following activation functions can be used:

 - Identity
 - Sigmoid
 - Softmax
 - Softplus
 - TanH

### Loss Functions
Depending on the learning task (e.g. classification), the following loss functions can be selected:
 - Mean Squared Error
 - Cross Entropy

### Layers
Multiple layer types enable different designs at compile time while layer sizes are set at runtime:
 - Fully Connected Layer
 - Dropout Layer

### Training
To archive good results, the following training methods can be used in combination with different methods to calculate learning rates depending on the number of rounds:
 - Stochastic Gradient Descent (optional: batch training, L2 regularization)
 - L-BFGS (optional: L2 regularization)

### Helpers
To make it easier to plug nntlib into existing architectures, some helpers are already implemented:
 - Iterator Adaptors (avoids copying of data, e.g. while training set generation)
 - ForEach for Multiple Iterators
 - Tuple Helpers (e.g. join, apply)

### TODO
The following features are missing:

 - Convolutional Layers (unlikely to get implemented because I don't need those)
 - More Training Methods
 - Multi-Threading
 - Tests
 - Serialization

## Requirements
To build and use nntlib, the following equipment is required:

 - C++14 compiler and stdlib
 - [Eigen 3 headers](http://eigen.tuxfamily.org/)
 - Optional for docs: [cldoc](https://jessevdk.github.io/cldoc/)

## Usage
nntlib does not need to be precompiled because it is a header-only library. There are no link-time dependencies apart the C++14 standard library. See `examples` for some small programs. Because of the heavy usage of templates and very generic code it is recommended to active compiler optimization to get performant code.

To build the examples, use:

    make examples

To build the docs, use:

    make doc

## Inspiration
This work is inspired by:

 - [tiny-cnn](https://github.com/nyanp/tiny-cnn)

## License
```
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

