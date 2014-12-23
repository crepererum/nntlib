BUILDDIR ?= target
CLDOC ?= cldoc
CXX ?= g++
CXXFLAGS = -std=c++14 -Iinclude

all: examples doc

examples: examples/sin.cpp
	mkdir -p $(BUILDDIR)/examples
	$(CXX) $(CXXFLAGS) examples/sin.cpp -o $(BUILDDIR)/examples/sin

doc: include/nntlib/*.hpp
	mkdir -p $(BUILDDIR)
	$(CLDOC) generate $(CXXFLAGS) -- --output $(BUILDDIR)/doc include/nntlib/*.hpp

clean:
	rm -rf target

.PHONY: all doc examples clean

