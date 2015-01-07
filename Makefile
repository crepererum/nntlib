BUILDDIR ?= target
CLDOC ?= cldoc
CXX ?= g++
CXXFLAGS = -std=c++14 -Iinclude -Iextern/etl/include
CXXFLAGS_EXTRA_EXAMPLES = -O2
EXAMPLES = $(addprefix $(BUILDDIR)/, $(basename $(wildcard examples/*.cpp)))

all: examples doc

examples: $(EXAMPLES)

$(BUILDDIR)/examples/%: examples/%.cpp include/nntlib/*.hpp
	mkdir -p $(BUILDDIR)/examples
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_EXTRA_EXAMPLES) $< -o $@

doc: include/nntlib/*.hpp
	mkdir -p $(BUILDDIR)
	$(CLDOC) generate $(CXXFLAGS) -- --output $(BUILDDIR)/doc include/nntlib/*.hpp

clean:
	rm -rf target

.PHONY: all doc examples clean

