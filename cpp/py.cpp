// MIT License
// 
// Copyright (c) 2020 Edward D. Lee, Bryan C. Daniels
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define BOOST_TEST_DYN_LINK
#include <stdio.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "samplers.hpp"

namespace py = boost::python;
namespace np = boost::python::numpy;

// thin wrappers (for keeping default args -- doesn't work)
//void (Metropolis::*generate_sample1)(int, int, int) = &Metropolis::generate_sample;

struct potts3_pickle_suite : py::pickle_suite
{
    static py::tuple getinitargs(Potts3 &w) {
        return py::make_tuple(w.n, w.multipliers2ndarray(), -1);
    }
};

BOOST_PYTHON_MODULE(samplers_ext) {
    using namespace boost::python;
    Py_Initialize();
    np::initialize();

    class_<Potts3>("BoostPotts3", init<int, np::ndarray, int>())
        .def("generate_sample", &Potts3::generate_sample)
        .def("fetch_sample", &Potts3::fetch_sample)
        .def_pickle(potts3_pickle_suite())
    ;
    
    class_<Ising>("BoostIsing", init<int, np::ndarray, int>())
        .def("generate_sample", &Ising::generate_sample)
        .def("fetch_sample", &Ising::fetch_sample)
    ;
};
