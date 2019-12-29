//
//  py.cpp
//  cppsamplers
//
//  Created by Eddie on 12/28/19.
//  Copyright © 2019 Santa Fe Institute. All rights reserved.
//

#define BOOST_TEST_DYN_LINK
#include <stdio.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "samplers.hpp"

namespace np = boost::python::numpy;

// thin wrappers (for keeping default args -- doesn't work)
//void (Metropolis::*generate_sample1)(int, int, int) = &Metropolis::generate_sample;


BOOST_PYTHON_MODULE(samplers_ext) {
    using namespace boost::python;
    Py_Initialize();
    np::initialize();

    class_<Metropolis>("Metropolis", init<int, np::ndarray, int>())
        .def("generate_sample", &Metropolis::generate_sample)
        .def("fetch_sample", &Metropolis::fetch_sample)
    ;
};
