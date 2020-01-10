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


//        .def("__getinitargs__", +[](Potts3 const& self) {
//            return py::make_tuple();
//        })
//        .def("__getstate__", &Potts3::get_state)
//        .def("__setstate__", &Potts3::set_state)
