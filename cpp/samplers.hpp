//
//  samplers.hpp
//  cppsamplers
//
//  Created by Eddie on 12/28/19.
//  Copyright © 2019 Santa Fe Institute. All rights reserved.
//

#ifndef samplers_hpp
#define samplers_hpp

#include <stdio.h>
#include <array>
#include <vector>
#include <random>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iterator>
#include <algorithm>

#endif /* samplers_hpp */

namespace py = boost::python;
namespace np = boost::python::numpy;

// Metropolis sampling for Ising model
class Metropolis {
public:
    int n;                                  // system size
    std::vector<double> multipliers;        // fields and couplings
    std::vector<std::vector<double>> couplingMat;
    std::vector<std::vector<int>> sample;   // stored sample
    int seed;
    std::mt19937 rd;
    std::uniform_real_distribution<double> unitrng;
    
    Metropolis();
    Metropolis(int, std::vector<double>, int=-1);
    Metropolis(int, np::ndarray, int=-1);
    double calc_e(std::vector<int> const&);
    double sample_metropolis(std::vector<int>&, int const);
    void generate_sample(int const,
                         int const,
                         int const,
                         bool const=false);
    np::ndarray fetch_sample();
    std::vector<double> means();
    void print();

private:
    void init_sample(int const);
};
