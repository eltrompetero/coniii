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


// Base sampling class
class Sampler {
public:
    int n;                                  // system size
    std::vector<std::vector<double>> couplingMat;
    std::vector<double> multipliers;        // fields and couplings
    std::vector<std::vector<int>> sample;   // stored sample
    int seed;
    std::mt19937_64 rd;
    std::uniform_real_distribution<double> unitrng;

    Sampler();
    Sampler(int, std::vector<double>, int=-1);  // init with multipliers
    Sampler(int, np::ndarray, int=-1);
    
    virtual double calc_e(std::vector<int> const&) = 0;
    virtual double sample_metropolis(std::vector<int>&, int const) = 0;
    void generate_sample(int const,
                         int const,
                         int const,
                         bool const=false);
    void generate_cond_sample(np::ndarray,
                              np::ndarray,
                              int const,
                              int const,
                              int const,
                              bool const=false);
    np::ndarray fetch_sample();
    std::vector<double> means();
    void print(int const);
    void readin_fixed_set(std::vector<int>&, std::vector<int>&, np::ndarray, np::ndarray);

private:
    virtual std::vector<int> init_sample() = 0;
};


// sampling for Ising model
class Ising : public Sampler {
public:
    Ising();
    Ising(int, std::vector<double>, int=-1);
    Ising(int, np::ndarray, int=-1);
    
    double calc_e(std::vector<int> const&);
    double sample_metropolis(std::vector<int>&, int const);
    void readin_multipliers(np::ndarray);
private:
    std::vector<int> init_sample();
};


// 3-state Potts sampling
class Potts3 : public Sampler {
public:
    Potts3();
    Potts3(int, std::vector<double>, int=-1);
    Potts3(int, np::ndarray, int=-1);
    
    double calc_e(std::vector<int> const&);
    double sample_metropolis(std::vector<int>&, int const);
    py::tuple get_value();
    py::tuple get_state();
    void set_state(py::tuple);
    np::ndarray multipliers2ndarray();
private:
    std::uniform_int_distribution<int> staterng;
    std::vector<int> init_sample();
    void readin_multipliers(np::ndarray);
};
