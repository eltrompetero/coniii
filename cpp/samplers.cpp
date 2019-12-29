//
//  samplers.cpp
//  cppsamplers
//
//  Created by Eddie on 12/28/19.
//  Copyright © 2019 Santa Fe Institute. All rights reserved.
//

#include "samplers.hpp"

Metropolis::Metropolis() {};

// Parameters
// size : int
// multipliers : vector<double>, maxent parameters
// seed : int
Metropolis::Metropolis(int new_n,
                       std::vector<double> new_multipliers,
                       int new_seed) {
    assert (new_multipliers.size()==(new_n+new_n*(new_n-1)/2));
    
    n = new_n;
    multipliers = new_multipliers;
    couplingMat = std::vector<std::vector<double>> (n, std::vector<double>(n,0));
    int counter = 0;
    seed = new_seed;
    unitrng = std::uniform_real_distribution<double> (0,1);
    
    // setup seed
    if (new_seed==-1) {
        rd = std::mt19937(std::random_device{}());
    } else {
        rd = std::mt19937(seed);
    }
    
    // setup couplings by copying to a matrix
    for (int i=0; i<(n-1); ++i) {
        for (int j=i+1; j<n; ++j) {
            couplingMat[i][j] = couplingMat[j][i] = multipliers[counter+n];
            counter++;
        }
    }
    // check that diagonal is zero
    for (int i=0; i<n; ++i) {
        assert (couplingMat[i][i]==0);
    }
};

Metropolis::Metropolis(int new_n,
                       np::ndarray new_multipliers,
                       int new_seed) {
    // read in numpy array
    int input_size = new_multipliers.shape(0);
    assert (input_size==(new_n+new_n*(new_n-1)/2));
    double* input_ptr = reinterpret_cast<double*>(new_multipliers.get_data());
    multipliers = std::vector<double>(input_size);
    for (int i = 0; i < input_size; ++i)
        multipliers[i] = *(input_ptr + i);

    n = new_n;
    couplingMat = std::vector<std::vector<double>> (n, std::vector<double>(n,0));
    int counter = 0;
    seed = new_seed;
    unitrng = std::uniform_real_distribution<double> (0,1);

    // setup seed
    if (new_seed==-1) {
        rd = std::mt19937(std::random_device{}());
    } else {
        rd = std::mt19937(seed);
    }

    // setup couplings by copying to a matrix
    for (int i=0; i<(n-1); ++i) {
        for (int j=i+1; j<n; ++j) {
            couplingMat[i][j] = couplingMat[j][i] = multipliers[counter+n];
            counter++;
        }
    }
    // check that diagonal is zero
    for (int i=0; i<n; ++i) {
        assert (couplingMat[i][i]==0);
    }
};

// Initialize samples with random -1, 1 values
void Metropolis::init_sample(int const n_samples) {
    std::uniform_int_distribution<int> intrng(0,1);
    sample = std::vector<std::vector<int>>(n_samples, std::vector<int>(n));
    
    for (int i=0; i<n_samples; ++i) {
        for (int j=0; j<n; ++j) {
            if (intrng(rd)) {
                sample[i][j] = -1;
            } else {
                sample[i][j] = 1;
            }
        }
    }
    return;
};

double Metropolis::calc_e(std::vector<int> const &s) {
    double e = 0.0;
    int counter = 0;
    for (int i=0; i<(n-1); ++i) {
        e -= multipliers[i] * s[i];
        for (int j=i+1; j<n; ++j) {
            // couplings
            e -= multipliers[counter+n] * s[i] * s[j];
            counter++;
        }
    }
    e -= multipliers[n-1] * s[n-1];
    return e;
};

// Metropolis sample on single state which may be altered in the function.
double Metropolis::sample_metropolis(std::vector<int> &s, int const randix) {
    double de = 0.0;
    
    s[randix] *= -1;
    
    de = -2 * multipliers[randix] * s[randix];
    for (int i=0; i<n; ++i) {
        de -= 2 * couplingMat[randix][i] * s[randix] * s[i];
    }
    
    if ( (de>0) and (unitrng(rd)>exp(-de))) {
        s[randix] *= -1;
        return 0.0;
    }
    return de;
};

void Metropolis::generate_sample(int const n_samples,
                                 int const burn_in,
                                 int const n_iters,
                                 bool const systematic_iter) {
    std::uniform_int_distribution<> unitrng(0,1);
    std::uniform_int_distribution<int> intrng(0,n-1);
    sample = std::vector<std::vector<int>> (n_samples, std::vector<int>(n,0));
    std::vector<int> s(n);  // state that we are randomly flipping
    double e;  // energy of current state
    int randix;
    int counter = 0;
    
    // initialize random starting vector
    for (int i=0; i<n; ++i) {
        if (unitrng(rd)) {
            s[i] = -1;
        } else {s[i] = 1;}
    }
    e = calc_e(s);
    
    // generate random samples
    // burn in
    for (int i=0; i<burn_in; ++i) {
        if (systematic_iter) {
            randix = counter%n;
        } else {
            randix = intrng(rd);
        }
        e += sample_metropolis(s, randix);
        counter++;
    }
    // record samples
    for (int i=0; i<n_samples; ++i) {
        for (int j=0; j<n_iters; ++j) {
            if (systematic_iter) {
                randix = counter%n;
            } else {
                randix = intrng(rd);
            }
            e += sample_metropolis(s, randix);
            counter++;
        }
        // copy vector
        for (int j=0; j<n; ++j) {
            sample[i] = s;
        }
    }
};

np::ndarray Metropolis::fetch_sample() {
    // flatten sample
    std::vector<int> flatsample;
    flatsample.reserve(sample.size() * n);
    for (std::vector<std::vector<int>>::iterator it=sample.begin(); it!=sample.end(); ++it) {
        for (std::vector<int>::iterator itt=(*it).begin(); itt!=(*it).end(); ++itt) {
            flatsample.emplace_back(*itt);
        }
    }
    
    // create numpy array
    int sample_size = sample.size();
    py::tuple shape = py::make_tuple(sample_size, n);
    py::tuple stride = py::make_tuple(sizeof(std::vector<int>::value_type)*n,
                                      sizeof(std::vector<int>::value_type));
    np::dtype dt = np::dtype::get_builtin<int>();
    np::ndarray output = np::from_data(&flatsample[0], dt, shape, stride, py::object()); 
    np::ndarray output_array = output.copy();

    return output_array;
};

// calculate mean of each column in sample
std::vector<double> Metropolis::means() {
    std::vector<double> m(n, 0);
    
    // for each column
    for (int i=0; i<n; ++i) {
        // sum over each row value
        for (int j=0; j<sample.size(); ++j) {
            m[i] += sample[j][i];
        }
        m[i] /= sample.size();
    }
    return m;
};

// show some information about the sample
void Metropolis::print() {
    std::cout << "Info:\nPreview of sample:\n";
    for (int i=0; i<10; ++i){
        for (int j=0; j<n; ++j) {
            std::cout << sample[i][j] << ", ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    
    std::vector<double> m = means();
    std::cout << "Means:\n";
    for (int i=0; i<n; ++i) {
        std::cout << m[i] << ", ";
    }
    std::cout << '\n';
    return;
};
