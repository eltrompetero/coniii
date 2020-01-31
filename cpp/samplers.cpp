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

#include "samplers.hpp"

Sampler::Sampler() {
    // setup ndarray
    py::tuple shape = py::make_tuple(0);
    np::dtype dt = np::dtype::get_builtin<double>();
};

// Parameters
// size : int
// multipliers : vector<double>, maxent parameters
// seed : int
Sampler::Sampler(int new_n,
                 std::vector<double> new_multipliers,
                 int new_seed) {
    n = new_n;
    multipliers = new_multipliers;
    couplingMat = std::vector<std::vector<double>> (n, std::vector<double>(n,0));
    int counter = 0;
    seed = new_seed;
    unitrng = std::uniform_real_distribution<double> (0,1);
    
    // setup ndarray
    py::tuple shape = py::make_tuple(new_multipliers.size());
    np::dtype dt = np::dtype::get_builtin<double>();
    
    // setup seed
    if (new_seed==-1) {
        rd = std::mt19937_64(std::random_device{}());
    } else {
        rd = std::mt19937_64(seed);
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

// basic setup that must be supplemented in derived classes
Sampler::Sampler(int new_n,
                 np::ndarray new_multipliers,
                 int new_seed) {
    n = new_n;

    couplingMat = std::vector<std::vector<double>> (n, std::vector<double>(n,0));
    seed = new_seed;
    unitrng = std::uniform_real_distribution<double> (0,1);
    
    // setup ndarray
//    py::tuple shape = py::make_tuple(new_multipliers.size(0));
//    np::dtype dt = np::dtype::get_builtin<double>();
//    ndmultipliers = np::zeros(shape, dt);

    // setup seed
    if (new_seed==-1) {
        rd = std::mt19937_64(std::random_device{}());
    } else {
        rd = std::mt19937_64(seed);
    }
};

void Sampler::generate_sample(int const n_samples,
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
    s = init_sample();
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
};//end generate_sample

np::ndarray Sampler::fetch_sample() {
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
std::vector<double> Sampler::means() {
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
void Sampler::print(int const nshow) {
    std::cout << "Info:\nPreview of sample:\n";
    for (int i=0; i<nshow; ++i){
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
    std::cout << "\n\n";
    return;
};

// ================
// Ising class
// ================
Ising::Ising(int n, std::vector<double> multipliers, int seed)
 : Sampler(n, multipliers, seed) {
     assert (multipliers.size()==(n+n*(n-1)/2));
 };

Ising::Ising(int n, np::ndarray new_multipliers, int seed)
 : Sampler(n, new_multipliers, seed) {
     int counter = 0;

     // read in numpy array
     readin_multipliers(new_multipliers);

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

void Ising::readin_multipliers(np::ndarray new_multipliers) {
    int input_size = new_multipliers.shape(0);
    assert (input_size==(n+n*(n-1)/2));
    double* input_ptr = reinterpret_cast<double*>(new_multipliers.get_data());
    multipliers = std::vector<double>(input_size);
    for (int i = 0; i < input_size; ++i) {
        multipliers[i] = *(input_ptr + i);
    }
};

// Initialize samples with random -1, 1 values
std::vector<int> Ising::init_sample() {
    std::uniform_int_distribution<int> intrng(0,1);
    std::vector<int> s(n);
    
    for (int i=0; i<n; ++i) {
        if (intrng(rd)) {
            s[i] = -1;
        } else {
            s[i] = 1;
        }
    }
    return s;
};

double Ising::calc_e(std::vector<int> const &s) {
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
};//end calc_e

// Ising sample on single state which may be altered in the function.
double Ising::sample_metropolis(std::vector<int> &s, int const randix) {
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
};//end sample_metropolis


// ================
// Potts3 class
// ================
Potts3::Potts3(int n, std::vector<double> new_multipliers, int seed)
 : Sampler(n, new_multipliers, seed) {
     staterng = std::uniform_int_distribution<int> (1,2);
     
     assert(multipliers.size()==(n*3+n*(n-1)/2));
 };

Potts3::Potts3(int n, np::ndarray new_multipliers, int seed)
    : Sampler(n, new_multipliers, seed) {
    int counter = 0;
    staterng = std::uniform_int_distribution<int> (1,2);

    // read in numpy array
    readin_multipliers(new_multipliers);

    // setup couplings by copying to a matrix
    for (int i=0; i<(n-1); ++i) {
        for (int j=i+1; j<n; ++j) {
            couplingMat[i][j] = couplingMat[j][i] = multipliers[counter+n*3];
            counter++;
        }
    }
    // check that diagonal is zero
    for (int i=0; i<n; ++i) {
        assert (couplingMat[i][i]==0);
    }
};

void Potts3::readin_multipliers(np::ndarray new_multipliers) {
    int input_size = new_multipliers.shape(0);
    assert (input_size==(n*3+n*(n-1)/2));
    double* input_ptr = reinterpret_cast<double*>(new_multipliers.get_data());
    multipliers = std::vector<double>(input_size);
    for (int i = 0; i < input_size; ++i) {
        multipliers[i] = *(input_ptr + i);
    }
};

std::vector<int> Potts3::init_sample() {
    std::uniform_int_distribution<int> intrng(0,2);
    std::vector<int> s(n);
    
    for (int i=0; i<n; ++i) {
        s[i] = intrng(rd);
    }
    return s;
};

// The three possible states are 0, 1, 2.
// The couplings add a term when the spins agree.
double Potts3::calc_e(std::vector<int> const &s) {
    double e = 0.0;
    int counter = 0;
    for (int i=0; i<(n-1); ++i) {
        // mean biases
        if (s[i]==0) {
            e -= multipliers[i];
        } else if (s[i]==1) {
            e -= multipliers[i+n];
        } else {
            e -= multipliers[i+2*n];
        }
        // couplings
        for (int j=i+1; j<n; ++j) {
            // couplings
            if (s[i]==s[j]) {
                e -= multipliers[counter+3*n];
            }
            counter++;
        }
    }
    // magnetization for last spin
    if (s[n-1]==0) {
        e -= multipliers[n-1];
    } else if (s[n-1]==1) {
        e -= multipliers[2*n-1];
    } else {
        e -= multipliers[3*n-1];
    }
    return e;
};//end calc_e

// Ising sample on single state which may be altered in the function.
double Potts3::sample_metropolis(std::vector<int> &s, int const randix) {
    double de = 0.0;
    int oState = s[randix];
    
    s[randix] = (staterng(rd)+oState)%3;
    
    // fields
    de += multipliers[oState*n+randix];  // remove old field
    de -= multipliers[s[randix]*n+randix];  // add new field
    
    // couplings
    for (int i=0; i<n; ++i) {
        if (oState==s[i]) {
            de += couplingMat[randix][i];
        }
        if (s[randix]==s[i]) {
            de -= couplingMat[randix][i];
        }
    }
    
    if ( (de>0) and (unitrng(rd)>exp(-de))) {
        s[randix] = oState;
        return 0.0;
    }
    return de;
};//end sample_metropolis

// convert std::vector of multipliers to a Python ndarray
np::ndarray Potts3::multipliers2ndarray() {
    py::tuple shape = py::make_tuple(multipliers.size());
    py::tuple stride = py::make_tuple(sizeof(std::vector<double>::value_type));
    np::dtype dt = np::dtype::get_builtin<double>();
    np::ndarray ndmultipliers = np::from_data(&multipliers[0], dt, shape, stride, py::object());
    return ndmultipliers;
};
