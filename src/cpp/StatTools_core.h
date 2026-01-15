#ifndef STATTOOLS_CORE_H
#define STATTOOLS_CORE_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

// Core statistical and mathematical functions
// These functions are shared between C API and pybind11 implementations

double get_exponential_dist_value(double lambda);
double get_gauss_dist_value();
std::vector<double> get_exp_dist_vector(double lambda, int size);
std::vector<double> get_poisson_thread(std::vector<double> input_vector, double divisor = 1);
std::vector<double> cumsum(std::vector<double> input_vector);
std::vector<double> core(std::vector<double> p_thread_cumsum, std::vector<double> C, std::vector<double> requests);
std::vector<double> model(std::vector<double> input_vector, std::vector<double> U, double C0_global = -1.0);

#endif // STATTOOLS_CORE_H
