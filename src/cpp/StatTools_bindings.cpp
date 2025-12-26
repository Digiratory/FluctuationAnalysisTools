#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Import the original C++ functions from StatTools_C_API.cpp
// We'll need to extract the core functions and expose them via pybind11

// Core functions from the original implementation
double get_exponential_dist_value(double lambda) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<double> dist(1.0/lambda);
    return dist(gen);
}

double get_gauss_dist_value() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

std::vector<double> get_exp_dist_vector(double lambda, int size) {
    std::vector<double> exp_vector;
    for(int i = 0; i < size; i++)
        exp_vector.push_back(get_exponential_dist_value(lambda));
    return exp_vector;
}

std::vector<double> get_poisson_thread(std::vector<double> input_vector, double divisor = 1) {
    std::vector<double> output_vector;
    for (double i : input_vector) {
        for (double exp_val : get_exp_dist_vector(divisor / i, std::ceil(i / divisor))) {
            output_vector.push_back(exp_val);
        }
    }
    return output_vector;
}

std::vector<double> cumsum(std::vector<double> input_vector) {
    std::vector<double> cumsum_vector(input_vector.size());
    cumsum_vector[0] = input_vector[0];
    for (int i = 1; i < input_vector.size(); i++) {
        cumsum_vector[i] = cumsum_vector[i-1] + input_vector[i];
    }
    return cumsum_vector;
}

std::vector<double> core(std::vector<double> p_thread_cumsum, std::vector<double> C, std::vector<double> requests) {
    std::vector<double> waiting_curve;

    for (double c : C) {
        int event_done = 0;
        std::vector<double> T_free;
        std::vector<double> T_waiting;
        double T_service = 1.0 / c;

        for (double event : p_thread_cumsum) {
            if (event_done == 0) {
                T_free.push_back(T_service + event);
                T_waiting.push_back(0.0);
                event_done++;
            } else {
                if (event < T_free.back()) {
                    T_waiting.push_back(T_free.back() - event);
                    T_free.push_back(T_service + T_free.back());
                    event_done++;
                } else {
                    T_free.push_back(T_service + event);
                    T_waiting.push_back(0.0);
                    event_done++;
                }
            }
        }

        double T_waiting_total = 0.0;
        for (double tw : T_waiting) {
            T_waiting_total += tw;
        }

        waiting_curve.push_back(T_waiting_total / event_done);
    }

    return waiting_curve;
}

std::vector<double> model(std::vector<double> input_vector, std::vector<double> U, double C0_global = -1.0) {
    std::vector<double> poisson_thread = get_poisson_thread(input_vector);
    std::vector<double> p_thread_cumsum = cumsum(poisson_thread);
    std::vector<double> requests;
    double requests_sum = 0.0;

    for (int i = 0; i < poisson_thread.size(); i++) {
        requests.push_back(1);
        requests_sum += 1.0;
    }

    double C0;
    if (C0_global < 0) {
        C0 = requests_sum / (p_thread_cumsum.back() - p_thread_cumsum[0]);
    } else {
        C0 = C0_global;
    }

    std::vector<double> C;
    for (int i = 0; i < U.size(); i++) {
        C.push_back(C0 / U[i]);
    }

    std::vector<double> curve = core(p_thread_cumsum, C, requests);
    return curve;
}

// pybind11 bindings
PYBIND11_MODULE(StatTools_bindings, m) {
    m.doc() = "Modern pybind11 bindings for StatTools C/C++ functions";

    // Waiting time calculation function
    m.def("get_waiting_time", [](py::array_t<double> input_vector, py::array_t<double> U, double C0_input) {
        // Convert numpy arrays to std::vector
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> u_vec(U.data(), U.data() + U.size());

        // Call the model function
        std::vector<double> result = model(input_vec, u_vec, C0_input);

        // Return as numpy array
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Calculate average waiting time curve for given input vector and utilization factors",
          py::arg("input_vector"), py::arg("U"), py::arg("C0_input") = -1.0);

    // Random value generators
    m.def("get_exponential_dist_value", &get_exponential_dist_value,
          "Generate a single exponential distribution value", py::arg("lambda"));

    m.def("get_gauss_dist_value", &get_gauss_dist_value,
          "Generate a single Gaussian distribution value");

    // Vector generators
    m.def("get_exp_dist_vector", [](double lambda, int size) {
        std::vector<double> result = get_exp_dist_vector(lambda, size);
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Generate a vector of exponential distribution values",
          py::arg("lambda"), py::arg("size"));

    m.def("get_poisson_thread", [](py::array_t<double> input_vector, double divisor) {
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> result = get_poisson_thread(input_vec, divisor);
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Generate Poisson thread from input vector",
          py::arg("input_vector"), py::arg("divisor") = 1.0);

    // Utility functions
    m.def("cumsum", [](py::array_t<double> input_vector) {
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> result = cumsum(input_vec);
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Compute cumulative sum of input vector", py::arg("input_vector"));

    m.def("model", [](py::array_t<double> input_vector, py::array_t<double> U, double C0_global) {
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> u_vec(U.data(), U.data() + U.size());
        std::vector<double> result = model(input_vec, u_vec, C0_global);
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Main model function for queueing theory calculations",
          py::arg("input_vector"), py::arg("U"), py::arg("C0_global") = -1.0);
}
