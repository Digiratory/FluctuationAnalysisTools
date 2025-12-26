#include "StatTools_core.h"

double get_exponential_dist_value(double lambda)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<double> dist(1.0/lambda);
    return dist(gen);
}

double get_gauss_dist_value()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

std::vector<double> get_exp_dist_vector(double lambda, int size)
{
    std::vector<double> exp_vector;
    for(int i = 0; i < size; i++)
        exp_vector.push_back(get_exponential_dist_value(lambda));
    return exp_vector;
}

std::vector<double> get_poisson_thread(std::vector<double> input_vector, double divisor)
{
    std::vector<double> output_vector;
    for (double i : input_vector) {
        for (double exp_val : get_exp_dist_vector(divisor / i, std::ceil(i / divisor))) {
            output_vector.push_back(exp_val);
        }
    }
    return output_vector;
}

std::vector<double> cumsum(std::vector<double> input_vector)
{
    std::vector<double> cumsum_vector(input_vector.size());
    cumsum_vector[0] = input_vector[0];
    for (int i = 1; i < input_vector.size(); i++) {
        cumsum_vector[i] = cumsum_vector[i-1] + input_vector[i];
    }
    return cumsum_vector;
}

std::vector<double> core(std::vector<double> p_thread_cumsum, std::vector<double> C, std::vector<double> requests)
{
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

std::vector<double> model(std::vector<double> input_vector, std::vector<double> U, double C0_global)
{
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
