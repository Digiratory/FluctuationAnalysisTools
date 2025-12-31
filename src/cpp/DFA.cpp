/*
Detrended Fluctuation Analysis (DFA) allows us to estimate
Hurst paramter of a given vector.

Requirements:
    1. GSL - GNU Scientific Library : https://www.gnu.org/software/gsl/

To compile include following flags:
    -I *gsl_path*\gsl\include\,
    -L *gsl_path*\gsl\lib\,
    -I *gsl_path*\gsl\bin\,
    -lgsl

Basic usage:
    DFA Vector.txt

Expected output:
    Input vector: Vector.txt
    H estimated: 1.23752
    Elapsed: 0.007
*/

#include <stdio.h>
#include <gsl/gsl_multifit.h>
#include <stdbool.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;

vector<double> polynomialfit(int obs, int degree, vector<double> dx, vector<double> dy)
{
  /*
    RETURNS LOWEST DEGREE FIRST!
  */
    vector<double> store;

    gsl_multifit_linear_workspace *ws;
    gsl_matrix *cov, *X;
    gsl_vector *y, *c;
    double chisq;

    int i, j;

    X = gsl_matrix_alloc(obs, degree);
    y = gsl_vector_alloc(obs);
    c = gsl_vector_alloc(degree);
    cov = gsl_matrix_alloc(degree, degree);

    for(i=0; i < obs; i++) {
        for(j=0; j < degree; j++) {
        gsl_matrix_set(X, i, j, pow(dx[i], j));
        }
        gsl_vector_set(y, i, dy[i]);
    }

    ws = gsl_multifit_linear_alloc(obs, degree);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);

    for(i=0; i < degree; i++)
    {
        store.push_back(gsl_vector_get(c, i));
    }

    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);
    return store;
}

vector<double> polyval(vector<int> dx, vector<double> coeffs, int degree){

    vector<double> result;

    if (degree == 2)
    {
        for (int i : dx){
        result.push_back(coeffs[0] + coeffs[1] * i + coeffs[2] * i * i);
        }
    }
    else if (degree == 1)
    {
        for (int i : dx){
        result.push_back(coeffs[0] + coeffs[1] * i);
        }
    }

    return result;
}


vector<double> cumsum(vector<double> input_vector){
    vector<double> cumsum_vector(input_vector.size());
    cumsum_vector[0] = input_vector[0];
    for (int i=1; i<input_vector.size(); i++){
        cumsum_vector[i] = cumsum_vector[i-1] + input_vector[i];
    }

    return cumsum_vector;
}


double DFA(vector<double> arr, int degree, bool root){

    int s_max = (int)(arr.size() / 4);

    vector<double> log_s_max;

    for (double n=1.6; n<log(s_max); n+=0.5){
        log_s_max.push_back(n);
    }

    double arr_mean = 0;
    for (double n : arr){
        arr_mean += n;
    }
    arr_mean /= arr.size();

    for (int i=0; i< arr.size(); i++){
        arr[i] -= arr_mean;
    }

    vector<double> Y_cumsum = cumsum(arr);

    vector<double> x_Axis;
    vector<double> y_Axis;

    for (double step: log_s_max){

        int floor_step = floor(exp(step));

        vector<int> s;
        for (int i=1; i <= floor_step; i ++){
            s.push_back(i);
        }
        vector<int> indices;
        vector<double> indices_double;
        for (int s_i: s){
            indices.push_back(0);
            indices_double.push_back(0.0);
        }

        vector<double> Y_cumsum_s;
        for (int s_i:s){
            Y_cumsum_s.push_back(0.0);
        }

        int cycles_amount = floor(arr.size() / s.size());

        double F_q_s_sum = 0;

        for (int i=1; i < cycles_amount; i++){

            int _val;
            for (int k=0; k < s.size(); k++){
                _val = s[k] - (i + 0.5) * s.size();
                indices[k] = _val;
                indices_double[k] = (double)_val;
                Y_cumsum_s[k] = Y_cumsum[s[k]];
            }

            vector<double> coeffs = polynomialfit(indices.size(), degree+1, indices_double, Y_cumsum_s);

            vector<double> current_trend = polyval(indices, coeffs, degree);

            vector<double> F_2;
            for (int i=0; i < Y_cumsum_s.size(); i ++){
                F_2.push_back(pow((Y_cumsum_s[i] - current_trend[i]), 2));
            }

            double F_2_sum = 0;
            for (double y : F_2){
                F_2_sum += y;
            }
            F_2_sum /= s.size();
            F_q_s_sum += pow(F_2_sum, (degree / 2));

            for (int j=0; j < s.size(); j ++){
                s[j] += floor_step;
            }

            int kk = 1;
        }

        double F1 = pow(((1.0 / cycles_amount) * F_q_s_sum), (1.0/degree));


        x_Axis.push_back(log(floor_step));

        if (root){
            y_Axis.push_back(log(F1 / sqrt(s.size())));
        }
        else{
            y_Axis.push_back(log(F1));
        }
    }

    vector<double> last_fit = polynomialfit(x_Axis.size(), 2, x_Axis, y_Axis);

    return last_fit[1];
}


int main(int argc, char *argv[])
{
    if (argc > 1){
        std::cout << "Input vector: "<<argv[1] << endl;
        vector<double> numbers;
        ifstream inputFile(argv[1]);

        if (inputFile.good()) {
            double current_number;
            while (inputFile >> current_number){
                numbers.push_back(current_number);
            }
            inputFile.close();
        }

        int degree = 2;

        double h;
        auto t1 = chrono::high_resolution_clock::now();
        h = DFA(numbers, degree, false);
        auto t2 = chrono::high_resolution_clock::now();

        auto elapsed = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
        std::cout << "H estimated: "<<h << endl;
        std::cout << "Elapsed: " <<elapsed.count() / 1000.0<< endl;

        return 0;
    }
    else{
        std::cout<< "NO INPUT FILE!" << std::endl;
    }
}
