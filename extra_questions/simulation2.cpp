#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/banded.hpp>

#include <iostream>
#include <cassert>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "integration.hpp"

namespace ublas = boost::numeric::ublas;


template <typename Precision>
void time_deriv(ublas::vector<Precision> const& x, ublas::vector<Precision> & dx ) {
    dx = x;
    Precision m = -0.1;
    auto f = [&m] (Precision const& element) -> Precision {m += 0.1; return -10 * std::pow(element - m, 3);};
    std::transform(dx.begin(), dx.end(), dx.begin(), f);
}


template <typename Precision>
void matrix_error_jacobian(ublas::vector<Precision> const& x, Precision const& delta_t, ublas::matrix<Precision> & jacob) {
    for (size_t i = 0; i < 50; ++i) {
        for (size_t j = 0; j < 50; ++j) {
            if (i != j) {
                jacob(i, j) = 0;
            } else {
                jacob(i, j) = -30 * pow(x(i) - i * 0.1, 2) * delta_t - 1;
            }
        }
    }
}

template <typename Precision>
void vector_error_jacobian(ublas::vector<Precision> const& x, Precision const& delta_t, ublas::vector<Precision> & jacob) {
    for (size_t i = 0; i < 50; ++i) {
        jacob(i) = -30 * pow(x(i) - i * 0.1, 2) * delta_t - 1;
    }
}


int main(int argc, char *argv[]) {
    typedef double Precision;

    assert(argc == 3);
    size_t N = std::stoi(argv[1]);
    double T = std::stod(argv[2]);
    Precision delta_t = T / N;

    // start timer
    auto t_start = std::chrono::high_resolution_clock::now();

    // initial values
    ublas::vector<Precision> prev_x(50);
    std::iota(prev_x.begin(), prev_x.end(), 1);
    prev_x *= 0.01;

    ublas::vector<Precision> next_x(50);

    // do actual calculations, and write results immediately to the output file
    std::ofstream outputfile("extra_sim2.txt");
    for (size_t n = 0; n < N; ++n) {
        if (n % 100 == 0) { // only print when T = 0, 1, 2, 3, ...
            outputfile << n * T / N << ' ';
            outputfile << prev_x(0) << ' ' << prev_x(24) << ' ' << prev_x(49) << std::endl;
        }
        //integration::backward_euler(prev_x, next_x, time_deriv<Precision>, matrix_error_jacobian<Precision>, delta_t);
        integration::backward_euler(prev_x, next_x, time_deriv<Precision>, vector_error_jacobian<Precision>, delta_t);
        prev_x = next_x;
    }
    outputfile << T << ' ';
    outputfile << prev_x(0) << ' ' << prev_x(24) << ' ' << prev_x(49) << std::endl;


    // output timing results
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " << std::chrono::duration<double>(t_end-t_start).count() << " seconds" << std::endl;
        
}