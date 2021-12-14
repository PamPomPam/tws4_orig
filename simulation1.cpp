#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>

#include <iostream>
#include <cassert>
#include <fstream>
#include <chrono>

#include "siqrd.hpp"
#include "integration.hpp"

namespace ublas = boost::numeric::ublas;


template <typename Precision>
void write_to_file(ublas::matrix<Precision> SIQRD, std::string filename, Precision T, int N) {
    assert(SIQRD.size1() == N+1);
    assert(SIQRD.size2() == 5);
    std::ofstream outputfile(filename);
    for (size_t i = 0; i < N + 1; ++i) {
        outputfile << i * T / N << ' ';
        for (size_t j = 0; j < 5; ++j) { outputfile << SIQRD(i, j) << ' ';}
        outputfile << std::endl;
    }
}


int main(int argc, char *argv[]) {
    assert(argc == 3);
    int N = std::stoi(argv[1]);
    double T = std::stof(argv[2]);

    ublas::matrix<double> SIQRD(N + 1, 5);
    ublas::matrix_row<ublas::matrix<double>> initial_conditions(SIQRD, 0);
    initial_conditions <<= 100, 5, 0, 0, 0;

    // beta, mu, gamma, alpha, delta, S0, I0
    ublas::vector<double> p(5);
    p <<= 0.5, 0, 0.2, 0.005, 0;
    //p <<= 10.0, 0.0, 10.0, 1.0, 0.0; // can be used to compare to fortran code
    
    auto t_start = std::chrono::high_resolution_clock::now();
    siqrd::time_deriv<double> myderiv(p);
    siqrd::error_jacob<double> myjacob(p);
    for (int i =0; i < 100; ++i) {
        integration::backward_euler(SIQRD, myderiv, myjacob, T);
        //write_to_file(SIQRD, "fwe_no_measures.txt", T, N);
    
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << " - Execution time: " << std::chrono::duration<double>(t_end-t_start).count() << std::endl;
    
    
    /*
    p(4) = 0.2;
    myderiv.change_p(p);
    siqrd::error_jacob<double> myjacob(p);
    integration::backward_euler(SIQRD, myderiv, myjacob, T);
    write_to_file(SIQRD, "bwe_quarantine.txt", T, N);

    p(4) = 0.9;
    myderiv.change_p(p);
    integration::heun_method(SIQRD, myderiv, T);
    write_to_file(SIQRD, "heun_lockdown.txt", T, N);
    */
}