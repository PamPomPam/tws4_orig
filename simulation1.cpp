
/* this program can be executed with the following commands:

make simulation1
./simulation1 100 100

*/
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
#include "IO.hpp"

namespace ublas = boost::numeric::ublas;



int main(int argc, char *argv[]) {
    typedef double Precision; // the precision of floating point numbers

    assert(argc == 3);
    int N = std::stoi(argv[1]);
    Precision T = std::stof(argv[2]);
    Precision delta_t = T / N;

    ublas::matrix<Precision> SIQRD(N + 1, 5); // matrix that will store all x's at each moment
    ublas::matrix_row<ublas::matrix<Precision>> initial_conditions(SIQRD, 0);
    initial_conditions <<= 100, 5, 0, 0, 0;

    // beta, mu, gamma, alpha, delta = siqrd-model parameters
    ublas::vector<Precision> p(5);
    p <<= 0.5, 0, 0.2, 0.005, 0;
    //p <<= 10.0, 0.0, 10.0, 1.0, 0.0; // can be used to compare to fortran code
    

    siqrd::Time_deriv<Precision> myderiv_fwe(p);
    integration::forward_euler(SIQRD, myderiv_fwe, delta_t);
    IO::write_siqrd(SIQRD, "fwe_no_measures.txt", T, N);

    p(4) = 0.2;
    siqrd::Time_deriv<Precision> myderiv_bwe(p);
    siqrd::Error_jacob<Precision> myjacob_bwe(p);
    integration::backward_euler(SIQRD, myderiv_bwe, myjacob_bwe, delta_t);
    IO::write_siqrd(SIQRD, "bwe_quarantine.txt", T, N);

    p(4) = 0.9;
    siqrd::Time_deriv<Precision> myderiv_heun(p);
    integration::heun_method(SIQRD, myderiv_heun, delta_t);
    IO::write_siqrd(SIQRD, "heun_lockdown.txt", T, N);
}