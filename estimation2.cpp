
/* this program can be executed with the following commands:

make estimation2
./estimation2

*/



#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>

#include <iostream>
#include <cassert>
#include <fstream>
#include <iomanip>

#include "siqrd.hpp"
#include "integration.hpp"
#include "optimization.hpp"
#include "IO.hpp"

namespace ublas = boost::numeric::ublas;

int main(int argc, char *argv[]) {
    typedef double Precision; // the precision of floating point numbers
    int N;
    int m;
    Precision T;
    ublas::matrix<Precision> observations(IO::load_observations("observations1.in", N, m, T));

    // initial guess for model parameters
    ublas::vector<Precision> p0(5);
    p0 <<= 0.32, 0.03, 0.151, 0.004, 0.052;

    ublas::identity_matrix<Precision> I(5); 

    // warmup
    std::cout << "Warm-up:" << std::endl;
    siqrd::LSE<Precision> my_eval_func_bwe(observations, T, integration::bwe, 8); // objective function backward euler
    optimization::BFGS(my_eval_func_bwe, p0, I, 1e-7, 1., 1e-4);
    std::cout << "------------------------------------" << std::endl;
    for (int i = 0; i < 5; ++i) {std::cout << std::endl;}


    std::cout << "Forward Euler:" << std::endl;
    p0 <<= 0.32, 0.03, 0.151, 0.004, 0.052;
    siqrd::LSE<Precision> my_eval_func_fwe(observations, T, integration::fwe, 8); // objective function forward euler
    optimization::BFGS(my_eval_func_fwe, p0, I, 1e-7, 1., 1e-4);


    std::cout << "------------------------------------" << std::endl;
    std::cout << "Backward Euler:" << std::endl;
    p0 <<= 0.32, 0.03, 0.151, 0.004, 0.052;
    optimization::BFGS(my_eval_func_bwe, p0, I, 1e-7, 1., 1e-4);
    

    std::cout << "------------------------------------" << std::endl;
    std::cout << "Heun's method:" << std::endl;
    p0 <<= 0.32, 0.03, 0.151, 0.004, 0.052;
    siqrd::LSE<Precision> my_eval_func_heun(observations, T, integration::heun, 8); // objective function heun method
    optimization::BFGS(my_eval_func_heun, p0, I, 1e-7, 1., 1e-4);

}

