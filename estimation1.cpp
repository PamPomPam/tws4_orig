
/* this program can be executed with the following commands:

make estimation1
./estimation1

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
    ublas::identity_matrix<Precision> I(5); 


    // initial guess for model parameters
    ublas::vector<Precision> p(5);
    p <<= 0.32, 0.03, 0.151, 0.004, 0.052;

    std::cout << "Observations1:" << std::endl;
    siqrd::LSE<Precision> my_eval_func1(observations, T, integration::heun, 8); // objective function
    optimization::BFGS(my_eval_func1, p, I, 1e-7, 1., 1e-4);

    // now simulate with the new-found optimal p
    // the easiest way to do this is to just evaluate my_eval_func again
    my_eval_func1(p);
    IO::write_siqrd(my_eval_func1.simulations_, "estimation_obs1.txt", T, (N-1)*8); 
    


    ublas::matrix<Precision> observations2(IO::load_observations("observations2.in", N, m, T));

    // initial guess for model parameters
    p <<= 0.5, 0.08, 0.04, 0.004, 0.09;
    
    std::cout << "Observations2:" << std::endl;
    siqrd::LSE<Precision> my_eval_func2(observations2, T, integration::heun, 8); // objective function
    optimization::BFGS(my_eval_func2, p, I, 1e-7, 1., 1e-4);

    // now simulate with the new-found optimal p
    // the easiest way to do this is to just evaluate my_eval_func again
    my_eval_func2(p);
    IO::write_siqrd(my_eval_func2.simulations_, "estimation_obs2.txt", T, (N-1)*8); 
    




}

