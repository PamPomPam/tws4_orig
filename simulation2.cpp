
/* this program can be executed with the following commands:

make simulation2
./simulation2 50000 500

(this will probably take a very long time)
*/


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

#include "integration.hpp"
#include "IO.hpp"

namespace ublas = boost::numeric::ublas;

// function that can be used to calculate time derivative in the functions from integration.hpp
template <typename Precision>
void function_time_deriv(ublas::vector<Precision> const& x, ublas::vector<Precision> & dx ) {
    dx = x;
    Precision m = -0.1;
    auto f = [&m] (Precision const& element) -> Precision {m += 0.1; return -10 * std::pow(element - m, 3);};
    std::transform(dx.begin(), dx.end(), dx.begin(), f);
}

// functor that can be used to calculate time derivative in the functions from integration.hpp
template <typename Precision>
struct functor_time_deriv {
    functor_time_deriv() : myvec{50} {
        std::iota(myvec.begin(), myvec.end(), 0);
        myvec *= 0.1;
    }

    template <typename V1, typename V2>
    void operator()(V1 const& x, V2 &  dx) const {
        dx = x - myvec;
        dx = - 10. * element_prod(element_prod(dx, dx), dx);
    }

    ublas::vector<Precision> myvec;
};


// functor that can be used in the backward euler from integration.hpp
// calculates the jacobi-matrix from the backwards error
// = jacobi-matrix from (X_k + delta_t * X_(K+1)' - X_(k+1)) with respect to X_(k+1)
template <typename Precision>
struct functor_error_jacobian {
    functor_error_jacobian() : myvec{50} {
        std::iota(myvec.begin(), myvec.end(), 0);
        myvec *= 0.1;
    }

    template <typename V, typename M>
    void operator()(V const& x, Precision const& delta_t, M & jacob) const {
        for (size_t i = 0; i < 50; ++i) {
            for (size_t j = 0; j < 50; ++j) {
                if (i != j) {
                    jacob(i, j) = 0;
                } else {
                    jacob(i, j) = -30 * pow(x(i) - myvec(i), 2) * delta_t - 1;
                }
            }
        }
    }

    ublas::vector<Precision> myvec;
};




int main(int argc, char *argv[]) {
    typedef double Precision; // the precision of floating point numbers

    assert(argc == 3);
    size_t N = std::stoi(argv[1]);
    Precision T = std::stod(argv[2]);
    Precision delta_t = T / N;


    ublas::matrix<Precision> X(N + 1, 50);

    // initial conditions
    ublas::matrix_row<ublas::matrix<Precision>> initial_conditions(X, 0);
    std::iota(initial_conditions.begin(), initial_conditions.end(), 1);
    initial_conditions *= 0.01;

    // lambda expression that can be used to calculate time derivative in the functions from integration.hpp
    auto lambda_time_deriv = [] ( \
    ublas::vector<Precision> const& x, ublas::vector<Precision> & dx) -> void {
        for (size_t i = 0; i < 50; ++i) {
            dx(i) = - 10 * std::pow(x(i) - 0.1 * i, 3);
        }
    };

    // do forward euler simulation
    integration::forward_euler(X, function_time_deriv<Precision>, delta_t);
    IO::write_sim2(X, "fwe_sim2.txt", T, N);

    // instantiate functors 
    functor_time_deriv<Precision> deriv_inst;
    functor_error_jacobian<Precision> jacob_inst;

    // backward euler simulation
    integration::backward_euler(X, deriv_inst, jacob_inst, delta_t);
    IO::write_sim2(X, "bwe_sim2.txt", T, N);

    // heun method simulation
    integration::heun_method(X, lambda_time_deriv, delta_t);
    IO::write_sim2(X, "heun_sim2.txt", T, N);
}