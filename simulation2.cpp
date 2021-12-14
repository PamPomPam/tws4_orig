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

namespace ublas = boost::numeric::ublas;

template <typename Precision>
void function_time_deriv(ublas::vector<Precision> const& x, ublas::vector<Precision> & dx ) {
    dx = x;
    Precision m = -0.1;
    auto f = [&m] (Precision const& element) -> Precision {m += 0.1; return -10 * std::pow(element - m, 3);};
    std::transform(dx.begin(), dx.end(), dx.begin(), f);
}



template <typename Precision>
struct functor_time_deriv {
    functor_time_deriv() : myvec{50} {
        std::iota(myvec.begin(), myvec.end(), 0);
        myvec *= 0.1;
        std::cout << myvec << std::endl;
    }

    template <typename V1, typename V2>
    void operator()(V1 const& x, V2 &  dx) const {
        dx = x - myvec;
        dx = - 10. * element_prod(element_prod(dx, dx), dx);
    }

    ublas::vector<Precision> myvec;
};



template <typename Precision>
struct functor_error_jacobian {
    functor_error_jacobian() : myvec{50} {
        std::iota(myvec.begin(), myvec.end(), 0);
        myvec *= 0.1;
        std::cout << myvec << std::endl;
    }

    template <typename V, typename M>
    void operator()(V const& x, Precision const& delta_t, M & jacob) const {
        
        Precision temp;
        for (size_t i = 0; i < 50; ++i) {
            for (size_t j = 0; j < 50; ++j) {
                if (i != j) {
                    jacob(i, j) = 0;
                } else {
                    jacob(i, j) = -30 * pow(x(i) - myvec(i), 2) * x(i) - 1;
                }
            }
        }

        //ublas::vector<Precision> dx = x - myvec;
        //dx = - 30. * delta_t * element_prod(element_prod(dx, dx), x) - ublas::scalar_vector<Precision>(50) ;
        //ublas::diagonal_matrix<Precision> jacobian(dx.size(), dx.data());

        std::cout << "jacobian: " << jacob << std::endl;
    }

    ublas::vector<Precision> myvec;
};





int main(int argc, char *argv[]) {
    assert(argc == 3);
    int N = std::stoi(argv[1]);
    double T = std::stod(argv[2]);

    // beta, mu, gamma, alpha, delta, S0, I0
    ublas::vector<double> p(5);
    //p <<= 0.5, 0, 0.2, 0.005, 0;
    p <<= 10.0, 0.0, 10.0, 1.0, 0.0;

    ublas::matrix<double> X(N + 1, 50);
    ublas::matrix_row<ublas::matrix<double>> initial_conditions(X, 0);
    std::iota(initial_conditions.begin(), initial_conditions.end(), 1);
    initial_conditions *= 0.01;


    auto lambda_time_deriv = [] ( \
    ublas::matrix_row<ublas::matrix<double>> const& x, ublas::vector<double> & dx) -> void {
        for (size_t i = 0; i < 50; ++i) {
            dx(i) = - 10 * std::pow(x(i) - 0.1 * i, 3);
        }
    };


    functor_time_deriv<double> d2;



    integration::forward_euler(X, d2, T);

    for (size_t i = 0; i < N + 1; ++i) {
        std::cout << std::endl << i * T / N << ' ';
        for (size_t j = 0; j < 5; ++j) { std::cout << X(i, j) << ' ';}
    }
    std::cout << std::endl;


    integration::forward_euler(X, lambda_time_deriv, T);
    
    //integration::heun_method(SIQRD, myderiv, T);
    //integration::heun_method(SIQRD, derivative5, T);
    //backward_euler(SIQRD, myderiv, myjacobian, T);
    


    // print results
    for (size_t i = 0; i < N + 1; ++i) {
        std::cout << std::endl << i * T / N << ' ';
        for (size_t j = 0; j < 5; ++j) { std::cout << X(i, j) << ' ';}
    }
    std::cout << std::endl;

}