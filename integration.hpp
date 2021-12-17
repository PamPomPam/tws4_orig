#ifndef integration_hpp
#define integration_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>

namespace ublas = boost::numeric::ublas;


namespace integration {

    enum method {fwe, bwe, heun};


    template <typename Precision, typename Time_derivative>
    void forward_euler(ublas::matrix<Precision>& X, Time_derivative const& time_deriv, Precision delta_t) {
        /*
        Takes in a matrix X (dimension N x K) where each row represents the state at a moment in time
        the first row (x) should already contain the initial condition
        this function fills all the other rows using timederivative time_deriv and timestep delta_t
        (using the forward euler method)

        time_deriv should be function that takes in 2 vectors, x and dx, of size K
        and assigns to dx the derivative of x with respect to time
        */
        size_t N = X.size1() - 1;
        size_t K = X.size2();
        ublas::vector<Precision> dx(K);
        for (size_t n = 0; n < N ; ++n) {
            ublas::matrix_row<ublas::matrix<Precision>>next_x(X, n+1);
            ublas::matrix_row<ublas::matrix<Precision>>prev_x(X, n);
            time_deriv(prev_x, dx);
            next_x.assign(prev_x + delta_t * dx);
        }
    }

    template <typename Precision, typename Time_derivative>
    void heun_method(ublas::matrix<Precision>& X, Time_derivative const& time_deriv, Precision delta_t) {
        // same as forward_euler, but using heun's method instead of the forward euler one
        size_t N = X.size1() - 1;
        size_t K = X.size2();
        ublas::vector<Precision> dx1(K);
        ublas::vector<Precision> dx2(K);

        for (size_t n = 0; n < N; ++n) {
            ublas::matrix_row<ublas::matrix<Precision>>next_x(X, n+1);
            ublas::matrix_row<ublas::matrix<Precision>>prev_x(X, n);

            time_deriv(prev_x, dx1);
            time_deriv(prev_x + delta_t * dx1, dx2);
            next_x.assign(prev_x + delta_t / 2 * (dx1 + dx2));
        }
    }

    template <typename Precision, typename Time_derivative, typename Error_Jacobian>
    void backward_euler(ublas::matrix<Precision>& X, Time_derivative const& time_deriv, Error_Jacobian const& error_jacob, Precision delta_t) {
        /* same as forward_euler/heun_method, but using backward euler
        Newton's method is used to solve the system of nonlinear equations
        Newton's method makes use of the jacobi-matrix, which is calculated using error_jacobian

        error_jacobian takes in a vector of size K (x), the timestep delta_t and a matrix of size (K x K)
        and fills up that matrix with the jacobi-matrix of (X_k + delta_t * X_(K+1)' - X_(k+1)) with respect to X_(k+1)
        */ 
        
        size_t N = X.size1() - 1;
        size_t K = X.size2();
        double conv_crit = 1e-8;
        ublas::vector<Precision> dx(K);
        ublas::vector<Precision> guessed_x(K);
        ublas::vector<Precision> error(K);
        ublas::matrix<Precision> jacobian(K, K);
        ublas::permutation_matrix<size_t> pm(K);
        ublas::permutation_matrix<size_t> pm_copy(K);
 
        for (size_t n = 0; n < N; ++n) {
            ublas::matrix_row<ublas::matrix<Precision>> next_x(X, n+1);
            ublas::matrix_row<ublas::matrix<Precision>> prev_x(X, n);

            time_deriv(prev_x, dx);
            guessed_x.assign(prev_x + delta_t * dx);
            time_deriv(guessed_x, dx);
            error.assign(prev_x + delta_t * dx - guessed_x); 

            while (norm_2(error) / norm_2(guessed_x) > conv_crit) {
                error_jacob(guessed_x, delta_t, jacobian);

                pm.assign(pm_copy);
                ublas::lu_factorize(jacobian, pm);
                ublas::lu_substitute(jacobian, pm, error);
                guessed_x -= error;
                time_deriv(guessed_x, dx);
                error.assign(prev_x + delta_t * dx - guessed_x); 
            }
            next_x.assign(guessed_x);
        }
    }
}


#endif
