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
    template <typename Precision, typename Time_derivative>
    void forward_euler(ublas::matrix<Precision>& X, Time_derivative const& time_deriv, Precision T) {
        size_t N = X.size1() - 1;
        size_t K = X.size2();
        Precision delta_t = T / N;
        ublas::vector<Precision> dx(K);
        for (size_t n = 0; n < N ; ++n) {
            ublas::matrix_row<ublas::matrix<Precision>>next_x(X, n+1);
            ublas::matrix_row<ublas::matrix<Precision>>prev_x(X, n);
            time_deriv(prev_x, dx);
            next_x = prev_x + delta_t * dx;
        }
    }

    template <typename Precision, typename Time_derivative>
    void heun_method(ublas::matrix<Precision>& X, Time_derivative const& time_deriv, Precision T) {
        size_t N = X.size1() - 1;
        size_t K = X.size2();
        Precision delta_t = T / N;
        ublas::vector<Precision> dx1(K);
        ublas::vector<Precision> dx2(K);

        for (size_t n = 0; n < N; ++n) {
            ublas::matrix_row<ublas::matrix<Precision>>next_x(X, n+1);
            ublas::matrix_row<ublas::matrix<Precision>>prev_x(X, n);

            time_deriv(prev_x, dx1);
            time_deriv(prev_x + delta_t * dx1, dx2);
            next_x = prev_x + delta_t / 2 * (dx1 + dx2);
        }
    }

    template <typename Precision, typename Time_derivative, typename Error_Jacobian>
    void backward_euler(ublas::matrix<Precision>& X, Time_derivative const& time_deriv, Error_Jacobian const& error_jacob, Precision T) {
        size_t N = X.size1() - 1;
        size_t K = X.size2();
        double conv_crit = 1e-10;
        Precision delta_t = T / N;
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
            guessed_x = prev_x + delta_t * dx;
            time_deriv(guessed_x, dx);
            error = prev_x + delta_t * dx - guessed_x; 

            while (norm_2(error) / norm_2(guessed_x) > conv_crit) {
                error_jacob(guessed_x, delta_t, jacobian);

                pm = pm_copy;
                ublas::lu_factorize(jacobian, pm);
                ublas::lu_substitute(jacobian, pm, error);
                guessed_x -= error;
                time_deriv(guessed_x, dx);
                error = prev_x + delta_t * dx - guessed_x; 
            }

            next_x = guessed_x;
        }

    }

}



#endif
