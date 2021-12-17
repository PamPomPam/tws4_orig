#ifndef integration_hpp
#define integration_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>
#include <type_traits>

namespace ublas = boost::numeric::ublas;


namespace integration {
    template <typename Precision, typename Time_derivative>
    void forward_euler(ublas::vector<Precision> const& prev_x, ublas::vector<Precision> & next_x, \
     Time_derivative const& time_deriv, Precision const& delta_t) {
        /*
        Takes in a vector prev_x (size K) which is the state at a moment in time
        and fills up vector next_x (also size K) with the next state

        time_deriv should be function that takes in 2 vectors, x and dx, of size K
        and assigns to dx the derivative of x with respect to time
        */
        time_deriv(prev_x, next_x);
        next_x = prev_x + delta_t * next_x;
    }

    template <typename Precision, typename Time_derivative>
    void heun_method(ublas::vector<Precision> const& prev_x, ublas::vector<Precision> & next_x, \
     Time_derivative const& time_deriv, Precision const& delta_t) {
        // same as forward_euler, but using heun's method instead of the forward euler one
        ublas::vector<Precision> dx;

        time_deriv(prev_x, dx);
        time_deriv(prev_x + delta_t * dx, next_x);
        next_x = prev_x + delta_t / 2 * (dx + next_x);
    }
    
    template <typename Precision, typename Time_derivative>
    void backward_euler(ublas::vector<Precision> const& prev_x, ublas::vector<Precision> & next_x, Time_derivative const& time_deriv, \
     void (*error_jacob) (ublas::vector<Precision>const&, Precision const&, ublas::matrix<Precision>&), Precision const& delta_t) {
        /*
        same as forward_euler/heun_method, but using backward euler
        Newton's method is used to solve the system of nonlinear equations
        Newton's method makes use of the jacobi-matrix, which is calculated using error_jacobian

        error_jacobian takes in a vector of size K (x), the timestep delta_t and a matrix of size (K x K)
        and fills up that matrix with the jacobi-matrix of (X_k + delta_t * X_(K+1)' - X_(k+1)) with respect to X_(k+1)
        */ 
        size_t K = prev_x.size();

        ublas::vector<Precision> dx(K);
        ublas::vector<Precision> guessed_x(K);
        ublas::vector<Precision> error(K);
        ublas::matrix<Precision> jacobian(K, K);
        ublas::permutation_matrix<size_t> pm(K);
        ublas::permutation_matrix<size_t> pm_copy(K);

        time_deriv(prev_x, dx);
        guessed_x = prev_x + delta_t * dx;
        time_deriv(guessed_x, dx);
        error = prev_x + delta_t * dx - guessed_x; 
        while (norm_2(error) / norm_2(guessed_x) > 1e-10) {
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

    template <typename Precision, typename Time_derivative>
    void backward_euler(ublas::vector<Precision> const& prev_x, ublas::vector<Precision> & next_x, \
     Time_derivative const& time_deriv, void (*error_jacob) (ublas::vector<Precision>const&, Precision const&, ublas::vector<Precision>&), Precision const& delta_t) {
        size_t K = prev_x.size();

        ublas::vector<Precision> dx(K);
        ublas::vector<Precision> guessed_x(K);
        ublas::vector<Precision> error(K);
        ublas::vector<Precision> jacobian(K, K);

        time_deriv(prev_x, dx);
        guessed_x = prev_x + delta_t * dx;
        time_deriv(guessed_x, dx);
        error = prev_x + delta_t * dx - guessed_x; 
        while (norm_2(error) / norm_2(guessed_x) > 1e-10) {
            error_jacob(guessed_x, delta_t, jacobian);
            guessed_x -= element_div(error, jacobian);
            time_deriv(guessed_x, dx);
            error = prev_x + delta_t * dx - guessed_x; 
        }
        next_x = guessed_x;
    
    }

}


#endif
