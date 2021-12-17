#ifndef optimization_hpp
#define optimization_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>
#include <chrono>

namespace ublas = boost::numeric::ublas;


namespace optimization {
    
    template <typename Precision, typename Function, typename Inputvector>
    ublas::vector<Precision> finite_diff_gradient(Function f, Inputvector x, Precision f0) {
        // approximates gradient df/dx using finite difference method
        // f takes in a vector of size K and returns a scalar
        // f0 is f(x) = scalar
        size_t m = x.size();
        Precision eps = 1e-8;
        ublas::vector<Precision> result(m);
        for (size_t i = 0; i < m; ++i) {
            x(i) += eps;
            result(i) = (f(x) - f0) / eps;
            x(i) -= eps;
        }
        return result;
    }


    template <typename Precision, typename Function, typename Inputvector>
    Precision Line_search(Function & f, Inputvector const& x0, Inputvector const& dx, \
     Inputvector const& grad_f, Precision & f0, Precision const& eta_initial, Precision const& line_search_c) {
        // performs line_search of objective function f from startpoint x0 (vector size K) in direction dx (vector size K)
        // using Wolfe condition

        // f takes in a vector of size K and returns a scalar
        // grad_f (gradient df/dx = vector size K) and f0 (f(x0) = scalar) are necessary to determine whether Wolfe condition is satisfied
        // eta_initial (scalar), the initial guess for the size of the step
        // line_search_c (scalar), determines how strict Wolfe condition is

        // returns scalar eta, such that eta * dx is the desired step
        // stores function value of new x in f0
        Precision eta = eta_initial;
        Precision new_f;
        
        new_f = f(x0 + eta * dx);
        while (new_f > f0 + line_search_c * eta * ublas::inner_prod(grad_f, dx)) {
            
            eta /= 2;
            new_f = f(x0 + eta * dx);
        } 
        f0 = new_f;
        return eta;
    }


    template <typename Precision, typename Function, typename M>
    void BFGS(Function & f, ublas::vector<Precision> & x, M const& initial_B, \
     Precision const& optim_tol, Precision const& eta_initial, Precision const& line_search_c) {
        // minimizes objective function f using BFGS-method with initial guess/starpoint x

        // x is a vector size m
        // f takes in a vector of size m and returns a scalar
        // initial_B is a matrix size (m x m) and is the initial guess for B
        // optim_tol (scalar) tolerance for BFGS-convergence criterion
        // eta_initial (scalar) parameter for line_search
        // line_search_c (scalar) parameter for line_search
 
        // changes x into it's optimal value
        size_t m = x.size();

        ublas::matrix<Precision> B(initial_B);
        ublas::matrix<Precision> B_copy(m, m);
        
        ublas::permutation_matrix<size_t> pm(m);
        ublas::permutation_matrix<size_t> pm_copy(m);
        ublas::vector<Precision> search_dir(m);
        ublas::vector<Precision> gradient(m);
        ublas::vector<Precision> new_gradient(m);

        Precision error;
        ublas::vector<Precision> y(m);
        ublas::vector<Precision> s(m);
        
        Precision eta;

        ublas::matrix<Precision> temp_m(m, m);
        ublas::matrix<Precision> temp_m2(m, m);
        ublas::vector<Precision> temp_v(m);
        Precision temp_c;
        Precision temp_c2;

        auto t_start = std::chrono::high_resolution_clock::now();
    
        error = f(x);
        search_dir.assign(finite_diff_gradient(f, x, error));
        gradient.assign(search_dir);

        
        int k = 0;
        s <<= 1e70; // makes sure 
        while (ublas::norm_2(s) / ublas::norm_2(x) > optim_tol) {

            pm.assign(pm_copy);

            B_copy.assign(B);

            ublas::lu_factorize(B, pm);
            ublas::lu_substitute(B, pm, search_dir);
            B.assign(B_copy);
            search_dir.assign(-search_dir);

            eta = Line_search(f, x, search_dir, gradient, error, eta_initial, line_search_c);
            s.assign(eta * search_dir);
            x += s;

            search_dir.assign(finite_diff_gradient(f, x, error));

            y.assign(search_dir - gradient);

            gradient.assign(search_dir);

            temp_v.assign(prod(B, s));
            temp_m.assign(outer_prod(temp_v, temp_v));
            temp_c = inner_prod(s, temp_v);

            temp_m2.assign(outer_prod(y, y));
            temp_c2 = inner_prod(y, s);

            B.assign(B - (temp_m / temp_c) + (temp_m2 / temp_c2));

            k++;
        }

        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << " - Number of BFGS iterations: " << k << std::endl;
        std::cout << " - Execution time: " << std::chrono::duration<double>(t_end-t_start).count() << " seconds" << std::endl;
        std::cout << " - Obtained parameters: (" << x(0);
        for (size_t i = 1; i < m; ++i) {std::cout << "," << x(i);}
        std::cout << ")" << std::endl << std::endl;

    }

}


#endif
