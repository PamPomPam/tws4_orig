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
    Precision Line_search(Function & f, Inputvector const& x0, Inputvector const& dx, Inputvector const& grad_f, Precision& f0) {
        Precision eta = 1;
        Precision c1 = 1e-4;
        Precision new_f;

        new_f = f(x0 + eta * dx);
        std::cout << "new linesearch: initial eval " << new_f  << " with target " << f0 + c1 * eta * ublas::inner_prod(grad_f, dx) << std::endl;
        std::cout << "getting crit as " << f0 << ' ' << ublas::inner_prod(grad_f, dx) << std::endl;
        std::cout << "more specifically" << f0 << ' ' << c1 * eta * ublas::inner_prod(grad_f, dx) << std::endl;
        while (new_f > f0 + c1 * eta * ublas::inner_prod(grad_f, dx)) {
            eta /= 2;
            new_f = f(x0 + eta * dx);
            std::cout << "new eta + eval " << eta << ' ' << new_f  << " with target" << f0 + c1 * eta * ublas::inner_prod(grad_f, dx) << std::endl;
        }
        f0 = new_f;
        
        return eta;
    }


    template <typename Precision, typename Function>
    void BFGS(Function f, ublas::vector<Precision> x) {
        size_t m = x.size();

        ublas::identity_matrix<Precision> I(m);
        ublas::matrix<Precision> B(I);
        ublas::matrix<Precision> B_copy(m, m);
        
        ublas::permutation_matrix<size_t> pm(m);
        ublas::permutation_matrix<size_t> pm_copy(m);
        ublas::vector<Precision> search_dir(m);
        ublas::vector<Precision> gradient(m);
        ublas::vector<Precision> new_gradient(m);

        Precision error;
        ublas::vector<Precision> y(m);
        ublas::vector<Precision> s(m);
        
        Precision tol = 1e-7;
        Precision eta;

        ublas::matrix<Precision> temp_m(m, m);
        ublas::matrix<Precision> temp_m2(m, m);
        ublas::vector<Precision> temp_v(m);
        Precision temp_c;
        Precision temp_c2;


        error = f(x);

        search_dir = finite_diff_gradient(f, x, error);
        gradient = search_dir;
        std::cout << "grad" << gradient << std::endl;

        auto t_start = std::chrono::high_resolution_clock::now();

        int k = 0;
        s <<= 1000000;
        while (ublas::norm_2(s) / ublas::norm_2(x) > tol) {
            std::cout << std::endl << "New iteration:" << std::endl;
            
            pm = pm_copy;
            B_copy = B;
            ublas::lu_factorize(B, pm);
            ublas::lu_substitute(B, pm, search_dir);
            B = B_copy;
            search_dir = -search_dir;
            std::cout << "searchdir: " << search_dir << std::endl;

            eta = Line_search(f, x, search_dir, gradient, error);
            std::cout << "eta: " << eta << std::endl;
            s = eta * search_dir;
            x += s;

            search_dir = finite_diff_gradient(f, x, error);
            std::cout << "grad: " << search_dir << std::endl;
            y = search_dir - gradient;
            gradient = search_dir;


            temp_v = prod(B, s);
            temp_m = outer_prod(temp_v, temp_v);
            temp_c = inner_prod(s, temp_v);

            temp_m2 = outer_prod(y, y);
            temp_c2 = inner_prod(y, s);

            


            B = B - (temp_m / temp_c) + (temp_m2 / temp_c2);
            //std::cout << "guessed B " << B << std::endl;
            //for (size_t i = 0; i < m; ++i) {
            //    ublas::matrix_row<ublas::matrix<Precision>> lala(B, i);
            //    std::cout << lala << std::endl;
            //}
            
            k++;
            std::cout << "guessed x + error" << x << " " << error << std::endl;
        }

        auto t_end = std::chrono::high_resolution_clock::now();


        std::cout << std::endl << " - Number of BFGS iterations: " << k << std::endl;
        std::cout << " - Execution time: " << std::chrono::duration<double>(t_end-t_start).count() << "seconds" << std::endl;
        std::cout << " - Obtained parameters: (" << x(0);
        for (size_t i = 1; i < m; ++i) {std::cout << "," << x(i);}
        std::cout << ")" << std::endl;

    }

}


#endif
