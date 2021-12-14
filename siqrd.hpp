#ifndef siqrd_hpp
#define siqrd_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>
#include <cmath>

#include "integration.hpp"

namespace ublas = boost::numeric::ublas;


namespace siqrd {

    template <typename Precision>
    struct time_deriv {
    public:
        time_deriv(ublas::vector<Precision> const& p) : p_(p) {}

        template <typename V1, typename V2>
        void operator()(V1 const& x, V2& dx) const {
            assert(x.size() == 5);
            dx(0) = - p_(0) * x(0) * x(1) / (x(0) + x(1) + x(3)) + p_(1) * x(3);
            dx(1) = (p_(0) * x(0) / (x(0) + x(1) + x(3)) - p_(2) - p_(4) - p_(3)) * x(1);
            dx(2) = p_(4) * x(1) - (p_(2) + p_(3)) * x(2);
            dx(3) = p_(2) * (x(1) + x(2)) - p_(1) * x(3);
            dx(4) = p_(3) * (x(1) + x(2));
        }

        template <typename V>
        void change_p(V const& p) {p_ = p;}

    private:
        ublas::vector<Precision> p_;
    };
    

    template <typename Precision>
    struct error_jacob {
    public:
        error_jacob(ublas::vector<Precision> p) : p_(p) {}

        template <typename V, typename M>
        void operator()(V const& x, Precision const& delta_t, M & jacob) const {
            assert(x.size() == 5);
            Precision c1 = delta_t * p_(0) * x(1) * x(0) / std::pow(x(0) + x(1) + x(3), 2);
            Precision c2 = delta_t * p_(0) / (x(0) + x(1) + x(3));

            jacob <<= c1 - x(1) * c2 - 1., c1 - x(0) * c2, 0., c1 + p_(1) * delta_t, 0., \
                    x(1) * c2 - c1, x(0) * c2 - c1 - (p_(2) + p_(4) + p_(3)) * delta_t - 1., 0., -c1, 0., \
                    0., p_(4)*delta_t, -(p_(2)+p_(3)) * delta_t - 1., 0., 0., \
                    0., p_(2)*delta_t, p_(2)*delta_t, - p_(1)*delta_t - 1., 0., \
                    0., p_(3)*delta_t, p_(3)*delta_t, 0., - 1.;
        }
    private:
        ublas::vector<Precision> p_;
    };


    template <typename Precision>
    struct LSE {
    public:
        LSE(ublas::matrix<Precision> observations, Precision T) : observations_(observations), \
            T_(T), N1_(observations_.size1()), N2_((N1_-1) * 8), K_(observations_.size2()), \
            simulations_{N2_+1, K_} {
                ublas::matrix_row<ublas::matrix<Precision>> observ_initial(observations_, 0);
                ublas::matrix_row<ublas::matrix<Precision>> simul_initial(simulations_, 0);
                simul_initial = observ_initial;
                Precision P = sum(observ_initial);
                multiplication_constant_ = 1 / (P*P*T_);
            }
        template <typename Inputvector>
        Precision operator()(Inputvector const& p) {


            siqrd::time_deriv<Precision> myderiv(p);
            //siqrd::error_jacob<double> myjacob(p);

            integration::forward_euler(simulations_, myderiv, T_);
            //heun_method(SIQRD,myderiv, T);
            //backward_euler(SIQRD, myderiv, myjacob, T);

            Precision total = 0;
            for (size_t i = 0; i < N1_; ++i) {
                ublas::matrix_row<ublas::matrix<Precision>>observ(observations_, i);
                ublas::matrix_row<ublas::matrix<Precision>>simul(simulations_, i * 8);
                
                total += std::pow(ublas::norm_2(observ - simul), 2) * multiplication_constant_; // multiplication_constant_;
                
            }
            
            //total *= multiplication_constant_;
            if (std::isnan(total)) { total = std::numeric_limits<float>::infinity(); }
            std::cout << total << std::endl;
            std::cout << multiplication_constant_ << std::endl;
            std::cout << "tf" << std::endl;
            exit(0);
            return total;
        }
    private:
        size_t N1_;
        size_t N2_;
        size_t K_;
        Precision T_;
        Precision multiplication_constant_;
        ublas::matrix<Precision> observations_;
        ublas::matrix<Precision> simulations_;
        
    };



}; // namespace siqrd

#endif
