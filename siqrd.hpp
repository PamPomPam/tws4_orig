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
    struct Time_deriv {
        // functor that can be used in the functions from integration.hpp
        // calculates the derivative according to the SIQRD-model
        // class member p_ is the set of parameters (beta, mu, gamma, alpha, delta)
    public:
        Time_deriv(ublas::vector<Precision> const& p) : p_(p) {}

        template <typename V1, typename V2>
        void operator()(V1 const& x, V2& dx) const {
            // x = [S, I, Q, R, D], dx will be [S', I', Q', R', D']
            assert(x.size() == 5);
            dx(0) = - p_(0) * x(0) * x(1) / (x(0) + x(1) + x(3)) + p_(1) * x(3);
            dx(1) = (p_(0) * x(0) / (x(0) + x(1) + x(3)) - p_(2) - p_(4) - p_(3)) * x(1);
            dx(2) = p_(4) * x(1) - (p_(2) + p_(3)) * x(2);
            dx(3) = p_(2) * (x(1) + x(2)) - p_(1) * x(3);
            dx(4) = p_(3) * (x(1) + x(2));
        }

    private:
        ublas::vector<Precision> p_;
    };
    

    template <typename Precision>
    struct Error_jacob {
        // functor that can be used in the backward euler function from integration.hpp
        // calculates the jacobi-matrix from the backwards error according to the SIQRD-model
        // = jacobi-matrix from (X_k + delta_t * X_(K+1)' - X_(k+1)) with respect to X_(k+1)
        // class member p_ is the set of parameters (beta, mu, gamma, alpha, delta)
    public:
        Error_jacob(ublas::vector<Precision> p) : p_(p) {}

        template <typename V, typename M>
        void operator()(V const& x, Precision const& delta_t, M & jacob) const {
            assert(x.size() == 5);

            // some constants that reappear in the jacobi-matrix
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
        // functor that can be used as an objective function for an minimization problem
        // calculates in function of p (=the five SIQRD-model parameters beta, mu, gamma, alpha, delta)
        // the least squares error between a set of given observations and a simulation given these parameters

        // the simulation happens with a time step of 1/8 day, while the observations have a timestep of 1 day
        
        // this function will be called multiple times, so arguments that remain the same 
        // such as the matrix of observations, the matrix dimensions, the initial conditions of the simulation
        // are stored as class members
    public:
        LSE(ublas::matrix<Precision> observations, Precision T, integration::method method, int nr_steps) :  observations_(observations), \
            N1_(observations.size1()) , N2_((N1_-1) * nr_steps), K_(observations.size2()), \
            simulations_{N2_+1, K_}, method_(method), nr_steps_(nr_steps) {
                // instantiate all member variables correctly

                // initial conditions for simulations should be same as observations
                ublas::matrix_row<ublas::matrix<Precision>> observ_initial(observations_, 0);
                ublas::matrix_row<ublas::matrix<Precision>> simul_initial(simulations_, 0);
                simul_initial.assign(observ_initial);
                
                Precision P = sum(observ_initial);
                multiplication_constant_ = 1 / (P*P*T);
                delta_t_ = T / N2_;
            }

        template <typename Inputvector>
        Precision operator()(Inputvector const& p) {
            // instantiate functor for time derivative
            Time_deriv<Precision> myderiv(p);

            // use correct time integration method
            switch(method_) {
            case integration::fwe:
                {
                    integration::forward_euler(simulations_, myderiv, delta_t_);
                    break;
                }
            case integration::bwe:
                {
                    Error_jacob<Precision> myjacob(p);
                    integration::backward_euler(simulations_, myderiv, myjacob, delta_t_);
                    break;
                }
            case integration::heun:
                {
                    integration::heun_method(simulations_, myderiv, delta_t_);
                    break;
                }
            default:
                std::cout << "invalid method" << std::endl;
                exit(1);
            }
            
            // now calculate the error between the simulation and the observations
            Precision total = 0;
            for (size_t i = 0; i < N1_; ++i) {
                ublas::matrix_row<ublas::matrix<Precision>>observ(observations_, i);
                ublas::matrix_row<ublas::matrix<Precision>>simul(simulations_, i * 8);

                total += std::pow(ublas::norm_2(observ - simul), 2);
            }
            
            // if total is nan (due to inrealistic parameters), change it to a large value
            if (std::isnan(total)) { total = 1e100; }

            return total * multiplication_constant_;
        }
    private:
        size_t N1_; // number of timesteps in oberservations
        size_t N2_; // number of timesteps in simulations
        size_t K_; // nr of columns of observations = 5 (S, I, Q, R, D)
        int nr_steps_; // number of steps per day 
        Precision delta_t_; // size of timestep 
        Precision multiplication_constant_; // 1 / (P * P * T_), with P population size, remains constant forever
        ublas::matrix<Precision> observations_;
        integration::method method_; // forward euler, backward euler or heun method
    public:
        ublas::matrix<Precision> simulations_;
    };


}; // namespace siqrd

#endif
