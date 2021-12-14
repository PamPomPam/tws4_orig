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

namespace ublas = boost::numeric::ublas;


/*template <typename Precision>
struct siqrd_LSE {
public:
    siqrd_LSE(ublas::matrix<Precision> observations, Precision T) : observations_(observations), T_(T) {}
    template <typename Inputvector>
    Precision operator()(Inputvector const& p) {

        //for (size_t i = 0; i < p.size(); ++i) {
        //    if (p(i) < -0.01) {
                return 1000000;
            }
        }

        size_t N1 = observations_.size1();
        size_t N2 = (N1-1) * 8;
        size_t m = observations_.size2();
        ublas::matrix<Precision> SIQRD(N2+1, m);

        ublas::matrix_row<ublas::matrix<Precision>>observ_initial(observations_, 0);
        ublas::matrix_row<ublas::matrix<double>> simul_initial(SIQRD, 0);
        simul_initial = observ_initial;
        Precision P = sum(observ_initial);

        siqrd::time_deriv<Precision> myderiv(p);
        //siqrd::error_jacob<double> myjacob(p);
        integration::forward_euler(SIQRD, myderiv, T_);
        //heun_method(SIQRD,myderiv, T);
        //backward_euler(SIQRD, myderiv, myjacob, T);
        for (size_t i = 0; i < N2 + 1; ++i) {
            std::cout << std::endl << i * T_ / N2 << ' ';
            for (size_t j = 0; j < 5; ++j) { std::cout << SIQRD(i, j) << ' ';}
        }
        std::cout << std::endl;
        ublas::matrix_row<ublas::matrix<Precision>>tempie(SIQRD, N2-1);
        //std::cout << "last row of siqrd: " << tempie << std::endl;
        //ublas::matrix_row<ublas::matrix<Precision>>tempie2(observations, N2-1);
        

        Precision total = 0;
        for (size_t i = 0; i < N1; ++i) {
            ublas::matrix_row<ublas::matrix<Precision>>observ(observations_, i);
            ublas::matrix_row<ublas::matrix<Precision>>simul(SIQRD, i * 8);
            std::cout << "Comparing: "<< std::endl << observ << std::endl;
            std::cout << simul << std::endl << std::endl;
            
            total += std::pow(ublas::norm_2(observ - simul), 2);
        }
        
        total /= P * P * T_;
        if (std::isnan(total)) {
            total = std::numeric_limits<float>::infinity();
        }
        std::cout << "total: " << total << std::endl;
        exit(0);
        return total;
    }
private:
    ublas::matrix<Precision> observations_;
    Precision T_;
};*/

int main(int argc, char *argv[]) {
    typedef double Precision;
    std::ifstream file("ownobservations.in");
    std::string line, element;

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> element;
    size_t N = stoi(element);
    iss >> element;
    size_t m = stoi(element);

    Precision T;

    ublas::matrix<Precision> observations(N, m);
    size_t n_ = 0;
    size_t m_ = 0;
    while (std::getline(file, line)) {
        m_ = 0;
        std::istringstream iss(line);
        iss >> element; // first argument is time: we don't need this
        if (n_ == N-1) {T = stod(element);}
        while (iss >> element) {
            observations(n_, m_) = stod(element);
            m_++;
        }
        n_++;
    }

    

    ublas::vector<Precision> x0(5);
    x0 <<= 11.0, 0.0, 10.0, 1.0, 0.5;
    //x0 <<= 0.32, 0.03, 0.151, 0.004, 0.052;

    siqrd::LSE<Precision> my_eval_func(observations, T);


    //std::cout << std::fixed;
    //std::cout << std::setprecision(15);

    optimization::BFGS(my_eval_func, x0);

}

