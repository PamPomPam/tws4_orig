#ifndef IO_hpp
#define IO_hpp

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>
#include <fstream>

namespace IO {
    template <typename Precision>
    void write_siqrd(ublas::matrix<Precision> SIQRD, std::string filename, Precision T, int N) {
        // write output from siqrd_model to a file in the correct format
        assert(SIQRD.size1() == N+1);
        assert(SIQRD.size2() == 5);
        std::ofstream outputfile(filename);
        for (size_t i = 0; i < N + 1; ++i) {
            outputfile << i * T / N << ' ';
            for (size_t j = 0; j < 5; ++j) { outputfile << SIQRD(i, j) << ' ';}
            outputfile << std::endl;
        }
    }

    
    template <typename Precision>
    void write_sim2(ublas::matrix<Precision> X, std::string filename, Precision T, int N) {
        // writes output from simulation2 in the correct format to a file
        assert(X.size1() == N+1);
        assert(X.size2() == 50);
        std::ofstream outputfile(filename);
        for (size_t i = 0; i < N + 1; ++i) {
            if (N % 100 == 0) { // only print when T = 0, 1, 2, 3, ...
                outputfile << i * T / N << ' ';
                outputfile << X(i, 0) << ' ' << X(i, 24) << ' ' << X(i, 49) << std::endl;
            }
        }
    }


    template <typename Precision>
    ublas::matrix<Precision> load_observations(std::string const& filename, int & N, int & m, Precision& T) {
        // loads in data from file with name 'filename'
        // and stores it in the matrix that it returns
        // N will contain the number of timesteps, m = 5 the length of x for a single timestep
        // T will contain the time of the last observation

        std::ifstream file(filename);
        std::string line, element;
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> element;
        N = stoi(element);
        iss >> element;
        m = stoi(element);
        ublas::matrix<Precision> observations(N, m);
        int n_ = 0;
        int m_ = 0;
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
        return observations;
    }

}

#endif
