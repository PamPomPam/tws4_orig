
C = g++ # compiler
FLAGS = -O3 -funroll-loops -march=native -std=c++17


default: simulation1 simulation2 estimation1 estimation2

simulation1: integration.hpp siqrd.hpp simulation1.cpp
	$(C) $(FLAGS) -o simulation1 simulation1.cpp
simulation2: integration.hpp simulation2.cpp
	$(C) $(FLAGS) -o simulation2 simulation2.cpp
estimation1: integration.hpp siqrd.hpp optimization.hpp estimation1.cpp
	$(C) $(FLAGS) -o estimation1 estimation1.cpp
estimation2: integration.hpp siqrd.hpp optimization.hpp estimation2.cpp
	$(C) $(FLAGS) -o estimation2 estimation2.cpp
