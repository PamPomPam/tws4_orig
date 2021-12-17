C = g++ # compiler
FLAGS = -O3 -funroll-loops -march=native -std=c++17

default: simulation2

simulation2: integration.hpp simulation2.cpp
	$(C) $(FLAGS) -o simulation2 simulation2.cpp

clean:
	rm -f extra_sim2.txt simulation2 a.out
