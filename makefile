default: simulation1 simulation2 estimation1

simulation1: integration.hpp siqrd.hpp simulation1.cpp
	g++ -o simulation1 simulation1.cpp
simulation2: integration.hpp simulation2.cpp
	g++ -o simulation2 simulation2.cpp
estimation1: integration.hpp siqrd.hpp optimization.hpp estimation1.cpp
	g++ -o estimation1 estimation1.cpp
estimation2:
