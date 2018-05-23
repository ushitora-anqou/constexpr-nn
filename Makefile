main: main.cpp test.cpp
	g++ -std=c++17 -g -O0 -o $@ main.cpp -Wall

