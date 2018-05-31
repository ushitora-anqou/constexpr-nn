main: main.cpp test.cpp mnist.cpp
	g++ -std=c++17 -g -O0 -o $@ main.cpp -Wall -I./vendor/Sprout -fconstexpr-depth=1024

