main: main.cpp test.cpp mnist.cpp
	clang++ -std=c++17 -g -O0 -o $@ main.cpp -Wall -I./vendor/Sprout \
		-fconstexpr-depth=2147483647 -fconstexpr-steps=2147483647
