main: main.cpp test.cpp mnist.cpp
	clang++ -std=c++17 -g -O0 -o $@ main.cpp -Wall -I./vendor/Sprout -I./vendor/stb \
		-fconstexpr-depth=-1 -fconstexpr-steps=-1
