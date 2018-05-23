#include <iostream>
#include <tuple>

// N-by-M matrix
template <size_t N, size_t M>
struct Matrix {
    float data[N][M];

    constexpr Matrix() : data{} {};
    constexpr Matrix<M, N> transposed() const
    {
        Matrix<M, N> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++) ret.data[j][i] = data[i][j];
        return ret;
    }

    constexpr std::tuple<size_t, size_t> shape() const
    {
        return std::make_tuple(N, M);
    }

    // to get an element
    constexpr float operator()(size_t i, size_t j) const { return data[i][j]; }
    constexpr float& operator()(size_t i, size_t j) { return data[i][j]; }
    // for vector-like matrix
    constexpr float operator()(size_t i) const { return data[0][i]; }
    constexpr float& operator()(size_t i) { return data[0][i]; }
    constexpr float operator[](size_t i) const { return (*this)(i); }
    constexpr float& operator[](size_t i) { return (*this)(i); }
};

// Vector is an alias of special Matrix
template <size_t N>
using Vector = Matrix<1, N>;

template <size_t lN, size_t lM, size_t rN, size_t rM>
constexpr bool operator==(const Matrix<lN, lM>& lhs, const Matrix<rN, rM>& rhs)
{
    if constexpr (lN != rN || lM != rM) return false;

    for (size_t i = 0; i < lN; i++)
        for (size_t j = 0; j < lM; j++)
            if (lhs.data[i][j] != rhs.data[i][j]) return false;
    return true;
}

template <size_t lN, size_t lM, size_t rN, size_t rM>
constexpr bool operator!=(const Matrix<lN, lM>& lhs, const Matrix<rN, rM>& rhs)
{
    return !(lhs == rhs);
}

#include "test.cpp"

/////////////////////
//// main
/////////////////////

int main() {}
