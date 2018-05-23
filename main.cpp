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

    constexpr float operator()(size_t i, size_t j) const { return data[i][j]; }
    constexpr float& operator()(size_t i, size_t j) { return data[i][j]; }
    constexpr bool operator==(const Matrix<N, M>& rhs) const
    {
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                if (data[i][j] != rhs.data[i][j]) return false;
        return true;
    }
};

constexpr std::tuple<Matrix<2, 2>, Matrix<2, 2>> test()
{
    Matrix<2, 2> mat;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    return std::make_tuple(mat, mat.transposed());
}

int main()
{
    constexpr auto ret = test();
    static_assert(std::get<0>(ret)(0, 1) == std::get<1>(ret)(1, 0));
}
