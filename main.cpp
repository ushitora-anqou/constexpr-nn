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

    constexpr float operator()(size_t i, size_t j) const { return data[i][j]; }
    constexpr float& operator()(size_t i, size_t j) { return data[i][j]; }
};

template <size_t lN, size_t lM, size_t rN, size_t rM>
constexpr bool operator==(const Matrix<lN, lM>& lhs, const Matrix<rN, rM>& rhs)
{
    if constexpr (lN != lM || rN != rM) return false;

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

///

constexpr void test_matrix_shape()
{
    Matrix<2, 3> mat;
    static_assert(mat.shape() == std::tuple<size_t, size_t>(2, 3));
}

constexpr std::tuple<Matrix<2, 2>, Matrix<2, 2>> test_matrix_equal_detail1()
{
    Matrix<2, 2> mat;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    return std::make_tuple(mat, mat);
}

constexpr std::tuple<Matrix<2, 2>, Matrix<2, 2>> test_matrix_equal_detail2()
{
    Matrix<2, 2> mat;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    Matrix<2, 2> mat1;
    mat(0, 0) = 1;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    return std::make_tuple(mat, mat1);
}

constexpr std::tuple<Matrix<2, 2>, Matrix<2, 4>> test_matrix_equal_detail3()
{
    Matrix<2, 2> mat;
    Matrix<2, 4> mat1;
    return std::make_tuple(mat, mat1);
}

constexpr void test_matrix_equal()
{
    {
        constexpr auto ret = test_matrix_equal_detail1();
        static_assert(std::get<0>(ret) == std::get<1>(ret));
    }
    {
        constexpr auto ret = test_matrix_equal_detail2();
        static_assert(std::get<0>(ret) != std::get<1>(ret));
    }
    {
        constexpr auto ret = test_matrix_equal_detail3();
        static_assert(std::get<0>(ret) != std::get<1>(ret));
    }
}

constexpr std::tuple<Matrix<2, 2>, Matrix<2, 2>> test_matrix_transposed_detail()
{
    Matrix<2, 2> mat;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    return std::make_tuple(mat, mat.transposed());
}

constexpr void test_matrix_transposed()
{
    constexpr auto ret = test_matrix_transposed_detail();
    static_assert(std::get<0>(ret)(0, 0) == std::get<1>(ret)(0, 0));
    static_assert(std::get<0>(ret)(0, 1) == std::get<1>(ret)(1, 0));
    static_assert(std::get<0>(ret)(1, 0) == std::get<1>(ret)(0, 1));
    static_assert(std::get<0>(ret)(1, 1) == std::get<1>(ret)(1, 1));
}

int main() {}
