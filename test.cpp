//////////////////
/// tests
//////////////////

constexpr void test_matrix_shape()
{
    Matrix<2, 3> mat;
    static_assert(mat.shape() == std::tuple<size_t, size_t>(2, 3));
}

constexpr auto test_matrix_equal1_detail()
{
    Matrix<2, 3> mat;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(0, 2) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    mat(1, 2) = 1;
    return std::make_tuple(mat, mat);
}

constexpr auto test_matrix_equal2_detail()
{
    Matrix<2, 3> mat;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(0, 2) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    mat(1, 2) = 1;
    Matrix<2, 3> mat1;
    mat1(0, 0) = 0;
    mat1(0, 1) = 1;
    mat1(0, 2) = 1;
    mat1(1, 0) = 2;
    mat1(1, 1) = 3;
    mat1(1, 2) = 0;
    return std::make_tuple(mat, mat1);
}

constexpr auto test_matrix_equal3_detail()
{
    Matrix<2, 2> mat;
    Matrix<2, 4> mat1;
    return std::make_tuple(mat, mat1);
}

constexpr void test_matrix_equal1()
{
    constexpr auto ret = test_matrix_equal1_detail();
    static_assert(std::get<0>(ret) == std::get<1>(ret));
}

constexpr void test_matrix_equal2()
{
    constexpr auto ret = test_matrix_equal2_detail();
    static_assert(std::get<0>(ret) != std::get<1>(ret));
}

constexpr void test_matrix_equal3()
{
    constexpr auto ret = test_matrix_equal3_detail();
    static_assert(std::get<0>(ret) != std::get<1>(ret));
}

constexpr auto test_matrix_transposed_detail()
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

constexpr auto test_vector1_detail()
{
    Vector<2> vec;
    vec[0] = 0;
    vec[1] = 1;
    return std::make_tuple(vec, vec);
}

constexpr void test_vector1()
{
    constexpr auto ret = test_vector1_detail();
    static_assert(std::get<0>(ret) == std::get<1>(ret));
}

constexpr auto test_vector2_detail()
{
    Vector<2> vec;
    vec[0] = 0;
    vec[1] = 1;
    Vector<2> vec2;
    vec[0] = 1;
    vec[1] = 1;
    return std::make_tuple(vec, vec2);
}

constexpr void test_vector2()
{
    constexpr auto ret = test_vector2_detail();
    static_assert(std::get<0>(ret) != std::get<1>(ret));
}
