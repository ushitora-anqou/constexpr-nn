//////////////////
/// tests
//////////////////

HOOLIB_CONSTEXPR void test_matrix_shape()
{
    HOOLIB_STATIC_ASSERT(Matrix<2, 3>::shape() == std::make_pair(2ul, 3ul));
}

HOOLIB_CONSTEXPR auto test_matrix_equal1_detail()
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

HOOLIB_CONSTEXPR auto test_matrix_equal2_detail()
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

HOOLIB_CONSTEXPR auto test_matrix_equal3_detail()
{
    Matrix<2, 2> mat;
    Matrix<2, 4> mat1;
    return std::make_tuple(mat, mat1);
}

HOOLIB_CONSTEXPR void test_matrix_equal1()
{
    HOOLIB_CONSTEXPR auto ret = test_matrix_equal1_detail();
    HOOLIB_STATIC_ASSERT(std::get<0>(ret) == std::get<1>(ret));
}

HOOLIB_CONSTEXPR void test_matrix_equal2()
{
    HOOLIB_CONSTEXPR auto ret = test_matrix_equal2_detail();
    HOOLIB_STATIC_ASSERT(std::get<0>(ret) != std::get<1>(ret));
}

HOOLIB_CONSTEXPR void test_matrix_equal3()
{
    HOOLIB_CONSTEXPR auto ret = test_matrix_equal3_detail();
    HOOLIB_STATIC_ASSERT(std::get<0>(ret) != std::get<1>(ret));
}

HOOLIB_CONSTEXPR auto test_matrix_transposed_detail()
{
    Matrix<2, 2> mat;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    return std::make_tuple(mat, mat.transposed());
}

HOOLIB_CONSTEXPR void test_matrix_transposed()
{
    HOOLIB_CONSTEXPR auto ret = test_matrix_transposed_detail();
    HOOLIB_STATIC_ASSERT(std::get<0>(ret)(0, 0) == std::get<1>(ret)(0, 0));
    HOOLIB_STATIC_ASSERT(std::get<0>(ret)(0, 1) == std::get<1>(ret)(1, 0));
    HOOLIB_STATIC_ASSERT(std::get<0>(ret)(1, 0) == std::get<1>(ret)(0, 1));
    HOOLIB_STATIC_ASSERT(std::get<0>(ret)(1, 1) == std::get<1>(ret)(1, 1));
}

HOOLIB_CONSTEXPR auto test_matrix_add_detail()
{
    Matrix<2, 2> mat, mat2, mat3;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    mat2(0, 0) = 1;
    mat2(0, 1) = 2;
    mat2(1, 0) = 3;
    mat2(1, 1) = 4;
    mat3(0, 0) = 1;
    mat3(0, 1) = 3;
    mat3(1, 0) = 5;
    mat3(1, 1) = 7;

    return std::make_tuple(mat + mat2, mat3);
}

HOOLIB_CONSTEXPR void test_matrix_add()
{
    HOOLIB_CONSTEXPR auto ret = test_matrix_add_detail();
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 0) - std::get<1>(ret)(0, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 1) - std::get<1>(ret)(0, 1)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 0) - std::get<1>(ret)(1, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 1) - std::get<1>(ret)(1, 1)) < 1e-5);
}

HOOLIB_CONSTEXPR auto test_matrix_mul_detail()
{
    Matrix<2, 2> mat, mat2;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 3;
    mat2(0, 0) = 0;
    mat2(0, 1) = 2;
    mat2(1, 0) = 4;
    mat2(1, 1) = 6;

    return std::make_tuple(mat * 2, mat2);
}

HOOLIB_CONSTEXPR void test_matrix_mul()
{
    HOOLIB_CONSTEXPR auto ret = test_matrix_mul_detail();
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 0) - std::get<1>(ret)(0, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 1) - std::get<1>(ret)(0, 1)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 0) - std::get<1>(ret)(1, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 1) - std::get<1>(ret)(1, 1)) < 1e-5);
}

HOOLIB_CONSTEXPR auto test_matrix_dot_detail()
{
    Matrix<2, 3> mat;
    Matrix<3, 2> mat2;
    Matrix<2, 2> ans;
    mat(0, 0) = 0;
    mat(0, 1) = 1;
    mat(0, 2) = -1;
    mat(1, 0) = 2;
    mat(1, 1) = -3;
    mat(1, 2) = 3;
    mat2(0, 0) = 0;
    mat2(0, 1) = 1;
    mat2(1, 0) = 0.5;
    mat2(1, 1) = 3;
    mat2(2, 0) = 1;
    mat2(2, 1) = -3;
    ans(0, 0) = -0.5;
    ans(0, 1) = 6;
    ans(1, 0) = 1.5;
    ans(1, 1) = -16;

    return std::make_tuple(mat.dot(mat2), ans);
}

HOOLIB_CONSTEXPR void test_matrix_dot()
{
    HOOLIB_CONSTEXPR auto ret = test_matrix_dot_detail();
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 0) - std::get<1>(ret)(0, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 1) - std::get<1>(ret)(0, 1)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 0) - std::get<1>(ret)(1, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 1) - std::get<1>(ret)(1, 1)) < 1e-5);
}

HOOLIB_CONSTEXPR auto test_vector1_detail()
{
    Vector<2> vec;
    vec[0] = 0;
    vec[1] = 1;
    return std::make_tuple(vec, vec);
}

HOOLIB_CONSTEXPR void test_vector1()
{
    HOOLIB_CONSTEXPR auto ret = test_vector1_detail();
    HOOLIB_STATIC_ASSERT(std::get<0>(ret) == std::get<1>(ret));
}

HOOLIB_CONSTEXPR auto test_vector2_detail()
{
    Vector<2> vec;
    vec[0] = 0;
    vec[1] = 1;
    Vector<2> vec2;
    vec[0] = 1;
    vec[1] = 1;
    return std::make_tuple(vec, vec2);
}

HOOLIB_CONSTEXPR void test_vector2()
{
    HOOLIB_CONSTEXPR auto ret = test_vector2_detail();
    HOOLIB_STATIC_ASSERT(std::get<0>(ret) != std::get<1>(ret));
}

HOOLIB_CONSTEXPR auto test_linear1_detail()
{
    Matrix<2, 3> W;
    W(0, 0) = 0;
    W(0, 1) = 1;
    W(0, 2) = 1;
    W(1, 0) = 2;
    W(1, 1) = 3;
    W(1, 2) = 1;
    Vector<3> b;
    b[0] = 1;
    b[1] = 2;
    b[2] = 2;

    Linear<2, 2, 3> l;
    l.W = W;
    l.b = b;

    Matrix<2, 2> src;
    src(0, 0) = -1;
    src(0, 1) = -2;
    src(1, 0) = 0.5;
    src(1, 1) = -0.5;
    auto ret = l.forward(src);

    Matrix<2, 3> ans = src.dot(W) + b;
    return std::make_tuple(ret, ans);
}

HOOLIB_CONSTEXPR void test_linear1()
{
    HOOLIB_CONSTEXPR auto ret = test_linear1_detail();
    HOOLIB_STATIC_ASSERT(std::get<0>(ret) == std::get<1>(ret));
}

HOOLIB_CONSTEXPR auto test_relu_detail()
{
    Matrix<2, 2> mat, mat2;
    mat(0, 0) = -1;
    mat(0, 1) = -2;
    mat(1, 0) = 0.5;
    mat(1, 1) = -0.5;
    mat2(0, 0) = 0;
    mat2(0, 1) = 0;
    mat2(1, 0) = 0.5;
    mat2(1, 1) = -0;

    return std::make_tuple(ReLU::forward(mat), mat2);
}

HOOLIB_CONSTEXPR void test_relu()
{
    HOOLIB_CONSTEXPR auto ret = test_relu_detail();
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 0) - std::get<1>(ret)(0, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(0, 1) - std::get<1>(ret)(0, 1)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 0) - std::get<1>(ret)(1, 0)) < 1e-5);
    HOOLIB_STATIC_ASSERT(
        std::abs(std::get<0>(ret)(1, 1) - std::get<1>(ret)(1, 1)) < 1e-5);
}

HOOLIB_CONSTEXPR auto test_softmax_cross_entropy_detail()
{
    Matrix<2, 2> mat, mat2;
    mat(0, 0) = -1;
    mat(0, 1) = -2;
    mat(1, 0) = 0.5;
    mat(1, 1) = -0.5;
    mat2(0, 0) = 0;
    mat2(0, 1) = 1;
    mat2(1, 0) = 1;
    mat2(1, 1) = 0;

    SoftmaxCrossEntropy<2, 2> sce;
    return std::make_tuple(sce.forward(mat, mat2), 0.8132616281509399);
}

HOOLIB_CONSTEXPR void test_softmax_cross_entropy()
{
    HOOLIB_CONSTEXPR auto ret = test_softmax_cross_entropy_detail();
    HOOLIB_STATIC_ASSERT(std::abs(std::get<0>(ret) - std::get<1>(ret)) < 1e-5);
}
