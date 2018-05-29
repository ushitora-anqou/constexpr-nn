#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

#include <sprout/math/exp.hpp>
#include <sprout/math/sqrt.hpp>
#include <sprout/random/mersenne_twister.hpp>
#include <sprout/random/normal_distribution.hpp>
#include <sprout/random/unique_seed.hpp>

#define HOOLIB_CONSTEXPR constexpr
//#define HOOLIB_CONSTEXPR
#define HOOLIB_STATIC_ASSERT static_assert
//#define HOOLIB_STATIC_ASSERT

//////////////////////
/// Random
//////////////////////
struct Random {
    sprout::random::mt19937 randgen;

    constexpr Random(size_t seed) : randgen(seed) {}

    constexpr float normal_dist(float mean, float std)
    {
        auto dist = sprout::random::normal_distribution(mean, std);
        return dist(randgen);
    }
};

///////////////////////
// N-by-M matrix
//////////////////////
// Vector is an alias of special Matrix
template <size_t N, size_t M>
struct Matrix;
template <size_t N>
using Vector = Matrix<1, N>;
template <size_t N, size_t M>
struct Matrix {
    float data[N][M];

    HOOLIB_CONSTEXPR Matrix() : data{} {};
    HOOLIB_CONSTEXPR Matrix<M, N> transposed() const
    {
        Matrix<M, N> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++) ret.data[j][i] = data[i][j];
        return ret;
    }

    HOOLIB_CONSTEXPR std::tuple<size_t, size_t> shape() const
    {
        return std::make_tuple(N, M);
    }

    // to get an element
    HOOLIB_CONSTEXPR float operator()(size_t i, size_t j) const
    {
        return data[i][j];
    }
    HOOLIB_CONSTEXPR float& operator()(size_t i, size_t j)
    {
        return data[i][j];
    }
    // for vector-like matrix
    HOOLIB_CONSTEXPR float operator()(size_t i) const { return data[0][i]; }
    HOOLIB_CONSTEXPR float& operator()(size_t i) { return data[0][i]; }
    HOOLIB_CONSTEXPR float operator[](size_t i) const { return (*this)(i); }
    HOOLIB_CONSTEXPR float& operator[](size_t i) { return (*this)(i); }

    template <size_t rN>
    HOOLIB_CONSTEXPR Matrix<N, M> operator+(const Matrix<rN, M>& rhs) const
    {
        HOOLIB_STATIC_ASSERT(rN == 1 || rN == N);
        if
            HOOLIB_CONSTEXPR(rN == 1)
            {
                // broadcast
                Matrix<N, M> ret;
                for (size_t i = 0; i < N; i++)
                    for (size_t j = 0; j < M; j++)
                        ret(i, j) = (*this)(i, j) + rhs[j];
                return ret;
            }
        else {
            Matrix<N, M> ret;
            for (size_t i = 0; i < N; i++)
                for (size_t j = 0; j < M; j++)
                    ret(i, j) = (*this)(i, j) + rhs(i, j);
            return ret;
        }
    }

    HOOLIB_CONSTEXPR Matrix<N, M> operator*(float scalar) const
    {
        Matrix<N, M> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++) ret(i, j) = (*this)(i, j) * scalar;
        return ret;
    }

    template <size_t L>
    HOOLIB_CONSTEXPR Matrix<N, L> dot(const Matrix<M, L>& rhs) const
    {
        Matrix<N, L> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < L; j++)
                for (size_t k = 0; k < M; k++)
                    ret(i, j) += (*this)(i, k) * rhs(k, j);
        return ret;
    }
};

template <size_t lN, size_t lM, size_t rN, size_t rM>
HOOLIB_CONSTEXPR bool operator==(const Matrix<lN, lM>& lhs,
                                 const Matrix<rN, rM>& rhs)
{
    if
        HOOLIB_CONSTEXPR(lN != rN || lM != rM) return false;

    for (size_t i = 0; i < lN; i++)
        for (size_t j = 0; j < lM; j++)
            if (lhs.data[i][j] != rhs.data[i][j]) return false;
    return true;
}

template <size_t lN, size_t lM, size_t rN, size_t rM>
HOOLIB_CONSTEXPR bool operator!=(const Matrix<lN, lM>& lhs,
                                 const Matrix<rN, rM>& rhs)
{
    return !(lhs == rhs);
}

/////////////////////
//// NN component
/////////////////////
template <size_t BatchSize, size_t InSize, size_t OutSize>
struct Linear {
    Matrix<InSize, OutSize> W, dW;
    Vector<OutSize> b, db;
    Matrix<BatchSize, InSize> x;

    HOOLIB_CONSTEXPR Linear() {}

    HOOLIB_CONSTEXPR Matrix<BatchSize, OutSize> forward(
        const Matrix<BatchSize, InSize>& src)
    {
        x = src;
        return src.dot(W) + b;
    }

    HOOLIB_CONSTEXPR Matrix<BatchSize, InSize> backward(
        const Matrix<BatchSize, OutSize>& src)
    {
        auto dx = src.dot(W.transposed());
        dW = x.transposed().dot(src);
        db = Vector<OutSize>();
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < OutSize; j++) db[j] += src(i, j);
        return dx;
    }

    template <class SGD>
    void update(SGD sgd)
    {
        W = sgd(W, dW);
        b = sgd(b, db);
    }
};

struct ReLU {
    template <size_t BatchSize, size_t InSize>
    static HOOLIB_CONSTEXPR Matrix<BatchSize, InSize> forward(
        Matrix<BatchSize, InSize> src)
    {
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                if (src(i, j) < 0) src(i, j) = 0;
        return src;
    }

    template <size_t BatchSize, size_t InSize>
    static HOOLIB_CONSTEXPR Matrix<BatchSize, InSize> backward(
        Matrix<BatchSize, InSize> src)
    {
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                if (src(i, j) < 0) src(i, j) = 0;
        return src;
    }
};

template <size_t BatchSize, size_t InSize>
struct SoftmaxCrossEntropy {
    Matrix<BatchSize, InSize> y, t;

    HOOLIB_CONSTEXPR float forward(Matrix<BatchSize, InSize> src,
                                   const Matrix<BatchSize, InSize>& t)
    {
        // process softmax
        for (size_t i = 0; i < BatchSize; i++) {
            // c = max src[i]
            float c = src(i, 0);
            for (size_t j = 0; j < InSize; j++) c = std::max(c, src(i, j));

            // src[i] = exp(src[i] - c)
            // s = sum src[i]
            float s = 0;
            for (size_t j = 0; j < InSize; j++) {
                src(i, j) = sprout::math::exp(src(i, j) - c);
                s += src(i, j);
            }

            // src[i] /= s
            for (size_t j = 0; j < InSize; j++) src(i, j) /= s;
        }
        this->y = src;
        this->t = t;

        // process cross-entropy
        float loss_sum = 0;
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                loss_sum += t(i, j) * std::log(src(i, j));

        return -loss_sum / BatchSize;
    }

    HOOLIB_CONSTEXPR Matrix<BatchSize, InSize> backward()
    {
        Matrix<BatchSize, InSize> dx;
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                dx(i, j) = (y(i, j) - t(i, j)) / BatchSize;
        return dx;
    }
};

#include "test.cpp"

/////////////////////
//// main
/////////////////////

template <size_t BatchSize, size_t InSize, size_t NUnit, size_t NOut>
struct MLP3 {
    Linear<BatchSize, InSize, NUnit> l1;
    Linear<BatchSize, NUnit, NUnit> l2;
    Linear<BatchSize, NUnit, NOut> l3;
    SoftmaxCrossEntropy<BatchSize, NOut> sce;

    HOOLIB_CONSTEXPR MLP3()
    {
        // He init
        Random rand = SPROUT_UNIQUE_SEED;
        for (size_t i = 0; i < InSize; i++)
            for (size_t j = 0; j < NUnit; j++)
                l1.W(i, j) = rand.normal_dist(0, std::sqrt(2. / InSize));
        for (size_t i = 0; i < NUnit; i++)
            for (size_t j = 0; j < NUnit; j++)
                l2.W(i, j) = rand.normal_dist(0, std::sqrt(2. / NUnit));
        for (size_t i = 0; i < NUnit; i++)
            for (size_t j = 0; j < NUnit; j++)
                l3.W(i, j) = rand.normal_dist(0, std::sqrt(2. / NUnit));
    }

    static constexpr size_t get_batch_size() { return BatchSize; }

    HOOLIB_CONSTEXPR auto predict(const Matrix<BatchSize, InSize>& src)
    {
        auto h1 = ReLU::forward(l1.forward(src));
        auto h2 = ReLU::forward(l2.forward(h1));
        auto h3 = ReLU::forward(l3.forward(h2));
        return h3;
    }

    HOOLIB_CONSTEXPR auto forward(const Matrix<BatchSize, InSize>& src,
                                  const Matrix<BatchSize, NOut>& t)
    {
        auto h3 = predict(src);
        float loss = sce.forward(h3, t);
        return loss;
    }

    HOOLIB_CONSTEXPR void backward()
    {
        auto h1 = sce.backward();
        auto h2 = l3.backward(ReLU::backward(h1));
        auto h3 = l2.backward(ReLU::backward(h2));
        auto h4 = l1.backward(ReLU::backward(h3));
    }

    template <class SGD>
    HOOLIB_CONSTEXPR void update(SGD sgd)
    {
        l1.update(sgd);
        l2.update(sgd);
        l3.update(sgd);
    }
};

template <class NN, size_t N, size_t InSize>
HOOLIB_CONSTEXPR auto predict(NN nn, const Matrix<N, InSize>& src)
{
    HOOLIB_STATIC_ASSERT(N <= NN::get_batch_size());
    Matrix<NN::get_batch_size(), InSize> batch;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < InSize; j++) batch(i, j) = src(i, j);
    auto res = nn.predict(batch);

    auto ret = std::array<size_t, N>();
    for (size_t i = 0; i < N; i++) {
        size_t argmax = 0;
        for (size_t j = 0; j < std::get<1>(res.shape()); j++)
            if (res(i, argmax) < res(i, j)) argmax = j;
        ret[i] = argmax;
    }

    return ret;
}

struct SGD {
    float lr;

    HOOLIB_CONSTEXPR SGD(float alr) : lr(alr) {}

    template <size_t BatchSize, size_t Size>
    Matrix<BatchSize, Size> operator()(const Matrix<BatchSize, Size>& W,
                                       const Matrix<BatchSize, Size>& dW)
    {
        return W + dW * -lr;
    }
};

#include "mnist.cpp"

HOOLIB_CONSTEXPR float train()
{
    constexpr size_t batch_size = 1;
    constexpr auto& train_x = MNIST_TRAIN_SAMPLES_X;
    constexpr auto& train_t = MNIST_TRAIN_SAMPLES_T;
    constexpr size_t train_sample_num = 1, train_sample_size = 764;
    MLP3<batch_size, train_sample_size, 100, 10> nn;

    for (size_t bi = 0; bi < train_sample_num / batch_size; bi++) {
        Matrix<batch_size, train_sample_size> batch;
        Matrix<batch_size, 10> onehot_t;
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < train_sample_size; j++)
                batch(i, j) = train_x[bi * batch_size + i][j];
            onehot_t(i, train_t[bi * batch_size + i]) = 1;
        }
        nn.forward(batch, onehot_t);
        nn.backward();
        nn.update(SGD(0.1));
    }

    constexpr auto& test_x = MNIST_TEST_SAMPLES_X;
    constexpr auto& test_t = MNIST_TEST_SAMPLES_T;
    constexpr size_t test_sample_num = 1, test_sample_size = 764;

    size_t correct_count = 0;
    for (size_t bi = 0; bi < test_sample_num / batch_size; bi++) {
        Matrix<batch_size, test_sample_size> batch;
        for (size_t i = 0; i < batch_size; i++)
            for (size_t j = 0; j < test_sample_size; j++)
                batch(i, j) = test_x[bi * batch_size + i][j];
        auto y = predict(nn, batch);

        bool all_true = true;
        for (size_t i = 0; i < batch_size; i++)
            if (test_t[bi * batch_size + i] != y[i]) all_true = false;
        if (all_true) correct_count++;
    }

    return static_cast<float>(correct_count) / test_sample_num;
}

int main()
{
    Random rand = SPROUT_UNIQUE_SEED;
    std::cout << rand.normal_dist(0, 1) << std::endl;
    std::cout << rand.normal_dist(0, 1) << std::endl;

    std::cout << train() << std::endl;
}
