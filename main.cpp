#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>

#include <sprout/math/exp.hpp>
#include <sprout/math/log.hpp>
#include <sprout/math/sqrt.hpp>
#include <sprout/random/mersenne_twister.hpp>
#include <sprout/random/normal_distribution.hpp>
#include <sprout/random/unique_seed.hpp>

#define HOOLIB_ENABLE_CONSTEXPR

#ifdef HOOLIB_ENABLE_CONSTEXPR
#define HOOLIB_CONSTEXPR constexpr
#define HOOLIB_STATIC_ASSERT static_assert
#define HOOOLIB_IF_CONSTEXPR if constexpr
#else
#define HOOLIB_CONSTEXPR
#define HOOLIB_STATIC_ASSERT
#define HOOOLIB_IF_CONSTEXPR if
#endif

//////////////////////
/// Random
//////////////////////
/*
struct Random {
    sprout::random::mt19937 randgen;

    constexpr Random(uint64_t seed) : randgen(seed) {}

    constexpr float uniform_dist()
    {
        return static_cast<float>(randgen() - randgen.min()) /
               static_cast<float>(randgen.max() - randgen.min());
    }

    constexpr float normal_dist(float mean, float std)
    {
        constexpr double pi = 3.1415926535897932384626433832795028f;
        double z = sprout::math::sqrt(-2. * sprout::math::log(uniform_dist())) *
                   sprout::math::sin(2. * pi * uniform_dist());
        return z * std + mean;
        // auto dist = sprout::random::normal_distribution<float>(mean, std);
        // return dist(randgen);
    }
};
*/
#include "normal_distribution.cpp"
struct Random {
    static constexpr size_t SIZE =
        sizeof(RANDOM_NORMAL_DIST_TABLE) / sizeof(float);
    size_t index;

    constexpr Random(uint64_t seed) : index(seed % SIZE) {}

    constexpr float normal_dist(float mean, float std)
    {
        float ret = RANDOM_NORMAL_DIST_TABLE[index] * std + mean;
        index = (index + 1) % SIZE;
        return ret;
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
        if (rN == 1) {
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

    // broadcast
    HOOLIB_CONSTEXPR Matrix<N, M> operator+(float rhs) const
    {
        Matrix<N, M> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++) ret(i, j) = (*this)(i, j) + rhs;
        return ret;
    }

    template <class R>
    HOOLIB_CONSTEXPR Matrix<N, M> operator-(const R& rhs) const
    {
        return *this + rhs * -1;
    }

    HOOLIB_CONSTEXPR Matrix<N, M> operator*(float scalar) const
    {
        Matrix<N, M> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++) ret(i, j) = (*this)(i, j) * scalar;
        return ret;
    }

    HOOLIB_CONSTEXPR Matrix<N, M> operator/(float scalar) const
    {
        return *this * (1. / scalar);
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

    HOOLIB_CONSTEXPR Matrix<N, M> operator*(const Matrix<N, M>& rhs) const
    {
        Matrix<N, M> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                ret(i, j) = (*this)(i, j) * rhs(i, j);
        return ret;
    }

    HOOLIB_CONSTEXPR Matrix<N, M> operator/(const Matrix<N, M>& rhs) const
    {
        Matrix<N, M> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                ret(i, j) = (*this)(i, j) / rhs(i, j);
        return ret;
    }
};

template <size_t N, size_t M>
HOOLIB_CONSTEXPR Matrix<N, M> sqrt(const Matrix<N, M>& src)
{
    Matrix<N, M> ret;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            ret(i, j) = sprout::math::sqrt(src(i, j));
    return ret;
}

template <size_t lN, size_t lM, size_t rN, size_t rM>
HOOLIB_CONSTEXPR bool operator==(const Matrix<lN, lM>& lhs,
                                 const Matrix<rN, rM>& rhs)
{
    if (lN != rN || lM != rM) return false;

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
                loss_sum += t(i, j) * sprout::math::log(src(i, j));

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

//#include "test.cpp"

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
                l1.W(i, j) =
                    rand.normal_dist(0, sprout::math::sqrt(2. / InSize));
        for (size_t i = 0; i < NUnit; i++)
            for (size_t j = 0; j < NUnit; j++)
                l2.W(i, j) =
                    rand.normal_dist(0, sprout::math::sqrt(2. / NUnit));
        for (size_t i = 0; i < NUnit; i++)
            for (size_t j = 0; j < NOut; j++)
                l3.W(i, j) =
                    rand.normal_dist(0, sprout::math::sqrt(2. / NUnit));
    }

    static constexpr size_t get_batch_size() { return BatchSize; }

    HOOLIB_CONSTEXPR auto predict(const Matrix<BatchSize, InSize>& src)
    {
        auto h1 = ReLU::forward(l1.forward(src));
        auto h2 = ReLU::forward(l2.forward(h1));
        auto h3 = l3.forward(h2);
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
        auto h2 = l3.backward(h1);
        auto h3 = l2.backward(ReLU::backward(h2));
        auto h4 = l1.backward(ReLU::backward(h3));
    }

    template <class OptimizerGenerator>
    HOOLIB_CONSTEXPR auto generate_optimizers(OptimizerGenerator gen)
    {
        return std::make_tuple(gen(l1.W, l1.dW), gen(l1.b, l1.db),
                               gen(l2.W, l2.dW), gen(l2.b, l2.db),
                               gen(l3.W, l3.dW), gen(l3.b, l3.db));
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

template <class Mat>
struct Adam {
    float alpha, beta1, beta2, beta1_t, beta2_t, eps;
    Mat &W, &dW;
    Mat m, v;

    HOOLIB_CONSTEXPR Adam(Mat& aW, Mat& adW, float a, float b1, float b2,
                          float e)
        : alpha(a),
          beta1(b1),
          beta2(b2),
          beta1_t(b1),
          beta2_t(b2),
          eps(e),
          W(aW),
          dW(adW)
    {
    }

    HOOLIB_CONSTEXPR void operator()()
    {
        m = m * beta1 + dW * (1 - beta1);
        v = v * beta2 + dW * dW * (1 - beta2);
        auto m_hat = m / (1 - beta1_t);
        auto v_hat = v / (1 - beta2_t);
        W = W - (m_hat * alpha) / (sqrt(v_hat) + eps);
        beta1_t *= beta1;
        beta2_t *= beta2;
    }
};

struct AdamGenerator {
    float alpha, beta1, beta2, eps;
    HOOLIB_CONSTEXPR AdamGenerator(float a = 0.001, float b1 = 0.9,
                                   float b2 = 0.999, float e = 10e-8)
        : alpha(a), beta1(b1), beta2(b2), eps(e)
    {
    }

    template <class Mat>
    HOOLIB_CONSTEXPR Adam<Mat> operator()(Mat& W, Mat& dW)
    {
        return Adam<Mat>(W, dW, alpha, beta1, beta2, eps);
    }
};

template <class Mat>
struct SGD {
    float lr;
    Mat &W, &dW;

    HOOLIB_CONSTEXPR SGD(Mat& aW, Mat& adW, float alr) : lr(alr), W(aW), dW(adW)
    {
    }

    HOOLIB_CONSTEXPR void operator()() { W = W + dW * -lr; }
};

struct SGDGenerator {
    float lr;

    HOOLIB_CONSTEXPR SGDGenerator(float alr = 0.1) : lr(alr) {}

    template <class Mat>
    HOOLIB_CONSTEXPR SGD<Mat> operator()(Mat& W, Mat& dW)
    {
        return SGD<Mat>(W, dW, lr);
    }
};

struct CallOptimizer {
    template <class Optimizer, class... Tail>
    HOOLIB_CONSTEXPR void operator()(Optimizer&& op, Tail&&... tail)
    {
        op();
        (*this)(std::forward<Tail>(tail)...);
    }

    template <class Optimizer>
    HOOLIB_CONSTEXPR void operator()(Optimizer&& op)
    {
        op();
    }
};

#include "mnist.cpp"

struct TrainResult {
    float train_loss;
    float test_accuracy;

    HOOLIB_CONSTEXPR TrainResult() : train_loss(0), test_accuracy(0) {}
};

template <size_t Epoch>
HOOLIB_CONSTEXPR auto train()
{
    constexpr size_t batch_size = 1;
    constexpr auto& train_x = MNIST_TRAIN_SAMPLES_X;
    constexpr auto& train_t = MNIST_TRAIN_SAMPLES_T;
    constexpr size_t train_sample_num = 2, train_sample_size = 764;
    std::array<TrainResult, Epoch> ret;
    MLP3<batch_size, train_sample_size, 100, 10> nn;
    auto optimizers = nn.generate_optimizers(AdamGenerator());

    for (size_t i = 0; i < Epoch; i++) {
        float train_loss = 0;
        for (size_t bi = 0; bi < train_sample_num / batch_size; bi++) {
            Matrix<batch_size, train_sample_size> batch;
            Matrix<batch_size, 10> onehot_t;
            for (size_t i = 0; i < batch_size; i++) {
                for (size_t j = 0; j < train_sample_size; j++)
                    batch(i, j) = train_x[bi * batch_size + i][j];
                onehot_t(i, train_t[bi * batch_size + i]) = 1;
            }
            float loss = nn.forward(batch, onehot_t);
            train_loss += loss;

            nn.backward();
            std::apply(CallOptimizer(), optimizers);
        }
        train_loss /= (train_sample_num / batch_size);

        constexpr auto& test_x = MNIST_TEST_SAMPLES_X;
        constexpr auto& test_t = MNIST_TEST_SAMPLES_T;
        constexpr size_t test_sample_num = 2, test_sample_size = 764;

        size_t correct_count = 0;
        for (size_t bi = 0; bi < test_sample_num / batch_size; bi++) {
            Matrix<batch_size, test_sample_size> batch;
            for (size_t i = 0; i < batch_size; i++)
                for (size_t j = 0; j < test_sample_size; j++)
                    batch(i, j) = test_x[bi * batch_size + i][j];
            auto y = predict(nn, batch);

            for (size_t i = 0; i < batch_size; i++)
                if (test_t[bi * batch_size + i] == y[i]) correct_count++;
        }

        ret[i].train_loss = train_loss;
        ret[i].test_accuracy =
            static_cast<float>(correct_count) / test_sample_num;
    }

    return ret;
}

int main()
{
    constexpr auto res = train<100>();
    for (auto&& r : res)
        std::cout << std::setprecision(10) << "train loss: " << r.train_loss
                  << std::endl
                  << "test accuracy: " << r.test_accuracy << std::endl;
}
