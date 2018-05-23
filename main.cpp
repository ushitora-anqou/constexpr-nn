#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

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

    template <size_t rN>
    constexpr Matrix<N, M> operator+(const Matrix<rN, M>& rhs) const
    {
        static_assert(rN == 1 || rN == N);
        if constexpr (rN == 1) {
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

    constexpr Matrix<N, M> operator*(float scalar) const
    {
        Matrix<N, M> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++) ret(i, j) = (*this)(i, j) * scalar;
        return ret;
    }

    template <size_t L>
    constexpr Matrix<N, L> dot(const Matrix<M, L>& rhs) const
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

/////////////////////
//// NN component
/////////////////////
template <size_t BatchSize, size_t InSize, size_t OutSize>
struct Linear {
    Matrix<InSize, OutSize> W, dW;
    Vector<OutSize> b, db;
    Matrix<BatchSize, InSize> x;

    constexpr Linear() {}

    constexpr Matrix<BatchSize, OutSize> forward(
        const Matrix<BatchSize, InSize>& src)
    {
        x = src;
        return src.dot(W) + b;
    }

    constexpr Matrix<BatchSize, InSize> backward(
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
    static constexpr Matrix<BatchSize, InSize> forward(
        Matrix<BatchSize, InSize> src)
    {
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                if (src(i, j) < 0) src(i, j) = 0;
        return src;
    }

    template <size_t BatchSize, size_t InSize>
    static constexpr Matrix<BatchSize, InSize> backward(
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

    constexpr float forward(Matrix<BatchSize, InSize> src,
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
                src(i, j) = std::exp(src(i, j) - c);
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

    constexpr Matrix<BatchSize, InSize> backward()
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

    static constexpr size_t get_batch_size() { return BatchSize; }

    constexpr auto forward(const Matrix<BatchSize, 764>& src)
    {
        constexpr auto h1 = ReLU::forward(l1.forward(src));
        constexpr auto h2 = ReLU::forward(l2.forward(l1));
        constexpr auto h3 = ReLU::forward(l3.forward(l2));
        constexpr float loss = sce.forward(h3);
        return std::make_tuple(h3, loss);
    }

    constexpr void backward(float in = 1)
    {
        constexpr auto h1 = sce.backward(in);
        constexpr auto h2 = l3.backward(ReLU::backward(h1));
        constexpr auto h3 = l2.backward(ReLU::backward(h2));
        constexpr auto h4 = l1.backward(ReLU::backward(h3));
    }

    template <class SGD>
    constexpr void update(SGD sgd)
    {
        l1.update(sgd);
        l2.update(sgd);
        l3.update(sgd);
    }
};

template <class NN, size_t N, size_t InSize>
constexpr auto predict(NN nn, const Matrix<N, InSize>& src)
{
    static_assert(N <= NN::get_batch_size());
    Matrix<NN::get_batch_size, InSize> in;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < InSize; j++) in(i, j) = src(i, j);
    constexpr auto res = nn.forward(in);

    auto ret = std::array<size_t, N>();
    for (size_t i = 0; i < N; i++) {
        size_t argmax = 0;
        for (size_t j = 0; j < res.shape()[1]; j++)
            if (std::get<0>(res)(i, argmax) < std::get<0>(res)(i, j))
                argmax = j;
        ret[i] = argmax;
    }

    return ret;
}

int main() {}
