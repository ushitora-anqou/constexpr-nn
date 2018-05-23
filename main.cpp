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
template <size_t InSize, size_t OutSize>
struct Linear {
    Matrix<InSize, OutSize> W;
    Vector<OutSize> b;

    constexpr Linear() : W{}, b{} {}

    template <size_t BatchSize>
    constexpr Matrix<BatchSize, OutSize> forward(
        const Matrix<BatchSize, InSize>& src)
    {
        return src.dot(W) + b;
    }
};

template <size_t Size>
struct ReLU {
    template <size_t BatchSize>
    constexpr Matrix<BatchSize, Size> forward(Matrix<BatchSize, Size> src)
    {
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < Size; j++)
                if (src(i, j) < 0) src(i, j) = 0;
        return src;
    }
};

template <size_t InSize>
struct SoftmaxCrossEntropy {
    template <size_t BatchSize>
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

        // process cross-entropy
        float loss_sum = 0;
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                loss_sum += t(i, j) * std::log(src(i, j));

        return -loss_sum / BatchSize;
    }
};

#include "test.cpp"

/////////////////////
//// main
/////////////////////

int main() {}
