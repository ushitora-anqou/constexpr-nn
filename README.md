# constexpr-nn

## What is this?

This is a program to calculate backpropagation of a neural network (MLP) **at its compile time** by using C++17 `constexpr`.

## Why did you do it?

It's fun.

## Why did you do it in actual?

It's a lot of fun.

## Requirements

- C++17 compiler
    - Patched Clang is the best. I'll describe it later.
- Python3 and Chainer
- Lots of memory
- Moderate CPU
- Wisdom, courage, love and patience.

## Patch Clang

Build Clang according to [Clang official documents](http://llvm.org/docs/GettingStarted.html#getting-started-quickly-a-summary) but in between execute `svn patch clang.diff`.
I'll show an example.

```
cd where-you-want-llvm-to-live
svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm
cd llvm/tools
svn co http://llvm.org/svn/llvm-project/cfe/trunk clang
svn patch where-you-cloned-constexpr-nn/clang.diff
cd ../..
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../llvm
make -j6
```

You also have to modify `Makefile` to use patched `clang++`.

## Configuration

- epoch, batch size to train and to test.
    - `main.cpp`
    - `constexpr size_t epoch = 1, train_batch_size = 100, test_batch_size = 100;`
- number of MNIST samples to train and test
    - `mnist.py`
    - `train_sample_num = 15000`
    - `test_sample_num = 15000`
    - The maxima are both 60000.
- test at the compile time or runtime
    - `main.cpp`
    - uncomment `//#define CNSTNN_GET_TEST_ACC_AT_COMPILE_TIME` to test at its compile time.

## How to play

- `git clone`
- `git submodule init`
- `git submodule update`
- `python mnist.py`
- `python random_table.py`
- `make`
    - This process may take a long time. This is a good time to think about your future.
    - About 13 hours were needed to run 1 epoch of 8000 samples (train 4000 + test 4000) using Intel Core i7-3630QM.
- `./main`
    - It'll print out the train loss and test accuracy.
        - Of course you need to uncomment `#define CNSTNN_GET_TEST_ACC_AT_COMPILE_TIME` to print the compile-time test accuracy. If you disabled it, then runtime test accuracy will be there.
    - `./main filename` will predict the picture you specified.

## License

MIT.

<hr />

## なんこれ

C++17のconstexprを使ってニューラルネットワークを **コンパイル時に** 学習させようというプロジェクト。

## 必要なもの

- C++17が動くコンパイラ
    - まともに動かすためには`clang++`が必要。`g++`だとnewしたメモリをdeleteしてくれない。
        - どうやら`g++`は`constexpr`指定された関数をメモ化するようだ。
    - さらにちゃんと動かすためにはClangに`clang.diff`のパッチを当てたものが必要。Clang内部で使用されている`unsigned int`のサイズ（典型的には32bit）では、実行ステップ数やコールスタックフレームの数が間に合わないため。詳しくは後述。
- Python3とChainer
- たくさんのメモリ
- そこそこのCPU
- 知恵と勇気と愛と忍耐

## Clangにパッチを当てる

基本的には[公式ドキュメント](http://llvm.org/docs/GettingStarted.html#getting-started-quickly-a-summary)を参考にすれば良い。途中で`svn patch clang.diff`を行う。以下典型的な手順。

```
cd where-you-want-llvm-to-live
svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm
cd llvm/tools
svn co http://llvm.org/svn/llvm-project/cfe/trunk clang
svn patch where-you-cloned-constexpr-nn/clang.diff
cd ../..
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../llvm
make -j6
```

パッチを当てたClangを使うように`Makefile`を書き換える必要がある。

## 設定

いくつかユーザーによって設定を変えるべき場所がある。

- エポック数・訓練サンプルのバッチサイズ・テストサンプルのバッチサイズ。
    - `main.cpp`の`main`関数冒頭。
    - `constexpr size_t epoch = 1, train_batch_size = 100, test_batch_size = 100;`
- 訓練・テストに使用するMNISTサンプルの数。
    - `mnist.py`
    - `train_sample_num = 15000`
    - `test_sample_num = 15000`
    - 最大60000
- テストをコンパイル時に行うか、実行時に行うか。
    - `main.cpp`
    - `//#define CNSTNN_GET_TEST_ACC_AT_COMPILE_TIME`のコメント指定を外すとコンパイル時に行う。

## あそびかた

- `git clone`
- `git submodule init`
- `git submodule update`
- `python mnist.py`
- `python random_table.py`
- `make`
    - 時間がかかる可能性があるので注意。他の仕事をすませると良い。散歩するとか。
    - Intel Core i7-3630QMで、8000サンプル（train 4000 + test 4000）を1 epochでおよそ13時間。
- `./main`
    - 訓練時のlossとテスト時のaccuracyが出力される。
        - ただしテスト時のaccuracyを出力させるためには`//#define CNSTNN_GET_TEST_ACC_AT_COMPILE_TIME`のコメントアウトが必要。
    - `./main filename`とすると外部から画像を読み込んで、その画像を判定する。

## License

MIT.

<hr />
